#include "compute/deconstruct_workflow.hpp"

#include "core/codec/codec.hpp"
#include "core/utils/sequence_utils.hpp"
#include "core/utils/threading_utils.hpp"
#include "compute/snarl_finder.hpp"
#include "compute/traversal_query.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace gfaz {
namespace {

using namespace gfaz::tquery;

// Strip a trailing ":<start>-<end>" decimal subrange from a reference name,
// returning the PanSN base (sample#hap#seq). Mirrors how walk reference names
// are formed -- walk_reference_name appends ":start-end" -- so a user-friendly
// "CHM13#0#chrY" resolves the stored "CHM13#0#chrY:0-57227415". Returns the
// input unchanged when there is no such suffix.
std::string strip_trailing_subrange(const std::string &s) {
  const size_t colon = s.rfind(':');
  if (colon == std::string::npos || colon == 0)
    return s;
  const size_t dash = s.find('-', colon + 1);
  if (dash == std::string::npos || colon + 1 == dash || dash + 1 == s.size())
    return s;
  for (size_t i = colon + 1; i < dash; ++i)
    if (!std::isdigit(static_cast<unsigned char>(s[i])))
      return s;
  for (size_t i = dash + 1; i < s.size(); ++i)
    if (!std::isdigit(static_cast<unsigned char>(s[i])))
      return s;
  return s.substr(0, colon);
}

// Parse the <start> of a trailing ":<start>-<end>" subrange (0 if absent). Used
// to place VCF POS in the reference contig's coordinate frame when the reference
// path is a subrange of its contig (e.g. a path name "...:2771644-26682252").
uint64_t trailing_subrange_start(const std::string &s) {
  const std::string base = strip_trailing_subrange(s);
  if (base.size() == s.size())
    return 0; // no subrange suffix
  uint64_t v = 0;
  for (size_t i = base.size() + 1; i < s.size() && s[i] != '-'; ++i)
    v = v * 10 + static_cast<uint64_t>(s[i] - '0');
  return v;
}

// ---------------------------------------------------------------------------
// Segment sequence access (forward storage; reverse handled per node id sign).
// ---------------------------------------------------------------------------
struct SegmentSeqs {
  std::string concat;
  std::vector<uint64_t> offset; // offset[n] = start of node (n+1) in concat
  std::vector<uint32_t> lengths;
  uint32_t num_nodes = 0;

  uint32_t length_of(NodeId signed_node) const {
    const uint32_t n = abs_node_id(signed_node);
    if (n == 0 || n > num_nodes)
      return 0;
    return lengths[n - 1];
  }

  // Append the (uppercased, orientation-resolved) sequence of one node.
  void append(std::string &out, NodeId signed_node) const {
    const uint32_t n = abs_node_id(signed_node);
    if (n == 0 || n > num_nodes)
      return;
    const char *s = concat.data() + offset[n - 1];
    const size_t len = offset[n] - offset[n - 1];
    if (signed_node >= 0) {
      const size_t base = out.size();
      out.append(s, len);
      for (size_t i = base; i < out.size(); ++i)
        out[i] = static_cast<char>(std::toupper(static_cast<unsigned char>(out[i])));
    } else {
      std::string sub(s, len);
      for (char &c : sub)
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
      reverse_complement_inplace(sub);
      out += sub;
    }
  }

  // Last base of a node as seen along its traversal orientation.
  char last_base(NodeId signed_node) const {
    const uint32_t n = abs_node_id(signed_node);
    if (n == 0 || n > num_nodes)
      return 'N';
    const char *s = concat.data() + offset[n - 1];
    const size_t len = offset[n] - offset[n - 1];
    if (len == 0)
      return 'N';
    if (signed_node >= 0)
      return static_cast<char>(std::toupper(static_cast<unsigned char>(s[len - 1])));
    return complement_base(
        static_cast<char>(std::toupper(static_cast<unsigned char>(s[0]))));
  }
};

SegmentSeqs load_segments(const CompressedData &data) {
  SegmentSeqs seg;
  seg.concat = Codec::zstd_decompress_string(data.segment_sequences_zstd);
  seg.lengths = Codec::zstd_decompress_uint32_vector(data.segment_seq_lengths_zstd);
  seg.num_nodes = static_cast<uint32_t>(seg.lengths.size());
  seg.offset.assign(seg.num_nodes + 1, 0);
  for (uint32_t i = 0; i < seg.num_nodes; ++i)
    seg.offset[i + 1] = seg.offset[i] + seg.lengths[i];
  if (seg.offset[seg.num_nodes] != seg.concat.size())
    throw std::runtime_error(
        "deconstruct: segment sequence length sum does not match payload");
  return seg;
}

// Per-slice identity used to form VCF sample columns.
struct SliceIdentity {
  std::vector<std::string> names;     // reference-lookup name per slice
  std::vector<std::string> samples;   // grouping key per slice
  std::vector<uint32_t> haps;         // haplotype index per slice
  std::vector<uint64_t> ref_starts;   // 0-based subrange start in the contig
  size_t num_paths = 0;
};

SliceIdentity build_slice_identity(const CompressedData &data,
                                   GroupingMode grouping) {
  SliceIdentity id;
  const size_t num_paths = data.sequence_lengths.size();
  const size_t num_walks = data.walk_lengths.size();
  id.num_paths = num_paths;
  id.names.reserve(num_paths + num_walks);
  id.samples.reserve(num_paths + num_walks);
  id.haps.reserve(num_paths + num_walks);
  id.ref_starts.reserve(num_paths + num_walks);

  if (num_paths) {
    std::vector<std::string> path_names =
        load_path_names(data, num_paths, "deconstruct");
    for (const std::string &name : path_names) {
      const PansnParts p = parse_pansn_path_name(name);
      uint32_t hap = 0;
      if (p.has_hap) {
        try {
          hap = static_cast<uint32_t>(std::stoul(p.hap));
        } catch (...) {
          hap = 0;
        }
      }
      id.names.push_back(name);
      id.samples.push_back(path_group_key(name, grouping));
      id.haps.push_back(hap);
      // P-lines carry no structured subrange; honor one encoded in the name.
      id.ref_starts.push_back(trailing_subrange_start(name));
    }
  }

  if (num_walks) {
    const WalkIdentityColumns w =
        load_walk_identity(data, num_walks, "deconstruct");
    for (size_t i = 0; i < num_walks; ++i) {
      id.names.push_back(walk_reference_name(w.samples[i], w.haps[i], w.seqs[i],
                                             w.starts[i], w.ends[i]));
      id.samples.push_back(
          walk_group_key(w.samples[i], w.haps[i], w.seqs[i], grouping));
      id.haps.push_back(w.haps[i]);
      // W-line seqStart is authoritative; a sentinel (-1) means "from 0".
      id.ref_starts.push_back(w.starts[i] > 0 ? static_cast<uint64_t>(w.starts[i])
                                              : 0);
    }
  }

  return id;
}

struct DecodedHaplotype {
  std::vector<NodeId> nodes;
  // Ordered (anchor ordinal, position-in-nodes) for same-orientation reference
  // anchors and opposite-orientation anchors, respectively. Opposite anchors are
  // only used when a whole local block runs reverse to the reference.
  std::vector<std::pair<uint32_t, uint32_t>> anchor_hits;
  std::vector<std::pair<uint32_t, uint32_t>> reverse_anchor_hits;
  std::unordered_map<uint32_t, uint32_t> forward_anchor_pos;
  std::unordered_map<uint32_t, uint32_t> reverse_anchor_pos;
};

struct VcfRecord {
  uint64_t pos = 0;
  std::string line;
};

std::string vcf_contig_name_for_reference(const std::string &reference_name) {
  const PansnParts p = parse_pansn_path_name(reference_name);
  if (p.has_seq && !p.seq.empty())
    return p.seq;
  return reference_name;
}

// VCF CHROM / ##contig name for a reference. Default: the bare sequence name
// (e.g. "chr1"). With graph-info (-a), the full PanSN base name including
// sample#hap (e.g. "CHM13#0#chr1", trailing :start-end stripped) for vg parity.
std::string vcf_chrom_name(const std::string &reference_name, bool graph_info) {
  if (graph_info)
    return strip_trailing_subrange(reference_name);
  return vcf_contig_name_for_reference(reference_name);
}

// Append one oriented node to a graph-path string in vg's AT/snarl-id spelling:
// '>' for forward, '<' for reverse, followed by the (1-based) node number. The
// number is gfaz's own node id (the .gfaz node space); original GFA segment
// names are not retained.
void append_oriented_node(std::string &s, NodeId n) {
  s += (n >= 0) ? '>' : '<';
  s += std::to_string(n >= 0 ? n : -n);
}

std::string format_af(uint64_t ac, uint64_t an) {
  if (an == 0)
    return "0";
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%g", static_cast<double>(ac) /
                                            static_cast<double>(an));
  return std::string(buf);
}

// Append the per-column GT genotype fields ("\tGT\t<gt>...") for one VCF record.
// Shared by the linear and snarl contig writers so genotypes are emitted
// identically. Phased '|' join; '.' for missing/conflict; under a megasite
// guard alleles collapse to 0/1.
void append_gt_columns(std::ostringstream &line,
                       const std::vector<std::vector<int>> &column_slot_allele,
                       bool guard, const DeconstructOptions &options) {
  if (!options.emit_gt)
    return;
  line << "\tGT";
  for (const std::vector<int> &slot_alleles : column_slot_allele) {
    line << '\t';
    for (size_t h = 0; h < slot_alleles.size(); ++h) {
      if (h)
        line << '|';
      const int a = slot_alleles[h];
      if (a < 0) // missing or conflict
        line << '.';
      else if (guard)
        line << (a == 0 ? 0 : 1);
      else
        line << a;
    }
  }
}

// Append the AC/AN/AF/NS INFO field for one VCF record (ac is per-allele counts,
// an the total called alleles, ns the sample count). Shared by both contig
// writers; under a megasite guard only the single collapsed alt count is shown.
void append_info_fields(std::ostringstream &line,
                        const std::vector<uint64_t> &ac, uint64_t an,
                        uint64_t ns, bool guard) {
  std::ostringstream info;
  info << "AC=";
  for (size_t a = 1; a < ac.size(); ++a) {
    if (a > 1)
      info << ',';
    info << (guard ? an - ac[0] : ac[a]);
    if (guard)
      break;
  }
  info << ";AN=" << an << ";AF=";
  for (size_t a = 1; a < ac.size(); ++a) {
    if (a > 1)
      info << ',';
    info << format_af(guard ? an - ac[0] : ac[a], an);
    if (guard)
      break;
  }
  info << ";NS=" << ns;
  line << info.str();
}

// Write the CHROM/POS/ID/REF/ALT/QUAL/FILTER head of one VCF record into `line`
// (through the trailing "\t.\t.\t" before INFO) and return POS. `prev` is the
// reference base preceding the site, used to left-anchor indels; under a
// megasite guard the record collapses to a single symbolic <CPX> alt. Shared by
// both contig writers so REF/ALT are formatted identically.
uint64_t write_record_head(std::ostringstream &line,
                           const std::string &contig_name, bool guard,
                           bool substitution, uint64_t interior_start, char prev,
                           const std::string &ref_allele,
                           const std::vector<std::string> &alleles,
                           uint64_t ref_start, const std::string &id_field = ".") {
  uint64_t pos;
  std::string ref_field;
  std::vector<std::string> alt_fields;
  if (guard) {
    // Megasite: collapse to a single symbolic record.
    pos = substitution ? interior_start + 1 : interior_start;
    ref_field = substitution
                    ? (ref_allele.empty() ? std::string("N") : ref_allele)
                    : std::string(1, prev);
    alt_fields.push_back("<CPX>");
  } else if (substitution) {
    pos = interior_start + 1;
    ref_field = ref_allele;
    for (size_t a = 1; a < alleles.size(); ++a)
      alt_fields.push_back(alleles[a]);
  } else {
    pos = interior_start; // anchor base position (1-based)
    ref_field = std::string(1, prev) + ref_allele;
    for (size_t a = 1; a < alleles.size(); ++a)
      alt_fields.push_back(std::string(1, prev) + alleles[a]);
  }

  // Shift into the reference contig's coordinate frame when the reference path
  // is a subrange of its contig (ref_start == 0 for whole-contig references).
  pos += ref_start;

  line << contig_name << '\t' << pos << '\t' << id_field << '\t' << ref_field
       << '\t';
  for (size_t a = 0; a < alt_fields.size(); ++a) {
    if (a)
      line << ',';
    line << alt_fields[a];
  }
  line << "\t.\t.\t";
  return pos;
}

// Deconstruct one reference contig. Appends VCF records to `records`, returns
// the contig length (sum of reference segment lengths).
uint64_t deconstruct_contig(
    const SegmentSeqs &seg,
    const std::vector<HapSlice> &slices, uint32_t ref_slice,
    const std::vector<uint32_t> &sample_slices, // non-ref slices to decode
    // columns[c][slot] = local slice ids covering haplotype `slot` of column c
    const std::vector<std::vector<std::vector<uint32_t>>> &columns,
    const std::vector<int32_t> &rules_first,
    const std::vector<int32_t> &rules_second, const RuleLeafCache &rule_cache,
    uint32_t min_rule_id, uint32_t max_rule_id, int delta_round,
    const DeconstructOptions &options, const std::string &contig_name,
    uint64_t ref_start, std::vector<VcfRecord> &records) {
  // --- Pass 1: reference profile ---
  std::vector<NodeId> ref_nodes;
  {
    std::vector<NodeId> scratch;
    auto visit = [&](NodeId node) { ref_nodes.push_back(node); };
    stream_decoded_nodes(slices[ref_slice], delta_round, min_rule_id,
                         max_rule_id, rules_first, rules_second, rule_cache,
                         scratch, visit);
  }

  const size_t ref_len = ref_nodes.size();
  std::vector<uint64_t> offset_at(ref_len + 1, 0);
  for (size_t i = 0; i < ref_len; ++i)
    offset_at[i + 1] = offset_at[i] + seg.length_of(ref_nodes[i]);
  // Contig length spans [0, ref_start + walk_length): the subrange end, so the
  // declared length stays >= every emitted POS.
  const uint64_t contig_length = ref_start + offset_at[ref_len];

  // Anchors = reference nodes occurring exactly once. Ordinal = reference order.
  std::unordered_map<uint32_t, int64_t> occ; // abs id -> ref index or -1 if dup
  occ.reserve(ref_len * 2 + 1);
  for (size_t i = 0; i < ref_len; ++i) {
    const uint32_t id = abs_node_id(ref_nodes[i]);
    auto it = occ.find(id);
    if (it == occ.end())
      occ.emplace(id, static_cast<int64_t>(i));
    else
      it->second = -1;
  }
  std::vector<uint32_t> anchor_ref_index; // ordinal -> ref index
  std::unordered_map<NodeId, uint32_t> anchor_node_to_ord;
  std::unordered_map<NodeId, uint32_t> reverse_anchor_node_to_ord;
  anchor_ref_index.reserve(ref_len);
  anchor_node_to_ord.reserve(ref_len * 2 + 1);
  reverse_anchor_node_to_ord.reserve(ref_len * 2 + 1);
  for (size_t i = 0; i < ref_len; ++i) {
    const uint32_t id = abs_node_id(ref_nodes[i]);
    auto it = occ.find(id);
    if (it != occ.end() && it->second == static_cast<int64_t>(i)) {
      const uint32_t ord = static_cast<uint32_t>(anchor_ref_index.size());
      anchor_node_to_ord.emplace(ref_nodes[i], ord);
      reverse_anchor_node_to_ord.emplace(static_cast<NodeId>(-ref_nodes[i]),
                                         ord);
      anchor_ref_index.push_back(static_cast<uint32_t>(i));
    }
  }
  const uint32_t num_anchors = static_cast<uint32_t>(anchor_ref_index.size());
  if (num_anchors < 2)
    return contig_length; // nothing to call against

  // --- Pass 2a: decode each non-ref slice, extract anchor subsequence ---
  const size_t num_samples_slices = sample_slices.size();
  std::vector<DecodedHaplotype> decoded(num_samples_slices);

  {
    ScopedOMPThreads omp_scope(options.num_threads);
    const int T = std::max(1, omp_scope.effective_threads());
#pragma omp parallel num_threads(T)
    {
      std::vector<NodeId> scratch;
#pragma omp for schedule(dynamic, 16)
      for (long long si = 0; si < static_cast<long long>(num_samples_slices);
           ++si) {
        const uint32_t slice_id = sample_slices[si];
        DecodedHaplotype &dh = decoded[si];
        dh.nodes.clear();
        auto visit = [&](NodeId node) { dh.nodes.push_back(node); };
        stream_decoded_nodes(slices[slice_id], delta_round, min_rule_id,
                             max_rule_id, rules_first, rules_second, rule_cache,
                             scratch, visit);
        for (uint32_t pos = 0; pos < dh.nodes.size(); ++pos) {
          auto it = anchor_node_to_ord.find(dh.nodes[pos]);
          if (it != anchor_node_to_ord.end()) {
            dh.anchor_hits.emplace_back(it->second, pos);
            continue;
          }
          auto rit = reverse_anchor_node_to_ord.find(dh.nodes[pos]);
          if (rit != reverse_anchor_node_to_ord.end())
            dh.reverse_anchor_hits.emplace_back(rit->second, pos);
        }
        dh.forward_anchor_pos.reserve(dh.anchor_hits.size() * 2 + 1);
        for (size_t k = 1; k < dh.anchor_hits.size(); ++k) {
          const auto prev = dh.anchor_hits[k - 1];
          const auto cur = dh.anchor_hits[k];
          if (cur.first > prev.first) {
            dh.forward_anchor_pos.emplace(prev.first, prev.second);
            dh.forward_anchor_pos.emplace(cur.first, cur.second);
          }
        }
        dh.reverse_anchor_pos.reserve(dh.reverse_anchor_hits.size() * 2 + 1);
        for (size_t k = 1; k < dh.reverse_anchor_hits.size(); ++k) {
          const auto prev = dh.reverse_anchor_hits[k - 1];
          const auto cur = dh.reverse_anchor_hits[k];
          if (cur.first < prev.first) {
            dh.reverse_anchor_pos.emplace(prev.first, prev.second);
            dh.reverse_anchor_pos.emplace(cur.first, cur.second);
          }
        }
      }
    }
  }

  // --- Pass 2b: breakpoints = anchors no haplotype bridges ---
  std::vector<uint8_t> skipped(num_anchors, 0);
  auto mark_skipped_between = [&](
      const std::vector<std::pair<uint32_t, uint32_t>> &hits) {
    for (size_t k = 1; k < hits.size(); ++k) {
      const uint32_t o_prev = hits[k - 1].first;
      const uint32_t o_cur = hits[k].first;
      const uint32_t lo = std::min(o_prev, o_cur);
      const uint32_t hi = std::max(o_prev, o_cur);
      if (lo == hi)
        continue;
      for (uint32_t o = lo + 1; o < hi; ++o)
        skipped[o] = 1;
    }
  };
  for (const DecodedHaplotype &dh : decoded) {
    mark_skipped_between(dh.anchor_hits);
    mark_skipped_between(dh.reverse_anchor_hits);
  }
  std::vector<uint32_t> breakpoints; // ordinals, ascending
  breakpoints.reserve(num_anchors);
  for (uint32_t o = 0; o < num_anchors; ++o)
    if (!skipped[o])
      breakpoints.push_back(o);
  if (breakpoints.size() < 2)
    return contig_length;

  const size_t num_columns = columns.size();

  // Resolve the allele of one haplotype slot from its covering slices:
  //   -1 = missing (no covering slice spans this site)
  //   -2 = conflict (covering slices disagree on the allele)
  auto resolve_slot = [](const std::vector<uint32_t> &slot,
                         const std::vector<int> &slice_allele) -> int {
    int allele = -1;
    for (uint32_t local : slot) {
      const int a = slice_allele[local];
      if (a < 0)
        continue;
      if (allele < 0)
        allele = a;
      else if (allele != a)
        return -2;
    }
    return allele;
  };

  // --- Pass 3 + assembly: one record per varying site ---
  for (size_t b = 0; b + 1 < breakpoints.size(); ++b) {
    const uint32_t src_ord = breakpoints[b];
    const uint32_t sink_ord = breakpoints[b + 1];
    const uint32_t src_ref_index = anchor_ref_index[src_ord];
    const uint32_t sink_ref_index = anchor_ref_index[sink_ord];

    // Reference interior allele string.
    std::string ref_allele;
    for (uint32_t i = src_ref_index + 1; i < sink_ref_index; ++i)
      seg.append(ref_allele, ref_nodes[i]);

    // Map allele string -> allele index. Reference is allele 0.
    std::map<std::string, int> allele_index;
    allele_index.emplace(ref_allele, 0);
    std::vector<std::string> alleles; // index -> string
    alleles.push_back(ref_allele);

    // Allele assigned to each sample slice (-1 == missing/no span/complex).
    std::vector<int> slice_allele(num_samples_slices, -1);

    for (size_t si = 0; si < num_samples_slices; ++si) {
      const DecodedHaplotype &dh = decoded[si];
      std::string allele;
      auto fs = dh.forward_anchor_pos.find(src_ord);
      auto fe = dh.forward_anchor_pos.find(sink_ord);
      if (fs != dh.forward_anchor_pos.end() &&
          fe != dh.forward_anchor_pos.end() && fs->second < fe->second) {
        for (uint32_t p = fs->second + 1; p < fe->second; ++p)
          seg.append(allele, dh.nodes[p]);
      } else {
        auto rs = dh.reverse_anchor_pos.find(src_ord);
        auto re = dh.reverse_anchor_pos.find(sink_ord);
        if (rs == dh.reverse_anchor_pos.end() ||
            re == dh.reverse_anchor_pos.end() || re->second >= rs->second)
          continue; // does not span this site
        for (uint32_t p = rs->second; p-- > re->second + 1;)
          seg.append(allele, static_cast<NodeId>(-dh.nodes[p]));
      }
      auto ai = allele_index.find(allele);
      int idx;
      if (ai == allele_index.end()) {
        idx = static_cast<int>(alleles.size());
        allele_index.emplace(allele, idx);
        alleles.push_back(allele);
      } else {
        idx = ai->second;
      }
      slice_allele[si] = idx;
    }

    if (alleles.size() < 2)
      continue; // no variation at this site

    // Substitution iff every allele has the same length as the reference.
    bool substitution = true;
    for (size_t a = 1; a < alleles.size() && substitution; ++a)
      substitution = (alleles[a].size() == ref_allele.size());

    const uint64_t interior_start = offset_at[src_ref_index + 1];
    const uint64_t span =
        offset_at[sink_ref_index] - offset_at[src_ref_index + 1];
    const bool guard =
        options.max_site_length != 0 && span > options.max_site_length;

    // Resolve each haplotype slot to a single allele, then count per slot
    // (not per slice) so a sample's tiling contigs collapse to its ploidy.
    std::vector<std::vector<int>> column_slot_allele(num_columns);
    std::vector<uint64_t> ac(alleles.size(), 0);
    uint64_t an = 0;
    uint64_t ns = 0;
    for (size_t c = 0; c < num_columns; ++c) {
      column_slot_allele[c].reserve(columns[c].size());
      bool has = false;
      for (const auto &slot : columns[c]) {
        const int a = resolve_slot(slot, slice_allele);
        column_slot_allele[c].push_back(a);
        if (a >= 0) {
          ++an;
          ++ac[a];
          has = true;
        }
      }
      if (has)
        ++ns;
    }

    const char prev = seg.last_base(ref_nodes[src_ref_index]);
    std::ostringstream line;
    const uint64_t pos =
        write_record_head(line, contig_name, guard, substitution, interior_start,
                          prev, ref_allele, alleles, ref_start);

    // INFO
    append_info_fields(line, ac, an, ns, guard);

    append_gt_columns(line, column_slot_allele, guard, options);

    records.push_back(VcfRecord{pos, line.str()});
  }

  return contig_length;
}

// ---------------------------------------------------------------------------
// Snarl-based deconstruction (topology-driven sites).
//
// Sites are superbubbles enumerated from the stored L-line links (see
// snarl_finder.cpp), projected onto the reference for coordinates. Alleles are
// observed by streaming each sample traversal exactly once: a small state
// machine captures only the interior between a snarl's boundary node-sides
// (forward or, for inversions, reversed) -- whole paths are never retained.
// ---------------------------------------------------------------------------

// One captured interior: the node stretch a sample spelled through a snarl,
// already normalized to the reference (forward) orientation. The interior nodes
// live in a single shared flat pool (interior_pool); each record just points at
// its slice [off, off + len). Storing observations as fixed 16-byte records over
// a flat pool -- rather than a std::vector<NodeId> per observation -- avoids one
// heap allocation per observation. At chromosome scale there is roughly one
// observation per (sample x snarl): hundreds of millions of them, with an
// average interior of ~1 node, so the per-vector control block + malloc chunk
// dominated peak RSS.
struct ObsRec {
  uint32_t snarl = 0;
  uint32_t slice_local = 0;
  uint32_t off = 0; // start index in interior_pool
  uint32_t len = 0; // number of interior nodes
};

uint64_t deconstruct_contig_snarl(
    const SegmentSeqs &seg, const DoubledGraph &g, const SegmentGraph &seg_graph,
    const std::vector<HapSlice> &slices, uint32_t ref_slice,
    const std::vector<uint32_t> &sample_slices,
    const std::vector<std::vector<std::vector<uint32_t>>> &columns,
    const std::vector<int32_t> &rules_first,
    const std::vector<int32_t> &rules_second, const RuleLeafCache &rule_cache,
    uint32_t min_rule_id, uint32_t max_rule_id, int delta_round,
    const DeconstructOptions &options, const std::string &contig_name,
    uint64_t ref_start, std::vector<VcfRecord> &records) {
  // --- Pass 1: reference profile ---
  std::vector<NodeId> ref_nodes;
  {
    std::vector<NodeId> scratch;
    auto visit = [&](NodeId node) { ref_nodes.push_back(node); };
    stream_decoded_nodes(slices[ref_slice], delta_round, min_rule_id,
                         max_rule_id, rules_first, rules_second, rule_cache,
                         scratch, visit);
  }
  const size_t ref_len = ref_nodes.size();
  std::vector<uint64_t> offset_at(ref_len + 1, 0);
  for (size_t i = 0; i < ref_len; ++i)
    offset_at[i + 1] = offset_at[i] + seg.length_of(ref_nodes[i]);
  // Contig length spans [0, ref_start + walk_length): the subrange end, so the
  // declared length stays >= every emitted POS.
  const uint64_t contig_length = ref_start + offset_at[ref_len];

  // --- Pass 2: topology-based snarls projected on the reference ---
  // vg-compat uses the global biconnected decomposition (top-level snarls,
  // matching vg's default granularity, with cyclic-reference blocks dropped).
  // Default --snarl keeps the leaf-superbubble superset.
  std::vector<ReferenceSnarl> snarls =
      options.vg_compat ? find_reference_snarls_top_level(seg_graph, ref_nodes)
                        : find_reference_snarls(g, ref_nodes);

  const size_t num_snarls = snarls.size();
  if (num_snarls == 0)
    return contig_length;

  // Boundary lookup for on-the-fly capture. Forward entrance/exit are the
  // oriented boundary node-sides; reversed entrance/exit handle a sample that
  // traverses the snarl on the opposite strand (inversion).
  std::unordered_map<uint32_t, uint32_t> fwd_entrance; // vertex -> snarl idx
  std::unordered_map<uint32_t, uint32_t> rev_entrance;
  std::vector<uint32_t> fwd_exit(num_snarls), rev_exit(num_snarls);
  fwd_entrance.reserve(num_snarls * 2 + 1);
  rev_entrance.reserve(num_snarls * 2 + 1);
  for (uint32_t s = 0; s < num_snarls; ++s) {
    fwd_entrance.emplace(DoubledGraph::vid(snarls[s].start_node), s);
    fwd_exit[s] = DoubledGraph::vid(snarls[s].end_node);
    rev_entrance.emplace(DoubledGraph::vid(-snarls[s].end_node), s);
    rev_exit[s] = DoubledGraph::vid(-snarls[s].start_node);
  }

  // --- Pass 3: observe alleles, streaming each sample slice once ---
  const size_t num_samples_slices = sample_slices.size();
  std::vector<ObsRec> observations;
  std::vector<NodeId> interior_pool;
  {
    ScopedOMPThreads omp_scope(options.num_threads);
    const int T = std::max(1, omp_scope.effective_threads());
    std::vector<std::vector<ObsRec>> per_thread(T);
    std::vector<std::vector<NodeId>> per_thread_pool(T);
#pragma omp parallel num_threads(T)
    {
      const int tid =
#ifdef _OPENMP
          omp_get_thread_num();
#else
          0;
#endif
      std::vector<ObsRec> &local = per_thread[tid];
      std::vector<NodeId> &pool = per_thread_pool[tid];
      std::vector<NodeId> scratch;
#pragma omp for schedule(dynamic, 16)
      for (long long si = 0; si < static_cast<long long>(num_samples_slices);
           ++si) {
        const uint32_t slice_id = sample_slices[si];
        uint32_t open = DoubledGraph::kInvalid;
        bool reversed = false;
        uint32_t exit_v = 0;
        std::vector<NodeId> interior;
        auto open_at = [&](uint32_t v) {
          auto f = fwd_entrance.find(v);
          if (f != fwd_entrance.end()) {
            open = f->second;
            reversed = false;
            exit_v = fwd_exit[open];
            interior.clear();
            return;
          }
          auto r = rev_entrance.find(v);
          if (r != rev_entrance.end()) {
            open = r->second;
            reversed = true;
            exit_v = rev_exit[open];
            interior.clear();
          }
        };
        auto visit = [&](NodeId node) {
          const uint32_t v = DoubledGraph::vid(node);
          if (open == DoubledGraph::kInvalid) {
            open_at(v);
            return;
          }
          if (v == exit_v) {
            // Append this interior to the thread's flat pool (normalized to the
            // reference / forward orientation) and record only its extent.
            const uint32_t off = static_cast<uint32_t>(pool.size());
            if (!reversed) {
              pool.insert(pool.end(), interior.begin(), interior.end());
            } else {
              for (auto it = interior.rbegin(); it != interior.rend(); ++it)
                pool.push_back(static_cast<NodeId>(-*it));
            }
            local.push_back(ObsRec{open, static_cast<uint32_t>(si), off,
                                   static_cast<uint32_t>(pool.size() - off)});
            open = DoubledGraph::kInvalid;
            interior.clear();
            // The exit boundary may itself open the next (chained) snarl.
            open_at(v);
            return;
          }
          interior.push_back(node);
        };
        stream_decoded_nodes(slices[slice_id], delta_round, min_rule_id,
                             max_rule_id, rules_first, rules_second, rule_cache,
                             scratch, visit);
        // A slice still "open" at end never reached the exit: no clean span.
      }
    }
    // Merge per-thread records and interior pools into the two shared flat
    // arrays, draining and freeing each thread's buffers as we go so that the
    // per-thread and merged copies never both reside in memory at full size.
    size_t total_recs = 0, total_pool = 0;
    for (int t = 0; t < T; ++t) {
      total_recs += per_thread[t].size();
      total_pool += per_thread_pool[t].size();
    }
    // off/len index the interior pool; the observation index k (below) and the
    // obs_by_snarl entries index the observations array. Both are 32-bit, so a
    // contig whose observation count (~haplotypes x snarls-spanned) or interior
    // pool would exceed 2^32 is rejected cleanly rather than silently wrapping.
    // Current inputs sit ~30x under this (chr1: ~142M); widen these indices to
    // 64-bit only when a real input needs it (doubles obs_by_snarl).
    if (total_pool > std::numeric_limits<uint32_t>::max())
      throw std::runtime_error(
          "deconstruct: snarl interior pool exceeds 2^32 nodes; "
          "widen ObsRec off/len to 64-bit");
    if (total_recs > std::numeric_limits<uint32_t>::max())
      throw std::runtime_error(
          "deconstruct: snarl observation count exceeds 2^32; "
          "widen the observation index / obs_by_snarl to 64-bit");
    observations.reserve(total_recs);
    interior_pool.reserve(total_pool);
    for (int t = 0; t < T; ++t) {
      const uint32_t base = static_cast<uint32_t>(interior_pool.size());
      std::vector<NodeId> &pl = per_thread_pool[t];
      interior_pool.insert(interior_pool.end(), pl.begin(), pl.end());
      std::vector<NodeId>().swap(pl);
      std::vector<ObsRec> &rc = per_thread[t];
      for (ObsRec r : rc) {
        r.off += base;
        observations.push_back(r);
      }
      std::vector<ObsRec>().swap(rc);
    }
  }

  // Group observations by snarl for per-site assembly. The allele numbering
  // assigned below is first-seen order over each snarl's observations, so that
  // order must not depend on Pass 3's thread scheduling. Sort each snarl's
  // observations by their source slice; this reproduces the single-thread order
  // (Pass 3 streams slices in ascending order, keeping one slice's observations
  // contiguous), making the whole VCF thread-count invariant.
  std::vector<std::vector<uint32_t>> obs_by_snarl(num_snarls);
  for (uint32_t k = 0; k < observations.size(); ++k)
    obs_by_snarl[observations[k].snarl].push_back(k);
  for (auto &ks : obs_by_snarl)
    std::stable_sort(ks.begin(), ks.end(), [&](uint32_t a, uint32_t b) {
      return observations[a].slice_local < observations[b].slice_local;
    });

  const size_t num_columns = columns.size();
  auto resolve_slot = [](const std::vector<uint32_t> &slot,
                         const std::vector<int> &slice_allele) -> int {
    int allele = -1;
    for (uint32_t local : slot) {
      const int a = slice_allele[local];
      if (a < 0)
        continue;
      if (allele < 0)
        allele = a;
      else if (allele != a)
        return -2;
    }
    return allele;
  };

  // --- Pass 4: assemble + emit one record per varying snarl ---
  // Per-snarl iterations are independent: every working buffer is either
  // iteration-local or a thread-private scratch (slice_allele, reset via
  // touched). Records are written to per-thread buffers and merged afterward;
  // the caller re-sorts records by POS, so emission order does not matter.
  {
    ScopedOMPThreads omp_scope(options.num_threads);
    const int T = std::max(1, omp_scope.effective_threads());
    std::vector<std::vector<VcfRecord>> per_thread(T);
#pragma omp parallel num_threads(T)
    {
      const int tid =
#ifdef _OPENMP
          omp_get_thread_num();
#else
          0;
#endif
      std::vector<VcfRecord> &local_records = per_thread[tid];
      // Thread-private allele scratch, reset between snarls via `touched`.
      std::vector<int> slice_allele(num_samples_slices, -1);
#pragma omp for schedule(dynamic, 16)
      for (long long sll = 0; sll < static_cast<long long>(num_snarls); ++sll) {
        const uint32_t s = static_cast<uint32_t>(sll);
        const uint32_t src_ref_index = snarls[s].start_ref_index;
        const uint32_t sink_ref_index = snarls[s].end_ref_index;

        std::string ref_allele;
        for (uint32_t i = src_ref_index + 1; i < sink_ref_index; ++i)
          seg.append(ref_allele, ref_nodes[i]);

        std::map<std::string, int> allele_index;
        allele_index.emplace(ref_allele, 0);
        std::vector<std::string> alleles;
        alleles.push_back(ref_allele);
        // For -a (AT field): the interior-pool extent (off,len) of the first
        // observation that introduced each ALT allele, used to spell its
        // traversal. Index 0 (REF) is a placeholder; the REF traversal comes
        // from ref_nodes. Only populated when emit_at.
        std::vector<std::pair<uint32_t, uint32_t>> allele_src;
        if (options.emit_at)
          allele_src.emplace_back(0u, 0u);

        // Reset only the slots we touch (slice_allele is reused across snarls).
        std::vector<uint32_t> touched;
        touched.reserve(obs_by_snarl[s].size());
        for (uint32_t k : obs_by_snarl[s]) {
          const ObsRec &obs = observations[k];
          std::string allele;
          for (uint32_t i = 0; i < obs.len; ++i)
            seg.append(allele, interior_pool[obs.off + i]);
          auto ai = allele_index.find(allele);
          int idx;
          if (ai == allele_index.end()) {
            idx = static_cast<int>(alleles.size());
            allele_index.emplace(allele, idx);
            alleles.push_back(allele);
            if (options.emit_at)
              allele_src.emplace_back(obs.off, obs.len);
          } else {
            idx = ai->second;
          }
          slice_allele[obs.slice_local] = idx;
          touched.push_back(obs.slice_local);
        }

        if (alleles.size() < 2) {
          for (uint32_t t : touched)
            slice_allele[t] = -1;
          continue;
        }

        bool substitution = true;
        for (size_t a = 1; a < alleles.size() && substitution; ++a)
          substitution = (alleles[a].size() == ref_allele.size());

        const uint64_t interior_start = offset_at[src_ref_index + 1];
        const uint64_t span =
            offset_at[sink_ref_index] - offset_at[src_ref_index + 1];
        const bool guard =
            options.max_site_length != 0 && span > options.max_site_length;

        std::vector<std::vector<int>> column_slot_allele(num_columns);
        std::vector<uint64_t> ac(alleles.size(), 0);
        uint64_t an = 0;
        uint64_t ns = 0;
        for (size_t c = 0; c < num_columns; ++c) {
          column_slot_allele[c].reserve(columns[c].size());
          bool has = false;
          for (const auto &slot : columns[c]) {
            const int a = resolve_slot(slot, slice_allele);
            column_slot_allele[c].push_back(a);
            if (a >= 0) {
              ++an;
              ++ac[a];
              has = true;
            }
          }
          if (has)
            ++ns;
        }

        const char prev = seg.last_base(ref_nodes[src_ref_index]);
        // -a: snarl boundary id (>src>sink in gfaz's node space) for the ID
        // column; otherwise "." (the default).
        std::string id_field = ".";
        if (options.emit_at) {
          id_field.clear();
          append_oriented_node(id_field, ref_nodes[src_ref_index]);
          append_oriented_node(id_field, ref_nodes[sink_ref_index]);
        }
        std::ostringstream line;
        const uint64_t pos = write_record_head(line, contig_name, guard,
                                               substitution, interior_start,
                                               prev, ref_allele, alleles,
                                               ref_start, id_field);
        append_info_fields(line, ac, an, ns, guard);
        // -a: AT (allele traversal), one '>'/'<'-oriented node path per allele
        // (REF first), each = src boundary + interior + sink boundary, in
        // gfaz's 1-based node space. Skipped under the megasite guard, which
        // already collapses the alleles to a single symbolic <CPX> record.
        if (options.emit_at && !guard) {
          line << ";AT=";
          for (size_t a = 0; a < alleles.size(); ++a) {
            if (a)
              line << ',';
            std::string at;
            append_oriented_node(at, ref_nodes[src_ref_index]);
            if (a == 0) {
              for (uint32_t i = src_ref_index + 1; i < sink_ref_index; ++i)
                append_oriented_node(at, ref_nodes[i]);
            } else {
              const auto &ext = allele_src[a];
              for (uint32_t i = 0; i < ext.second; ++i)
                append_oriented_node(at, interior_pool[ext.first + i]);
            }
            append_oriented_node(at, ref_nodes[sink_ref_index]);
            line << at;
          }
        }
        append_gt_columns(line, column_slot_allele, guard, options);
        local_records.push_back(VcfRecord{pos, line.str()});

        for (uint32_t t : touched)
          slice_allele[t] = -1;
      }
    }
    for (auto &v : per_thread)
      for (auto &r : v)
        records.push_back(std::move(r));
  }

  return contig_length;
}

} // namespace

void deconstruct_to_vcf(const CompressedData &data,
                        const DeconstructOptions &options, std::ostream &out) {
  if (options.reference_names.empty() && options.reference_prefixes.empty())
    throw std::runtime_error(
        "deconstruct: at least one reference (-r) or prefix (-P) required");

  SegmentSeqs seg = load_segments(data);

  Rulebook rulebook = load_rulebook(data);
  const std::vector<int32_t> &rules_first = rulebook.rules_first;
  const std::vector<int32_t> &rules_second = rulebook.rules_second;

  LoadedTraversals loaded = load_traversals(data);
  const std::vector<HapSlice> &slices = loaded.slices;

  SliceIdentity ident = build_slice_identity(data, options.grouping);
  if (ident.names.size() != slices.size())
    throw std::runtime_error("deconstruct: slice/identity count mismatch");

  // Resolve reference names to slice ids. Exact match wins; otherwise fall back
  // to a PanSN base-name match that ignores the ":start-end" subrange, so a
  // friendly "CHM13#0#chrY" resolves the stored "CHM13#0#chrY:0-57227415"
  // (vg accepts the base name via -P; gfaz should too).
  std::unordered_map<std::string, uint32_t> name_to_slice;
  std::unordered_map<std::string, std::vector<uint32_t>> base_to_slices;
  name_to_slice.reserve(slices.size() * 2 + 1);
  base_to_slices.reserve(slices.size() * 2 + 1);
  for (uint32_t i = 0; i < slices.size(); ++i) {
    name_to_slice.emplace(ident.names[i], i);
    base_to_slices[strip_trailing_subrange(ident.names[i])].push_back(i);
  }

  std::vector<uint32_t> ref_slices;
  std::vector<std::string> ref_display_names; // parallel to ref_slices (CHROM src)
  std::vector<uint8_t> is_ref(slices.size(), 0);
  auto add_ref = [&](uint32_t slice_id, const std::string &display) {
    if (is_ref[slice_id])
      return; // already selected (dedup across -r / -P)
    is_ref[slice_id] = 1;
    ref_slices.push_back(slice_id);
    ref_display_names.push_back(display);
  };

  // Explicit -r names: exact match, else PanSN base-name fallback.
  for (const std::string &rn : options.reference_names) {
    auto it = name_to_slice.find(rn);
    if (it != name_to_slice.end()) {
      add_ref(it->second, rn);
      continue;
    }
    auto bit = base_to_slices.find(strip_trailing_subrange(rn));
    if (bit == base_to_slices.end())
      throw std::runtime_error("deconstruct: reference path not found: " + rn);
    if (bit->second.size() != 1)
      throw std::runtime_error(
          "deconstruct: reference '" + rn + "' matches " +
          std::to_string(bit->second.size()) +
          " subrange fragments; specify the exact name including :start-end");
    add_ref(bit->second.front(), rn);
  }

  // -P prefixes (vg parity): every stored name beginning with the prefix becomes
  // a reference contig. Slices are scanned in container order for determinism.
  for (const std::string &pfx : options.reference_prefixes) {
    size_t matched = 0;
    for (uint32_t i = 0; i < slices.size(); ++i) {
      if (ident.names[i].rfind(pfx, 0) == 0) { // names[i] starts with pfx
        add_ref(i, ident.names[i]);
        ++matched;
      }
    }
    if (matched == 0)
      throw std::runtime_error(
          "deconstruct: no reference path matches prefix: " + pfx);
  }

  // Each reference contig must be a single path/walk: a contig split into
  // multiple subrange fragments would emit duplicate CHROM blocks. Detect and
  // reject (stitching fragments into one reference is not yet supported).
  {
    std::unordered_map<std::string, std::string> chrom_owner;
    for (size_t r = 0; r < ref_slices.size(); ++r) {
      const std::string chrom =
          vcf_chrom_name(ref_display_names[r], options.emit_at);
      auto ins = chrom_owner.emplace(chrom, ref_display_names[r]);
      if (!ins.second)
        throw std::runtime_error(
            "deconstruct: reference contig '" + chrom +
            "' is split into multiple paths ('" + ins.first->second + "', '" +
            ref_display_names[r] +
            "'); fragmented references are not yet supported");
    }
  }

  // Flatten non-ref slices; assign each a dense local index used by per-site
  // allele arrays.
  std::vector<uint32_t> sample_slices;
  std::unordered_map<uint32_t, uint32_t> global_to_local;
  for (uint32_t i = 0; i < slices.size(); ++i) {
    if (is_ref[i])
      continue;
    global_to_local.emplace(i, static_cast<uint32_t>(sample_slices.size()));
    sample_slices.push_back(i);
  }

  // VCF sample columns. Each column has one or more haplotype slots; each slot
  // lists the (local) slices that cover that haplotype. When grouping by
  // sample, slots correspond to distinct haplotype indices, so a sample's
  // contigs that tile the reference collapse into the correct ploidy rather
  // than inflating it. For SampleHap / PerPathWalk there is a single slot per
  // column. std::map keeps columns and slots in deterministic order.
  const bool slot_by_hap = (options.grouping == GroupingMode::Sample);
  std::map<std::string, std::map<uint32_t, std::vector<uint32_t>>> col_slot_map;
  for (uint32_t i = 0; i < slices.size(); ++i) {
    if (is_ref[i])
      continue;
    const uint32_t slot_key = slot_by_hap ? ident.haps[i] : 0u;
    col_slot_map[ident.samples[i]][slot_key].push_back(global_to_local.at(i));
  }
  std::vector<std::string> column_names;
  std::vector<std::vector<std::vector<uint32_t>>> columns_local;
  column_names.reserve(col_slot_map.size());
  columns_local.reserve(col_slot_map.size());
  for (auto &ckv : col_slot_map) {
    column_names.push_back(ckv.first);
    std::vector<std::vector<uint32_t>> slots;
    slots.reserve(ckv.second.size());
    for (auto &skv : ckv.second)
      slots.push_back(std::move(skv.second));
    columns_local.push_back(std::move(slots));
  }

  // Rule-leaf cache (shared, read-only during decoding). Disabled by default
  // for deconstruct: every sample slice is streamed exactly once, so eagerly
  // expanding the whole rulebook to leaf arrays costs build time + up to a GiB
  // of RAM without paying off. Measured on chr1/chrY, a zero budget is both
  // faster and ~1.2 GiB smaller than the former 1 GiB default, with byte-
  // identical output. Set GFAZ_DECONSTRUCT_RULE_CACHE_BYTES to re-enable it.
  const uint32_t min_rule_id = rulebook.min_rule_id;
  const uint32_t max_rule_id = rulebook.max_rule_id;
  RuleLeafCache rule_cache = make_rule_cache(
      min_rule_id, rules_first, rules_second, "GFAZ_DECONSTRUCT_RULE_CACHE_BYTES",
      /*default_budget=*/0);

  const int delta_round = data.delta_round;

  // Topology-based mode: build the bidirected node-side graph once from links.
  // vg-compat additionally needs the undirected segment graph for the global
  // biconnected (top-level snarl) decomposition.
  DoubledGraph snarl_graph;
  SegmentGraph seg_graph;
  if (options.use_snarls) {
    snarl_graph = build_doubled_graph_from_links(data, seg.num_nodes);
    if (options.vg_compat)
      seg_graph = build_segment_graph_from_links(data, seg.num_nodes);
    if (data.num_links == 0)
      std::cerr << "Warning: deconstruct --snarl: container has no L-line "
                   "links; no snarls can be called.\n";
  }

  // Process each reference contig, collecting records + contig lengths.
  struct ContigOut {
    std::string name;
    uint64_t length;
    std::vector<VcfRecord> records;
  };
  std::vector<ContigOut> contigs;
  contigs.reserve(ref_slices.size());

  for (size_t r = 0; r < ref_slices.size(); ++r) {
    ContigOut co;
    co.name = vcf_chrom_name(ref_display_names[r], options.emit_at);
    const uint64_t ref_start = ident.ref_starts[ref_slices[r]];
    co.length =
        options.use_snarls
            ? deconstruct_contig_snarl(
                  seg, snarl_graph, seg_graph, slices, ref_slices[r],
                  sample_slices, columns_local, rules_first, rules_second,
                  rule_cache, min_rule_id, max_rule_id, delta_round, options,
                  co.name, ref_start, co.records)
            : deconstruct_contig(
                  seg, slices, ref_slices[r], sample_slices, columns_local,
                  rules_first, rules_second, rule_cache, min_rule_id,
                  max_rule_id, delta_round, options, co.name, ref_start,
                  co.records);
    std::sort(co.records.begin(), co.records.end(),
              [](const VcfRecord &a, const VcfRecord &b) {
                return a.pos < b.pos;
              });
    contigs.push_back(std::move(co));
  }

  // --- VCF header ---
  out << "##fileformat=VCFv4.2\n";
  out << "##FILTER=<ID=PASS,Description=\"All filters passed\">\n";
  for (const ContigOut &co : contigs)
    out << "##contig=<ID=" << co.name << ",length=" << co.length << ">\n";
  out << "##INFO=<ID=AC,Number=A,Type=Integer,Description=\"Allele count in "
         "genotypes\">\n";
  out << "##INFO=<ID=AN,Number=1,Type=Integer,Description=\"Total number of "
         "alleles in called genotypes\">\n";
  out << "##INFO=<ID=AF,Number=A,Type=Float,Description=\"Allele frequency\">\n";
  out << "##INFO=<ID=NS,Number=1,Type=Integer,Description=\"Number of samples "
         "with data\">\n";
  if (options.emit_at) {
    out << "##INFO=<ID=AT,Number=R,Type=String,Description=\"Allele Traversal "
           "as path in graph (gfaz 1-based node ids)\">\n";
  }
  if (options.max_site_length != 0) {
    out << "##ALT=<ID=CPX,Description=\"Complex region exceeding "
           "max-site-length\">\n";
  }
  if (options.emit_gt) {
    out << "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n";
  }
  out << "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO";
  if (options.emit_gt) {
    out << "\tFORMAT";
    for (const std::string &name : column_names)
      out << '\t' << name;
  }
  out << '\n';

  for (const ContigOut &co : contigs)
    for (const VcfRecord &rec : co.records)
      out << rec.line << '\n';
}

} // namespace gfaz
