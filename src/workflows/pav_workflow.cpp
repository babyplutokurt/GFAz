#include "workflows/pav_workflow.hpp"

#include "codec/codec.hpp"
#include "utils/threading_utils.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
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

// Bottom-up cache of fully-expanded rule leaf sequences (forward direction).
// A rule is "ready" iff its leaf sequence fits in budget after recursively
// expanding its children. Reverse expansion is derived by iterating the
// cached leaves in reverse and negating each sign.
struct RuleLeafCache {
  std::vector<std::vector<int32_t>> forward;
  std::vector<uint8_t> ready;
  uint32_t min_rule_id = 0;
  uint32_t max_rule_id = 0;
  size_t bytes_used = 0;
  size_t budget_bytes = 0;
};

bool build_rule_recursive(uint32_t rule_id, RuleLeafCache &cache,
                          const std::vector<int32_t> &first,
                          const std::vector<int32_t> &second) {
  const uint32_t idx = rule_id - cache.min_rule_id;
  if (cache.ready[idx])
    return true;

  const int32_t a = first[idx];
  const int32_t b = second[idx];

  std::vector<int32_t> leaves;

  auto append_child = [&](int32_t child) -> bool {
    const uint32_t abs_c = static_cast<uint32_t>(std::abs(child));
    if (abs_c >= cache.min_rule_id && abs_c < cache.max_rule_id) {
      if (!build_rule_recursive(abs_c, cache, first, second))
        return false;
      const uint32_t cidx = abs_c - cache.min_rule_id;
      const std::vector<int32_t> &cl = cache.forward[cidx];
      const size_t projected = leaves.size() + cl.size();
      if (projected * sizeof(int32_t) > cache.budget_bytes)
        return false;
      if (child >= 0) {
        leaves.insert(leaves.end(), cl.begin(), cl.end());
      } else {
        leaves.reserve(projected);
        for (auto it = cl.rbegin(); it != cl.rend(); ++it)
          leaves.push_back(-*it);
      }
    } else {
      leaves.push_back(child);
    }
    return true;
  };

  if (!append_child(a))
    return false;
  if (!append_child(b))
    return false;

  const size_t needed = leaves.size() * sizeof(int32_t);
  if (cache.bytes_used + needed > cache.budget_bytes)
    return false;

  cache.bytes_used += needed;
  cache.forward[idx] = std::move(leaves);
  cache.ready[idx] = 1;
  return true;
}

void build_rule_cache(RuleLeafCache &cache,
                      const std::vector<int32_t> &first,
                      const std::vector<int32_t> &second) {
  if (cache.budget_bytes == 0)
    return;
  for (uint32_t rid = cache.min_rule_id; rid < cache.max_rule_id; ++rid) {
    build_rule_recursive(rid, cache, first, second);
  }
}

size_t resolve_rule_cache_budget() {
  if (const char *env = std::getenv("GFAZ_PAV_RULE_CACHE_BYTES")) {
    char *end = nullptr;
    const long long parsed = std::strtoll(env, &end, 10);
    if (end != env && *end == '\0' && parsed >= 0)
      return static_cast<size_t>(parsed);
  }
  // Default: 1 GiB. Plenty for HPRC-scale grammars; small grammars use less.
  return static_cast<size_t>(1) << 30;
}

template <typename Visitor>
void expand_rule_visit(uint32_t rule_id, bool reverse,
                       const std::vector<int32_t> &first,
                       const std::vector<int32_t> &second, uint32_t min_id,
                       uint32_t max_id, const RuleLeafCache &cache,
                       Visitor &visit) {
  const uint32_t idx = rule_id - min_id;
  if (cache.ready[idx]) {
    const std::vector<int32_t> &leaves = cache.forward[idx];
    if (!reverse) {
      for (int32_t leaf : leaves)
        visit(leaf);
    } else {
      for (auto it = leaves.rbegin(); it != leaves.rend(); ++it)
        visit(static_cast<int32_t>(-*it));
    }
    return;
  }

  const int32_t a = first[idx];
  const int32_t b = second[idx];

  if (!reverse) {
    const uint32_t abs_a = static_cast<uint32_t>(std::abs(a));
    if (abs_a >= min_id && abs_a < max_id)
      expand_rule_visit(abs_a, a < 0, first, second, min_id, max_id, cache,
                        visit);
    else
      visit(a);

    const uint32_t abs_b = static_cast<uint32_t>(std::abs(b));
    if (abs_b >= min_id && abs_b < max_id)
      expand_rule_visit(abs_b, b < 0, first, second, min_id, max_id, cache,
                        visit);
    else
      visit(b);
  } else {
    const uint32_t abs_b = static_cast<uint32_t>(std::abs(b));
    if (abs_b >= min_id && abs_b < max_id)
      expand_rule_visit(abs_b, b >= 0, first, second, min_id, max_id, cache,
                        visit);
    else
      visit(static_cast<int32_t>(-b));

    const uint32_t abs_a = static_cast<uint32_t>(std::abs(a));
    if (abs_a >= min_id && abs_a < max_id)
      expand_rule_visit(abs_a, a >= 0, first, second, min_id, max_id, cache,
                        visit);
    else
      visit(static_cast<int32_t>(-a));
  }
}

template <typename Visitor>
void stream_hap_leaves(const int32_t *encoded, size_t encoded_len,
                       uint32_t min_rule_id, uint32_t max_rule_id,
                       const std::vector<int32_t> &rules_first,
                       const std::vector<int32_t> &rules_second,
                       const RuleLeafCache &cache, Visitor &visit) {
  for (size_t i = 0; i < encoded_len; ++i) {
    const NodeId node = encoded[i];
    const uint32_t abs_id = static_cast<uint32_t>(std::abs(node));
    if (abs_id >= min_rule_id && abs_id < max_rule_id) {
      expand_rule_visit(abs_id, node < 0, rules_first, rules_second,
                        min_rule_id, max_rule_id, cache, visit);
    } else {
      visit(node);
    }
  }
}

void decode_one_haplotype_general(const int32_t *encoded, size_t encoded_len,
                                  uint32_t original_len, int delta_round,
                                  uint32_t min_rule_id, uint32_t max_rule_id,
                                  const std::vector<int32_t> &rules_first,
                                  const std::vector<int32_t> &rules_second,
                                  const RuleLeafCache &cache,
                                  std::vector<NodeId> &decoded) {
  decoded.clear();
  decoded.reserve(original_len);
  auto push = [&](int32_t v) { decoded.push_back(v); };
  stream_hap_leaves(encoded, encoded_len, min_rule_id, max_rule_id,
                    rules_first, rules_second, cache, push);
  for (int r = 0; r < delta_round; ++r) {
    for (size_t i = 1; i < decoded.size(); ++i)
      decoded[i] = decoded[i] + decoded[i - 1];
  }
}

struct HapSlice {
  const int32_t *encoded = nullptr;
  uint32_t enc_len = 0;
  uint32_t orig_len = 0;
};

void build_slices(const std::vector<int32_t> &flat,
                  const std::vector<uint32_t> &lengths,
                  const std::vector<uint32_t> &original_lengths,
                  std::vector<HapSlice> &out) {
  size_t offset = 0;
  for (size_t i = 0; i < lengths.size(); ++i) {
    const uint32_t enc_len = lengths[i];
    const uint32_t orig_len =
        (i < original_lengths.size()) ? original_lengths[i] : enc_len;
    if (offset + enc_len > flat.size())
      throw std::runtime_error("pav: encoded traversal block is truncated");
    out.push_back(HapSlice{flat.data() + offset, enc_len, orig_len});
    offset += enc_len;
  }
}

uint32_t abs_node_id(NodeId node) {
  return static_cast<uint32_t>(node < 0 ? -static_cast<int64_t>(node) : node);
}

std::vector<PavRange> read_bed(const std::string &path) {
  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("pav: failed to open BED file: " + path);

  std::vector<PavRange> ranges;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#')
      continue;
    std::stringstream ss(line);
    PavRange r;
    ss >> r.chrom >> r.start >> r.end;
    if (!ss || r.chrom.empty())
      throw std::runtime_error("pav: malformed BED line: " + line);
    ss >> r.name;
    if (r.name.empty())
      r.name = r.chrom + ":" + std::to_string(r.start) + "-" +
               std::to_string(r.end);
    if (r.end < r.start)
      throw std::runtime_error("pav: BED end before start: " + line);
    ranges.push_back(std::move(r));
  }
  if (ranges.empty())
    throw std::runtime_error("pav: BED file contains no ranges");
  return ranges;
}

void reconstruct_strings(const std::string &concat,
                         const std::vector<uint32_t> &lengths,
                         std::vector<std::string> &out,
                         const char *label) {
  out.clear();
  out.reserve(lengths.size());
  size_t off = 0;
  for (uint32_t len : lengths) {
    if (off + len > concat.size())
      throw std::runtime_error(std::string("pav: truncated ") + label +
                               " string column");
    out.push_back(concat.substr(off, len));
    off += len;
  }
}

std::vector<std::string> decompress_strings(const ZstdCompressedBlock &strings,
                                            const ZstdCompressedBlock &lengths,
                                            const char *label) {
  std::vector<std::string> out;
  reconstruct_strings(Codec::zstd_decompress_string(strings),
                      Codec::zstd_decompress_uint32_vector(lengths), out,
                      label);
  return out;
}

void strip_pansn_coords_inplace(std::string &s) {
  const size_t colon = s.rfind(':');
  if (colon == std::string::npos || colon == 0)
    return;
  const size_t dash = s.find('-', colon + 1);
  if (dash == std::string::npos || colon + 1 == dash || dash + 1 == s.size())
    return;
  for (size_t i = colon + 1; i < dash; ++i)
    if (!std::isdigit(static_cast<unsigned char>(s[i])))
      return;
  for (size_t i = dash + 1; i < s.size(); ++i)
    if (!std::isdigit(static_cast<unsigned char>(s[i])))
      return;
  s.erase(colon);
}

struct PansnParts {
  std::string sample;
  std::string hap;
  std::string seq;
  bool has_hap = false;
  bool has_seq = false;
};

PansnParts parse_pansn_path_name(const std::string &name) {
  PansnParts p;
  const size_t h1 = name.find('#');
  if (h1 == std::string::npos) {
    p.sample = name;
    strip_pansn_coords_inplace(p.sample);
    return p;
  }
  p.sample = name.substr(0, h1);
  const size_t h2 = name.find('#', h1 + 1);
  if (h2 == std::string::npos) {
    p.hap = name.substr(h1 + 1);
    p.has_hap = !p.hap.empty();
    strip_pansn_coords_inplace(p.hap);
    return p;
  }
  p.hap = name.substr(h1 + 1, h2 - h1 - 1);
  p.seq = name.substr(h2 + 1);
  p.has_hap = !p.hap.empty();
  p.has_seq = !p.seq.empty();
  strip_pansn_coords_inplace(p.seq);
  return p;
}

std::string path_group_key(const std::string &name, GroupingMode mode) {
  const PansnParts p = parse_pansn_path_name(name);
  switch (mode) {
  case GroupingMode::Sample:
    return p.sample;
  case GroupingMode::SampleHap:
    return p.has_hap ? (p.sample + "#" + p.hap) : p.sample;
  case GroupingMode::SampleHapSeq:
    if (p.has_hap && p.has_seq)
      return p.sample + "#" + p.hap + "#" + p.seq;
    if (p.has_hap)
      return p.sample + "#" + p.hap;
    return p.sample;
  case GroupingMode::PerPathWalk:
  default:
    return name;
  }
}

std::string walk_group_key(const std::string &sample, uint32_t hap,
                           const std::string &seq, GroupingMode mode) {
  switch (mode) {
  case GroupingMode::Sample:
    return sample;
  case GroupingMode::SampleHap:
    return sample + "#" + std::to_string(hap);
  case GroupingMode::SampleHapSeq:
    return sample + "#" + std::to_string(hap) + "#" + seq;
  case GroupingMode::PerPathWalk:
  default:
    return sample + "#" + std::to_string(hap) + "#" + seq;
  }
}

std::string walk_reference_name(const std::string &sample, uint32_t hap,
                                const std::string &seq, int64_t start,
                                int64_t end) {
  std::string name = sample + "#" + std::to_string(hap) + "#" + seq;
  if (start != -1 || end != -1)
    name += ":" + std::to_string(start) + "-" + std::to_string(end);
  return name;
}

struct TraversalMetadata {
  std::vector<std::string> path_names;
  std::vector<std::string> walk_names;
  std::vector<std::string> group_names;
  // group_of_slice[s] = group id assigned to slice s.
  std::vector<uint32_t> group_of_slice;
  std::vector<std::vector<uint32_t>> groups;
};

TraversalMetadata build_metadata(const CompressedData &data,
                                 GroupingMode grouping) {
  TraversalMetadata meta;
  const size_t num_paths = data.sequence_lengths.size();
  const size_t num_walks = data.walk_lengths.size();
  const size_t total = num_paths + num_walks;

  if (num_paths) {
    meta.path_names = decompress_strings(data.names_zstd, data.name_lengths_zstd,
                                         "path name");
    if (meta.path_names.size() != num_paths)
      throw std::runtime_error("pav: path name count mismatch");
  }

  std::vector<std::string> keys;
  keys.reserve(total);
  for (const std::string &name : meta.path_names)
    keys.push_back(path_group_key(name, grouping));

  if (num_walks) {
    std::vector<std::string> samples =
        decompress_strings(data.walk_sample_ids_zstd,
                           data.walk_sample_id_lengths_zstd, "walk sample");
    std::vector<std::string> seqs =
        decompress_strings(data.walk_seq_ids_zstd,
                           data.walk_seq_id_lengths_zstd, "walk seq");
    std::vector<uint32_t> haps =
        Codec::zstd_decompress_uint32_vector(data.walk_hap_indices_zstd);
    std::vector<int64_t> starts =
        Codec::decompress_varint_int64(data.walk_seq_starts_zstd, num_walks);
    std::vector<int64_t> ends =
        Codec::decompress_varint_int64(data.walk_seq_ends_zstd, num_walks);
    if (samples.size() != num_walks || seqs.size() != num_walks ||
        haps.size() != num_walks || starts.size() != num_walks ||
        ends.size() != num_walks)
      throw std::runtime_error("pav: walk metadata count mismatch");
    meta.walk_names.reserve(num_walks);
    for (size_t i = 0; i < num_walks; ++i) {
      meta.walk_names.push_back(
          walk_reference_name(samples[i], haps[i], seqs[i], starts[i], ends[i]));
      keys.push_back(walk_group_key(samples[i], haps[i], seqs[i], grouping));
    }
  }

  meta.group_of_slice.assign(total, 0);
  std::unordered_map<std::string, uint32_t> key_to_gid;
  key_to_gid.reserve(total * 2 + 1);
  for (uint32_t i = 0; i < static_cast<uint32_t>(total); ++i) {
    const std::string &key = keys[i];
    auto it = key_to_gid.find(key);
    uint32_t gid = 0;
    if (it == key_to_gid.end()) {
      gid = static_cast<uint32_t>(meta.group_names.size());
      key_to_gid.emplace(key, gid);
      meta.group_names.push_back(key);
      meta.groups.emplace_back();
    } else {
      gid = it->second;
    }
    meta.groups[gid].push_back(i);
    meta.group_of_slice[i] = gid;
  }

  return meta;
}

template <typename Visitor>
void stream_decoded_nodes(const HapSlice &slice, int delta_round,
                          uint32_t min_rule_id, uint32_t max_rule_id,
                          const std::vector<int32_t> &rules_first,
                          const std::vector<int32_t> &rules_second,
                          const RuleLeafCache &cache,
                          std::vector<NodeId> &decoded, Visitor &visit) {
  if (delta_round == 0) {
    auto leaf_visit = [&](int32_t leaf) { visit(leaf); };
    stream_hap_leaves(slice.encoded, slice.enc_len, min_rule_id, max_rule_id,
                      rules_first, rules_second, cache, leaf_visit);
  } else if (delta_round == 1) {
    int32_t prev = 0;
    auto leaf_visit = [&](int32_t leaf) {
      prev += leaf;
      visit(prev);
    };
    stream_hap_leaves(slice.encoded, slice.enc_len, min_rule_id, max_rule_id,
                      rules_first, rules_second, cache, leaf_visit);
  } else {
    decode_one_haplotype_general(slice.encoded, slice.enc_len, slice.orig_len,
                                 delta_round, min_rule_id, max_rule_id,
                                 rules_first, rules_second, cache, decoded);
    for (NodeId node : decoded)
      visit(node);
  }
}

} // namespace

PavResult compute_pav(const CompressedData &data, const PavOptions &options) {
  PavResult result;
  result.ranges = read_bed(options.bed_path);

  const std::vector<uint32_t> segment_lengths =
      Codec::zstd_decompress_uint32_vector(data.segment_seq_lengths_zstd);
  const uint32_t num_nodes = static_cast<uint32_t>(segment_lengths.size());

  std::vector<int32_t> rules_first =
      Codec::zstd_decompress_int32_vector(data.rules_first_zstd);
  std::vector<int32_t> rules_second =
      Codec::zstd_decompress_int32_vector(data.rules_second_zstd);
  Codec::delta_decode_int32(rules_first);
  Codec::delta_decode_int32(rules_second);

  std::vector<int32_t> paths_flat =
      Codec::zstd_decompress_int32_vector(data.paths_zstd);
  std::vector<int32_t> walks_flat =
      Codec::zstd_decompress_int32_vector(data.walks_zstd);

  std::vector<HapSlice> slices;
  slices.reserve(data.sequence_lengths.size() + data.walk_lengths.size());
  build_slices(paths_flat, data.sequence_lengths, data.original_path_lengths,
               slices);
  build_slices(walks_flat, data.walk_lengths, data.original_walk_lengths,
               slices);

  TraversalMetadata meta = build_metadata(data, options.grouping);
  result.group_names = meta.group_names;
  if (result.group_names.empty())
    return result;

  const uint32_t min_rule_id = data.min_rule_id();
  const uint32_t max_rule_id =
      min_rule_id + static_cast<uint32_t>(rules_first.size());
  const int delta_round = data.delta_round;

  // Tier 2 #5: bottom-up rule-leaf cache. Built single-threaded; read-only
  // during slice decoding so no synchronisation is needed.
  RuleLeafCache rule_cache;
  rule_cache.min_rule_id = min_rule_id;
  rule_cache.max_rule_id = max_rule_id;
  rule_cache.budget_bytes = resolve_rule_cache_budget();
  rule_cache.forward.assign(rules_first.size(), {});
  rule_cache.ready.assign(rules_first.size(), 0);
  build_rule_cache(rule_cache, rules_first, rules_second);

  // Map BED chrom -> reference slice id.
  std::unordered_map<std::string, uint32_t> path_name_to_slice;
  path_name_to_slice.reserve((meta.path_names.size() + meta.walk_names.size()) *
                                 2 +
                             1);
  for (uint32_t i = 0; i < static_cast<uint32_t>(meta.path_names.size()); ++i)
    path_name_to_slice.emplace(meta.path_names[i], i);
  const uint32_t walk_slice_offset =
      static_cast<uint32_t>(meta.path_names.size());
  for (uint32_t i = 0; i < static_cast<uint32_t>(meta.walk_names.size()); ++i)
    path_name_to_slice.emplace(meta.walk_names[i], walk_slice_offset + i);

  std::unordered_map<std::string, std::vector<uint32_t>> ranges_by_chrom;
  for (uint32_t i = 0; i < static_cast<uint32_t>(result.ranges.size()); ++i) {
    ranges_by_chrom[result.ranges[i].chrom].push_back(i);
  }

  const size_t num_windows = result.ranges.size();
  const size_t num_groups = result.group_names.size();
  result.denominators.assign(num_windows, 0);
  result.numerators.assign(num_windows * num_groups, 0);

  // Validate every BED chrom resolves to a slice; collect (chrom, slice_id)
  // and a per-slice flag identifying reference targets.
  const size_t num_slices = slices.size();
  std::vector<std::vector<uint32_t> *> slice_to_ref_stream(num_slices, nullptr);
  std::vector<std::vector<uint32_t>> ref_streams; // one per distinct chrom
  ref_streams.reserve(ranges_by_chrom.size());
  std::vector<std::pair<std::string, uint32_t>> chrom_list;
  chrom_list.reserve(ranges_by_chrom.size());

  for (const auto &entry : ranges_by_chrom) {
    auto pit = path_name_to_slice.find(entry.first);
    if (pit == path_name_to_slice.end())
      throw std::runtime_error("pav: BED reference path not found: " +
                               entry.first);
    chrom_list.emplace_back(entry.first, pit->second);
  }
  ref_streams.resize(chrom_list.size());
  for (size_t i = 0; i < chrom_list.size(); ++i) {
    slice_to_ref_stream[chrom_list[i].second] = &ref_streams[i];
  }

  // -------------------------------------------------------------------------
  // Pass 1 (Tier 1 #1, #3): parallel over slices, lock-free.
  //   - For every slice produce a sorted-unique list of visited node ids.
  //   - For reference slices additionally emit the ordered node-id stream so
  //     the BED sweep in pass 3 needs no further rule expansion.
  // -------------------------------------------------------------------------
  std::vector<std::vector<uint32_t>> slice_nodes(num_slices);

  {
    ScopedOMPThreads omp_scope(options.num_threads);
    const int T = std::max(1, omp_scope.effective_threads());

#pragma omp parallel num_threads(T)
    {
      std::vector<NodeId> local_decoded;
      std::vector<uint32_t> local_nodes;

#pragma omp for schedule(dynamic, 16)
      for (long long sll = 0; sll < static_cast<long long>(num_slices); ++sll) {
        const uint32_t s = static_cast<uint32_t>(sll);
        std::vector<uint32_t> *ref_stream = slice_to_ref_stream[s];
        local_nodes.clear();

        if (ref_stream) {
          auto visit = [&](NodeId signed_node) {
            const uint32_t node = abs_node_id(signed_node);
            if (node == 0 || node > num_nodes)
              return;
            local_nodes.push_back(node);
            ref_stream->push_back(node);
          };
          stream_decoded_nodes(slices[s], delta_round, min_rule_id,
                               max_rule_id, rules_first, rules_second,
                               rule_cache, local_decoded, visit);
        } else {
          auto visit = [&](NodeId signed_node) {
            const uint32_t node = abs_node_id(signed_node);
            if (node == 0 || node > num_nodes)
              return;
            local_nodes.push_back(node);
          };
          stream_decoded_nodes(slices[s], delta_round, min_rule_id,
                               max_rule_id, rules_first, rules_second,
                               rule_cache, local_decoded, visit);
        }

        std::sort(local_nodes.begin(), local_nodes.end());
        local_nodes.erase(
            std::unique(local_nodes.begin(), local_nodes.end()),
            local_nodes.end());
        slice_nodes[s] = std::move(local_nodes);
        local_nodes = std::vector<uint32_t>();
      }
    }
  }

  // We no longer need the rule cache after slice decoding.
  rule_cache.forward = std::vector<std::vector<int32_t>>();
  rule_cache.ready = std::vector<uint8_t>();
  rule_cache.bytes_used = 0;

  // -------------------------------------------------------------------------
  // Pass 2 (Tier 2 #4): build per-group sorted-unique node lists, then a
  // CSR-shaped node->groups index. CSR avoids the per-node vector overhead
  // (24 bytes empty * num_nodes) of the previous representation.
  // -------------------------------------------------------------------------
  std::vector<std::vector<uint32_t>> group_nodes(num_groups);
  {
    ScopedOMPThreads omp_scope(options.num_threads);
    const int T = std::max(1, omp_scope.effective_threads());

#pragma omp parallel for schedule(dynamic, 1) num_threads(T)
    for (long long gll = 0; gll < static_cast<long long>(num_groups); ++gll) {
      const uint32_t gid = static_cast<uint32_t>(gll);
      auto &gn = group_nodes[gid];
      size_t total = 0;
      for (uint32_t s : meta.groups[gid])
        total += slice_nodes[s].size();
      gn.reserve(total);
      for (uint32_t s : meta.groups[gid])
        gn.insert(gn.end(), slice_nodes[s].begin(), slice_nodes[s].end());
      std::sort(gn.begin(), gn.end());
      gn.erase(std::unique(gn.begin(), gn.end()), gn.end());
    }
  }

  // Free per-slice node lists; they are no longer needed.
  slice_nodes = std::vector<std::vector<uint32_t>>();

  // Build CSR: node_offsets[node..node+1] indexes into node_to_group_ids.
  std::vector<uint64_t> node_offsets(static_cast<size_t>(num_nodes) + 2, 0);
  for (const auto &gn : group_nodes) {
    for (uint32_t node : gn) {
      if (node != 0 && node <= num_nodes)
        ++node_offsets[node + 1];
    }
  }
  for (size_t i = 1; i < node_offsets.size(); ++i)
    node_offsets[i] += node_offsets[i - 1];

  const uint64_t total_entries = node_offsets[node_offsets.size() - 1];
  std::vector<uint32_t> node_to_group_ids(total_entries);
  {
    std::vector<uint64_t> cursor(node_offsets.begin(),
                                 node_offsets.end() - 1);
    for (uint32_t gid = 0; gid < static_cast<uint32_t>(num_groups); ++gid) {
      for (uint32_t node : group_nodes[gid]) {
        if (node != 0 && node <= num_nodes)
          node_to_group_ids[cursor[node]++] = gid;
      }
    }
  }
  group_nodes = std::vector<std::vector<uint32_t>>();

  // -------------------------------------------------------------------------
  // Pass 3 (Tier 1 #1 payoff): parallel over distinct BED chroms, sweep using
  // the cached reference node streams. No rule expansion happens here.
  // -------------------------------------------------------------------------
  {
    ScopedOMPThreads omp_scope(options.num_threads);
    const int T = std::max(1, omp_scope.effective_threads());

#pragma omp parallel num_threads(T)
    {
      std::vector<std::pair<uint32_t, uint32_t>> local_window_nodes;

#pragma omp for schedule(dynamic, 1)
      for (long long cll = 0;
           cll < static_cast<long long>(chrom_list.size()); ++cll) {
        const std::string &chrom = chrom_list[cll].first;
        const std::vector<uint32_t> &ref_stream = ref_streams[cll];

        auto rit = ranges_by_chrom.find(chrom);
        if (rit == ranges_by_chrom.end())
          continue;

        std::vector<uint32_t> range_ids = rit->second;
        std::sort(range_ids.begin(), range_ids.end(),
                  [&](uint32_t a, uint32_t b) {
                    return std::tie(result.ranges[a].start,
                                    result.ranges[a].end, a) <
                           std::tie(result.ranges[b].start,
                                    result.ranges[b].end, b);
                  });

        local_window_nodes.clear();
        size_t next_range = 0;
        uint64_t offset = 0;
        for (uint32_t node : ref_stream) {
          if (node == 0 || node > num_nodes)
            continue;
          const uint64_t len = segment_lengths[node - 1];
          const uint64_t node_start = offset;
          const uint64_t node_end = offset + len;
          while (next_range < range_ids.size() &&
                 result.ranges[range_ids[next_range]].end <= node_start) {
            ++next_range;
          }
          for (size_t j = next_range; j < range_ids.size(); ++j) {
            const PavRange &r = result.ranges[range_ids[j]];
            if (r.start >= node_end)
              break;
            if (r.end > node_start && r.start < node_end)
              local_window_nodes.emplace_back(range_ids[j], node);
          }
          offset = node_end;
        }

        std::sort(local_window_nodes.begin(), local_window_nodes.end());
        local_window_nodes.erase(
            std::unique(local_window_nodes.begin(),
                        local_window_nodes.end()),
            local_window_nodes.end());

        for (const auto &wn : local_window_nodes) {
          const uint32_t wid = wn.first;
          const uint32_t node = wn.second;
          const uint64_t len = segment_lengths[node - 1];
#pragma omp atomic
          result.denominators[wid] += len;
          const uint64_t start = node_offsets[node];
          const uint64_t end = node_offsets[node + 1];
          for (uint64_t k = start; k < end; ++k) {
            const uint32_t gid = node_to_group_ids[k];
            const size_t idx =
                static_cast<size_t>(wid) * num_groups + gid;
#pragma omp atomic
            result.numerators[idx] += len;
          }
        }
      }
    }
  }

  return result;
}

} // namespace gfaz
