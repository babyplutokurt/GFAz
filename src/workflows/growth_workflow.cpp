#include "workflows/growth_workflow.hpp"

#include "codec/codec.hpp"
#include "utils/threading_utils.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace gfaz {

namespace {

// Recursively expand a rule, calling `visit` once per leaf in expansion order
// (left-to-right of the original rule, possibly reversed). Templated on the
// visitor so the compiler can inline the per-leaf action through recursion.
template <typename Visitor>
void expand_rule_visit(uint32_t rule_id, bool reverse,
                       const std::vector<int32_t> &first,
                       const std::vector<int32_t> &second, uint32_t min_id,
                       uint32_t max_id, Visitor &visit) {
  const uint32_t idx = rule_id - min_id;
  const int32_t a = first[idx];
  const int32_t b = second[idx];

  if (!reverse) {
    const uint32_t abs_a = static_cast<uint32_t>(std::abs(a));
    if (abs_a >= min_id && abs_a < max_id)
      expand_rule_visit(abs_a, a < 0, first, second, min_id, max_id, visit);
    else
      visit(a);

    const uint32_t abs_b = static_cast<uint32_t>(std::abs(b));
    if (abs_b >= min_id && abs_b < max_id)
      expand_rule_visit(abs_b, b < 0, first, second, min_id, max_id, visit);
    else
      visit(b);
  } else {
    const uint32_t abs_b = static_cast<uint32_t>(std::abs(b));
    if (abs_b >= min_id && abs_b < max_id)
      expand_rule_visit(abs_b, b >= 0, first, second, min_id, max_id, visit);
    else
      visit(static_cast<int32_t>(-b));

    const uint32_t abs_a = static_cast<uint32_t>(std::abs(a));
    if (abs_a >= min_id && abs_a < max_id)
      expand_rule_visit(abs_a, a >= 0, first, second, min_id, max_id, visit);
    else
      visit(static_cast<int32_t>(-a));
  }
}

// Stream the leaves of one haplotype's encoded slice through `visit` in
// path order (delta-domain leaves; visitor is responsible for inverse-delta).
template <typename Visitor>
void stream_hap_leaves(const int32_t *encoded, size_t encoded_len,
                       uint32_t min_rule_id, uint32_t max_rule_id,
                       const std::vector<int32_t> &rules_first,
                       const std::vector<int32_t> &rules_second,
                       Visitor &visit) {
  for (size_t i = 0; i < encoded_len; ++i) {
    const NodeId node = encoded[i];
    const uint32_t abs_id = static_cast<uint32_t>(std::abs(node));
    if (abs_id >= min_rule_id && abs_id < max_rule_id) {
      expand_rule_visit(abs_id, node < 0, rules_first, rules_second,
                        min_rule_id, max_rule_id, visit);
    } else {
      visit(node);
    }
  }
}

// Fallback materializer for delta_round >= 2: expands rules into `decoded`,
// then runs `delta_round` inverse-delta passes.
void decode_one_haplotype_general(const int32_t *encoded, size_t encoded_len,
                                  uint32_t original_len, int delta_round,
                                  uint32_t min_rule_id, uint32_t max_rule_id,
                                  const std::vector<int32_t> &rules_first,
                                  const std::vector<int32_t> &rules_second,
                                  std::vector<NodeId> &decoded) {
  decoded.clear();
  decoded.reserve(original_len);
  auto push = [&](int32_t v) { decoded.push_back(v); };
  stream_hap_leaves(encoded, encoded_len, min_rule_id, max_rule_id,
                    rules_first, rules_second, push);
  for (int r = 0; r < delta_round; ++r) {
    for (size_t i = 1; i < decoded.size(); ++i) {
      decoded[i] = decoded[i] + decoded[i - 1];
    }
  }
}

// P(node with coverage c is in a random size-k subset of N) under union mode.
// Closed form: 1 - C(N-c, k) / C(N, k), computed as a stable product of
// fractions to avoid overflow for large N.
double frac_covered_union(uint32_t N, uint32_t c, uint32_t k) {
  if (c == 0 || k == 0)
    return 0.0;
  if (c >= N)
    return 1.0;
  if (N - c < k)
    return 1.0;
  double ratio = 1.0;
  for (uint32_t i = 0; i < c; ++i) {
    ratio *= static_cast<double>(N - k - i) / static_cast<double>(N - i);
  }
  return 1.0 - ratio;
}

uint32_t infer_num_nodes(const CompressedData &data) {
  std::vector<uint32_t> seg_lens =
      Codec::zstd_decompress_uint32_vector(data.segment_seq_lengths_zstd);
  return static_cast<uint32_t>(seg_lens.size());
}

struct HapSlice {
  const int32_t *encoded;
  uint32_t enc_len;
  uint32_t orig_len;
};

// Strip a trailing ":start-end" suffix in place (digits on both sides of '-',
// preceded by ':'). Mirrors Panacus's PATHID_COORDS = ^(.+):([0-9]+)-([0-9]+)$.
void strip_pansn_coords_inplace(std::string &s) {
  const size_t colon = s.rfind(':');
  if (colon == std::string::npos || colon == 0)
    return;
  const size_t dash = s.find('-', colon + 1);
  if (dash == std::string::npos)
    return;
  if (colon + 1 == dash || dash + 1 == s.size())
    return;
  for (size_t i = colon + 1; i < dash; ++i)
    if (!std::isdigit(static_cast<unsigned char>(s[i])))
      return;
  for (size_t i = dash + 1; i < s.size(); ++i)
    if (!std::isdigit(static_cast<unsigned char>(s[i])))
      return;
  s.erase(colon);
}

// Parts of a PanSN path name after Panacus's from_str + clear_coords().
// `has_hap`/`has_seq` mirror PathSegment's Option<haplotype>/Option<seqid>.
struct PansnParts {
  std::string sample;
  std::string hap;
  std::string seq;
  bool has_hap = false;
  bool has_seq = false;
};

// Panacus regex: ^([^#]+)(#[^#]+)?(#[^#].*)?$; ":start-end" is stripped from
// the last populated field (sample | hap | seq).
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
  auto take_two_field = [&](const std::string &hap_raw) {
    if (hap_raw.empty()) {
      // Invalid PanSN ("sample#"): fall back to single-field key.
      return;
    }
    p.hap = hap_raw;
    p.has_hap = true;
    strip_pansn_coords_inplace(p.hap);
  };
  if (h2 == std::string::npos) {
    take_two_field(name.substr(h1 + 1));
    return p;
  }
  if (h2 == h1 + 1) {
    // "sample##...": doesn't match PanSN at all; keep sample only.
    return p;
  }
  const std::string hap_raw = name.substr(h1 + 1, h2 - h1 - 1);
  std::string seq_raw = name.substr(h2 + 1);
  if (seq_raw.empty() || seq_raw[0] == '#') {
    // Group-3 regex fails; Panacus backtracks to the 2-field match.
    take_two_field(hap_raw);
    return p;
  }
  p.hap = hap_raw;
  p.has_hap = true;
  p.seq = std::move(seq_raw);
  p.has_seq = true;
  strip_pansn_coords_inplace(p.seq);
  return p;
}

// Build the group key for a P-line under the requested grouping mode. Mirrors
// Panacus: default uses id(), -H uses "{sample}#{hap_or_empty}", -S uses
// "{sample}". `SampleHapSeq` here is Panacus default (PathSegment::id()).
std::string path_group_key(const PansnParts &p, GroupingMode mode) {
  switch (mode) {
  case GroupingMode::Sample:
    return p.sample;
  case GroupingMode::SampleHap:
    // Matches Panacus: format!("{}#{}", sample, hap.unwrap_or("")).
    return p.sample + "#" + (p.has_hap ? p.hap : std::string());
  case GroupingMode::SampleHapSeq:
  default:
    if (p.has_hap) {
      return p.has_seq ? (p.sample + "#" + p.hap + "#" + p.seq)
                       : (p.sample + "#" + p.hap);
    }
    if (p.has_seq)
      return p.sample + "#*#" + p.seq;
    return p.sample;
  }
}

// Build the group key for a W-line (sample/hap/seq already separate columns).
std::string walk_group_key(const std::string &sample, uint32_t hap,
                           const std::string &seq, GroupingMode mode) {
  switch (mode) {
  case GroupingMode::Sample:
    return sample;
  case GroupingMode::SampleHap:
    return sample + "#" + std::to_string(hap);
  case GroupingMode::SampleHapSeq:
  default:
    return sample + "#" + std::to_string(hap) + "#" + seq;
  }
}

void build_slices(const std::vector<int32_t> &flat,
                  const std::vector<uint32_t> &lengths,
                  const std::vector<uint32_t> &original_lengths,
                  std::vector<HapSlice> &out) {
  size_t offset = 0;
  for (size_t i = 0; i < lengths.size(); ++i) {
    const uint32_t enc_len = lengths[i];
    const uint32_t orig_len =
        (i < original_lengths.size()) ? original_lengths[i] : enc_len;
    if (offset + enc_len > flat.size()) {
      throw std::runtime_error("growth: encoded haplotype block is truncated");
    }
    out.push_back(HapSlice{flat.data() + offset, enc_len, orig_len});
    offset += enc_len;
  }
}

} // namespace

GrowthResult compute_growth(const CompressedData &data, int num_threads,
                            GroupingMode mode) {
  GrowthResult result;

  const uint32_t num_nodes = infer_num_nodes(data);
  result.num_nodes = num_nodes;

  const size_t num_paths = data.sequence_lengths.size();
  const size_t num_walks = data.walk_lengths.size();
  const size_t total_slices = num_paths + num_walks;
  if (total_slices == 0)
    return result;

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

  const uint32_t min_rule_id = data.min_rule_id();
  const uint32_t max_rule_id =
      min_rule_id + static_cast<uint32_t>(rules_first.size());
  const int delta_round = data.delta_round;

  // Unified slice vector: [paths ... walks]. Index order is preserved so the
  // slice-to-group mapping below can address walks with offset num_paths.
  std::vector<HapSlice> slices;
  slices.reserve(total_slices);
  build_slices(paths_flat, data.sequence_lengths, data.original_path_lengths,
               slices);
  build_slices(walks_flat, data.walk_lengths, data.original_walk_lengths,
               slices);

  // Build groups[] -- each group is a list of slice indices sharing the same
  // haplotype identity per the selected GroupingMode. Each group is processed
  // as a unit downstream so nodes visited across its slices are only counted
  // once toward that group's coverage.
  std::vector<std::vector<uint32_t>> groups;
  if (mode == GroupingMode::PerPathWalk) {
    groups.resize(total_slices);
    for (size_t i = 0; i < total_slices; ++i)
      groups[i].push_back(static_cast<uint32_t>(i));
  } else {
    // Derive a key per slice under the requested mode, then dedup first-seen.
    std::vector<std::string> keys;
    keys.reserve(total_slices);

    // Paths: decompress P-line names and parse PanSN parts.
    std::string names_concat;
    std::vector<uint32_t> name_lens;
    if (num_paths > 0) {
      names_concat = Codec::zstd_decompress_string(data.names_zstd);
      name_lens = Codec::zstd_decompress_uint32_vector(data.name_lengths_zstd);
      if (name_lens.size() != num_paths) {
        throw std::runtime_error(
            "growth: path name count does not match number of paths");
      }
    }
    size_t name_off = 0;
    for (size_t i = 0; i < num_paths; ++i) {
      const uint32_t L = name_lens[i];
      std::string name = (name_off + L <= names_concat.size())
                             ? names_concat.substr(name_off, L)
                             : std::string();
      name_off += L;
      keys.push_back(path_group_key(parse_pansn_path_name(name), mode));
    }

    // Walks: already have (sample, hap, seqid) as separate columns.
    std::string sample_concat, seq_concat;
    std::vector<uint32_t> sample_lens, seq_lens;
    std::vector<uint32_t> hap_indices;
    if (num_walks > 0) {
      sample_concat = Codec::zstd_decompress_string(data.walk_sample_ids_zstd);
      sample_lens =
          Codec::zstd_decompress_uint32_vector(data.walk_sample_id_lengths_zstd);
      seq_concat = Codec::zstd_decompress_string(data.walk_seq_ids_zstd);
      seq_lens =
          Codec::zstd_decompress_uint32_vector(data.walk_seq_id_lengths_zstd);
      hap_indices =
          Codec::zstd_decompress_uint32_vector(data.walk_hap_indices_zstd);
      if (sample_lens.size() != num_walks || seq_lens.size() != num_walks ||
          hap_indices.size() != num_walks) {
        throw std::runtime_error(
            "growth: walk metadata column count does not match number of walks");
      }
    }
    size_t s_off = 0, q_off = 0;
    for (size_t i = 0; i < num_walks; ++i) {
      const uint32_t SL = sample_lens[i];
      const uint32_t QL = seq_lens[i];
      std::string sample = (s_off + SL <= sample_concat.size())
                               ? sample_concat.substr(s_off, SL)
                               : std::string();
      std::string seq = (q_off + QL <= seq_concat.size())
                            ? seq_concat.substr(q_off, QL)
                            : std::string();
      s_off += SL;
      q_off += QL;
      keys.push_back(walk_group_key(sample, hap_indices[i], seq, mode));
    }

    std::unordered_map<std::string, uint32_t> key_to_gid;
    key_to_gid.reserve(total_slices * 2);
    for (uint32_t i = 0; i < total_slices; ++i) {
      auto it = key_to_gid.find(keys[i]);
      uint32_t gid;
      if (it == key_to_gid.end()) {
        gid = static_cast<uint32_t>(groups.size());
        key_to_gid.emplace(std::move(keys[i]), gid);
        groups.emplace_back();
      } else {
        gid = it->second;
      }
      groups[gid].push_back(i);
    }
  }

  if (groups.empty())
    return result;
  if (groups.size() > UINT32_MAX) {
    throw std::runtime_error("growth: number of haplotype groups exceeds 2^32-1");
  }
  const uint32_t N = static_cast<uint32_t>(groups.size());
  result.num_haplotypes = N;

  ScopedOMPThreads omp_scope(num_threads);
  const int T = std::max(1, omp_scope.effective_threads());

  // Single shared coverage array; per-thread last-seen stamp filters duplicate
  // updates within one group (all slices of the group share one stamp).
  const size_t cov_len = static_cast<size_t>(num_nodes) + 1;
  std::vector<uint32_t> cov(cov_len, 0);

#pragma omp parallel num_threads(T)
  {
    std::vector<uint8_t> last_seen(cov_len, 0);
    uint8_t stamp = 0;
    // Materialization buffer used only when delta_round >= 2.
    std::vector<NodeId> decoded;
    if (delta_round >= 2)
      decoded.reserve(1024);

    auto bump_stamp = [&]() {
      if (stamp == 255) {
        std::fill(last_seen.begin(), last_seen.end(), 0);
        stamp = 0;
      }
      ++stamp;
    };

    // Per-group inline coverage update; lambda body so the OMP atomic binds to
    // the enclosing parallel region.
    auto cov_update = [&](int32_t signed_v) {
      const uint32_t a = static_cast<uint32_t>(
          signed_v < 0 ? -static_cast<int64_t>(signed_v) : signed_v);
      if (a == 0 || a > num_nodes)
        return;
      if (last_seen[a] != stamp) {
        last_seen[a] = stamp;
#pragma omp atomic
        cov[a] += 1;
      }
    };

    auto process_slice = [&](const HapSlice &s) {
      if (delta_round == 0) {
        auto visit = [&](int32_t leaf) { cov_update(leaf); };
        stream_hap_leaves(s.encoded, s.enc_len, min_rule_id, max_rule_id,
                          rules_first, rules_second, visit);
      } else if (delta_round == 1) {
        int32_t prev = 0;
        auto visit = [&](int32_t leaf) {
          prev += leaf;
          cov_update(prev);
        };
        stream_hap_leaves(s.encoded, s.enc_len, min_rule_id, max_rule_id,
                          rules_first, rules_second, visit);
      } else {
        decode_one_haplotype_general(s.encoded, s.enc_len, s.orig_len,
                                     delta_round, min_rule_id, max_rule_id,
                                     rules_first, rules_second, decoded);
        for (NodeId node : decoded)
          cov_update(node);
      }
    };

#pragma omp for schedule(dynamic, 1)
    for (long long g = 0; g < static_cast<long long>(groups.size()); ++g) {
      bump_stamp();
      const auto &group = groups[static_cast<size_t>(g)];
      for (uint32_t si : group) {
        process_slice(slices[si]);
      }
    }
  }
  paths_flat = std::vector<int32_t>{};
  walks_flat = std::vector<int32_t>{};
  slices = std::vector<HapSlice>{};

  // Build coverage histogram from the shared cov[] array.
  result.hist.assign(static_cast<size_t>(N) + 1, 0);
#pragma omp parallel num_threads(T)
  {
    std::vector<uint64_t> local_hist(static_cast<size_t>(N) + 1, 0);
#pragma omp for schedule(static) nowait
    for (long long v = 1; v <= static_cast<long long>(num_nodes); ++v) {
      const uint32_t c = cov[static_cast<size_t>(v)];
      if (c <= N)
        local_hist[c] += 1;
    }
#pragma omp critical
    {
      for (size_t c = 0; c <= N; ++c)
        result.hist[c] += local_hist[c];
    }
  }

  // Closed-form expectation: per k, sum over c of hist[c] * P(covered).
  result.growth.assign(static_cast<size_t>(N) + 1, 0.0);
#pragma omp parallel for num_threads(T) schedule(dynamic, 16)
  for (long long k = 1; k <= static_cast<long long>(N); ++k) {
    double sum = 0.0;
    for (uint32_t c = 1; c <= N; ++c) {
      const uint64_t hc = result.hist[c];
      if (hc == 0)
        continue;
      sum += static_cast<double>(hc) *
             frac_covered_union(N, c, static_cast<uint32_t>(k));
    }
    result.growth[static_cast<size_t>(k)] = sum;
  }

  return result;
}

} // namespace gfaz
