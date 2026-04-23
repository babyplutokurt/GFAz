#include "workflows/growth_workflow.hpp"

#include "codec/codec.hpp"
#include "utils/threading_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
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

GrowthResult compute_growth(const CompressedData &data, int num_threads) {
  GrowthResult result;

  const uint32_t num_nodes = infer_num_nodes(data);
  result.num_nodes = num_nodes;

  const size_t num_paths = data.sequence_lengths.size();
  const size_t num_walks = data.walk_lengths.size();
  const size_t total_haps = num_paths + num_walks;
  if (total_haps == 0)
    return result;
  if (total_haps > UINT32_MAX) {
    throw std::runtime_error("growth: number of haplotypes exceeds 2^32-1");
  }
  const uint32_t N = static_cast<uint32_t>(total_haps);
  result.num_haplotypes = N;

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

  std::vector<HapSlice> path_slices;
  path_slices.reserve(num_paths);
  build_slices(paths_flat, data.sequence_lengths, data.original_path_lengths,
               path_slices);
  std::vector<HapSlice> walk_slices;
  walk_slices.reserve(num_walks);
  build_slices(walks_flat, data.walk_lengths, data.original_walk_lengths,
               walk_slices);

  ScopedOMPThreads omp_scope(num_threads);
  const int T = std::max(1, omp_scope.effective_threads());

  // Single shared coverage array; per-thread last-seen stamp filters duplicate
  // updates within one hap.
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

    // Per-hap inline coverage update; lambda body so the OMP atomic binds to
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

    auto process_one = [&](const HapSlice &s) {
      bump_stamp();
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
    for (long long i = 0; i < static_cast<long long>(path_slices.size()); ++i) {
      process_one(path_slices[static_cast<size_t>(i)]);
    }
    // Implicit barrier at end of `for`; safe to release paths_flat and reset
    // per-thread stamps before the walks pass.

#pragma omp single
    {
      paths_flat = std::vector<int32_t>{};
      path_slices = std::vector<HapSlice>{};
    }
    // Implicit barrier at end of `single`.

    // Stamps from the path pass are still live in last_seen; with only 255
    // distinct values, a stale path-stamp could collide with a fresh
    // walk-stamp and falsely suppress an increment. Reset per-thread state.
    std::fill(last_seen.begin(), last_seen.end(), 0);
    stamp = 0;

#pragma omp for schedule(dynamic, 1)
    for (long long i = 0; i < static_cast<long long>(walk_slices.size()); ++i) {
      process_one(walk_slices[static_cast<size_t>(i)]);
    }
  }
  walks_flat = std::vector<int32_t>{};
  walk_slices = std::vector<HapSlice>{};

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
