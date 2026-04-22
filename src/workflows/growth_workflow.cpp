#include "workflows/growth_workflow.hpp"

#include "codec/codec.hpp"
#include "utils/threading_utils.hpp"

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

// Recursively expand a rule into delta-domain leaves.
// Mirrors expand_rule in extraction_workflow.cpp; isolated here so the growth
// workflow does not depend on the extraction translation unit.
void expand_rule(uint32_t rule_id, bool reverse,
                 const std::vector<int32_t> &first,
                 const std::vector<int32_t> &second, uint32_t min_id,
                 uint32_t max_id, std::vector<NodeId> &out) {
  const uint32_t idx = rule_id - min_id;
  const int32_t a = first[idx];
  const int32_t b = second[idx];

  if (!reverse) {
    const uint32_t abs_a = static_cast<uint32_t>(std::abs(a));
    if (abs_a >= min_id && abs_a < max_id)
      expand_rule(abs_a, a < 0, first, second, min_id, max_id, out);
    else
      out.push_back(a);

    const uint32_t abs_b = static_cast<uint32_t>(std::abs(b));
    if (abs_b >= min_id && abs_b < max_id)
      expand_rule(abs_b, b < 0, first, second, min_id, max_id, out);
    else
      out.push_back(b);
  } else {
    const uint32_t abs_b = static_cast<uint32_t>(std::abs(b));
    if (abs_b >= min_id && abs_b < max_id)
      expand_rule(abs_b, b >= 0, first, second, min_id, max_id, out);
    else
      out.push_back(-b);

    const uint32_t abs_a = static_cast<uint32_t>(std::abs(a));
    if (abs_a >= min_id && abs_a < max_id)
      expand_rule(abs_a, a >= 0, first, second, min_id, max_id, out);
    else
      out.push_back(-a);
  }
}

// Decode one encoded slice into original signed node IDs. The encoded slice
// holds delta-domain values mixed with rule references; we expand rules into
// delta-domain leaves and then run inverse delta `delta_round` times.
void decode_one_haplotype(const int32_t *encoded, size_t encoded_len,
                          uint32_t original_len, int delta_round,
                          uint32_t min_rule_id,
                          const std::vector<int32_t> &rules_first,
                          const std::vector<int32_t> &rules_second,
                          std::vector<NodeId> &decoded) {
  decoded.clear();
  decoded.reserve(original_len);

  const uint32_t max_rule_id =
      min_rule_id + static_cast<uint32_t>(rules_first.size());

  for (size_t i = 0; i < encoded_len; ++i) {
    const NodeId node = encoded[i];
    const uint32_t abs_id = static_cast<uint32_t>(std::abs(node));
    if (abs_id >= min_rule_id && abs_id < max_rule_id) {
      expand_rule(abs_id, node < 0, rules_first, rules_second, min_rule_id,
                  max_rule_id, decoded);
    } else {
      decoded.push_back(node);
    }
  }

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

// Number of segments == number of original 1-based node IDs. The segment
// sequence-length column is small relative to the segment sequences themselves
// and decompresses to a uint32 vector whose size is num_segments.
uint32_t infer_num_nodes(const CompressedData &data) {
  std::vector<uint32_t> seg_lens =
      Codec::zstd_decompress_uint32_vector(data.segment_seq_lengths_zstd);
  return static_cast<uint32_t>(seg_lens.size());
}

// Slice descriptor for one haplotype (path or walk) in the flat encoded array.
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
    const uint32_t orig_len = (i < original_lengths.size())
                                  ? original_lengths[i]
                                  : enc_len;
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
  const int delta_round = data.delta_round;

  std::vector<HapSlice> slices;
  slices.reserve(total_haps);
  build_slices(paths_flat, data.sequence_lengths, data.original_path_lengths,
               slices);
  build_slices(walks_flat, data.walk_lengths, data.original_walk_lengths,
               slices);

  ScopedOMPThreads omp_scope(num_threads);
  const int T = std::max(1, omp_scope.effective_threads());

  // Single shared coverage array; per-thread last-seen stamp filters duplicate
  // updates within one hap. Memory is 4 * num_nodes (cov) + T * num_nodes
  // (last_seen), instead of 2T * 4 * num_nodes.
  const size_t cov_len = static_cast<size_t>(num_nodes) + 1;
  std::vector<uint32_t> cov(cov_len, 0);

#pragma omp parallel num_threads(T)
  {
    // uint8_t stamps cycle 1..255; wrap forces a reset of last_seen. With ~N
    // haps total per thread (N << 1e6 in practice), resets are rare.
    std::vector<uint8_t> last_seen(cov_len, 0);
    uint8_t stamp = 0;
    std::vector<NodeId> decoded;
    decoded.reserve(1024);

#pragma omp for schedule(dynamic, 16)
    for (long long i = 0; i < static_cast<long long>(slices.size()); ++i) {
      if (stamp == 255) {
        std::fill(last_seen.begin(), last_seen.end(), 0);
        stamp = 0;
      }
      ++stamp;

      const HapSlice &s = slices[static_cast<size_t>(i)];
      decode_one_haplotype(s.encoded, s.enc_len, s.orig_len, delta_round,
                           min_rule_id, rules_first, rules_second, decoded);
      for (NodeId node : decoded) {
        const uint32_t v = static_cast<uint32_t>(
            node < 0 ? -static_cast<int64_t>(node) : node);
        if (v == 0 || v > num_nodes)
          continue;
        if (last_seen[v] != stamp) {
          last_seen[v] = stamp;
#pragma omp atomic
          cov[v] += 1;
        }
      }
    }
  }

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
