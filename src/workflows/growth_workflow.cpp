#include "workflows/growth_workflow.hpp"

#include "codec/codec.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <vector>

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

void accumulate_haplotypes(const std::vector<int32_t> &flat,
                           const std::vector<uint32_t> &lengths,
                           const std::vector<uint32_t> &original_lengths,
                           int delta_round, uint32_t min_rule_id,
                           const std::vector<int32_t> &rules_first,
                           const std::vector<int32_t> &rules_second,
                           uint32_t num_nodes, std::vector<uint32_t> &cov,
                           std::vector<uint32_t> &last_seen, uint32_t &hap_id) {
  std::vector<NodeId> decoded;
  decoded.reserve(1024);

  size_t offset = 0;
  for (size_t i = 0; i < lengths.size(); ++i) {
    ++hap_id;
    const size_t enc_len = lengths[i];
    const uint32_t orig_len = (i < original_lengths.size())
                                  ? original_lengths[i]
                                  : static_cast<uint32_t>(enc_len);
    if (offset + enc_len > flat.size()) {
      throw std::runtime_error("growth: encoded haplotype block is truncated");
    }
    decode_one_haplotype(flat.data() + offset, enc_len, orig_len, delta_round,
                         min_rule_id, rules_first, rules_second, decoded);
    offset += enc_len;

    for (NodeId node : decoded) {
      const uint32_t v =
          static_cast<uint32_t>(node < 0 ? -static_cast<int64_t>(node) : node);
      if (v == 0 || v > num_nodes)
        continue;
      if (last_seen[v] != hap_id) {
        last_seen[v] = hap_id;
        cov[v] += 1;
      }
    }
  }
}

} // namespace

GrowthResult compute_growth(const CompressedData &data) {
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

  std::vector<uint32_t> cov(static_cast<size_t>(num_nodes) + 1, 0);
  std::vector<uint32_t> last_seen(static_cast<size_t>(num_nodes) + 1, 0);

  uint32_t hap_id = 0;
  accumulate_haplotypes(paths_flat, data.sequence_lengths,
                        data.original_path_lengths, data.delta_round,
                        min_rule_id, rules_first, rules_second, num_nodes, cov,
                        last_seen, hap_id);
  accumulate_haplotypes(walks_flat, data.walk_lengths,
                        data.original_walk_lengths, data.delta_round,
                        min_rule_id, rules_first, rules_second, num_nodes, cov,
                        last_seen, hap_id);

  result.hist.assign(static_cast<size_t>(N) + 1, 0);
  for (uint32_t v = 1; v <= num_nodes; ++v) {
    result.hist[cov[v]] += 1;
  }

  result.growth.assign(static_cast<size_t>(N) + 1, 0.0);
  for (uint32_t k = 1; k <= N; ++k) {
    double sum = 0.0;
    for (uint32_t c = 1; c <= N; ++c) {
      const uint64_t hc = result.hist[c];
      if (hc == 0)
        continue;
      sum += static_cast<double>(hc) * frac_covered_union(N, c, k);
    }
    result.growth[k] = sum;
  }

  return result;
}

} // namespace gfaz
