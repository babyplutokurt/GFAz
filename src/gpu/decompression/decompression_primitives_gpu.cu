#include "gpu/decompression/decompression_primitives_gpu.hpp"

#include "codec/codec.hpp"
#include "gpu/core/codec_gpu.cuh"
#include "gpu/decompression/decompression_workflow_gpu_internal.hpp"
#include "utils/runtime_utils.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

namespace gpu_decompression {

namespace {

using Clock = std::chrono::high_resolution_clock;
using gfz::runtime_utils::elapsed_ms;

} // namespace

GpuTraversalRulebook prepare_gpu_traversal_rulebook(const CompressedData &data) {
  const auto start = Clock::now();

  std::vector<int32_t> first_host =
      Codec::zstd_decompress_int32_vector(data.rules_first_zstd);
  std::vector<int32_t> second_host =
      Codec::zstd_decompress_int32_vector(data.rules_second_zstd);
  Codec::delta_decode_int32(first_host);
  Codec::delta_decode_int32(second_host);

  GpuTraversalRulebook rulebook;
  rulebook.d_rules_first = thrust::device_vector<int32_t>(first_host.begin(),
                                                          first_host.end());
  rulebook.d_rules_second = thrust::device_vector<int32_t>(second_host.begin(),
                                                           second_host.end());
  rulebook.min_rule_id = data.min_rule_id();
  rulebook.num_rules = std::min(
      {data.total_rules(), rulebook.d_rules_first.size(),
       rulebook.d_rules_second.size()});
  rulebook.host_decode_ms = elapsed_ms(start, Clock::now());
  return rulebook;
}

GpuTraversalPayload prepare_gpu_traversal_payload(
    const ZstdCompressedBlock &encoded_block,
    const std::vector<uint32_t> &final_lengths) {
  const auto start = Clock::now();

  std::vector<int32_t> encoded_host =
      Codec::zstd_decompress_int32_vector(encoded_block);

  GpuTraversalPayload payload;
  payload.d_encoded =
      thrust::device_vector<int32_t>(encoded_host.begin(), encoded_host.end());
  payload.d_final_lengths =
      thrust::device_vector<uint32_t>(final_lengths.begin(), final_lengths.end());
  payload.host_decode_ms = elapsed_ms(start, Clock::now());
  return payload;
}

std::vector<int32_t> decode_gpu_traversal_to_host(
    const GpuTraversalPayload &payload, const GpuTraversalRulebook &rulebook,
    GpuDecompressionOptions options) {
  std::vector<int32_t> result;

  if (rulebook.num_rules == 0) {
    thrust::device_vector<uint64_t> d_offsets(payload.d_final_lengths.size());
    thrust::exclusive_scan(payload.d_final_lengths.begin(),
                           payload.d_final_lengths.end(), d_offsets.begin(),
                           uint64_t(0));
    thrust::device_vector<int32_t> d_decoded =
        gpu_codec::segmented_inverse_delta_decode_device_vec(
            payload.d_encoded, d_offsets,
            static_cast<uint32_t>(payload.d_final_lengths.size()),
            payload.d_encoded.size());
    result.resize(d_decoded.size());
    thrust::copy(d_decoded.begin(), d_decoded.end(), result.begin());
    return result;
  }

  if (options.use_legacy_full_decompression) {
    decompress_paths_gpu_legacy(payload.d_encoded, rulebook.d_rules_first,
                                rulebook.d_rules_second, rulebook.min_rule_id,
                                rulebook.num_rules, payload.d_final_lengths,
                                result);
  } else {
    decompress_paths_gpu_rolling(
        payload.d_encoded, rulebook.d_rules_first, rulebook.d_rules_second,
        rulebook.min_rule_id, rulebook.num_rules, payload.d_final_lengths,
        std::max<uint32_t>(1, options.traversals_per_chunk),
        std::max<size_t>(1, options.max_expanded_chunk_bytes), result);
  }

  return result;
}

void stream_gpu_traversal_to_host(
    const GpuTraversalPayload &payload, const GpuTraversalRulebook &rulebook,
    GpuDecompressionOptions options, RollingPathChunkConsumer consumer,
    RollingPathStreamOptions stream_options) {
  if (options.use_legacy_full_decompression) {
    throw std::runtime_error(
        "GPU rolling streaming is not available in legacy decompression mode");
  }

  stream_decompress_paths_gpu_rolling(
      payload.d_encoded, rulebook.d_rules_first, rulebook.d_rules_second,
      rulebook.min_rule_id, rulebook.num_rules, payload.d_final_lengths,
      std::max<uint32_t>(1, options.traversals_per_chunk), std::move(consumer),
      stream_options);
}

} // namespace gpu_decompression
