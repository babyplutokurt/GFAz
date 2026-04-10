#include "gpu/decompression/traversal_decode_gpu.hpp"

#include "gpu/core/codec_gpu.cuh"
#include "gpu/decompression/decompression_workflow_gpu_internal.hpp"
#include "utils/runtime_utils.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

namespace gpu_decompression {

namespace {

using Clock = std::chrono::high_resolution_clock;
using gfz::runtime_utils::elapsed_ms;

std::vector<int32_t>
decode_gpu_traversal_without_rules(const GpuTraversalPayload &payload) {
  thrust::device_vector<uint64_t> d_offsets(payload.d_final_lengths.size());
  thrust::exclusive_scan(payload.d_final_lengths.begin(),
                         payload.d_final_lengths.end(), d_offsets.begin(),
                         uint64_t(0));
  thrust::device_vector<int32_t> d_decoded =
      gpu_codec::segmented_inverse_delta_decode_device_vec(
          payload.d_encoded, d_offsets,
          static_cast<uint32_t>(payload.d_final_lengths.size()),
          payload.d_encoded.size());

  std::vector<int32_t> result(d_decoded.size());
  thrust::copy(d_decoded.begin(), d_decoded.end(), result.begin());
  return result;
}

void log_gpu_traversal_decode(
    const GpuTraversalPayload &payload, const GpuTraversalRulebook &rulebook,
    const Clock::time_point &decode_start,
    GpuTraversalDecompressionPath path) {
  if (!decompression_debug_enabled()) {
    return;
  }

  const char *path_label = "rolling materialized";
  if (path == GpuTraversalDecompressionPath::kLegacyMaterialized) {
    path_label = "legacy materialized";
  } else if (path == GpuTraversalDecompressionPath::kRollingDirectWriter) {
    path_label = "rolling direct writer";
  }

  std::cout << "[GPU Decompress] path=" << path_label
            << ", Zstd(host)=" << std::fixed << std::setprecision(2)
            << payload.host_decode_ms
            << " ms, decode_rules(host)=" << rulebook.host_decode_ms
            << " ms, expand(gpu)=" << elapsed_ms(decode_start, Clock::now())
            << " ms" << std::endl;
}

} // namespace

GpuTraversalDecompressionPath resolve_gpu_traversal_decompression_path(
    GpuDecompressionOptions options, bool direct_writer) {
  if (direct_writer) {
    return GpuTraversalDecompressionPath::kRollingDirectWriter;
  }
  if (options.use_legacy_full_decompression) {
    return GpuTraversalDecompressionPath::kLegacyMaterialized;
  }
  return GpuTraversalDecompressionPath::kRollingMaterialized;
}

std::vector<int32_t> decompress_gpu_traversal_materialized(
    const ZstdCompressedBlock &encoded_block,
    const std::vector<uint32_t> &final_lengths,
    const GpuTraversalRulebook &rulebook, GpuDecompressionOptions options,
    GpuTraversalDecodeStats *stats) {
  const GpuTraversalPayload payload =
      prepare_gpu_traversal_payload(encoded_block, final_lengths);
  const auto decode_start = Clock::now();
  if (stats != nullptr) {
    stats->payload_decode_ms = payload.host_decode_ms;
  }

  std::vector<int32_t> result;
  if (rulebook.num_rules == 0) {
    result = decode_gpu_traversal_without_rules(payload);
    if (stats != nullptr) {
      stats->expand_ms = elapsed_ms(decode_start, Clock::now());
    }
    log_gpu_traversal_decode(payload, rulebook, decode_start,
                             resolve_gpu_traversal_decompression_path(options,
                                                                      false));
    return result;
  }

  const GpuTraversalDecompressionPath path =
      resolve_gpu_traversal_decompression_path(options, false);
  if (path == GpuTraversalDecompressionPath::kLegacyMaterialized) {
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

  if (stats != nullptr) {
    stats->expand_ms = elapsed_ms(decode_start, Clock::now());
  }
  log_gpu_traversal_decode(payload, rulebook, decode_start, path);
  return result;
}

void decompress_gpu_traversal_rolling_direct_writer(
    const ZstdCompressedBlock &encoded_block,
    const std::vector<uint32_t> &final_lengths,
    const GpuTraversalRulebook &rulebook, GpuDecompressionOptions options,
    RollingPathChunkConsumer consumer,
    RollingPathStreamOptions stream_options, GpuTraversalDecodeStats *stats) {
  if (options.use_legacy_full_decompression) {
    throw std::runtime_error(
        "GPU traversal streaming is not available in legacy decompression mode");
  }

  const GpuTraversalDecompressionPath path =
      resolve_gpu_traversal_decompression_path(options, true);
  if (path != GpuTraversalDecompressionPath::kRollingDirectWriter) {
    throw std::runtime_error(
        "GPU traversal streaming is only available for the rolling direct-writer path");
  }

  const GpuTraversalPayload payload =
      prepare_gpu_traversal_payload(encoded_block, final_lengths);
  if (stats != nullptr) {
    stats->payload_decode_ms = payload.host_decode_ms;
  }

  const auto decode_start = Clock::now();
  stream_decompress_paths_gpu_rolling(
      payload.d_encoded, rulebook.d_rules_first, rulebook.d_rules_second,
      rulebook.min_rule_id, rulebook.num_rules, payload.d_final_lengths,
      std::max<uint32_t>(1, options.traversals_per_chunk), std::move(consumer),
      stream_options);
  if (stats != nullptr) {
    stats->expand_ms = elapsed_ms(decode_start, Clock::now());
  }
  log_gpu_traversal_decode(payload, rulebook, decode_start, path);
}

} // namespace gpu_decompression
