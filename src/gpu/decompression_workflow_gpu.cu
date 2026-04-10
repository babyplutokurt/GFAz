#include "codec/codec.hpp"
#include "gpu/codec_gpu.cuh"
#include "gpu/decompression_workflow_gpu.hpp"
#include "gpu/gfa_graph_gpu.hpp"
#include "gpu/metadata_codec_gpu.hpp"
#include "gpu/path_decompression_gpu_legacy.hpp"
#include "gpu/path_decompression_gpu_rolling.hpp"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

namespace gpu_decompression {

static bool g_debug_decompression = false;
using Clock = std::chrono::high_resolution_clock;

void set_gpu_decompression_debug(bool enabled) {
  g_debug_decompression = enabled;
}

bool decompression_debug_enabled() { return g_debug_decompression; }

static double elapsed_ms(const Clock::time_point &start,
                         const Clock::time_point &end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

namespace {

std::vector<int32_t> expand_sequence_gpu(
                                         const ZstdCompressedBlock &encoded_block,
                                         const std::vector<uint32_t> & /*encoded_lengths*/,
                                         const std::vector<uint32_t> &final_lengths,
                                         const CompressedData &data,
                                         GpuDecompressionOptions options) {
  auto start = Clock::now();

  std::vector<int32_t> encoded_host =
      Codec::zstd_decompress_int32_vector(encoded_block);
  std::vector<int32_t> first_host =
      Codec::zstd_decompress_int32_vector(data.rules_first_zstd);
  std::vector<int32_t> second_host =
      Codec::zstd_decompress_int32_vector(data.rules_second_zstd);

  auto zstd_end = Clock::now();
  Codec::delta_decode_int32(first_host);
  Codec::delta_decode_int32(second_host);

  thrust::device_vector<int32_t> d_encoded(encoded_host.begin(),
                                           encoded_host.end());
  thrust::device_vector<int32_t> d_rules_first(first_host.begin(),
                                               first_host.end());
  thrust::device_vector<int32_t> d_rules_second(second_host.begin(),
                                                second_host.end());
  thrust::device_vector<uint32_t> d_final_lengths(final_lengths.begin(),
                                                  final_lengths.end());

  const auto rules_end = Clock::now();
  const uint32_t min_rule_id = data.min_rule_id();
  const size_t num_rules =
      std::min({data.total_rules(), d_rules_first.size(), d_rules_second.size()});

  std::vector<int32_t> result;
  if (num_rules == 0) {
    thrust::device_vector<uint64_t> d_offsets(d_final_lengths.size());
    thrust::exclusive_scan(d_final_lengths.begin(), d_final_lengths.end(),
                           d_offsets.begin(),
                           uint64_t(0));
    thrust::device_vector<int32_t> d_decoded =
        gpu_codec::segmented_inverse_delta_decode_device_vec(
            d_encoded, d_offsets, static_cast<uint32_t>(d_final_lengths.size()),
            d_encoded.size());
    result.resize(d_decoded.size());
    thrust::copy(d_decoded.begin(), d_decoded.end(), result.begin());
  } else if (options.use_legacy_full_decompression) {
    decompress_paths_gpu_legacy(d_encoded, d_rules_first, d_rules_second,
                                min_rule_id, num_rules, d_final_lengths, result);
  } else {
    decompress_paths_gpu_rolling(
        d_encoded, d_rules_first, d_rules_second, min_rule_id, num_rules,
        d_final_lengths, std::max<uint32_t>(1, options.traversals_per_chunk),
        std::max<size_t>(1, options.max_expanded_chunk_bytes),
        result);
  }

  if (g_debug_decompression) {
    const auto end = Clock::now();
    std::cout << "[GPU Decompress] Zstd(host)=" << std::fixed
              << std::setprecision(2) << elapsed_ms(start, zstd_end)
              << " ms, decode_rules(host)=" << elapsed_ms(zstd_end, rules_end)
              << " ms, expand(gpu)=" << elapsed_ms(rules_end, end) << " ms"
              << std::endl;
  }

  return result;
}

} // namespace

FlattenedPaths decompress_paths_gpu(const CompressedData &data,
                                    GpuDecompressionOptions options) {
  FlattenedPaths result;
  if (!data.paths_zstd.payload.empty() && !data.sequence_lengths.empty()) {
    result.data = expand_sequence_gpu(data.paths_zstd, data.sequence_lengths,
                                      data.original_path_lengths, data,
                                      options);
  }
  result.lengths = data.original_path_lengths;
  return result;
}

GfaGraph_gpu decompress_to_gpu_layout(const CompressedData &data,
                                      GpuDecompressionOptions options) {
  auto start = Clock::now();

  GfaGraph_gpu result;
  result.paths = decompress_paths_gpu(data, options);
  if (!data.walks_zstd.payload.empty() && !data.walk_lengths.empty()) {
    std::vector<int32_t> walks = expand_sequence_gpu(
        data.walks_zstd, data.walk_lengths, data.original_walk_lengths, data,
        options);
    result.paths.data.insert(result.paths.data.end(), walks.begin(), walks.end());
    result.paths.lengths.insert(result.paths.lengths.end(),
                                data.original_walk_lengths.begin(),
                                data.original_walk_lengths.end());
  }

  decompress_graph_metadata_gpu(data, result);

  if (g_debug_decompression) {
    auto end = Clock::now();
    std::cout << "[GPU Decompress] Full decompression: " << std::fixed
              << std::setprecision(2) << elapsed_ms(start, end) << " ms"
              << std::endl;
  }

  return result;
}

GfaGraph decompress_to_host_graph(const CompressedData &data,
                                  GpuDecompressionOptions options) {
  return convert_from_gpu_layout(decompress_to_gpu_layout(data, options));
}

} // namespace gpu_decompression
