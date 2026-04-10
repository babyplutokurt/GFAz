#include "gpu/decompression/decompression_primitives_gpu.hpp"
#include "gpu/decompression/decompression_workflow_gpu.hpp"
#include "gpu/core/gfa_graph_gpu.hpp"
#include "gpu/compression/metadata_codec_gpu.hpp"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>

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

std::vector<int32_t> expand_sequence_gpu(const ZstdCompressedBlock &encoded_block,
                                         const std::vector<uint32_t> &final_lengths,
                                         const GpuTraversalRulebook &rulebook,
                                         GpuDecompressionOptions options) {
  const GpuTraversalPayload payload =
      prepare_gpu_traversal_payload(encoded_block, final_lengths);
  const auto prep_end = Clock::now();
  std::vector<int32_t> result =
      decode_gpu_traversal_to_host(payload, rulebook, options);

  if (g_debug_decompression) {
    const auto end = Clock::now();
    std::cout << "[GPU Decompress] Zstd(host)=" << std::fixed
              << std::setprecision(2) << payload.host_decode_ms
              << " ms, decode_rules(host)=" << rulebook.host_decode_ms
              << " ms, expand(gpu)=" << elapsed_ms(prep_end, end) << " ms"
              << std::endl;
  }

  return result;
}

} // namespace

FlattenedPaths decompress_paths_gpu(const CompressedData &data,
                                    GpuDecompressionOptions options) {
  FlattenedPaths result;
  const GpuTraversalRulebook rulebook = prepare_gpu_traversal_rulebook(data);
  if (!data.paths_zstd.payload.empty() && !data.sequence_lengths.empty()) {
    result.data = expand_sequence_gpu(data.paths_zstd,
                                      data.original_path_lengths, rulebook,
                                      options);
  }
  result.lengths = data.original_path_lengths;
  return result;
}

GfaGraph_gpu decompress_to_gpu_layout(const CompressedData &data,
                                      GpuDecompressionOptions options) {
  auto start = Clock::now();

  GfaGraph_gpu result;
  const GpuTraversalRulebook rulebook = prepare_gpu_traversal_rulebook(data);
  if (!data.paths_zstd.payload.empty() && !data.sequence_lengths.empty()) {
    result.paths.data = expand_sequence_gpu(data.paths_zstd,
                                            data.original_path_lengths, rulebook,
                                            options);
  }
  result.paths.lengths = data.original_path_lengths;
  if (!data.walks_zstd.payload.empty() && !data.walk_lengths.empty()) {
    std::vector<int32_t> walks = expand_sequence_gpu(
        data.walks_zstd, data.original_walk_lengths, rulebook, options);
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
