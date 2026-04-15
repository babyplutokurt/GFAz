#include "gpu/decompression/decompression_primitives_gpu.hpp"
#include "gpu/decompression/traversal_decode_gpu.hpp"
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

void append_decoded_traversal_column(
    std::vector<int32_t> &decoded_nodes, std::vector<uint32_t> &decoded_lengths,
    const gfaz::ZstdCompressedBlock &encoded_block,
    const std::vector<uint32_t> &final_lengths,
    const GpuTraversalRulebook &rulebook, GpuDecompressionOptions options) {
  if (encoded_block.payload.empty() || final_lengths.empty()) {
    return;
  }

  std::vector<int32_t> decoded = decompress_gpu_traversal_materialized(
      encoded_block, final_lengths, rulebook, options);
  decoded_nodes.insert(decoded_nodes.end(), decoded.begin(), decoded.end());
  decoded_lengths.insert(decoded_lengths.end(), final_lengths.begin(),
                         final_lengths.end());
}

} // namespace

FlattenedPaths decompress_paths_gpu(const gfaz::CompressedData &data,
                                    GpuDecompressionOptions options) {
  FlattenedPaths result;
  const GpuTraversalRulebook rulebook = prepare_gpu_traversal_rulebook(data);
  append_decoded_traversal_column(result.data, result.lengths, data.paths_zstd,
                                  data.original_path_lengths, rulebook,
                                  options);
  return result;
}

GfaGraph_gpu decompress_to_gpu_layout(const gfaz::CompressedData &data,
                                      GpuDecompressionOptions options) {
  auto start = Clock::now();

  GfaGraph_gpu result;
  const GpuTraversalRulebook rulebook = prepare_gpu_traversal_rulebook(data);
  append_decoded_traversal_column(result.paths.data, result.paths.lengths,
                                  data.paths_zstd, data.original_path_lengths,
                                  rulebook, options);
  append_decoded_traversal_column(result.paths.data, result.paths.lengths,
                                  data.walks_zstd, data.original_walk_lengths,
                                  rulebook, options);

  decompress_graph_metadata_gpu(data, result);

  if (g_debug_decompression) {
    auto end = Clock::now();
    std::cout << "[GPU Decompress] Full decompression: " << std::fixed
              << std::setprecision(2) << elapsed_ms(start, end) << " ms"
              << std::endl;
  }

  return result;
}

gfaz::GfaGraph decompress_to_host_graph(const gfaz::CompressedData &data,
                                  GpuDecompressionOptions options) {
  return convert_from_gpu_layout(decompress_to_gpu_layout(data, options));
}

} // namespace gpu_decompression
