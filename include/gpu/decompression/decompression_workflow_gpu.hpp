#ifndef DECOMPRESSION_WORKFLOW_GPU_HPP
#define DECOMPRESSION_WORKFLOW_GPU_HPP

#include "gpu/compression/compression_workflow_gpu.hpp"
#include "gpu/core/gfa_graph_gpu.hpp"
#include <cstdint>
#include <vector>

namespace gpu_decompression {

constexpr size_t kDefaultRollingOutputChunkBytes =
    1024ull * 1024ull * 1024ull;

struct GpuDecompressionOptions {
  uint32_t traversals_per_chunk = 4096;
  size_t rolling_output_chunk_bytes = kDefaultRollingOutputChunkBytes;
  bool use_legacy_full_decompression = false;
};

/**
 * GPU-accelerated path decompression (rule expansion).
 *
 * Decompresses the encoded path by:
 * 1. Zstd decompressing encoded_path, rules_first, rules_second on host
 * 2. Inverse delta-encoding the rules
 * 3. Iteratively expanding rules until all are raw nodes
 * 4. Inverse delta-encoding the path
 *
 * @param data gfaz::CompressedData with compressed P-line paths and rules
 * @return FlattenedPaths with decompressed path data and lengths
 */
FlattenedPaths decompress_paths_gpu(const gfaz::CompressedData &data,
                                    GpuDecompressionOptions options = {});

/**
 * Full GPU decompression: gfaz::CompressedData -> GfaGraph_gpu
 *
 * Decompresses all components:
 * - Paths (using GPU rule expansion)
 * - Path names and overlaps
 * - Segment sequences
 * - Links
 * - Optional fields
 *
 * @param data gfaz::CompressedData with all compressed fields
 * @return GfaGraph_gpu with fully decompressed data
 */
GfaGraph_gpu decompress_to_gpu_layout(const gfaz::CompressedData &data,
                                      GpuDecompressionOptions options = {});

/**
 * Full GPU decompression to host graph: gfaz::CompressedData -> gfaz::GfaGraph
 *
 * Uses the configured GPU path decompression mode (legacy whole-device or
 * rolling traversal expansion), then reconstructs the host-side graph model.
 *
 * @param data gfaz::CompressedData with all compressed fields
 * @return gfaz::GfaGraph with fully decompressed host-side data
 */
gfaz::GfaGraph decompress_to_host_graph(const gfaz::CompressedData &data,
                                  GpuDecompressionOptions options = {});

void set_gpu_decompression_debug(bool enabled);

} // namespace gpu_decompression

#endif // DECOMPRESSION_WORKFLOW_GPU_HPP
