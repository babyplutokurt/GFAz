#ifndef DECOMPRESSION_WORKFLOW_GPU_HPP
#define DECOMPRESSION_WORKFLOW_GPU_HPP

#include "gfa_graph_gpu.hpp"
#include "compression_workflow_gpu.hpp"
#include <vector>
#include <cstdint>

namespace gpu_decompression {

/**
 * GPU-accelerated path decompression (rule expansion).
 * 
 * Decompresses the encoded path by:
 * 1. nvComp ZSTD decompressing encoded_path, rules_first, rules_second
 * 2. Inverse delta-encoding the rules
 * 3. Iteratively expanding rules until all are raw nodes
 * 4. Inverse delta-encoding the path
 * 
 * @param data CompressedData_gpu with compressed paths and rules
 * @return FlattenedPaths with decompressed path data and lengths
 */
FlattenedPaths decompress_paths_gpu(const gpu_compression::CompressedData_gpu& data);

/**
 * Full GPU decompression: CompressedData_gpu -> GfaGraph_gpu
 * 
 * Decompresses all components:
 * - Paths (using GPU rule expansion)
 * - Path names and overlaps
 * - Segment sequences
 * - Links
 * - Optional fields
 * 
 * @param data CompressedData_gpu with all compressed fields
 * @return GfaGraph_gpu with fully decompressed data
 */
GfaGraph_gpu decompress_to_gpu_layout(const gpu_compression::CompressedData_gpu& data);

void set_gpu_decompression_debug(bool enabled);

} // namespace gpu_decompression

#endif // DECOMPRESSION_WORKFLOW_GPU_HPP
