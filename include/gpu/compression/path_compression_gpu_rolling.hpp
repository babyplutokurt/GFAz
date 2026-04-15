#ifndef PATH_COMPRESSION_GPU_ROLLING_HPP
#define PATH_COMPRESSION_GPU_ROLLING_HPP

#include "gpu/compression/compression_workflow_gpu.hpp"

namespace gpu_compression {

gfaz::CompressedData compress_gpu_traversals_rolling_scheduler(
    const FlattenedPaths &paths, uint32_t num_paths, int num_rounds,
    size_t chunk_bytes, GpuPathCompressionDebugInfo *debug_info);

} // namespace gpu_compression

#endif // PATH_COMPRESSION_GPU_ROLLING_HPP
