#ifndef PATH_COMPRESSION_GPU_LEGACY_HPP
#define PATH_COMPRESSION_GPU_LEGACY_HPP

#include "gpu/compression/compression_workflow_gpu.hpp"

namespace gpu_compression {

CompressedData compress_gpu_traversals_legacy_whole_device(
    const FlattenedPaths &paths, uint32_t num_paths, int num_rounds,
    GpuPathCompressionDebugInfo *debug_info);

} // namespace gpu_compression

#endif // PATH_COMPRESSION_GPU_LEGACY_HPP
