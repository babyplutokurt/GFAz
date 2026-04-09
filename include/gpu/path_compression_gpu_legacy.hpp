#ifndef PATH_COMPRESSION_GPU_LEGACY_HPP
#define PATH_COMPRESSION_GPU_LEGACY_HPP

#include "gpu/compression_workflow_gpu.hpp"

namespace gpu_compression {

CompressedData run_path_compression_gpu_full_device(const FlattenedPaths &paths,
                                                    uint32_t num_paths,
                                                    int num_rounds);

} // namespace gpu_compression

#endif // PATH_COMPRESSION_GPU_LEGACY_HPP
