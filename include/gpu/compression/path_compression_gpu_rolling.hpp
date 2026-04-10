#ifndef PATH_COMPRESSION_GPU_ROLLING_HPP
#define PATH_COMPRESSION_GPU_ROLLING_HPP

#include "gpu/compression/compression_workflow_gpu.hpp"

namespace gpu_compression {

CompressedData run_path_compression_gpu_rolling(const FlattenedPaths &paths,
                                                uint32_t num_paths,
                                                int num_rounds,
                                                size_t chunk_bytes);

} // namespace gpu_compression

#endif // PATH_COMPRESSION_GPU_ROLLING_HPP
