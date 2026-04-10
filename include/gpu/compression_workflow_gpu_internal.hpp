#ifndef COMPRESSION_WORKFLOW_GPU_INTERNAL_HPP
#define COMPRESSION_WORKFLOW_GPU_INTERNAL_HPP

#include "codec/codec.hpp"
#include "gpu/compression_workflow_gpu.hpp"
#include "gpu/path_chunk_planner.hpp"

#include <cstdint>
#include <thrust/device_vector.h>
#include <vector>

namespace gpu_compression {

bool scheduler_debug_enabled();
bool compression_debug_enabled();

ZstdCompressedBlock compress_uint32_gpu(const std::vector<uint32_t> &input,
                                        const char *label = "uint32_vec");

ZstdCompressedBlock compress_int32_gpu(const std::vector<int32_t> &input,
                                       const char *label = "int32_vec");

ZstdCompressedBlock
compress_int32_device_gpu(const thrust::device_vector<int32_t> &d_input,
                          const char *label = "int32_device");

CompressedData run_path_compression_gpu_full_device(const FlattenedPaths &paths,
                                                    uint32_t num_paths,
                                                    int num_rounds,
                                                    GpuPathCompressionDebugInfo
                                                        *debug_info = nullptr);

CompressedData run_path_compression_gpu_rolling(const FlattenedPaths &paths,
                                                uint32_t num_paths,
                                                int num_rounds,
                                                size_t chunk_bytes,
                                                GpuPathCompressionDebugInfo
                                                    *debug_info = nullptr);

} // namespace gpu_compression

#endif // COMPRESSION_WORKFLOW_GPU_INTERNAL_HPP
