#ifndef COMPRESSION_WORKFLOW_GPU_INTERNAL_HPP
#define COMPRESSION_WORKFLOW_GPU_INTERNAL_HPP

#include "gpu/codec_gpu_nvcomp.cuh"
#include "gpu/compression_workflow_gpu.hpp"
#include "gpu/path_chunk_planner.hpp"

#include <cstdint>
#include <thrust/device_vector.h>
#include <vector>

namespace gpu_compression {

bool scheduler_debug_enabled();
bool compression_debug_enabled();

gpu_codec::NvcompCompressedBlock
compress_uint32_gpu(const std::vector<uint32_t> &input,
                    const char *label = "uint32_vec");

gpu_codec::NvcompCompressedBlock
compress_int32_gpu(const std::vector<int32_t> &input,
                   const char *label = "int32_vec");

gpu_codec::NvcompCompressedBlock
compress_int32_device_gpu(const thrust::device_vector<int32_t> &d_input,
                          const char *label = "int32_device");

CompressedData_gpu run_path_compression_gpu_full_device(
    const FlattenedPaths &paths, int num_rounds);

CompressedData_gpu run_path_compression_gpu_rolling(const FlattenedPaths &paths,
                                                    int num_rounds,
                                                    size_t chunk_bytes);

} // namespace gpu_compression

#endif // COMPRESSION_WORKFLOW_GPU_INTERNAL_HPP
