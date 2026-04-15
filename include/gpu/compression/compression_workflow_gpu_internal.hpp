#ifndef COMPRESSION_WORKFLOW_GPU_INTERNAL_HPP
#define COMPRESSION_WORKFLOW_GPU_INTERNAL_HPP

#include "codec/codec.hpp"
#include "gpu/compression/compression_workflow_gpu.hpp"
#include "gpu/core/path_chunk_planner.hpp"

#include <cstdint>
#include <thrust/device_vector.h>
#include <vector>

namespace gpu_compression {

bool scheduler_debug_enabled();
bool compression_debug_enabled();

gfaz::ZstdCompressedBlock compress_uint32_gpu(const std::vector<uint32_t> &input,
                                        const char *label = "uint32_vec");

gfaz::ZstdCompressedBlock compress_int32_gpu(const std::vector<int32_t> &input,
                                       const char *label = "int32_vec");

gfaz::ZstdCompressedBlock
compress_int32_device_gpu(const thrust::device_vector<int32_t> &d_input,
                          const char *label = "int32_device");

gfaz::CompressedData compress_gpu_traversals_legacy_whole_device(
    const FlattenedPaths &paths, uint32_t num_paths, int num_rounds,
    GpuPathCompressionDebugInfo *debug_info = nullptr);

gfaz::CompressedData compress_gpu_traversals_rolling_scheduler(
    const FlattenedPaths &paths, uint32_t num_paths, int num_rounds,
    size_t chunk_bytes, GpuPathCompressionDebugInfo *debug_info = nullptr);

} // namespace gpu_compression

#endif // COMPRESSION_WORKFLOW_GPU_INTERNAL_HPP
