#ifndef DECOMPRESSION_WORKFLOW_GPU_INTERNAL_HPP
#define DECOMPRESSION_WORKFLOW_GPU_INTERNAL_HPP

#include "gpu/compression/compression_workflow_gpu.hpp"
#include "gpu/decompression/decompression_workflow_gpu.hpp"

#include <cstddef>
#include <cstdint>
#include <thrust/device_vector.h>
#include <vector>

namespace gpu_decompression {

bool decompression_debug_enabled();

void decompress_paths_gpu_legacy(
    const thrust::device_vector<int32_t> &d_encoded_path,
    const thrust::device_vector<int32_t> &d_rules_first,
    const thrust::device_vector<int32_t> &d_rules_second,
    uint32_t min_rule_id, size_t num_rules,
    const thrust::device_vector<uint32_t> &d_lens_final,
    std::vector<int32_t> &out_data);

void decompress_paths_gpu_rolling(
    const thrust::device_vector<int32_t> &d_encoded_path,
    const thrust::device_vector<int32_t> &d_rules_first,
    const thrust::device_vector<int32_t> &d_rules_second,
    uint32_t min_rule_id, size_t num_rules,
    const thrust::device_vector<uint32_t> &d_lens_final,
    uint32_t traversals_per_chunk, size_t rolling_output_chunk_bytes,
    std::vector<int32_t> &out_data);

} // namespace gpu_decompression

#endif // DECOMPRESSION_WORKFLOW_GPU_INTERNAL_HPP
