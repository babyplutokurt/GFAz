#ifndef PATH_DECOMPRESSION_GPU_ROLLING_HPP
#define PATH_DECOMPRESSION_GPU_ROLLING_HPP

#include <cstddef>
#include <cstdint>
#include <thrust/device_vector.h>
#include <vector>

namespace gpu_decompression {

void decompress_paths_gpu_rolling(
    const thrust::device_vector<int32_t> &d_encoded_path,
    const thrust::device_vector<int32_t> &d_rules_first,
    const thrust::device_vector<int32_t> &d_rules_second,
    uint32_t min_rule_id, size_t num_rules,
    const thrust::device_vector<uint32_t> &d_lens_final,
    uint32_t traversals_per_chunk, std::vector<int32_t> &out_data);

} // namespace gpu_decompression

#endif // PATH_DECOMPRESSION_GPU_ROLLING_HPP
