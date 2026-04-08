#include "gpu/codec_gpu.cuh"

namespace gpu_codec {

void remap_chunk_rule_ids_device_vec(
    thrust::device_vector<int32_t>& d_chunk_data,
    const thrust::device_vector<uint64_t>& d_new_indices,
    const thrust::device_vector<uint32_t>& d_reorder_map,
    uint32_t start_id,
    uint32_t num_rules_before_compact,
    uint32_t num_rules_after_compact) {
    if (d_chunk_data.empty()) return;
    
    int threads = 256;
    int blocks = (d_chunk_data.size() + threads - 1) / threads;
    
    remap_paths_kernel<<<blocks, threads>>>(
        thrust::raw_pointer_cast(d_chunk_data.data()), d_chunk_data.size(),
        thrust::raw_pointer_cast(d_new_indices.data()), start_id, num_rules_before_compact
    );
    CUDA_CHECK(cudaGetLastError());
    
    if (num_rules_after_compact > 0) {
        remap_paths_reorder_kernel<<<blocks, threads>>>(
            thrust::raw_pointer_cast(d_chunk_data.data()), d_chunk_data.size(),
            thrust::raw_pointer_cast(d_reorder_map.data()), start_id, num_rules_after_compact
        );
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace gpu_codec
