#include "gpu/codec_gpu.cuh"

namespace gpu_codec {

__global__ void expand_path_chunk_kernel(
    const int32_t *__restrict__ d_input,
    const int64_t *__restrict__ d_output_offsets,
    const int32_t *__restrict__ d_expanded_rules,
    const int64_t *__restrict__ d_rule_offsets,
    const int64_t *__restrict__ d_rule_sizes, int32_t *__restrict__ d_output,
    size_t chunk_start_idx, size_t chunk_end_idx, int64_t global_expanded_offset,
    uint32_t min_rule_id, uint32_t max_rule_id) {

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x + chunk_start_idx;
  if (idx >= chunk_end_idx)
    return;

  int32_t val = d_input[idx];
  uint32_t abs_val = static_cast<uint32_t>(val >= 0 ? val : -val);
  int64_t out_offset = d_output_offsets[idx] - global_expanded_offset;

  if (abs_val >= min_rule_id && abs_val < max_rule_id) {
    uint32_t rule_idx = abs_val - min_rule_id;
    int64_t rule_offset = d_rule_offsets[rule_idx];
    int64_t rule_size = d_rule_sizes[rule_idx];

    if (val >= 0) {
      for (int64_t i = 0; i < rule_size; i++) {
        d_output[out_offset + i] = d_expanded_rules[rule_offset + i];
      }
    } else {
      for (int64_t i = 0; i < rule_size; i++) {
        d_output[out_offset + i] =
            -d_expanded_rules[rule_offset + rule_size - 1 - i];
      }
    }
  } else {
    d_output[out_offset] = val;
  }
}

__global__ void segmented_inverse_delta_chunk_kernel(
    const int32_t *input, int32_t *output,
    const uint64_t *offsets,
    uint32_t chunk_seg_start, uint32_t chunk_seg_end,
    uint32_t total_segments, size_t total_nodes,
    uint64_t global_expanded_offset) {
  uint32_t seg = blockIdx.x * blockDim.x + threadIdx.x + chunk_seg_start;
  if (seg >= chunk_seg_end) return;

  uint64_t start = offsets[seg];
  uint64_t end = (seg + 1 < total_segments) ? offsets[seg + 1] : static_cast<uint64_t>(total_nodes);

  if (start >= end) return;  // Empty segment

  uint64_t local_start = start - global_expanded_offset;
  uint64_t local_end = end - global_expanded_offset;

  int32_t acc = input[local_start];
  output[local_start] = acc;
  for (uint64_t i = local_start + 1; i < local_end; ++i) {
    acc += input[i];
    output[i] = acc;
  }
}

void expand_and_inverse_decode_chunk_device(
    const thrust::device_vector<int32_t>& d_encoded_path,
    const thrust::device_vector<int64_t>& d_output_offsets,
    const thrust::device_vector<int32_t>& d_expanded_rules,
    const thrust::device_vector<int64_t>& d_rule_offsets,
    const thrust::device_vector<int64_t>& d_rule_sizes,
    thrust::device_vector<int32_t>& d_chunk_workspace,
    const thrust::device_vector<uint64_t>& d_offs_final,
    size_t chunk_encoded_begin, size_t chunk_encoded_end,
    int64_t chunk_expanded_begin, int64_t chunk_expanded_end,
    uint32_t chunk_segment_begin, uint32_t chunk_segment_end,
    uint32_t min_rule_id, uint32_t max_rule_id, size_t total_nodes) {
    
    size_t num_encoded = chunk_encoded_end - chunk_encoded_begin;
    size_t num_expanded = chunk_expanded_end - chunk_expanded_begin;
    d_chunk_workspace.resize(num_expanded);
    
    if (num_encoded == 0 || num_expanded == 0) return;
    
    int threads = 256;
    int blocks_expand = (num_encoded + threads - 1) / threads;
    expand_path_chunk_kernel<<<blocks_expand, threads>>>(
        thrust::raw_pointer_cast(d_encoded_path.data()),
        thrust::raw_pointer_cast(d_output_offsets.data()),
        thrust::raw_pointer_cast(d_expanded_rules.data()),
        thrust::raw_pointer_cast(d_rule_offsets.data()),
        thrust::raw_pointer_cast(d_rule_sizes.data()),
        thrust::raw_pointer_cast(d_chunk_workspace.data()),
        chunk_encoded_begin, chunk_encoded_end, chunk_expanded_begin,
        min_rule_id, max_rule_id
    );
    CUDA_CHECK(cudaGetLastError());
    
    size_t num_segments = chunk_segment_end - chunk_segment_begin;
    int blocks_inv = (num_segments + threads - 1) / threads;
    segmented_inverse_delta_chunk_kernel<<<blocks_inv, threads>>>(
        thrust::raw_pointer_cast(d_chunk_workspace.data()),
        thrust::raw_pointer_cast(d_chunk_workspace.data()), // in-place inverse delta decode
        thrust::raw_pointer_cast(d_offs_final.data()),
        chunk_segment_begin, chunk_segment_end,
        static_cast<uint32_t>(d_offs_final.size()),
        total_nodes,
        static_cast<uint64_t>(chunk_expanded_begin)
    );
    CUDA_CHECK(cudaGetLastError());
}

} // namespace gpu_codec
