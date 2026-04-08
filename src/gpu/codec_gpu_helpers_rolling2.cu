#include "gpu/codec_gpu.cuh"

namespace gpu_codec {

void rolling_expand_and_inverse_delta_decode(
    const thrust::device_vector<int32_t> &d_encoded_path,
    const thrust::device_vector<int32_t> &d_rules_first,
    const thrust::device_vector<int32_t> &d_rules_second,
    uint32_t min_rule_id, size_t num_rules,
    const thrust::device_vector<uint32_t> &d_lens_final,
    std::vector<int32_t> &h_result_data) {

  const uint32_t max_rule_id = min_rule_id + static_cast<uint32_t>(num_rules);
  const int threads = 256;

  thrust::device_vector<int64_t> d_rule_sizes(num_rules);
  {
    int blocks = (num_rules + threads - 1) / threads;
    compute_rule_final_sizes_kernel<<<blocks, threads>>>(
        thrust::raw_pointer_cast(d_rules_first.data()),
        thrust::raw_pointer_cast(d_rules_second.data()),
        thrust::raw_pointer_cast(d_rule_sizes.data()), num_rules, min_rule_id);
    CUDA_CHECK(cudaGetLastError());
  }

  thrust::device_vector<int64_t> d_rule_offsets(num_rules);
  thrust::exclusive_scan(d_rule_sizes.begin(), d_rule_sizes.end(),
                         d_rule_offsets.begin());

  int64_t total_expanded_size = d_rule_offsets.back() + d_rule_sizes.back();
  thrust::device_vector<int32_t> d_expanded_rules(total_expanded_size);
  {
    int blocks = (num_rules + threads - 1) / threads;
    expand_rules_to_buffer_kernel<<<blocks, threads>>>(
        thrust::raw_pointer_cast(d_rules_first.data()),
        thrust::raw_pointer_cast(d_rules_second.data()),
        thrust::raw_pointer_cast(d_rule_offsets.data()),
        thrust::raw_pointer_cast(d_expanded_rules.data()), num_rules,
        min_rule_id);
    CUDA_CHECK(cudaGetLastError());
  }

  size_t num_elements = d_encoded_path.size();
  FinalExpansionSizeOp size_op(thrust::raw_pointer_cast(d_rule_sizes.data()),
                               min_rule_id, max_rule_id);
  auto size_iter =
      thrust::make_transform_iterator(d_encoded_path.begin(), size_op);

  int64_t output_size =
      thrust::transform_reduce(d_encoded_path.begin(), d_encoded_path.end(),
                               size_op, (int64_t)0, thrust::plus<int64_t>());

  thrust::device_vector<int64_t> d_output_offsets(num_elements);
  thrust::exclusive_scan(size_iter, size_iter + num_elements,
                         d_output_offsets.begin());

  uint32_t num_segs_final = static_cast<uint32_t>(d_lens_final.size());
  thrust::device_vector<uint64_t> d_offs_final(num_segs_final);
  thrust::exclusive_scan(d_lens_final.begin(), d_lens_final.end(),
                         d_offs_final.begin(), uint64_t(0));

  thrust::device_vector<int64_t> d_encoded_offsets(num_segs_final);
  thrust::lower_bound(d_output_offsets.begin(), d_output_offsets.end(),
                      d_offs_final.begin(), d_offs_final.end(),
                      d_encoded_offsets.begin());

  std::vector<int64_t> h_encoded_offsets(num_segs_final);
  thrust::copy(d_encoded_offsets.begin(), d_encoded_offsets.end(),
               h_encoded_offsets.begin());
  std::vector<uint64_t> h_expanded_offsets(num_segs_final);
  thrust::copy(d_offs_final.begin(), d_offs_final.end(),
               h_expanded_offsets.begin());

  h_result_data.resize(output_size);

  thrust::device_vector<int32_t> d_chunk_workspace;

  uint32_t current_seg_begin = 0;
  for (uint32_t i = 0; i < num_segs_final; ++i) {
    int64_t seg_expanded_start = h_expanded_offsets[current_seg_begin];
    int64_t next_expanded_start = (i + 1 < num_segs_final) ? h_expanded_offsets[i + 1] : output_size;
    int64_t size_so_far = next_expanded_start - seg_expanded_start;

    bool split = false;
    // Chunking condition: 128 paths/walks OR 128 MB elements (32 million paths equivalent)
    if (i - current_seg_begin >= 128) split = true;
    if (size_so_far >= 32 * 1024 * 1024 && i > current_seg_begin) split = true;
    if (i + 1 == num_segs_final) split = true;

    if (split) {
      uint32_t chunk_seg_end = i + 1;
      int64_t chunk_encoded_begin = h_encoded_offsets[current_seg_begin];
      int64_t chunk_encoded_end = (chunk_seg_end < num_segs_final) ? h_encoded_offsets[chunk_seg_end] : d_encoded_path.size();
      int64_t chunk_expanded_begin = h_expanded_offsets[current_seg_begin];
      int64_t chunk_expanded_end = next_expanded_start;

      expand_and_inverse_decode_chunk_device(
          d_encoded_path, d_output_offsets, d_expanded_rules,
          d_rule_offsets, d_rule_sizes, d_chunk_workspace,
          d_offs_final, chunk_encoded_begin, chunk_encoded_end,
          chunk_expanded_begin, chunk_expanded_end,
          current_seg_begin, chunk_seg_end,
          min_rule_id, max_rule_id, output_size);

      thrust::copy(d_chunk_workspace.begin(), d_chunk_workspace.end(),
                   h_result_data.begin() + chunk_expanded_begin);

      current_seg_begin = chunk_seg_end;
    }
  }
}

} // namespace gpu_codec
