#include "gpu/core/codec_gpu.cuh"
#include "gpu/decompression/decompression_workflow_gpu.hpp"

#include <cuda_runtime.h>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/transform_reduce.h>

#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      throw std::runtime_error(std::string("CUDA error at ") + __FILE__ +      \
                               ":" + std::to_string(__LINE__) + " - " +        \
                               cudaGetErrorString(err));                       \
    }                                                                          \
  } while (0)

namespace gpu_codec {

struct ExpansionSizeOp_Deprecated {
  uint32_t min_rule_id;
  uint32_t max_rule_id;

  __host__ __device__ ExpansionSizeOp_Deprecated(uint32_t min_id,
                                                 size_t num_rules)
      : min_rule_id(min_id),
        max_rule_id(min_id + static_cast<uint32_t>(num_rules)) {}

  __host__ __device__ __forceinline__ int64_t operator()(int32_t val) const {
    uint32_t abs_val = static_cast<uint32_t>(val >= 0 ? val : -val);
    return (abs_val >= min_rule_id && abs_val < max_rule_id) ? 2 : 1;
  }
};

__global__ void expand_rules_kernel_deprecated(
    const int32_t *__restrict__ d_input, const int64_t *__restrict__ d_offsets,
    const int32_t *__restrict__ d_rules_first,
    const int32_t *__restrict__ d_rules_second, int32_t *__restrict__ d_output,
    size_t num_elements, uint32_t min_rule_id, uint32_t max_rule_id) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_elements) {
    return;
  }

  int32_t val = d_input[idx];
  uint32_t abs_val = static_cast<uint32_t>(val >= 0 ? val : -val);
  int64_t out_offset = d_offsets[idx];

  if (abs_val >= min_rule_id && abs_val < max_rule_id) {
    uint32_t rule_idx = abs_val - min_rule_id;
    int32_t first = d_rules_first[rule_idx];
    int32_t second = d_rules_second[rule_idx];

    if (val >= 0) {
      d_output[static_cast<size_t>(out_offset)] = first;
      d_output[static_cast<size_t>(out_offset + 1)] = second;
    } else {
      d_output[static_cast<size_t>(out_offset)] = -second;
      d_output[static_cast<size_t>(out_offset + 1)] = -first;
    }
  } else {
    d_output[static_cast<size_t>(out_offset)] = val;
  }
}

thrust::device_vector<int32_t> expand_path_device_vec_iterative(
    const thrust::device_vector<int32_t> &d_encoded_path,
    const thrust::device_vector<int32_t> &d_rules_first,
    const thrust::device_vector<int32_t> &d_rules_second, uint32_t min_rule_id,
    size_t num_rules) {
  if (d_encoded_path.empty()) {
    return thrust::device_vector<int32_t>();
  }

  const uint32_t max_rule_id = min_rule_id + static_cast<uint32_t>(num_rules);
  const int threads = 256;
  const int max_passes = 64;

  size_t initial_size = d_encoded_path.size();
  size_t buffer_capacity = initial_size * 4;

  thrust::device_vector<int32_t> d_buffer_a;
  thrust::device_vector<int32_t> d_buffer_b;
  d_buffer_a.reserve(buffer_capacity);
  d_buffer_b.reserve(buffer_capacity);
  d_buffer_a = d_encoded_path;

  thrust::device_vector<int32_t> *d_current = &d_buffer_a;
  thrust::device_vector<int32_t> *d_output = &d_buffer_b;

  thrust::device_vector<int64_t> d_offsets;
  d_offsets.reserve(buffer_capacity);

  for (int pass = 0; pass < max_passes; ++pass) {
    size_t num_elements = d_current->size();
    int blocks = (num_elements + threads - 1) / threads;

    ExpansionSizeOp_Deprecated size_op(min_rule_id, num_rules);
    auto size_iter =
        thrust::make_transform_iterator(d_current->begin(), size_op);

    d_offsets.resize(num_elements);
    thrust::exclusive_scan(size_iter, size_iter + num_elements,
                           d_offsets.begin());

    int64_t last_offset = d_offsets.back();
    int64_t last_size = size_op(d_current->back());
    size_t new_size = static_cast<size_t>(last_offset + last_size);

    if (new_size == num_elements) {
      break;
    }

    if (new_size > d_output->capacity()) {
      d_output->reserve(new_size * 2);
    }
    d_output->resize(new_size);

    expand_rules_kernel_deprecated<<<blocks, threads>>>(
        thrust::raw_pointer_cast(d_current->data()),
        thrust::raw_pointer_cast(d_offsets.data()),
        thrust::raw_pointer_cast(d_rules_first.data()),
        thrust::raw_pointer_cast(d_rules_second.data()),
        thrust::raw_pointer_cast(d_output->data()), num_elements, min_rule_id,
        max_rule_id);
    CUDA_CHECK(cudaGetLastError());

    std::swap(d_current, d_output);
  }

  return std::move(*d_current);
}

static constexpr int EXPANSION_STACK_CAPACITY = 256;

__global__ void compute_rule_final_sizes_kernel(
    const int32_t *__restrict__ d_rules_first,
    const int32_t *__restrict__ d_rules_second,
    int64_t *__restrict__ d_rule_sizes,
    size_t num_rules, uint32_t min_rule_id) {
  size_t rule_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (rule_idx >= num_rules) {
    return;
  }

  int32_t stack[EXPANSION_STACK_CAPACITY];
  int stack_ptr = 0;
  int64_t final_size = 0;

  stack[stack_ptr++] = d_rules_first[rule_idx];
  stack[stack_ptr++] = d_rules_second[rule_idx];

  while (stack_ptr > 0) {
    int32_t val = stack[--stack_ptr];
    uint32_t abs_val = static_cast<uint32_t>(val >= 0 ? val : -val);

    if (abs_val >= min_rule_id && abs_val < min_rule_id + num_rules) {
      uint32_t child_idx = abs_val - min_rule_id;
      if (stack_ptr + 2 <= EXPANSION_STACK_CAPACITY) {
        stack[stack_ptr++] = d_rules_first[child_idx];
        stack[stack_ptr++] = d_rules_second[child_idx];
      } else {
        final_size++;
      }
    } else {
      final_size++;
    }
  }

  d_rule_sizes[rule_idx] = final_size;
}

void compute_rule_final_sizes_device_vec(
    const thrust::device_vector<int32_t> &d_rules_first,
    const thrust::device_vector<int32_t> &d_rules_second,
    thrust::device_vector<int64_t> &d_rule_sizes, uint32_t min_rule_id) {
  const size_t num_rules = d_rule_sizes.size();
  if (num_rules == 0) {
    return;
  }

  const int threads = 256;
  const int blocks = (num_rules + threads - 1) / threads;
  compute_rule_final_sizes_kernel<<<blocks, threads>>>(
      thrust::raw_pointer_cast(d_rules_first.data()),
      thrust::raw_pointer_cast(d_rules_second.data()),
      thrust::raw_pointer_cast(d_rule_sizes.data()), num_rules, min_rule_id);
  CUDA_CHECK(cudaGetLastError());
}

__global__ void expand_rules_to_buffer_kernel(
    const int32_t *__restrict__ d_rules_first,
    const int32_t *__restrict__ d_rules_second,
    const int64_t *__restrict__ d_rule_offsets,
    int32_t *__restrict__ d_expanded_rules,
    size_t num_rules, uint32_t min_rule_id) {
  size_t rule_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (rule_idx >= num_rules) {
    return;
  }

  int64_t write_pos = d_rule_offsets[rule_idx];
  int32_t stack[EXPANSION_STACK_CAPACITY];
  int stack_ptr = 0;

  stack[stack_ptr++] = d_rules_second[rule_idx];
  stack[stack_ptr++] = d_rules_first[rule_idx];

  while (stack_ptr > 0) {
    int32_t val = stack[--stack_ptr];
    uint32_t abs_val = static_cast<uint32_t>(val >= 0 ? val : -val);

    if (abs_val >= min_rule_id && abs_val < min_rule_id + num_rules) {
      uint32_t child_idx = abs_val - min_rule_id;
      if (stack_ptr + 2 <= EXPANSION_STACK_CAPACITY) {
        if (val >= 0) {
          stack[stack_ptr++] = d_rules_second[child_idx];
          stack[stack_ptr++] = d_rules_first[child_idx];
        } else {
          stack[stack_ptr++] = -d_rules_first[child_idx];
          stack[stack_ptr++] = -d_rules_second[child_idx];
        }
      } else {
        d_expanded_rules[write_pos++] = val;
      }
    } else {
      d_expanded_rules[write_pos++] = val;
    }
  }
}

void expand_rules_to_buffer_device_vec(
    const thrust::device_vector<int32_t> &d_rules_first,
    const thrust::device_vector<int32_t> &d_rules_second,
    const thrust::device_vector<int64_t> &d_rule_offsets,
    thrust::device_vector<int32_t> &d_expanded_rules, uint32_t min_rule_id) {
  const size_t num_rules = d_rule_offsets.size();
  if (num_rules == 0) {
    return;
  }

  const int threads = 256;
  const int blocks = (num_rules + threads - 1) / threads;
  expand_rules_to_buffer_kernel<<<blocks, threads>>>(
      thrust::raw_pointer_cast(d_rules_first.data()),
      thrust::raw_pointer_cast(d_rules_second.data()),
      thrust::raw_pointer_cast(d_rule_offsets.data()),
      thrust::raw_pointer_cast(d_expanded_rules.data()), num_rules,
      min_rule_id);
  CUDA_CHECK(cudaGetLastError());
}

struct FinalExpansionSizeOp {
  const int64_t *d_rule_sizes;
  uint32_t min_rule_id;
  uint32_t max_rule_id;

  __host__ __device__ FinalExpansionSizeOp(const int64_t *sizes,
                                           uint32_t min_id, uint32_t max_id)
      : d_rule_sizes(sizes), min_rule_id(min_id), max_rule_id(max_id) {}

  __host__ __device__ __forceinline__ int64_t operator()(int32_t val) const {
    uint32_t abs_val = static_cast<uint32_t>(val >= 0 ? val : -val);
    if (abs_val >= min_rule_id && abs_val < max_rule_id) {
      return d_rule_sizes[abs_val - min_rule_id];
    }
    return 1;
  }
};

__global__ void expand_path_single_pass_kernel(
    const int32_t *__restrict__ d_input,
    const int64_t *__restrict__ d_output_offsets,
    const int32_t *__restrict__ d_expanded_rules,
    const int64_t *__restrict__ d_rule_offsets,
    const int64_t *__restrict__ d_rule_sizes, int32_t *__restrict__ d_output,
    size_t num_elements, uint32_t min_rule_id, uint32_t max_rule_id) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_elements) {
    return;
  }

  int32_t val = d_input[idx];
  uint32_t abs_val = static_cast<uint32_t>(val >= 0 ? val : -val);
  int64_t out_offset = d_output_offsets[idx];

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

thrust::device_vector<int32_t>
expand_path_device_vec(const thrust::device_vector<int32_t> &d_encoded_path,
                       const thrust::device_vector<int32_t> &d_rules_first,
                       const thrust::device_vector<int32_t> &d_rules_second,
                       uint32_t min_rule_id, size_t num_rules) {
  if (d_encoded_path.empty()) {
    return thrust::device_vector<int32_t>();
  }
  if (num_rules == 0) {
    return d_encoded_path;
  }

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
  int blocks = (num_elements + threads - 1) / threads;
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

  thrust::device_vector<int32_t> d_output(output_size);
  expand_path_single_pass_kernel<<<blocks, threads>>>(
      thrust::raw_pointer_cast(d_encoded_path.data()),
      thrust::raw_pointer_cast(d_output_offsets.data()),
      thrust::raw_pointer_cast(d_expanded_rules.data()),
      thrust::raw_pointer_cast(d_rule_offsets.data()),
      thrust::raw_pointer_cast(d_rule_sizes.data()),
      thrust::raw_pointer_cast(d_output.data()), num_elements, min_rule_id,
      max_rule_id);
  CUDA_CHECK(cudaGetLastError());

  return d_output;
}

__global__ void expand_path_chunk_kernel(
    const int32_t *__restrict__ d_input,
    const int64_t *__restrict__ d_output_offsets,
    const int32_t *__restrict__ d_expanded_rules,
    const int64_t *__restrict__ d_rule_offsets,
    const int64_t *__restrict__ d_rule_sizes, int32_t *__restrict__ d_output,
    size_t chunk_start_idx, size_t chunk_end_idx, int64_t global_expanded_offset,
    uint32_t min_rule_id, uint32_t max_rule_id) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x + chunk_start_idx;
  if (idx >= chunk_end_idx) {
    return;
  }

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
  if (seg >= chunk_seg_end) {
    return;
  }

  uint64_t start = offsets[seg];
  uint64_t end =
      (seg + 1 < total_segments) ? offsets[seg + 1] : (uint64_t)total_nodes;
  if (start >= end) {
    return;
  }

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

  if (num_encoded == 0 || num_expanded == 0) {
    return;
  }

  int threads = 256;
  int blocks_expand = (num_encoded + threads - 1) / threads;
  expand_path_chunk_kernel<<<blocks_expand, threads>>>(
      thrust::raw_pointer_cast(d_encoded_path.data()),
      thrust::raw_pointer_cast(d_output_offsets.data()),
      thrust::raw_pointer_cast(d_expanded_rules.data()),
      thrust::raw_pointer_cast(d_rule_offsets.data()),
      thrust::raw_pointer_cast(d_rule_sizes.data()),
      thrust::raw_pointer_cast(d_chunk_workspace.data()), chunk_encoded_begin,
      chunk_encoded_end, chunk_expanded_begin, min_rule_id, max_rule_id);
  CUDA_CHECK(cudaGetLastError());

  size_t num_segments = chunk_segment_end - chunk_segment_begin;
  int blocks_inv = (num_segments + threads - 1) / threads;
  segmented_inverse_delta_chunk_kernel<<<blocks_inv, threads>>>(
      thrust::raw_pointer_cast(d_chunk_workspace.data()),
      thrust::raw_pointer_cast(d_chunk_workspace.data()),
      thrust::raw_pointer_cast(d_offs_final.data()), chunk_segment_begin,
      chunk_segment_end, static_cast<uint32_t>(d_offs_final.size()),
      total_nodes, static_cast<uint64_t>(chunk_expanded_begin));
  CUDA_CHECK(cudaGetLastError());
}

RollingDecodeSchedule build_rolling_decode_schedule(
    const thrust::device_vector<int64_t> &d_output_offsets,
    const thrust::device_vector<uint32_t> &d_lens_final, size_t encoded_size,
    int64_t output_size, uint32_t traversals_per_chunk,
    size_t max_expanded_chunk_bytes) {
  RollingDecodeSchedule schedule;
  const uint32_t num_segs_final = static_cast<uint32_t>(d_lens_final.size());
  if (num_segs_final == 0) {
    schedule.output_size = output_size;
    return schedule;
  }

  thrust::device_vector<uint64_t> d_offs_final(num_segs_final);
  thrust::exclusive_scan(d_lens_final.begin(), d_lens_final.end(),
                         d_offs_final.begin(), uint64_t(0));

  thrust::device_vector<int64_t> d_encoded_offsets(num_segs_final);
  thrust::lower_bound(d_output_offsets.begin(), d_output_offsets.end(),
                      d_offs_final.begin(), d_offs_final.end(),
                      d_encoded_offsets.begin());

  schedule.encoded_offsets.resize(num_segs_final);
  thrust::copy(d_encoded_offsets.begin(), d_encoded_offsets.end(),
               schedule.encoded_offsets.begin());
  schedule.expanded_offsets.resize(num_segs_final);
  thrust::copy(d_offs_final.begin(), d_offs_final.end(),
               schedule.expanded_offsets.begin());
  schedule.output_size = output_size;

  uint32_t current_seg_begin = 0;
  for (uint32_t i = 0; i < num_segs_final; ++i) {
    const int64_t seg_expanded_start = schedule.expanded_offsets[current_seg_begin];
    const int64_t next_expanded_start =
        (i + 1 < num_segs_final) ? static_cast<int64_t>(schedule.expanded_offsets[i + 1])
                                 : output_size;
    const int64_t size_so_far = next_expanded_start - seg_expanded_start;

    bool split = false;
    if (i - current_seg_begin >= traversals_per_chunk) {
      split = true;
    }
    if (size_so_far >= static_cast<int64_t>(max_expanded_chunk_bytes) &&
        i > current_seg_begin) {
      split = true;
    }
    if (i + 1 == num_segs_final) {
      split = true;
    }

    if (!split) {
      continue;
    }

    const uint32_t chunk_seg_end = i + 1;
    const int64_t chunk_encoded_begin = schedule.encoded_offsets[current_seg_begin];
    const int64_t chunk_encoded_end =
        (chunk_seg_end < num_segs_final)
            ? schedule.encoded_offsets[chunk_seg_end]
            : static_cast<int64_t>(encoded_size);
    const int64_t chunk_expanded_begin =
        static_cast<int64_t>(schedule.expanded_offsets[current_seg_begin]);

    schedule.chunks.push_back(
        {current_seg_begin, chunk_seg_end, chunk_encoded_begin,
         chunk_encoded_end, chunk_expanded_begin, next_expanded_start});
    current_seg_begin = chunk_seg_end;
  }

  return schedule;
}

void rolling_expand_and_inverse_delta_decode(
    const thrust::device_vector<int32_t> &d_encoded_path,
    const thrust::device_vector<int32_t> &d_rules_first,
    const thrust::device_vector<int32_t> &d_rules_second,
    uint32_t min_rule_id, size_t num_rules,
    const thrust::device_vector<uint32_t> &d_lens_final,
    std::vector<int32_t> &h_result_data,
    uint32_t traversals_per_chunk) {
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

  const RollingDecodeSchedule schedule = build_rolling_decode_schedule(
      d_output_offsets, d_lens_final, d_encoded_path.size(), output_size,
      traversals_per_chunk,
      gpu_decompression::kDefaultRollingOutputChunkBytes);

  thrust::device_vector<uint64_t> d_offs_final(schedule.expanded_offsets.begin(),
                                               schedule.expanded_offsets.end());

  h_result_data.resize(static_cast<size_t>(schedule.output_size));
  thrust::device_vector<int32_t> d_chunk_workspace;

  for (const auto &chunk : schedule.chunks) {
    expand_and_inverse_decode_chunk_device(
        d_encoded_path, d_output_offsets, d_expanded_rules, d_rule_offsets,
        d_rule_sizes, d_chunk_workspace, d_offs_final, chunk.encoded_begin,
        chunk.encoded_end, chunk.expanded_begin, chunk.expanded_end,
        chunk.segment_begin, chunk.segment_end, min_rule_id, max_rule_id,
        static_cast<size_t>(schedule.output_size));

    thrust::copy(d_chunk_workspace.begin(), d_chunk_workspace.end(),
                 h_result_data.begin() + chunk.expanded_begin);
  }
}

} // namespace gpu_codec
