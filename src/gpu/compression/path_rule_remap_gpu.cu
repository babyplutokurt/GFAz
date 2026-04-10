#include "gpu/core/codec_gpu.cuh"

#include <cuda_runtime.h>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scatter.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <stdexcept>
#include <string>
#include <vector>

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

// Remap path IDs after rule compaction. Any surviving rule ID in the current
// round can be rewritten from its old offset to the compacted offset.
__global__ void remap_paths_kernel(
    int32_t *paths, size_t num_nodes,
    const uint64_t *new_indices,
    uint32_t start_id, uint32_t num_rules) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_nodes) {
    return;
  }

  int32_t val = paths[idx];
  int32_t abs_val = (val >= 0) ? val : -val;

  if ((uint32_t)abs_val >= start_id) {
    uint32_t offset = (uint32_t)abs_val - start_id;
    if (offset < num_rules) {
      uint64_t new_offset = new_indices[offset];
      uint32_t new_id = start_id + static_cast<uint32_t>(new_offset);
      paths[idx] = (val >= 0) ? (int32_t)new_id : -(int32_t)new_id;
    }
  }
}

// Remap path IDs after sorting round-local rules by 2-mer value.
__global__ void remap_paths_reorder_kernel(int32_t *paths, size_t num_nodes,
                                           const uint32_t *reorder_map,
                                           uint32_t start_id,
                                           uint32_t num_rules) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_nodes) {
    return;
  }

  int32_t val = paths[idx];
  int32_t abs_val = (val >= 0) ? val : -val;

  if ((uint32_t)abs_val >= start_id) {
    uint32_t offset = (uint32_t)abs_val - start_id;
    if (offset < num_rules) {
      uint32_t new_offset = reorder_map[offset];
      uint32_t new_id = start_id + new_offset;
      paths[idx] = (val >= 0) ? (int32_t)new_id : -(int32_t)new_id;
    }
  }
}

void compact_rules_and_remap_gpu(FlattenedPaths &paths,
                                 const std::vector<uint8_t> &rules_used,
                                 std::vector<uint64_t> &current_rules,
                                 uint32_t start_id) {
  if (rules_used.empty()) {
    return;
  }

  size_t num_rules = rules_used.size();
  thrust::device_vector<uint8_t> d_flags(rules_used);

  thrust::device_vector<uint64_t> d_flags_int(num_rules);
  thrust::transform(d_flags.begin(), d_flags.end(), d_flags_int.begin(),
                    [] __host__ __device__(uint8_t v) {
                      return v ? uint64_t(1) : uint64_t(0);
                    });
  thrust::device_vector<uint64_t> d_new_indices(num_rules);
  thrust::exclusive_scan(d_flags_int.begin(), d_flags_int.end(),
                         d_new_indices.begin());
  CUDA_CHECK(cudaDeviceSynchronize());

  if (!paths.data.empty()) {
    size_t num_nodes = paths.data.size();
    thrust::device_vector<int32_t> d_paths(paths.data.begin(), paths.data.end());

    int threads = 256;
    int blocks = (num_nodes + threads - 1) / threads;
    remap_paths_kernel<<<blocks, threads>>>(
        thrust::raw_pointer_cast(d_paths.data()), num_nodes,
        thrust::raw_pointer_cast(d_new_indices.data()), start_id,
        static_cast<uint32_t>(num_rules));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    thrust::copy(d_paths.begin(), d_paths.end(), paths.data.begin());
  }

  std::vector<uint64_t> compacted_rules;
  compacted_rules.reserve(num_rules);
  for (size_t i = 0; i < num_rules; ++i) {
    if (rules_used[i]) {
      compacted_rules.push_back(current_rules[i]);
    }
  }
  current_rules = std::move(compacted_rules);
}

void compact_rules_and_remap_gpu_device(FlattenedPathsDevice &paths,
                                        const std::vector<uint8_t> &rules_used,
                                        std::vector<uint64_t> &current_rules,
                                        uint32_t start_id) {
  if (rules_used.empty()) {
    return;
  }

  size_t num_rules = rules_used.size();
  thrust::device_vector<uint8_t> d_flags(rules_used);

  thrust::device_vector<uint64_t> d_flags_int(num_rules);
  thrust::transform(d_flags.begin(), d_flags.end(), d_flags_int.begin(),
                    [] __host__ __device__(uint8_t v) {
                      return v ? uint64_t(1) : uint64_t(0);
                    });
  thrust::device_vector<uint64_t> d_new_indices(num_rules);
  thrust::exclusive_scan(d_flags_int.begin(), d_flags_int.end(),
                         d_new_indices.begin());
  CUDA_CHECK(cudaDeviceSynchronize());

  if (!paths.data.empty()) {
    size_t num_nodes = paths.data.size();
    int threads = 256;
    int blocks = (num_nodes + threads - 1) / threads;
    remap_paths_kernel<<<blocks, threads>>>(
        thrust::raw_pointer_cast(paths.data.data()), num_nodes,
        thrust::raw_pointer_cast(d_new_indices.data()), start_id,
        static_cast<uint32_t>(num_rules));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  std::vector<uint64_t> compacted_rules;
  compacted_rules.reserve(num_rules);
  for (size_t i = 0; i < num_rules; ++i) {
    if (rules_used[i]) {
      compacted_rules.push_back(current_rules[i]);
    }
  }
  current_rules = std::move(compacted_rules);
}

void compact_rules_and_remap_device_vec(
    thrust::device_vector<int32_t> &d_data,
    const thrust::device_vector<uint8_t> &rules_used,
    thrust::device_vector<uint64_t> &rules, uint32_t start_id) {
  if (rules_used.empty()) {
    return;
  }

  size_t num_rules = rules_used.size();
  thrust::device_vector<uint64_t> d_flags_int(num_rules);
  thrust::transform(rules_used.begin(), rules_used.end(), d_flags_int.begin(),
                    [] __device__(uint8_t v) {
                      return v ? uint64_t(1) : uint64_t(0);
                    });

  thrust::device_vector<uint64_t> d_new_indices(num_rules);
  thrust::exclusive_scan(d_flags_int.begin(), d_flags_int.end(),
                         d_new_indices.begin());

  if (!d_data.empty()) {
    size_t num_nodes = d_data.size();
    int threads = 256;
    int blocks = (num_nodes + threads - 1) / threads;
    remap_paths_kernel<<<blocks, threads>>>(
        thrust::raw_pointer_cast(d_data.data()), num_nodes,
        thrust::raw_pointer_cast(d_new_indices.data()), start_id,
        static_cast<uint32_t>(num_rules));
    CUDA_CHECK(cudaGetLastError());
  }

  thrust::device_vector<uint64_t> compacted_rules(num_rules);
  auto end_it = thrust::copy_if(
      rules.begin(), rules.end(), rules_used.begin(), compacted_rules.begin(),
      [] __device__(uint8_t used) { return used != 0; });
  compacted_rules.resize(end_it - compacted_rules.begin());
  rules = std::move(compacted_rules);
}

void sort_rules_and_remap_gpu(FlattenedPaths &paths,
                              std::vector<uint64_t> &current_rules,
                              uint32_t start_id) {
  if (current_rules.empty()) {
    return;
  }

  size_t num_rules = current_rules.size();
  thrust::device_vector<uint64_t> d_rules(current_rules);
  thrust::device_vector<uint32_t> d_indices(num_rules);
  thrust::sequence(d_indices.begin(), d_indices.end());
  thrust::sort_by_key(d_rules.begin(), d_rules.end(), d_indices.begin());

  thrust::device_vector<uint32_t> d_reorder_map(num_rules);
  thrust::scatter(
      thrust::counting_iterator<uint32_t>(0),
      thrust::counting_iterator<uint32_t>(static_cast<uint32_t>(num_rules)),
      d_indices.begin(), d_reorder_map.begin());

  if (!paths.data.empty()) {
    size_t num_nodes = paths.data.size();
    thrust::device_vector<int32_t> d_paths(paths.data.begin(), paths.data.end());

    int threads = 256;
    int blocks = (num_nodes + threads - 1) / threads;
    remap_paths_reorder_kernel<<<blocks, threads>>>(
        thrust::raw_pointer_cast(d_paths.data()), num_nodes,
        thrust::raw_pointer_cast(d_reorder_map.data()), start_id,
        static_cast<uint32_t>(num_rules));
    CUDA_CHECK(cudaGetLastError());
    thrust::copy(d_paths.begin(), d_paths.end(), paths.data.begin());
  }

  thrust::copy(d_rules.begin(), d_rules.end(), current_rules.begin());
}

void sort_rules_and_remap_gpu_device(FlattenedPathsDevice &paths,
                                     std::vector<uint64_t> &current_rules,
                                     uint32_t start_id) {
  if (current_rules.empty()) {
    return;
  }

  size_t num_rules = current_rules.size();
  thrust::device_vector<uint64_t> d_rules(current_rules);
  thrust::device_vector<uint32_t> d_indices(num_rules);
  thrust::sequence(d_indices.begin(), d_indices.end());
  thrust::sort_by_key(d_rules.begin(), d_rules.end(), d_indices.begin());

  thrust::device_vector<uint32_t> d_reorder_map(num_rules);
  thrust::scatter(
      thrust::counting_iterator<uint32_t>(0),
      thrust::counting_iterator<uint32_t>(static_cast<uint32_t>(num_rules)),
      d_indices.begin(), d_reorder_map.begin());

  if (!paths.data.empty()) {
    size_t num_nodes = paths.data.size();
    int threads = 256;
    int blocks = (num_nodes + threads - 1) / threads;
    remap_paths_reorder_kernel<<<blocks, threads>>>(
        thrust::raw_pointer_cast(paths.data.data()), num_nodes,
        thrust::raw_pointer_cast(d_reorder_map.data()), start_id,
        static_cast<uint32_t>(num_rules));
    CUDA_CHECK(cudaGetLastError());
  }

  thrust::copy(d_rules.begin(), d_rules.end(), current_rules.begin());
}

void sort_rules_and_remap_device_vec(thrust::device_vector<int32_t> &d_data,
                                     thrust::device_vector<uint64_t> &rules,
                                     uint32_t start_id) {
  if (rules.empty()) {
    return;
  }

  size_t num_rules = rules.size();
  thrust::device_vector<uint32_t> d_indices(num_rules);
  thrust::sequence(d_indices.begin(), d_indices.end());
  thrust::sort_by_key(rules.begin(), rules.end(), d_indices.begin());

  thrust::device_vector<uint32_t> d_reorder_map(num_rules);
  thrust::scatter(
      thrust::counting_iterator<uint32_t>(0),
      thrust::counting_iterator<uint32_t>(static_cast<uint32_t>(num_rules)),
      d_indices.begin(), d_reorder_map.begin());

  if (!d_data.empty()) {
    size_t num_nodes = d_data.size();
    int threads = 256;
    int blocks = (num_nodes + threads - 1) / threads;
    remap_paths_reorder_kernel<<<blocks, threads>>>(
        thrust::raw_pointer_cast(d_data.data()), num_nodes,
        thrust::raw_pointer_cast(d_reorder_map.data()), start_id,
        static_cast<uint32_t>(num_rules));
    CUDA_CHECK(cudaGetLastError());
  }
}

void remap_chunk_rule_ids_device_vec(
    thrust::device_vector<int32_t>& d_chunk_data,
    const thrust::device_vector<uint64_t>& d_new_indices,
    const thrust::device_vector<uint32_t>& d_reorder_map,
    uint32_t start_id,
    uint32_t num_rules_before_compact,
    uint32_t num_rules_after_compact) {
  if (d_chunk_data.empty()) {
    return;
  }

  int threads = 256;
  int blocks = (d_chunk_data.size() + threads - 1) / threads;
  remap_paths_kernel<<<blocks, threads>>>(
      thrust::raw_pointer_cast(d_chunk_data.data()), d_chunk_data.size(),
      thrust::raw_pointer_cast(d_new_indices.data()), start_id,
      num_rules_before_compact);
  CUDA_CHECK(cudaGetLastError());

  if (num_rules_after_compact > 0) {
    remap_paths_reorder_kernel<<<blocks, threads>>>(
        thrust::raw_pointer_cast(d_chunk_data.data()), d_chunk_data.size(),
        thrust::raw_pointer_cast(d_reorder_map.data()), start_id,
        num_rules_after_compact);
    CUDA_CHECK(cudaGetLastError());
  }
}

} // namespace gpu_codec
