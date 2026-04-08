#include "gpu/codec_gpu.cuh"
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

#include <cuco/static_map.cuh>
#include <limits>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/unique.h>
#include <thrust/remove.h>

// CUDA error checking macro
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

FlattenedPathsDevice copy_paths_to_device(const FlattenedPaths &paths) {
  FlattenedPathsDevice device_paths;
  device_paths.data =
      thrust::device_vector<int32_t>(paths.data.begin(), paths.data.end());
  device_paths.lengths = thrust::device_vector<uint32_t>(paths.lengths.begin(),
                                                         paths.lengths.end());
  return device_paths;
}

void copy_paths_to_host(const FlattenedPathsDevice &device_paths,
                        FlattenedPaths &paths) {
  paths.data.resize(device_paths.data.size());
  paths.lengths.resize(device_paths.lengths.size());

  thrust::copy(device_paths.data.begin(), device_paths.data.end(),
               paths.data.begin());
  thrust::copy(device_paths.lengths.begin(), device_paths.lengths.end(),
               paths.lengths.begin());
}

void delta_encode_paths(FlattenedPaths &paths) {
  if (paths.data.empty()) {
    return; // Nothing to encode
  }

  const size_t num_elements = paths.data.size();

  // Allocate device memory
  int32_t *d_input = nullptr;
  int32_t *d_output = nullptr;

  CUDA_CHECK(cudaMalloc(&d_input, num_elements * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&d_output, num_elements * sizeof(int32_t)));

  // Copy input data to device
  CUDA_CHECK(cudaMemcpy(d_input, paths.data.data(),
                        num_elements * sizeof(int32_t),
                        cudaMemcpyHostToDevice));

  // CUB adjacent difference operation
  // First call: determine temporary storage requirements
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  cub::DeviceAdjacentDifference::SubtractLeftCopy(
      d_temp_storage, temp_storage_bytes, d_input, d_output,
      static_cast<int>(num_elements));

  // Allocate temporary storage
  CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  // Second call: perform actual delta encoding
  cub::DeviceAdjacentDifference::SubtractLeftCopy(
      d_temp_storage, temp_storage_bytes, d_input, d_output,
      static_cast<int>(num_elements));

  // Wait for GPU to finish
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy result back to host
  CUDA_CHECK(cudaMemcpy(paths.data.data(), d_output,
                        num_elements * sizeof(int32_t),
                        cudaMemcpyDeviceToHost));

  // Free device memory
  CUDA_CHECK(cudaFree(d_temp_storage));
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));
}

void delta_decode_paths(FlattenedPaths &paths) {
  if (paths.data.empty()) {
    return; // Nothing to decode
  }

  const size_t num_elements = paths.data.size();

  // Allocate device memory
  int32_t *d_input = nullptr;
  int32_t *d_output = nullptr;

  CUDA_CHECK(cudaMalloc(&d_input, num_elements * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&d_output, num_elements * sizeof(int32_t)));

  // Copy delta-encoded data to device
  CUDA_CHECK(cudaMemcpy(d_input, paths.data.data(),
                        num_elements * sizeof(int32_t),
                        cudaMemcpyHostToDevice));

  // CUB inclusive sum operation (restores original from deltas)
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_input,
                                d_output, static_cast<int>(num_elements));

  // Allocate temporary storage
  CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  // Second call: perform actual delta decoding
  cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_input,
                                d_output, static_cast<int>(num_elements));

  // Wait for GPU to finish
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy result back to host
  CUDA_CHECK(cudaMemcpy(paths.data.data(), d_output,
                        num_elements * sizeof(int32_t),
                        cudaMemcpyDeviceToHost));

  // Free device memory
  CUDA_CHECK(cudaFree(d_temp_storage));
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));
}

void delta_encode_paths_device(FlattenedPathsDevice &paths) {
  if (paths.data.empty()) {
    return;
  }

  const size_t num_elements = paths.data.size();
  int32_t *d_input = thrust::raw_pointer_cast(paths.data.data());

  thrust::device_vector<int32_t> d_output(num_elements);
  int32_t *d_output_ptr = thrust::raw_pointer_cast(d_output.data());

  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  cub::DeviceAdjacentDifference::SubtractLeftCopy(
      d_temp_storage, temp_storage_bytes, d_input, d_output_ptr,
      static_cast<int>(num_elements));

  CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  cub::DeviceAdjacentDifference::SubtractLeftCopy(
      d_temp_storage, temp_storage_bytes, d_input, d_output_ptr,
      static_cast<int>(num_elements));

  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaFree(d_temp_storage));

  paths.data = std::move(d_output);
}

void delta_encode_device_vec(thrust::device_vector<int32_t> &d_data) {
  if (d_data.empty()) {
    return;
  }

  const size_t num_elements = d_data.size();
  int32_t *d_input = thrust::raw_pointer_cast(d_data.data());

  thrust::device_vector<int32_t> d_output(num_elements);
  int32_t *d_output_ptr = thrust::raw_pointer_cast(d_output.data());

  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  cub::DeviceAdjacentDifference::SubtractLeftCopy(
      d_temp_storage, temp_storage_bytes, d_input, d_output_ptr,
      static_cast<int>(num_elements));

  CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  cub::DeviceAdjacentDifference::SubtractLeftCopy(
      d_temp_storage, temp_storage_bytes, d_input, d_output_ptr,
      static_cast<int>(num_elements));

  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaFree(d_temp_storage));

  d_data = std::move(d_output);
}

// ============================================================================
// Segmented (boundary-aware) operations
// ============================================================================

// Kernel: set is_first[offset[seg]] = 1 for each segment
__global__ void set_first_flags_kernel(const uint64_t *offsets,
                                       uint32_t num_segments,
                                       uint8_t *is_first,
                                       size_t total_nodes) {
  uint32_t seg = blockIdx.x * blockDim.x + threadIdx.x;
  if (seg >= num_segments) return;

  uint64_t start = offsets[seg];
  if (start < total_nodes) {
    is_first[start] = 1;
  }
}

// Kernel: set is_last[offset[seg+1]-1] = 1 for each segment
// For the last segment, the end is total_nodes.
__global__ void set_last_flags_kernel(const uint64_t *offsets,
                                      uint32_t num_segments,
                                      uint8_t *is_last,
                                      size_t total_nodes) {
  uint32_t seg = blockIdx.x * blockDim.x + threadIdx.x;
  if (seg >= num_segments) return;

  uint64_t end;
  if (seg + 1 < num_segments) {
    end = offsets[seg + 1];
  } else {
    end = static_cast<uint64_t>(total_nodes);
  }

  // Mark last element of this segment (skip empty segments)
  if (end > offsets[seg]) {
    is_last[end - 1] = 1;
  }
}

void compute_boundary_masks(
    const thrust::device_vector<uint64_t> &d_offsets,
    uint32_t num_segments,
    size_t total_nodes,
    thrust::device_vector<uint8_t> &d_is_first,
    thrust::device_vector<uint8_t> &d_is_last) {
  d_is_first.assign(total_nodes, 0);
  d_is_last.assign(total_nodes, 0);

  if (num_segments == 0 || total_nodes == 0) return;

  int threads = 256;
  int blocks = (num_segments + threads - 1) / threads;

  set_first_flags_kernel<<<blocks, threads>>>(
      thrust::raw_pointer_cast(d_offsets.data()),
      num_segments,
      thrust::raw_pointer_cast(d_is_first.data()),
      total_nodes);
  CUDA_CHECK(cudaGetLastError());

  set_last_flags_kernel<<<blocks, threads>>>(
      thrust::raw_pointer_cast(d_offsets.data()),
      num_segments,
      thrust::raw_pointer_cast(d_is_last.data()),
      total_nodes);
  CUDA_CHECK(cudaGetLastError());
}

// Kernel: segmented delta encoding
// is_first[idx] == 1 → output[idx] = input[idx] (first element of segment)
// Otherwise → output[idx] = input[idx] - input[idx-1]
__global__ void segmented_delta_encode_kernel(const int32_t *input,
                                              int32_t *output,
                                              size_t total_nodes,
                                              const uint8_t *is_first) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_nodes) return;

  if (idx == 0 || is_first[idx]) {
    output[idx] = input[idx];
  } else {
    output[idx] = input[idx] - input[idx - 1];
  }
}

void segmented_delta_encode_device_vec(
    thrust::device_vector<int32_t> &d_data,
    const thrust::device_vector<uint8_t> &d_is_first) {
  if (d_data.empty()) return;

  size_t total_nodes = d_data.size();
  thrust::device_vector<int32_t> d_output(total_nodes);

  int threads = 256;
  int blocks_n = (total_nodes + threads - 1) / threads;

  segmented_delta_encode_kernel<<<blocks_n, threads>>>(
      thrust::raw_pointer_cast(d_data.data()),
      thrust::raw_pointer_cast(d_output.data()),
      total_nodes,
      thrust::raw_pointer_cast(d_is_first.data()));
  CUDA_CHECK(cudaGetLastError());

  d_data = std::move(d_output);
}


struct AbsOp {
  __device__ __forceinline__ uint32_t operator()(int32_t a) const {
    uint32_t u_a = static_cast<uint32_t>(a);
    return (a >= 0) ? u_a : (0u - u_a);
  }
};

uint32_t find_max_abs_node(const FlattenedPaths &paths) {
  if (paths.data.empty()) {
    return 0;
  }

  const size_t num_elements = paths.data.size();

  // Allocate device memory
  int32_t *d_input = nullptr;
  uint32_t *d_output = nullptr;

  CUDA_CHECK(cudaMalloc(&d_input, num_elements * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&d_output, sizeof(uint32_t)));

  // Copy input data to device
  CUDA_CHECK(cudaMemcpy(d_input, paths.data.data(),
                        num_elements * sizeof(int32_t),
                        cudaMemcpyHostToDevice));

  // Setup transformation iterator
  AbsOp abs_op;
  auto d_transform_iter = thrust::make_transform_iterator(d_input, abs_op);

  // CUB Reduction (Max)
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_transform_iter,
                         d_output, static_cast<int>(num_elements));

  CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_transform_iter,
                         d_output, static_cast<int>(num_elements));

  CUDA_CHECK(cudaDeviceSynchronize());

  uint32_t max_val = 0;
  CUDA_CHECK(
      cudaMemcpy(&max_val, d_output, sizeof(uint32_t), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_temp_storage));
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));

  return max_val;
}

uint32_t find_max_abs_device(const thrust::device_vector<int32_t> &d_data) {
  if (d_data.empty()) {
    return 0;
  }

  const int32_t *d_input = thrust::raw_pointer_cast(d_data.data());
  size_t num_elements = d_data.size();

  uint32_t *d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_output, sizeof(uint32_t)));

  AbsOp abs_op;
  auto d_transform_iter = thrust::make_transform_iterator(d_input, abs_op);

  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_transform_iter,
                         d_output, static_cast<int>(num_elements));

  CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_transform_iter,
                         d_output, static_cast<int>(num_elements));

  CUDA_CHECK(cudaDeviceSynchronize());

  uint32_t max_val = 0;
  CUDA_CHECK(
      cudaMemcpy(&max_val, d_output, sizeof(uint32_t), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_temp_storage));
  CUDA_CHECK(cudaFree(d_output));

  return max_val;
}
// Helper for 2-mer packing
__device__ __forceinline__ uint64_t pack_2mer_gpu(int32_t first,
                                                  int32_t second) {
  return (static_cast<int64_t>(first) << 32) | static_cast<uint32_t>(second);
}

// Fused packing and canonicalization
__device__ __forceinline__ uint64_t canonical_pack_2mer_gpu(int32_t u,
                                                            int32_t v) {
  uint64_t forward = pack_2mer_gpu(u, v);
  uint64_t reverse = pack_2mer_gpu(-v, -u);
  return (forward < reverse) ? forward : reverse;
}

// Fully parallel kernel: One thread per node (except last one)
__global__ void generate_2mer_keys_flat_kernel(const int32_t *nodes,
                                               size_t total_nodes,
                                               uint64_t *keys_out) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  // We only go up to total_nodes - 2 (inclusive) because idx+1 must be valid
  if (idx >= total_nodes - 1)
    return;

  int32_t u = nodes[idx];
  int32_t v = nodes[idx + 1];
  keys_out[idx] = canonical_pack_2mer_gpu(u, v);
}

// Functor to mark duplicates
struct DuplicateMarker {
  const uint64_t *keys;
  uint32_t M;

  DuplicateMarker(const uint64_t *_keys, uint32_t _M) : keys(_keys), M(_M) {}

  __device__ uint8_t operator()(int i) const {
    bool left = (i > 0) && (keys[i] == keys[i - 1]);
    bool right = (i + 1 < M) && (keys[i] == keys[i + 1]);
    return (left || right) ? 1 : 0;
  }
};

// Functor for copy_if predicate
struct IsDuplicate {
  __host__ __device__ bool operator()(uint8_t f) const { return f != 0; }
};

static std::vector<uint64_t>
find_repeated_2mers_from_device(const int32_t *d_nodes, size_t total_nodes) {
  if (total_nodes < 2) {
    return {};
  }

  // 1. Generate Keys (M = total_nodes - 1)
  size_t M = total_nodes - 1;
  thrust::device_vector<uint64_t> d_keys(M);

  int threads_gen = 256;
  int blocks_gen = (M + threads_gen - 1) / threads_gen;

  generate_2mer_keys_flat_kernel<<<blocks_gen, threads_gen>>>(
      d_nodes, total_nodes, thrust::raw_pointer_cast(d_keys.data()));
  CUDA_CHECK(cudaGetLastError());

  // 2. Sort Keys (Radix Sort)
  thrust::sort(d_keys.begin(), d_keys.end());

  // 3. Mark Duplicates
  thrust::device_vector<uint8_t> d_dup(M);
  uint8_t *d_dup_ptr = thrust::raw_pointer_cast(d_dup.data());
  DuplicateMarker marker(thrust::raw_pointer_cast(d_keys.data()), (uint32_t)M);

  thrust::for_each(
      thrust::counting_iterator<int>(0), thrust::counting_iterator<int>((int)M),
      [marker, d_dup_ptr] __device__(int i) { d_dup_ptr[i] = marker(i); });

  // 4. Compact Duplicate Elements
  thrust::device_vector<uint64_t> d_dup_keys(M);
  auto end_it = thrust::copy_if(d_keys.begin(), d_keys.end(), d_dup.begin(),
                                d_dup_keys.begin(), IsDuplicate());

  size_t num_dups = end_it - d_dup_keys.begin();
  if (num_dups == 0)
    return {};

  d_dup_keys.resize(num_dups);

  // 5. Unique to get final set
  auto new_end = thrust::unique(d_dup_keys.begin(), d_dup_keys.end());
  d_dup_keys.resize(new_end - d_dup_keys.begin());

  // 6. Copy back to host
  std::vector<uint64_t> h_result(d_dup_keys.size());
  thrust::copy(d_dup_keys.begin(), d_dup_keys.end(), h_result.begin());

  return h_result;
}

std::vector<uint64_t> find_repeated_2mers(const FlattenedPaths &paths) {
  if (paths.data.empty() || paths.data.size() < 2) {
    return {};
  }

  size_t total_nodes = paths.data.size();

  // 1. Prepare Device Memory (Copy just nodes)
  thrust::device_vector<int32_t> d_nodes(paths.data.begin(), paths.data.end());

  return find_repeated_2mers_from_device(
      thrust::raw_pointer_cast(d_nodes.data()), total_nodes);
}

std::vector<uint64_t>
find_repeated_2mers_device(const FlattenedPathsDevice &paths) {
  if (paths.data.empty() || paths.data.size() < 2) {
    return {};
  }

  return find_repeated_2mers_from_device(
      thrust::raw_pointer_cast(paths.data.data()), paths.data.size());
}

thrust::device_vector<uint64_t>
find_repeated_2mers_device_vec(const thrust::device_vector<int32_t> &d_data) {
  if (d_data.size() < 2) {
    return {};
  }

  size_t total_nodes = d_data.size();
  size_t M = total_nodes - 1;

  thrust::device_vector<uint64_t> d_keys(M);

  int threads_gen = 256;
  int blocks_gen = (M + threads_gen - 1) / threads_gen;

  generate_2mer_keys_flat_kernel<<<blocks_gen, threads_gen>>>(
      thrust::raw_pointer_cast(d_data.data()), total_nodes,
      thrust::raw_pointer_cast(d_keys.data()));
  CUDA_CHECK(cudaGetLastError());

  thrust::sort(d_keys.begin(), d_keys.end());

  thrust::device_vector<uint8_t> d_dup(M);
  uint8_t *d_dup_ptr = thrust::raw_pointer_cast(d_dup.data());
  DuplicateMarker marker(thrust::raw_pointer_cast(d_keys.data()), (uint32_t)M);

  thrust::for_each(
      thrust::counting_iterator<int>(0), thrust::counting_iterator<int>((int)M),
      [marker, d_dup_ptr] __device__(int i) { d_dup_ptr[i] = marker(i); });

  thrust::device_vector<uint64_t> d_dup_keys(M);
  auto end_it = thrust::copy_if(d_keys.begin(), d_keys.end(), d_dup.begin(),
                                d_dup_keys.begin(), IsDuplicate());

  size_t num_dups = end_it - d_dup_keys.begin();
  if (num_dups == 0)
    return {};

  d_dup_keys.resize(num_dups);

  auto new_end = thrust::unique(d_dup_keys.begin(), d_dup_keys.end());
  d_dup_keys.resize(new_end - d_dup_keys.begin());

  return d_dup_keys;
}

// Boundary-aware 2-mer key generation: outputs UINT64_MAX sentinel at boundaries
__global__ void generate_2mer_keys_segmented_kernel(
    const int32_t *nodes, size_t total_nodes,
    uint64_t *keys_out,
    const uint8_t *is_last) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_nodes - 1) return;

  // Skip cross-boundary pairs: if idx is the last element of a segment,
  // then (nodes[idx], nodes[idx+1]) spans two segments.
  if (is_last[idx]) {
    keys_out[idx] = UINT64_MAX;  // Sentinel, filtered out later
    return;
  }

  int32_t u = nodes[idx];
  int32_t v = nodes[idx + 1];
  keys_out[idx] = canonical_pack_2mer_gpu(u, v);
}

struct IsNotSentinel {
  __host__ __device__ bool operator()(uint64_t key) const {
    return key != UINT64_MAX;
  }
};

thrust::device_vector<uint64_t> find_repeated_2mers_segmented_device_vec(
    const thrust::device_vector<int32_t> &d_data,
    const thrust::device_vector<uint8_t> &d_is_last) {
  thrust::device_vector<uint64_t> d_unique_keys;
  thrust::device_vector<uint32_t> d_counts;
  count_2mers_segmented_device_vec(d_data, d_is_last, d_unique_keys, d_counts);

  if (d_unique_keys.empty()) {
    return {};
  }

  thrust::device_vector<uint64_t> d_repeated(d_unique_keys.size());
  auto repeated_end = thrust::copy_if(
      d_unique_keys.begin(), d_unique_keys.end(), d_counts.begin(),
      d_repeated.begin(),
      [] __device__(uint32_t count) { return count >= 2; });
  d_repeated.resize(repeated_end - d_repeated.begin());
  return d_repeated;
}

void count_2mers_segmented_device_vec(
    const thrust::device_vector<int32_t> &d_data,
    const thrust::device_vector<uint8_t> &d_is_last,
    thrust::device_vector<uint64_t> &d_unique_keys,
    thrust::device_vector<uint32_t> &d_counts) {
  if (d_data.size() < 2) {
    d_unique_keys.clear();
    d_counts.clear();
    return;
  }

  size_t total_nodes = d_data.size();
  size_t M = total_nodes - 1;

  thrust::device_vector<uint64_t> d_keys(M);

  int threads_gen = 256;
  int blocks_gen = (M + threads_gen - 1) / threads_gen;

  generate_2mer_keys_segmented_kernel<<<blocks_gen, threads_gen>>>(
      thrust::raw_pointer_cast(d_data.data()), total_nodes,
      thrust::raw_pointer_cast(d_keys.data()),
      thrust::raw_pointer_cast(d_is_last.data()));
  CUDA_CHECK(cudaGetLastError());

  // Remove sentinel values before sorting
  auto valid_end = thrust::remove(d_keys.begin(), d_keys.end(), UINT64_MAX);
  size_t valid_count = valid_end - d_keys.begin();
  if (valid_count == 0) {
    d_unique_keys.clear();
    d_counts.clear();
    return;
  }
  d_keys.resize(valid_count);

  thrust::sort(d_keys.begin(), d_keys.end());

  d_unique_keys.resize(valid_count);
  d_counts.resize(valid_count);

  auto ones_begin = thrust::make_constant_iterator<uint32_t>(1);
  auto reduce_end = thrust::reduce_by_key(
      d_keys.begin(), d_keys.end(), ones_begin,
      d_unique_keys.begin(), d_counts.begin());

  size_t unique_count = reduce_end.first - d_unique_keys.begin();
  d_unique_keys.resize(unique_count);
  d_counts.resize(unique_count);
}

// --- GPU Hash Table Implementation (cuCollections) ---

using Key = uint64_t;
using Value = int32_t;

// Wrapper struct to hold the cuco map (since we pass void* to bindings)
struct GpuDict {
  using ProbingScheme =
      cuco::linear_probing<1, cuco::default_hash_function<Key>>;
  using MapType = cuco::static_map<Key, Value, cuco::extent<std::size_t>,
                                   cuda::thread_scope_device,
                                   thrust::equal_to<Key>, ProbingScheme>;
  MapType map;

  GpuDict(size_t n, cudaStream_t s)
      : map(n, // Initial capacity (will be rounded up by cuco)
            cuco::empty_key<Key>{std::numeric_limits<Key>::max()},
            cuco::empty_value<Value>{0}, {}, // Predicate
            {},                              // Probing scheme
            {},                              // Thread scope
            {},                              // Storage
            {},                              // Allocator
            s)                               // Stream
  {}
};

void *create_rule_table_gpu(const std::vector<uint64_t> &rules,
                            uint32_t start_id) {
  size_t num_rules = rules.size();
  if (num_rules == 0)
    return nullptr;

  // Use default stream for simplicity (or create one)
  cudaStream_t stream = 0;

  // Instantiate map on heap
  // We increase capacity slightly to ensure good load factor (e.g. 2x)
  // cuCollections handles valid extent logic, but passing 2*N is safer for
  // performance.
  size_t capacity = num_rules * 2;
  GpuDict *dict = new GpuDict(capacity, stream);

  // Prepare Keys
  thrust::device_vector<Key> d_keys(rules.begin(), rules.end());

  // Prepare Values (sequence starting from start_id)
  thrust::device_vector<Value> d_vals(num_rules);
  thrust::sequence(d_vals.begin(), d_vals.end(), start_id);

  // Insert
  // API: insert(first, last, stream) -> inserts keys? No we want key-value
  // pairs. The previous error showed: size_type insert(InputIt first, InputIt
  // last, ...); This inserts *pairs* if InputIt::value_type is convertible to
  // value_type (pair<Key, Value>). Or we use a zip iterator?

  // There is no explicit "insert(keys, values)" in the header snippet shown.
  // But there is:
  // template <typename InputIt> size_type insert(InputIt first, InputIt last,
  // stream) We need an iterator that yields cuco::pair<Key, Value>.

  auto zip_begin = thrust::make_zip_iterator(
      thrust::make_tuple(d_keys.begin(), d_vals.begin()));
  auto zip_end =
      thrust::make_zip_iterator(thrust::make_tuple(d_keys.end(), d_vals.end()));

  dict->map.insert(zip_begin, zip_end, stream);

  CUDA_CHECK(cudaGetLastError());

  return static_cast<void *>(dict);
}

void *
create_rule_table_gpu_from_device(const thrust::device_vector<uint64_t> &rules,
                                  uint32_t start_id) {
  size_t num_rules = rules.size();
  if (num_rules == 0)
    return nullptr;

  cudaStream_t stream = 0;
  size_t capacity = num_rules * 2;
  GpuDict *dict = new GpuDict(capacity, stream);

  thrust::device_vector<Value> d_vals(num_rules);
  thrust::sequence(d_vals.begin(), d_vals.end(), start_id);

  auto zip_begin = thrust::make_zip_iterator(
      thrust::make_tuple(rules.begin(), d_vals.begin()));
  auto zip_end =
      thrust::make_zip_iterator(thrust::make_tuple(rules.end(), d_vals.end()));

  dict->map.insert(zip_begin, zip_end, stream);
  CUDA_CHECK(cudaGetLastError());

  return static_cast<void *>(dict);
}

void free_rule_table_gpu(void *d_table_handle) {
  if (!d_table_handle)
    return;
  GpuDict *dict = static_cast<GpuDict *>(d_table_handle);
  delete dict;
}

// --- Encoding Kernels ---

// Step 1: Mark replacements
template <typename ViewType>
__global__ void mark_replacements_kernel(const int32_t *nodes, size_t num_nodes,
                                         uint8_t *flags, int32_t *new_values,
                                         ViewType view) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_nodes)
    return;

  // Pass 1: Independent Lookups
  if (idx < num_nodes - 1) {
    int32_t u = nodes[idx];
    int32_t v = nodes[idx + 1];
    uint64_t key = canonical_pack_2mer_gpu(u, v);

    // Lookup
    auto found = view.find(key);
    if (found != view.end()) {
      uint32_t rule_id = found->second;
      // It's a potential rule!

      // Check orientation:
      if (pack_2mer_gpu(u, v) == key) {
        new_values[idx] = (int32_t)rule_id;
      } else {
        new_values[idx] = -(int32_t)rule_id;
      }

      // Debug print for first replacement
      // if (idx < 100) printf("GPU MARK: idx=%lu, key=%lx, rule_id=%u,
      // new_val=%d\n", idx, key, rule_id, new_values[idx]);

      flags[idx] = 1;
    } else {
      new_values[idx] = nodes[idx];
      flags[idx] = 0;
    }
  } else {
    new_values[idx] = nodes[idx];
    flags[idx] = 0;
  }
}

// Step 2: Resolve overlaps (Evens Priority)
__global__ void resolve_overlaps_kernel(uint8_t *flags, size_t num_nodes,
                                        int pass // 0 for Evens, 1 for Odds
) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_nodes)
    return;

  // Only process nodes matching the pass parity
  if ((idx & 1) == pass) {
    // Read current flag status
    uint8_t f = flags[idx];

    if (f == 1) { // Candidate start of rule
      if (pass == 0) {
        // Pass 0 (Evens): Unconditional priority (mostly)
        // Consume next
        if (idx + 1 < num_nodes) {
          flags[idx + 1] = 2;
        }
      } else {
        // Pass 1 (Odds): Must check for conflicts with Evens
        bool conflict = false;

        // Check right neighbor (Even, idx+1)
        // If idx+1 is a START (1), we cannot fire.
        if (idx + 1 < num_nodes) {
          if (flags[idx + 1] == 1) {
            conflict = true;
          }
        }

        // Also check if we ourselves were consumed (idx is Odd, idx-1 is Even)
        // Wait, if f was read as 1, it means we weren't consumed YET?
        // No, flags is global volatile. But we read it into 'f' at start.
        // If Pass 0 ran, flags[idx] might be 2.
        // We checked 'if (f == 1)'. So we are not consumed.
        // NOTE: This relies on Pass 0 completing fully before Pass 1.

        if (conflict) {
          // Revert self to 0
          flags[idx] = 0;
        } else {
          // Fire: Consume next
          if (idx + 1 < num_nodes) {
            flags[idx + 1] = 2;
          }
        }
      }
    }
  }
}

// Step 3: SizeOp
struct SizeOp {
  __device__ int64_t operator()(uint8_t f) const { return (f == 2) ? 0 : 1; }
};

// Step 4: Scatter
__global__ void scatter_kernel(const int32_t *nodes,      // Original nodes
                               const int32_t *new_values, // Rule IDs
                               const uint8_t *flags,
                               const int64_t *scan_result,
                               int32_t *output, size_t num_nodes) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_nodes)
    return;

  uint8_t f = flags[idx];
  if (f == 2)
    return; // Consumed

  int64_t write_pos = scan_result[idx];

  // Debug scatter
  // if (write_pos < 10) printf("GPU SCATTER: idx=%lu, f=%d, write_pos=%d,
  // node=%d, new_val=%d\n", idx, f, write_pos, nodes[idx], new_values[idx]);

  if (f == 1) {
    // Use the rule ID
    output[static_cast<size_t>(write_pos)] = new_values[idx];
  } else {
    // f == 0: Use original node
    output[static_cast<size_t>(write_pos)] = nodes[idx];
  }
}

// Step 2.5: Mark used rules in the usage vector
__global__ void mark_used_rules_kernel(const uint8_t *flags,
                                       const int32_t *new_values,
                                       size_t num_nodes, int32_t *rule_flags,
                                       uint32_t start_id, uint32_t num_rules) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_nodes)
    return;

  // Check if this node is a replacement (flag == 1)
  if (flags[idx] == 1) {
    int32_t val = new_values[idx];
    // Use absolute value since val can be negative for reverse-oriented rules
    int32_t abs_val = (val >= 0) ? val : -val;

    // Check if value is within expected rule ID range
    if (abs_val >= (int32_t)start_id) {
      uint32_t offset = (uint32_t)abs_val - start_id;
      if (offset < num_rules) {
        // Mark as used. Concurrent writes of 1 are safe.
        rule_flags[offset] = 1;
      }
    }
  }
}

void apply_2mer_rules_gpu(FlattenedPaths &paths, void *d_table_handle,
                          std::vector<uint8_t> &rules_used, uint32_t start_id) {
  if (paths.data.empty() || !d_table_handle)
    return;

  size_t num_nodes = paths.data.size();
  GpuDict *dict = static_cast<GpuDict *>(d_table_handle);

  // Get Device View
  auto view = dict->map.ref(cuco::find);

  // Alloc temporary buffers and COPY input to device
  // Crucial Fix: paths.data is Host memory. We must copy to Device.
  thrust::device_vector<int32_t> d_nodes(paths.data.begin(), paths.data.end());

  thrust::device_vector<uint8_t> d_flags(num_nodes);
  thrust::device_vector<int32_t> d_new_values(num_nodes);

  // 1. Mark Candidates
  int threads = 256;
  int blocks = (num_nodes + threads - 1) / threads;
  mark_replacements_kernel<<<blocks, threads>>>(
      thrust::raw_pointer_cast(d_nodes.data()), // Use device pointer
      num_nodes, thrust::raw_pointer_cast(d_flags.data()),
      thrust::raw_pointer_cast(d_new_values.data()), view);
  CUDA_CHECK(cudaGetLastError());

  // 2. Resolve Overlaps (Pass 0: Evens, Pass 1: Odds)
  resolve_overlaps_kernel<<<blocks, threads>>>(
      thrust::raw_pointer_cast(d_flags.data()), num_nodes, 0);
  CUDA_CHECK(cudaGetLastError());

  resolve_overlaps_kernel<<<blocks, threads>>>(
      thrust::raw_pointer_cast(d_flags.data()), num_nodes, 1);
  CUDA_CHECK(cudaGetLastError());

  // 2.5: Record used rules
  size_t num_rules = rules_used.size();
  if (num_rules > 0) {
    thrust::device_vector<int32_t> d_rule_flags(num_rules, 0); // Init to 0

    mark_used_rules_kernel<<<blocks, threads>>>(
        thrust::raw_pointer_cast(d_flags.data()),
        thrust::raw_pointer_cast(d_new_values.data()), num_nodes,
        thrust::raw_pointer_cast(d_rule_flags.data()), start_id,
        (uint32_t)num_rules);
    CUDA_CHECK(cudaGetLastError());

    // Copy back to host
    thrust::copy(d_rule_flags.begin(), d_rule_flags.end(), rules_used.begin());
  }

  // 3. Prefix Sum to find write positions
  thrust::device_vector<int64_t> d_sizes(num_nodes);
  thrust::transform(d_flags.begin(), d_flags.end(), d_sizes.begin(), SizeOp());

  thrust::device_vector<int64_t> d_scan(num_nodes);
  thrust::exclusive_scan(d_sizes.begin(), d_sizes.end(), d_scan.begin());

  // Get total new size
  int64_t last_scan = d_scan.back();
  int64_t last_size = d_sizes.back();
  int64_t new_total_size = last_scan + last_size;

  // 4. Scatter
  thrust::device_vector<int32_t> d_output(static_cast<size_t>(new_total_size));
  scatter_kernel<<<blocks, threads>>>(
      thrust::raw_pointer_cast(d_nodes.data()), // Original nodes (NEW)
      thrust::raw_pointer_cast(d_new_values.data()),
      thrust::raw_pointer_cast(d_flags.data()),
      thrust::raw_pointer_cast(d_scan.data()),
      thrust::raw_pointer_cast(d_output.data()), num_nodes);
  CUDA_CHECK(cudaGetLastError());

  // 5. Swap / Update result
  paths.data.resize(static_cast<size_t>(new_total_size));
  thrust::copy(d_output.begin(), d_output.end(), paths.data.begin());

  // Treat as one huge path
  // paths.lengths.clear();
  // paths.lengths.push_back((uint32_t)new_total_size);
}

void apply_2mer_rules_gpu_device(FlattenedPathsDevice &paths,
                                 void *d_table_handle,
                                 std::vector<uint8_t> &rules_used,
                                 uint32_t start_id) {
  if (paths.data.empty() || !d_table_handle)
    return;

  size_t num_nodes = paths.data.size();
  GpuDict *dict = static_cast<GpuDict *>(d_table_handle);

  // Get Device View
  auto view = dict->map.ref(cuco::find);

  int32_t *d_nodes = thrust::raw_pointer_cast(paths.data.data());

  thrust::device_vector<uint8_t> d_flags(num_nodes);
  thrust::device_vector<int32_t> d_new_values(num_nodes);

  int threads = 256;
  int blocks = (num_nodes + threads - 1) / threads;

  mark_replacements_kernel<<<blocks, threads>>>(
      d_nodes, num_nodes, thrust::raw_pointer_cast(d_flags.data()),
      thrust::raw_pointer_cast(d_new_values.data()), view);
  CUDA_CHECK(cudaGetLastError());

  resolve_overlaps_kernel<<<blocks, threads>>>(
      thrust::raw_pointer_cast(d_flags.data()), num_nodes, 0);
  CUDA_CHECK(cudaGetLastError());

  resolve_overlaps_kernel<<<blocks, threads>>>(
      thrust::raw_pointer_cast(d_flags.data()), num_nodes, 1);
  CUDA_CHECK(cudaGetLastError());

  size_t num_rules = rules_used.size();
  if (num_rules > 0) {
    thrust::device_vector<int32_t> d_rule_flags(num_rules, 0);

    mark_used_rules_kernel<<<blocks, threads>>>(
        thrust::raw_pointer_cast(d_flags.data()),
        thrust::raw_pointer_cast(d_new_values.data()), num_nodes,
        thrust::raw_pointer_cast(d_rule_flags.data()), start_id,
        (uint32_t)num_rules);
    CUDA_CHECK(cudaGetLastError());

    thrust::copy(d_rule_flags.begin(), d_rule_flags.end(), rules_used.begin());
  }

  thrust::device_vector<int64_t> d_sizes(num_nodes);
  thrust::transform(d_flags.begin(), d_flags.end(), d_sizes.begin(), SizeOp());

  thrust::device_vector<int64_t> d_scan(num_nodes);
  thrust::exclusive_scan(d_sizes.begin(), d_sizes.end(), d_scan.begin());

  int64_t last_scan = d_scan.back();
  int64_t last_size = d_sizes.back();
  int64_t new_total_size = last_scan + last_size;

  thrust::device_vector<int32_t> d_output(static_cast<size_t>(new_total_size));
  scatter_kernel<<<blocks, threads>>>(
      d_nodes, thrust::raw_pointer_cast(d_new_values.data()),
      thrust::raw_pointer_cast(d_flags.data()),
      thrust::raw_pointer_cast(d_scan.data()),
      thrust::raw_pointer_cast(d_output.data()), num_nodes);
  CUDA_CHECK(cudaGetLastError());

  paths.data = std::move(d_output);
}

void apply_2mer_rules_device_vec(thrust::device_vector<int32_t> &d_data,
                                 void *d_table_handle,
                                 thrust::device_vector<uint8_t> &rules_used,
                                 uint32_t start_id) {
  if (d_data.empty() || !d_table_handle)
    return;

  size_t num_nodes = d_data.size();
  size_t num_rules = rules_used.size();
  GpuDict *dict = static_cast<GpuDict *>(d_table_handle);

  auto view = dict->map.ref(cuco::find);
  int32_t *d_nodes = thrust::raw_pointer_cast(d_data.data());

  thrust::device_vector<uint8_t> d_flags(num_nodes);
  thrust::device_vector<int32_t> d_new_values(num_nodes);

  int threads = 256;
  int blocks = (num_nodes + threads - 1) / threads;

  mark_replacements_kernel<<<blocks, threads>>>(
      d_nodes, num_nodes, thrust::raw_pointer_cast(d_flags.data()),
      thrust::raw_pointer_cast(d_new_values.data()), view);
  CUDA_CHECK(cudaGetLastError());

  resolve_overlaps_kernel<<<blocks, threads>>>(
      thrust::raw_pointer_cast(d_flags.data()), num_nodes, 0);
  CUDA_CHECK(cudaGetLastError());

  resolve_overlaps_kernel<<<blocks, threads>>>(
      thrust::raw_pointer_cast(d_flags.data()), num_nodes, 1);
  CUDA_CHECK(cudaGetLastError());

  if (num_rules > 0) {
    thrust::device_vector<int32_t> d_rule_flags(num_rules, 0);

    mark_used_rules_kernel<<<blocks, threads>>>(
        thrust::raw_pointer_cast(d_flags.data()),
        thrust::raw_pointer_cast(d_new_values.data()), num_nodes,
        thrust::raw_pointer_cast(d_rule_flags.data()), start_id,
        (uint32_t)num_rules);
    CUDA_CHECK(cudaGetLastError());

    thrust::transform(d_rule_flags.begin(), d_rule_flags.end(),
                      rules_used.begin(),
                      [] __device__(int32_t v) { return v ? 1 : 0; });
  }

  thrust::device_vector<int64_t> d_sizes(num_nodes);
  thrust::transform(d_flags.begin(), d_flags.end(), d_sizes.begin(), SizeOp());

  thrust::device_vector<int64_t> d_scan(num_nodes);
  thrust::exclusive_scan(d_sizes.begin(), d_sizes.end(), d_scan.begin());

  int64_t last_scan = d_scan.back();
  int64_t last_size = d_sizes.back();
  int64_t new_total_size = last_scan + last_size;

  thrust::device_vector<int32_t> d_output(static_cast<size_t>(new_total_size));
  scatter_kernel<<<blocks, threads>>>(
      d_nodes, thrust::raw_pointer_cast(d_new_values.data()),
      thrust::raw_pointer_cast(d_flags.data()),
      thrust::raw_pointer_cast(d_scan.data()),
      thrust::raw_pointer_cast(d_output.data()), num_nodes);
  CUDA_CHECK(cudaGetLastError());

  d_data = std::move(d_output);
}

// Boundary-aware mark replacements: skips pairs that cross segment boundaries
template <typename ViewType>
__global__ void mark_replacements_segmented_kernel(
    const int32_t *nodes, size_t num_nodes,
    uint8_t *flags, int32_t *new_values,
    ViewType view,
    const uint8_t *is_last) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_nodes) return;

  // Cannot form a pair at last position or at segment boundaries
  if (idx >= num_nodes - 1 || is_last[idx]) {
    new_values[idx] = nodes[idx];
    flags[idx] = 0;
    return;
  }

  int32_t u = nodes[idx];
  int32_t v = nodes[idx + 1];
  uint64_t key = canonical_pack_2mer_gpu(u, v);

  auto found = view.find(key);
  if (found != view.end()) {
    uint32_t rule_id = found->second;

    if (pack_2mer_gpu(u, v) == key) {
      new_values[idx] = (int32_t)rule_id;
    } else {
      new_values[idx] = -(int32_t)rule_id;
    }
    flags[idx] = 1;
  } else {
    new_values[idx] = nodes[idx];
    flags[idx] = 0;
  }
}

// Compute new per-segment lengths after scatter compaction
__global__ void compute_new_lengths_kernel(
    const int64_t *d_scan,
    const uint64_t *d_offsets,
    uint32_t *new_lengths,
    uint32_t num_segments,
    size_t old_total_nodes,
    int64_t new_total_size) {
  uint32_t seg = blockIdx.x * blockDim.x + threadIdx.x;
  if (seg >= num_segments) return;

  uint64_t start_offset = d_offsets[seg];
  uint64_t end_offset = (seg + 1 < num_segments)
                            ? d_offsets[seg + 1]
                            : static_cast<uint64_t>(old_total_nodes);

  int64_t start_write_pos =
      (start_offset < old_total_nodes) ? d_scan[start_offset] : new_total_size;
  int64_t end_write_pos;
  if (end_offset < old_total_nodes) {
    end_write_pos = d_scan[end_offset];
  } else {
    end_write_pos = new_total_size;
  }

  new_lengths[seg] = static_cast<uint32_t>(end_write_pos - start_write_pos);
}

thrust::device_vector<uint32_t> apply_2mer_rules_segmented_device_vec(
    thrust::device_vector<int32_t> &d_data,
    void *d_table_handle,
    thrust::device_vector<uint8_t> &rules_used,
    uint32_t start_id,
    const thrust::device_vector<uint64_t> &d_offsets,
    uint32_t num_segments) {
  // Fallback: return current lengths if nothing to do
  if (d_data.empty() || !d_table_handle) {
    thrust::device_vector<uint32_t> result(num_segments, 0);
    return result;
  }

  size_t num_nodes = d_data.size();
  size_t num_rules = rules_used.size();
  GpuDict *dict = static_cast<GpuDict *>(d_table_handle);

  auto view = dict->map.ref(cuco::find);
  int32_t *d_nodes = thrust::raw_pointer_cast(d_data.data());

  // Recompute boundary mask for current data size
  thrust::device_vector<uint8_t> d_is_last(num_nodes, 0);
  {
    int bthreads = 256;
    int bblocks = (num_segments + bthreads - 1) / bthreads;
    set_last_flags_kernel<<<bblocks, bthreads>>>(
        thrust::raw_pointer_cast(d_offsets.data()),
        num_segments,
        thrust::raw_pointer_cast(d_is_last.data()),
        num_nodes);
    CUDA_CHECK(cudaGetLastError());
  }

  thrust::device_vector<uint8_t> d_flags(num_nodes);
  thrust::device_vector<int32_t> d_new_values(num_nodes);

  int threads = 256;
  int blocks = (num_nodes + threads - 1) / threads;

  // 1. Mark replacements (boundary-aware)
  mark_replacements_segmented_kernel<<<blocks, threads>>>(
      d_nodes, num_nodes,
      thrust::raw_pointer_cast(d_flags.data()),
      thrust::raw_pointer_cast(d_new_values.data()),
      view,
      thrust::raw_pointer_cast(d_is_last.data()));
  CUDA_CHECK(cudaGetLastError());

  // 2. Resolve overlaps (same as non-segmented — operates on flags only)
  resolve_overlaps_kernel<<<blocks, threads>>>(
      thrust::raw_pointer_cast(d_flags.data()), num_nodes, 0);
  CUDA_CHECK(cudaGetLastError());

  resolve_overlaps_kernel<<<blocks, threads>>>(
      thrust::raw_pointer_cast(d_flags.data()), num_nodes, 1);
  CUDA_CHECK(cudaGetLastError());

  // 2.5. Mark used rules
  if (num_rules > 0) {
    thrust::device_vector<int32_t> d_rule_flags(num_rules, 0);

    mark_used_rules_kernel<<<blocks, threads>>>(
        thrust::raw_pointer_cast(d_flags.data()),
        thrust::raw_pointer_cast(d_new_values.data()), num_nodes,
        thrust::raw_pointer_cast(d_rule_flags.data()), start_id,
        (uint32_t)num_rules);
    CUDA_CHECK(cudaGetLastError());

    thrust::transform(d_rule_flags.begin(), d_rule_flags.end(),
                      rules_used.begin(),
                      [] __device__(int32_t v) { return v ? 1 : 0; });
  }

  // 3. Prefix scan for scatter write positions
  thrust::device_vector<int64_t> d_sizes(num_nodes);
  thrust::transform(d_flags.begin(), d_flags.end(), d_sizes.begin(), SizeOp());

  thrust::device_vector<int64_t> d_scan(num_nodes);
  thrust::exclusive_scan(d_sizes.begin(), d_sizes.end(), d_scan.begin());

  int64_t last_scan = d_scan.back();
  int64_t last_size = d_sizes.back();
  int64_t new_total_size = last_scan + last_size;

  // 4. Scatter
  thrust::device_vector<int32_t> d_output(static_cast<size_t>(new_total_size));
  scatter_kernel<<<blocks, threads>>>(
      d_nodes, thrust::raw_pointer_cast(d_new_values.data()),
      thrust::raw_pointer_cast(d_flags.data()),
      thrust::raw_pointer_cast(d_scan.data()),
      thrust::raw_pointer_cast(d_output.data()), num_nodes);
  CUDA_CHECK(cudaGetLastError());

  d_data = std::move(d_output);

  // 5. Compute new per-segment lengths from the scan array
  thrust::device_vector<uint32_t> d_new_lengths(num_segments);
  {
    int lthreads = 256;
    int lblocks = (num_segments + lthreads - 1) / lthreads;
    compute_new_lengths_kernel<<<lblocks, lthreads>>>(
        thrust::raw_pointer_cast(d_scan.data()),
        thrust::raw_pointer_cast(d_offsets.data()),
        thrust::raw_pointer_cast(d_new_lengths.data()),
        num_segments,
        num_nodes,
        new_total_size);
    CUDA_CHECK(cudaGetLastError());
  }

  return d_new_lengths;
}

// Kernel to remap paths using new IDs based on compaction
// Key insight: If a rule ID appears in the path, it MUST have been used
// (scatter only writes rule IDs for positions where flags==1, and
// mark_used_rules marks those exact rules). So we remap ALL rule IDs in the
// valid range.
__global__ void remap_paths_kernel(
    int32_t *paths, size_t num_nodes,
    const uint64_t *new_indices, // exclusive scan result: new_indices[i] =
                                 // count of used rules before i
    uint32_t start_id, uint32_t num_rules) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_nodes)
    return;

  int32_t val = paths[idx];
  int32_t abs_val = (val >= 0) ? val : -val;

  if ((uint32_t)abs_val >= start_id) {
    uint32_t offset = (uint32_t)abs_val - start_id;
    if (offset < num_rules) {
      // Remap using exclusive scan result
      // new_indices[offset] gives the position in the compacted array
      uint64_t new_offset = new_indices[offset];
      uint32_t new_id = start_id + (uint32_t)new_offset;

      paths[idx] = (val >= 0) ? (int32_t)new_id : -(int32_t)new_id;
    }
  }
}

// Predicate for rule compaction
struct IsRuleUsed {
  const uint8_t *flags;
  IsRuleUsed(const uint8_t *f) : flags(f) {}

  __host__ __device__ bool operator()(const int &idx) const {
    return flags[idx] != 0;
  }
};

void compact_rules_and_remap_gpu(FlattenedPaths &paths,
                                 const std::vector<uint8_t> &rules_used,
                                 std::vector<uint64_t> &current_rules,
                                 uint32_t start_id) {
  if (rules_used.empty())
    return;

  size_t num_rules = rules_used.size();

  // 1. Move rules_used to GPU
  thrust::device_vector<uint8_t> d_flags(rules_used);

  // 2. Exclusive Scan to get new indices (cast to int to avoid uint8 overflow)
  // Example: flags = [0, 1, 1, 0, 1]
  // scan   = [0, 0, 1, 2, 2] -> New offsets for used rules
  thrust::device_vector<uint64_t> d_flags_int(num_rules);
  thrust::transform(d_flags.begin(), d_flags.end(), d_flags_int.begin(),
                    [] __host__ __device__(uint8_t v) {
                      return v ? uint64_t(1) : uint64_t(0);
                    });
  thrust::device_vector<uint64_t> d_new_indices(num_rules);
  thrust::exclusive_scan(d_flags_int.begin(), d_flags_int.end(),
                         d_new_indices.begin());

  // Ensure scan completes before kernel
  CUDA_CHECK(cudaDeviceSynchronize());

  // 3. Remap Paths on GPU
  if (!paths.data.empty()) {
    size_t num_nodes = paths.data.size();
    thrust::device_vector<int32_t> d_paths(paths.data.begin(),
                                           paths.data.end());

    int threads = 256;
    int blocks = (num_nodes + threads - 1) / threads;

    remap_paths_kernel<<<blocks, threads>>>(
        thrust::raw_pointer_cast(d_paths.data()), num_nodes,
        thrust::raw_pointer_cast(d_new_indices.data()), start_id,
        (uint32_t)num_rules);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back modified paths
    thrust::copy(d_paths.begin(), d_paths.end(), paths.data.begin());
  }

  // 4. Compact Rules Vector (CPU side is easier given we need to resize input
  // vector) Actually, we can use thrust/algorithm to do this efficiently

  // Create a temporary vector for compacted rules
  std::vector<uint64_t> compacted_rules;
  compacted_rules.reserve(num_rules); // Max possible size

  for (size_t i = 0; i < num_rules; ++i) {
    if (rules_used[i]) {
      compacted_rules.push_back(current_rules[i]);
    }
  }

  // Replace original
  current_rules = std::move(compacted_rules);
}

void compact_rules_and_remap_gpu_device(FlattenedPathsDevice &paths,
                                        const std::vector<uint8_t> &rules_used,
                                        std::vector<uint64_t> &current_rules,
                                        uint32_t start_id) {
  if (rules_used.empty())
    return;

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
        (uint32_t)num_rules);
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
  if (rules_used.empty())
    return;

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
        (uint32_t)num_rules);
    CUDA_CHECK(cudaGetLastError());
  }

  thrust::device_vector<uint64_t> compacted_rules(num_rules);
  auto end_it = thrust::copy_if(
      rules.begin(), rules.end(), rules_used.begin(), compacted_rules.begin(),
      [] __device__(uint8_t used) { return used != 0; });

  compacted_rules.resize(end_it - compacted_rules.begin());
  rules = std::move(compacted_rules);
}

// Kernel to remap paths based on rule reordering
__global__ void
remap_paths_reorder_kernel(int32_t *paths, size_t num_nodes,
                           const uint32_t *reorder_map, // old_id -> new_id
                           uint32_t start_id, uint32_t num_rules) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_nodes)
    return;

  int32_t val = paths[idx];
  int32_t abs_val = (val >= 0) ? val : -val;

  if ((uint32_t)abs_val >= start_id) {
    uint32_t offset = (uint32_t)abs_val - start_id;
    if (offset < num_rules) {
      // Map old ID offset to new ID offset
      uint32_t new_offset = reorder_map[offset];
      uint32_t new_id = start_id + new_offset;

      paths[idx] = (val >= 0) ? (int32_t)new_id : -(int32_t)new_id;
    }
  }
}

void sort_rules_and_remap_gpu(FlattenedPaths &paths,
                              std::vector<uint64_t> &current_rules,
                              uint32_t start_id) {
  if (current_rules.empty())
    return;
  size_t num_rules = current_rules.size();

  // 1. Move rules to GPU
  thrust::device_vector<uint64_t> d_rules(current_rules);

  // 2. Generate indices [0, 1, ..., N-1] efficiently on GPU
  thrust::device_vector<uint32_t> d_indices(num_rules);
  thrust::sequence(d_indices.begin(), d_indices.end());

  // 3. Sort rules by key (d_rules becomes sorted, d_indices is permuted)
  // d_indices[new_idx] = old_idx
  thrust::sort_by_key(d_rules.begin(), d_rules.end(), d_indices.begin());

  // 4. Invert Permutation: we need reorder_map[old_idx] = new_idx
  thrust::device_vector<uint32_t> d_reorder_map(num_rules);
  thrust::scatter(thrust::counting_iterator<uint32_t>(0),
                  thrust::counting_iterator<uint32_t>(static_cast<uint32_t>(num_rules)),
                  d_indices.begin(),    // Map: positions (old_idx) to write to
                  d_reorder_map.begin() // Output
  );

  // 5. Remap Paths
  if (!paths.data.empty()) {
    size_t num_nodes = paths.data.size();
    thrust::device_vector<int32_t> d_paths(paths.data.begin(),
                                           paths.data.end());

    int threads = 256;
    int blocks = (num_nodes + threads - 1) / threads;

    remap_paths_reorder_kernel<<<blocks, threads>>>(
        thrust::raw_pointer_cast(d_paths.data()), num_nodes,
        thrust::raw_pointer_cast(d_reorder_map.data()), start_id,
        (uint32_t)num_rules);
    CUDA_CHECK(cudaGetLastError());

    // Copy back modified paths
    thrust::copy(d_paths.begin(), d_paths.end(), paths.data.begin());
  }

  // 6. Update Host Rules (copy back sorted rules)
  thrust::copy(d_rules.begin(), d_rules.end(), current_rules.begin());
}

void sort_rules_and_remap_gpu_device(FlattenedPathsDevice &paths,
                                     std::vector<uint64_t> &current_rules,
                                     uint32_t start_id) {
  if (current_rules.empty())
    return;
  size_t num_rules = current_rules.size();

  thrust::device_vector<uint64_t> d_rules(current_rules);

  thrust::device_vector<uint32_t> d_indices(num_rules);
  thrust::sequence(d_indices.begin(), d_indices.end());

  thrust::sort_by_key(d_rules.begin(), d_rules.end(), d_indices.begin());

  thrust::device_vector<uint32_t> d_reorder_map(num_rules);
  thrust::scatter(thrust::counting_iterator<uint32_t>(0),
                  thrust::counting_iterator<uint32_t>(static_cast<uint32_t>(num_rules)), d_indices.begin(),
                  d_reorder_map.begin());

  if (!paths.data.empty()) {
    size_t num_nodes = paths.data.size();
    int threads = 256;
    int blocks = (num_nodes + threads - 1) / threads;

    remap_paths_reorder_kernel<<<blocks, threads>>>(
        thrust::raw_pointer_cast(paths.data.data()), num_nodes,
        thrust::raw_pointer_cast(d_reorder_map.data()), start_id,
        (uint32_t)num_rules);
    CUDA_CHECK(cudaGetLastError());
  }

  thrust::copy(d_rules.begin(), d_rules.end(), current_rules.begin());
}

void sort_rules_and_remap_device_vec(thrust::device_vector<int32_t> &d_data,
                                     thrust::device_vector<uint64_t> &rules,
                                     uint32_t start_id) {
  if (rules.empty())
    return;
  size_t num_rules = rules.size();

  thrust::device_vector<uint32_t> d_indices(num_rules);
  thrust::sequence(d_indices.begin(), d_indices.end());

  thrust::sort_by_key(rules.begin(), rules.end(), d_indices.begin());

  thrust::device_vector<uint32_t> d_reorder_map(num_rules);
  thrust::scatter(thrust::counting_iterator<uint32_t>(0),
                  thrust::counting_iterator<uint32_t>(static_cast<uint32_t>(num_rules)), d_indices.begin(),
                  d_reorder_map.begin());

  if (!d_data.empty()) {
    size_t num_nodes = d_data.size();
    int threads = 256;
    int blocks = (num_nodes + threads - 1) / threads;

    remap_paths_reorder_kernel<<<blocks, threads>>>(
        thrust::raw_pointer_cast(d_data.data()), num_nodes,
        thrust::raw_pointer_cast(d_reorder_map.data()), start_id,
        (uint32_t)num_rules);
    CUDA_CHECK(cudaGetLastError());
  }
}

// Functor to extract first element from packed 2-mer
struct ExtractFirst {
  __device__ __forceinline__ int32_t operator()(uint64_t packed) const {
    return static_cast<int32_t>(packed >> 32);
  }
};

// Functor to extract second element from packed 2-mer
struct ExtractSecond {
  __device__ __forceinline__ int32_t operator()(uint64_t packed) const {
    return static_cast<int32_t>(packed & 0xFFFFFFFF);
  }
};

void split_and_delta_encode_rules_device_vec(
    const thrust::device_vector<uint64_t> &d_rules,
    thrust::device_vector<int32_t> &d_first,
    thrust::device_vector<int32_t> &d_second) {

  if (d_rules.empty()) {
    d_first.clear();
    d_second.clear();
    return;
  }

  size_t num_rules = d_rules.size();

  // Allocate output vectors for unpacked data
  thrust::device_vector<int32_t> d_first_raw(num_rules);
  thrust::device_vector<int32_t> d_second_raw(num_rules);

  // Extract first and second elements using transform
  thrust::transform(d_rules.begin(), d_rules.end(), d_first_raw.begin(),
                    ExtractFirst());
  thrust::transform(d_rules.begin(), d_rules.end(), d_second_raw.begin(),
                    ExtractSecond());

  // Delta-encode first elements
  d_first.resize(num_rules);
  {
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    cub::DeviceAdjacentDifference::SubtractLeftCopy(
        d_temp_storage, temp_storage_bytes,
        thrust::raw_pointer_cast(d_first_raw.data()),
        thrust::raw_pointer_cast(d_first.data()), static_cast<int>(num_rules));

    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    cub::DeviceAdjacentDifference::SubtractLeftCopy(
        d_temp_storage, temp_storage_bytes,
        thrust::raw_pointer_cast(d_first_raw.data()),
        thrust::raw_pointer_cast(d_first.data()), static_cast<int>(num_rules));

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_temp_storage));
  }

  // Delta-encode second elements
  d_second.resize(num_rules);
  {
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    cub::DeviceAdjacentDifference::SubtractLeftCopy(
        d_temp_storage, temp_storage_bytes,
        thrust::raw_pointer_cast(d_second_raw.data()),
        thrust::raw_pointer_cast(d_second.data()), static_cast<int>(num_rules));

    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    cub::DeviceAdjacentDifference::SubtractLeftCopy(
        d_temp_storage, temp_storage_bytes,
        thrust::raw_pointer_cast(d_second_raw.data()),
        thrust::raw_pointer_cast(d_second.data()), static_cast<int>(num_rules));

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_temp_storage));
  }
}

void run_compression_layer_2mer_gpu(
    FlattenedPaths &paths, uint32_t &next_starting_id, int num_rounds,
    std::map<uint32_t, uint64_t> &master_rulebook) {

  uint64_t accumulated_reduction = 0;

  for (int round_idx = 0; round_idx < num_rounds; ++round_idx) {
    // 1. Find Repeated 2-mers (Runs on GPU, returns vector to Host)
    // Optimization TODO: Keep data on GPU between steps.
    // Current: find_repeated_2mers usage copies paths to GPU, then result to
    // host.
    std::vector<uint64_t> current_rules = find_repeated_2mers(paths);

    if (current_rules.empty()) {
      // No more rules found
      break;
    }

    uint32_t num_rules_found = current_rules.size();

    // 2. Create Rule Hash Table on GPU
    // Current: create_rule_table_gpu copies rules from Host to GPU.
    void *table_ptr = create_rule_table_gpu(current_rules, next_starting_id);

    // 3. Apply Rules on GPU
    // Current: apply_2mer_rules_gpu copies paths to GPU, modifies, copies back.
    std::vector<uint8_t> rules_used(current_rules.size(), 0);

    size_t size_before = paths.data.size();
    apply_2mer_rules_gpu(paths, table_ptr, rules_used, next_starting_id);
    size_t size_after = paths.data.size();
    accumulated_reduction += (size_before - size_after);

    // Cleanup Table
    free_rule_table_gpu(table_ptr);

    // Step 4a: Compact
    compact_rules_and_remap_gpu(paths, rules_used, current_rules,
                                next_starting_id);

    // Step 4b: Sort & Reorder
    sort_rules_and_remap_gpu(paths, current_rules, next_starting_id);

    // 5. Update Master Rulebook
    // Add valid, sorted rules to the map
    for (size_t i = 0; i < current_rules.size(); ++i) {
      uint32_t rule_id = next_starting_id + i;
      master_rulebook[rule_id] = current_rules[i];
    }

    // 6. Update next_starting_id
    next_starting_id += current_rules.size();
  }
}

void run_compression_layer_2mer_gpu_device(
    FlattenedPathsDevice &paths, uint32_t &next_starting_id, int num_rounds,
    std::map<uint32_t, uint64_t> &master_rulebook) {
  delta_encode_paths_device(paths);
  for (int round_idx = 0; round_idx < num_rounds; ++round_idx) {
    std::vector<uint64_t> current_rules = find_repeated_2mers_device(paths);

    if (current_rules.empty()) {
      break;
    }

    void *table_ptr = create_rule_table_gpu(current_rules, next_starting_id);

    std::vector<uint8_t> rules_used(current_rules.size(), 0);
    apply_2mer_rules_gpu_device(paths, table_ptr, rules_used, next_starting_id);

    free_rule_table_gpu(table_ptr);

    compact_rules_and_remap_gpu_device(paths, rules_used, current_rules,
                                       next_starting_id);
    sort_rules_and_remap_gpu_device(paths, current_rules, next_starting_id);

    for (size_t i = 0; i < current_rules.size(); ++i) {
      uint32_t rule_id = next_starting_id + i;
      master_rulebook[rule_id] = current_rules[i];
    }

    next_starting_id += current_rules.size();
  }
}

// ============================================================================
// GPU Path Expansion (Decompression) - DEPRECATED Iterative Version
// Kept for performance comparison. Use expand_path_device_vec instead.
// ============================================================================

// Functor to compute expansion size (1 for raw nodes, 2 for rules)
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

// Kernel to expand one level of rules (deprecated version)
__global__ void expand_rules_kernel_deprecated(
    const int32_t *__restrict__ d_input, const int64_t *__restrict__ d_offsets,
    const int32_t *__restrict__ d_rules_first,
    const int32_t *__restrict__ d_rules_second, int32_t *__restrict__ d_output,
    size_t num_elements, uint32_t min_rule_id, uint32_t max_rule_id) {

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_elements)
    return;

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

/**
 * @deprecated Use expand_path_device_vec instead.
 *
 * Iterative multi-pass expansion. Each pass expands one level of rules.
 * Kept for performance comparison with the new single-pass algorithm.
 */
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

  // Ping-pong buffers
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

// ============================================================================
// GPU Path Expansion (Decompression) - Single-Pass with Pre-Expanded Rules
// ============================================================================

// Maximum stack depth for iterative rule expansion kernels.
// Each compression round can add at most 1 level of nesting, and rules
// are binary trees, so depth D requires at most D+1 stack entries.
// 256 supports up to 255 compression rounds which is far beyond practical use.
static constexpr int EXPANSION_STACK_CAPACITY = 256;

// Kernel to compute final expansion size of each rule (iterative on GPU)
// Each rule expands to 2^depth elements in the worst case
__global__ void
compute_rule_final_sizes_kernel(const int32_t *__restrict__ d_rules_first,
                                const int32_t *__restrict__ d_rules_second,
                                int64_t *__restrict__ d_rule_sizes,
                                size_t num_rules, uint32_t min_rule_id) {

  size_t rule_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (rule_idx >= num_rules)
    return;

  // Stack-based iterative expansion to count final size
  int32_t stack[EXPANSION_STACK_CAPACITY];
  int stack_ptr = 0;
  int64_t final_size = 0;

  // Push initial rule's children
  stack[stack_ptr++] = d_rules_first[rule_idx];
  stack[stack_ptr++] = d_rules_second[rule_idx];

  while (stack_ptr > 0) {
    int32_t val = stack[--stack_ptr];
    uint32_t abs_val = static_cast<uint32_t>(val >= 0 ? val : -val);

    if (abs_val >= min_rule_id && abs_val < min_rule_id + num_rules) {
      // It's a rule, push its children (with bounds check)
      uint32_t child_idx = abs_val - min_rule_id;
      if (stack_ptr + 2 <= EXPANSION_STACK_CAPACITY) {
        stack[stack_ptr++] = d_rules_first[child_idx];
        stack[stack_ptr++] = d_rules_second[child_idx];
      } else {
        // Stack overflow: treat this unexpanded rule as a single raw node.
        // This should never happen with reasonable compression rounds.
        final_size++;
      }
    } else {
      // It's a raw node
      final_size++;
    }
  }

  d_rule_sizes[rule_idx] = final_size;
}

// Kernel to expand rules into the pre-expanded buffer
__global__ void
expand_rules_to_buffer_kernel(const int32_t *__restrict__ d_rules_first,
                              const int32_t *__restrict__ d_rules_second,
                              const int64_t *__restrict__ d_rule_offsets,
                              int32_t *__restrict__ d_expanded_rules,
                              size_t num_rules, uint32_t min_rule_id) {

  size_t rule_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (rule_idx >= num_rules)
    return;

  int64_t write_pos = d_rule_offsets[rule_idx];

  // Stack-based expansion
  int32_t stack[EXPANSION_STACK_CAPACITY];
  int stack_ptr = 0;

  // Push in reverse order so we process first before second
  stack[stack_ptr++] = d_rules_second[rule_idx];
  stack[stack_ptr++] = d_rules_first[rule_idx];

  while (stack_ptr > 0) {
    int32_t val = stack[--stack_ptr];
    uint32_t abs_val = static_cast<uint32_t>(val >= 0 ? val : -val);

    if (abs_val >= min_rule_id && abs_val < min_rule_id + num_rules) {
      // It's a rule, push its children (reverse order) with bounds check
      uint32_t child_idx = abs_val - min_rule_id;
      if (stack_ptr + 2 <= EXPANSION_STACK_CAPACITY) {
        if (val >= 0) {
          stack[stack_ptr++] = d_rules_second[child_idx];
          stack[stack_ptr++] = d_rules_first[child_idx];
        } else {
          // Negative: reverse and negate
          stack[stack_ptr++] = -d_rules_first[child_idx];
          stack[stack_ptr++] = -d_rules_second[child_idx];
        }
      }
      // If stack would overflow, treat this unexpanded rule as a single
      // raw node. Write it directly so output matches the size kernel.
      else {
        d_expanded_rules[write_pos++] = val;
      }
    } else {
      // It's a raw node, write it
      d_expanded_rules[write_pos++] = val;
    }
  }
}

// Functor to compute final expansion size for path elements
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
    return 1; // Raw node
  }
};

// Kernel for final single-pass expansion of path
__global__ void expand_path_single_pass_kernel(
    const int32_t *__restrict__ d_input,
    const int64_t *__restrict__ d_output_offsets,
    const int32_t *__restrict__ d_expanded_rules,
    const int64_t *__restrict__ d_rule_offsets,
    const int64_t *__restrict__ d_rule_sizes, int32_t *__restrict__ d_output,
    size_t num_elements, uint32_t min_rule_id, uint32_t max_rule_id) {

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_elements)
    return;

  int32_t val = d_input[idx];
  uint32_t abs_val = static_cast<uint32_t>(val >= 0 ? val : -val);
  int64_t out_offset = d_output_offsets[idx];

  if (abs_val >= min_rule_id && abs_val < max_rule_id) {
    uint32_t rule_idx = abs_val - min_rule_id;
    int64_t rule_offset = d_rule_offsets[rule_idx];
    int64_t rule_size = d_rule_sizes[rule_idx];

    if (val >= 0) {
      // Copy expanded rule forward
      for (int64_t i = 0; i < rule_size; i++) {
        d_output[out_offset + i] = d_expanded_rules[rule_offset + i];
      }
    } else {
      // Copy expanded rule reversed and negated
      for (int64_t i = 0; i < rule_size; i++) {
        d_output[out_offset + i] =
            -d_expanded_rules[rule_offset + rule_size - 1 - i];
      }
    }
  } else {
    // Raw node
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
    // No rules, return as-is
    return d_encoded_path;
  }

  const uint32_t max_rule_id = min_rule_id + static_cast<uint32_t>(num_rules);
  const int threads = 256;

  // =========================================================================
  // PHASE 1: Pre-expand all rules (done once)
  // =========================================================================

  // 1a. Compute final expansion size for each rule
  thrust::device_vector<int64_t> d_rule_sizes(num_rules);
  {
    int blocks = (num_rules + threads - 1) / threads;
    compute_rule_final_sizes_kernel<<<blocks, threads>>>(
        thrust::raw_pointer_cast(d_rules_first.data()),
        thrust::raw_pointer_cast(d_rules_second.data()),
        thrust::raw_pointer_cast(d_rule_sizes.data()), num_rules, min_rule_id);
    CUDA_CHECK(cudaGetLastError());
  }

  // 1b. Compute offsets for pre-expanded rules buffer
  thrust::device_vector<int64_t> d_rule_offsets(num_rules);
  thrust::exclusive_scan(d_rule_sizes.begin(), d_rule_sizes.end(),
                         d_rule_offsets.begin());

  // 1c. Allocate and fill pre-expanded rules buffer
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

  // =========================================================================
  // PHASE 2: Single-pass expansion of the path
  // =========================================================================

  size_t num_elements = d_encoded_path.size();
  int blocks = (num_elements + threads - 1) / threads;

  // 2a. Compute output sizes using pre-computed rule sizes
  FinalExpansionSizeOp size_op(thrust::raw_pointer_cast(d_rule_sizes.data()),
                               min_rule_id, max_rule_id);
  auto size_iter =
      thrust::make_transform_iterator(d_encoded_path.begin(), size_op);

  // 2b. Compute total output size using transform_reduce (single pass)
  int64_t output_size =
      thrust::transform_reduce(d_encoded_path.begin(), d_encoded_path.end(),
                               size_op, (int64_t)0, thrust::plus<int64_t>());

  // 2c. Compute exclusive scan for output offsets
  thrust::device_vector<int64_t> d_output_offsets(num_elements);
  thrust::exclusive_scan(size_iter, size_iter + num_elements,
                         d_output_offsets.begin());

  // 2d. Allocate output and expand in single pass
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

thrust::device_vector<int32_t> inverse_delta_decode_device_vec(
    const thrust::device_vector<int32_t> &d_delta_encoded) {

  if (d_delta_encoded.empty()) {
    return thrust::device_vector<int32_t>();
  }

  thrust::device_vector<int32_t> d_result(d_delta_encoded.size());

  // Inverse delta is just prefix sum (inclusive scan)
  thrust::inclusive_scan(d_delta_encoded.begin(), d_delta_encoded.end(),
                         d_result.begin());

  return d_result;
}

// Segmented inverse delta decode: per-segment prefix sum.
// One thread per segment. Each thread reads its segment boundaries from offsets
// and does a sequential prefix sum within that segment.
__global__ void segmented_inverse_delta_kernel(const int32_t *input,
                                               int32_t *output,
                                               const uint64_t *offsets,
                                               uint32_t num_segments,
                                               size_t total_nodes) {
  uint32_t seg = blockIdx.x * blockDim.x + threadIdx.x;
  if (seg >= num_segments) return;

  uint64_t start = offsets[seg];
  uint64_t end;
  if (seg + 1 < num_segments) {
    end = offsets[seg + 1];
  } else {
    end = static_cast<uint64_t>(total_nodes);
  }

  if (start >= end) return;  // Empty segment

  // Prefix sum within this segment
  int32_t acc = input[start];
  output[start] = acc;
  for (uint64_t i = start + 1; i < end; ++i) {
    acc += input[i];
    output[i] = acc;
  }
}

thrust::device_vector<int32_t> segmented_inverse_delta_decode_device_vec(
    const thrust::device_vector<int32_t> &d_delta_encoded,
    const thrust::device_vector<uint64_t> &d_offsets,
    uint32_t num_segments,
    size_t total_nodes) {

  if (d_delta_encoded.empty()) {
    return thrust::device_vector<int32_t>();
  }

  thrust::device_vector<int32_t> d_result(total_nodes);

  int threads = 256;
  int blocks = (num_segments + threads - 1) / threads;
  segmented_inverse_delta_kernel<<<blocks, threads>>>(
      thrust::raw_pointer_cast(d_delta_encoded.data()),
      thrust::raw_pointer_cast(d_result.data()),
      thrust::raw_pointer_cast(d_offsets.data()),
      num_segments,
      total_nodes);
  CUDA_CHECK(cudaGetLastError());

  return d_result;
}

// ============================================================================
// GPU Bit-Packing for Orientations
// ============================================================================

// Kernel: each thread packs 8 orientation chars into 1 output byte
// '-' maps to bit=1, '+' (or anything else) maps to bit=0
__global__ void pack_orientations_kernel(const char *__restrict__ d_orients,
                                         uint8_t *__restrict__ d_packed,
                                         size_t num_orients, size_t num_bytes) {

  size_t byte_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (byte_idx >= num_bytes)
    return;

  uint8_t packed_byte = 0;
  size_t base_idx = byte_idx * 8;

// Pack 8 chars into 1 byte
#pragma unroll
  for (int bit = 0; bit < 8; ++bit) {
    size_t char_idx = base_idx + bit;
    if (char_idx < num_orients) {
      // '-' = bit 1, '+' = bit 0
      if (d_orients[char_idx] == '-') {
        packed_byte |= (1 << bit);
      }
    }
  }

  d_packed[byte_idx] = packed_byte;
}

std::vector<uint8_t> pack_orientations_gpu(const std::vector<char> &orients) {
  if (orients.empty()) {
    return std::vector<uint8_t>();
  }

  size_t num_orients = orients.size();
  size_t num_bytes = (num_orients + 7) / 8;

  // Copy orientations to device
  thrust::device_vector<char> d_orients(orients.begin(), orients.end());
  thrust::device_vector<uint8_t> d_packed(num_bytes);

  // Launch kernel
  const int threads = 256;
  const int blocks = (num_bytes + threads - 1) / threads;

  pack_orientations_kernel<<<blocks, threads>>>(
      thrust::raw_pointer_cast(d_orients.data()),
      thrust::raw_pointer_cast(d_packed.data()), num_orients, num_bytes);
  CUDA_CHECK(cudaGetLastError());

  // Copy result back to host
  std::vector<uint8_t> result(num_bytes);
  thrust::copy(d_packed.begin(), d_packed.end(), result.begin());

  return result;
}

// Kernel: each thread unpacks 1 byte into 8 orientation chars
__global__ void unpack_orientations_kernel(const uint8_t *__restrict__ d_packed,
                                           char *__restrict__ d_orients,
                                           size_t num_orients,
                                           size_t num_bytes) {

  size_t byte_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (byte_idx >= num_bytes)
    return;

  uint8_t packed_byte = d_packed[byte_idx];
  size_t base_idx = byte_idx * 8;

// Unpack 1 byte into 8 chars
#pragma unroll
  for (int bit = 0; bit < 8; ++bit) {
    size_t char_idx = base_idx + bit;
    if (char_idx < num_orients) {
      // bit=1 means '-', bit=0 means '+'
      bool is_minus = (packed_byte >> bit) & 1;
      d_orients[char_idx] = is_minus ? '-' : '+';
    }
  }
}

std::vector<char> unpack_orientations_gpu(const std::vector<uint8_t> &packed,
                                          size_t num_orients) {
  if (packed.empty() || num_orients == 0) {
    return std::vector<char>();
  }

  size_t num_bytes = packed.size();

  // Copy packed data to device
  thrust::device_vector<uint8_t> d_packed(packed.begin(), packed.end());
  thrust::device_vector<char> d_orients(num_orients);

  // Launch kernel
  const int threads = 256;
  const int blocks = (num_bytes + threads - 1) / threads;

  unpack_orientations_kernel<<<blocks, threads>>>(
      thrust::raw_pointer_cast(d_packed.data()),
      thrust::raw_pointer_cast(d_orients.data()), num_orients, num_bytes);
  CUDA_CHECK(cudaGetLastError());

  // Copy result back to host
  std::vector<char> result(num_orients);
  thrust::copy(d_orients.begin(), d_orients.end(), result.begin());

  return result;
}

} // namespace gpu_codec
// This temporary file contains the new helpers. We will merge them into codec_gpu.cu later.
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include "gpu/codec_gpu.cuh"

namespace gpu_codec {

void merge_counted_2mers_device_vec(
    thrust::device_vector<uint64_t>& d_keys,
    thrust::device_vector<uint32_t>& d_counts,
    thrust::device_vector<uint64_t>& d_unique_keys,
    thrust::device_vector<uint32_t>& d_total_counts) {
    if (d_keys.empty()) {
        d_unique_keys.clear();
        d_total_counts.clear();
        return;
    }
    
    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_counts.begin());
    
    d_unique_keys.resize(d_keys.size());
    d_total_counts.resize(d_keys.size());
    
    auto new_end = thrust::reduce_by_key(
        d_keys.begin(), d_keys.end(),
        d_counts.begin(),
        d_unique_keys.begin(), d_total_counts.begin()
    );
    
    d_unique_keys.resize(new_end.first - d_unique_keys.begin());
    d_total_counts.resize(new_end.second - d_total_counts.begin());
}

struct GreaterEqual {
    uint32_t threshold;
    GreaterEqual(uint32_t t) : threshold(t) {}
    __host__ __device__ bool operator()(uint32_t count) const {
        return count >= threshold;
    }
};

void filter_rules_by_count_device_vec(
    const thrust::device_vector<uint64_t>& d_unique_keys,
    const thrust::device_vector<uint32_t>& d_total_counts,
    uint32_t min_count,
    thrust::device_vector<uint64_t>& d_round_rules) {
    
    d_round_rules.resize(d_unique_keys.size());
    
    auto it = thrust::copy_if(
        d_unique_keys.begin(), d_unique_keys.end(),
        d_total_counts.begin(),
        d_round_rules.begin(),
        GreaterEqual(min_count)
    );
    
    d_round_rules.resize(it - d_round_rules.begin());
}

void or_reduce_rules_used_device_vec(
    const thrust::device_vector<uint8_t>& d_chunk_used,
    thrust::device_vector<uint8_t>& d_round_used) {
    
    thrust::transform(d_round_used.begin(), d_round_used.end(),
                      d_chunk_used.begin(),
                      d_round_used.begin(),
                      thrust::maximum<uint8_t>());
}

void build_rule_compaction_map_device_vec(
    const thrust::device_vector<uint8_t>& d_rules_used_round,
    thrust::device_vector<uint64_t>& d_new_indices) {
    
    size_t num_rules = d_rules_used_round.size();
    
    thrust::device_vector<uint64_t> d_flags_int(num_rules);
    thrust::transform(d_rules_used_round.begin(), d_rules_used_round.end(), d_flags_int.begin(),
                      [] __device__(uint8_t v) { return v ? uint64_t(1) : uint64_t(0); });
                      
    d_new_indices.resize(num_rules);
    thrust::exclusive_scan(d_flags_int.begin(), d_flags_int.end(), d_new_indices.begin());
}

void build_rule_reorder_map_device_vec(
    const thrust::device_vector<uint64_t>& d_rules,
    thrust::device_vector<uint32_t>& d_reorder_map) {
    
    size_t num_rules = d_rules.size();
    thrust::device_vector<uint32_t> d_indices(num_rules);
    thrust::sequence(d_indices.begin(), d_indices.end());
    
    thrust::device_vector<uint64_t> d_rules_copy = d_rules;
    thrust::sort_by_key(d_rules_copy.begin(), d_rules_copy.end(), d_indices.begin());
    
    d_reorder_map.resize(num_rules);
    thrust::scatter(thrust::counting_iterator<uint32_t>(0),
                    thrust::counting_iterator<uint32_t>(static_cast<uint32_t>(num_rules)), d_indices.begin(),
                    d_reorder_map.begin());
}

} // namespace gpu_codec
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
