#ifndef PATH_DECOMPRESSION_GPU_ROLLING_HPP
#define PATH_DECOMPRESSION_GPU_ROLLING_HPP

#include "gpu/codec_gpu.cuh"

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <thrust/device_vector.h>
#include <vector>

namespace gpu_decompression {

struct RollingPathHostBuffer {
  int32_t *host_nodes = nullptr;
  size_t node_capacity = 0;
  size_t node_count = 0;
  uint32_t segment_begin = 0;
  uint32_t segment_end = 0;
  int64_t expanded_begin = 0;
  int64_t expanded_end = 0;
  std::vector<uint32_t> lengths;
};

struct RollingPathPinnedHostBuffer {
  int32_t *host_nodes = nullptr;
  size_t node_capacity = 0;
  size_t node_count = 0;
  uint32_t segment_begin = 0;
  uint32_t segment_end = 0;
  int64_t expanded_begin = 0;
  int64_t expanded_end = 0;
  std::vector<uint32_t> lengths;
  cudaEvent_t ready = nullptr;
};

struct RollingPathDecodeContext {
  gpu_codec::RollingDecodeSchedule schedule;
  std::vector<uint32_t> lengths;
  thrust::device_vector<int64_t> d_output_offsets;
  thrust::device_vector<int64_t> d_rule_offsets;
  thrust::device_vector<int64_t> d_rule_sizes;
  thrust::device_vector<uint64_t> d_offs_final;
  thrust::device_vector<int32_t> d_expanded_rules;
  thrust::device_vector<int32_t> d_chunk_workspace;
  uint32_t min_rule_id = 0;
  uint32_t max_rule_id = 0;
};

struct RollingPathStreamOptions {
  size_t num_host_buffers = 2;
};

using RollingPathChunkConsumer =
    std::function<void(const RollingPathPinnedHostBuffer &)>;

RollingPathDecodeContext prepare_rolling_path_decode(
    const thrust::device_vector<int32_t> &d_encoded_path,
    const thrust::device_vector<int32_t> &d_rules_first,
    const thrust::device_vector<int32_t> &d_rules_second,
    uint32_t min_rule_id, size_t num_rules,
    const thrust::device_vector<uint32_t> &d_lens_final,
    uint32_t traversals_per_chunk);

void decode_rolling_path_chunk_to_device(
    const thrust::device_vector<int32_t> &d_encoded_path,
    RollingPathDecodeContext &context, size_t chunk_index);

void prepare_rolling_path_host_buffer(const RollingPathDecodeContext &context,
                                      size_t chunk_index,
                                      RollingPathHostBuffer &host_buffer);

void copy_rolling_path_chunk_to_host_buffer(
    const RollingPathDecodeContext &context, size_t chunk_index,
    RollingPathHostBuffer &host_buffer);

void ensure_rolling_path_pinned_host_buffer_capacity(
    RollingPathPinnedHostBuffer &host_buffer, size_t required_capacity);

void release_rolling_path_pinned_host_buffer(
    RollingPathPinnedHostBuffer &host_buffer);

void copy_rolling_path_chunk_to_pinned_host_async(
    const RollingPathDecodeContext &context, size_t chunk_index,
    RollingPathPinnedHostBuffer &host_buffer, cudaStream_t copy_stream);

void wait_for_rolling_path_pinned_host_buffer(
    const RollingPathPinnedHostBuffer &host_buffer);

void stream_decompress_paths_gpu_rolling(
    const thrust::device_vector<int32_t> &d_encoded_path,
    const thrust::device_vector<int32_t> &d_rules_first,
    const thrust::device_vector<int32_t> &d_rules_second,
    uint32_t min_rule_id, size_t num_rules,
    const thrust::device_vector<uint32_t> &d_lens_final,
    uint32_t traversals_per_chunk, RollingPathChunkConsumer consumer,
    RollingPathStreamOptions stream_options = {});

void decompress_paths_gpu_rolling(
    const thrust::device_vector<int32_t> &d_encoded_path,
    const thrust::device_vector<int32_t> &d_rules_first,
    const thrust::device_vector<int32_t> &d_rules_second,
    uint32_t min_rule_id, size_t num_rules,
    const thrust::device_vector<uint32_t> &d_lens_final,
    uint32_t traversals_per_chunk, std::vector<int32_t> &out_data);

} // namespace gpu_decompression

#endif // PATH_DECOMPRESSION_GPU_ROLLING_HPP
