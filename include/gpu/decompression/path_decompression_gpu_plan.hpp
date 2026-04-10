#ifndef PATH_DECOMPRESSION_GPU_PLAN_HPP
#define PATH_DECOMPRESSION_GPU_PLAN_HPP

#include "gpu/core/codec_gpu.cuh"

#include <cstddef>
#include <cstdint>
#include <thrust/device_vector.h>
#include <vector>

namespace gpu_decompression {

struct RollingPathChunkMetadata {
  size_t node_count = 0;
  uint32_t segment_begin = 0;
  uint32_t segment_end = 0;
  int64_t expanded_begin = 0;
  int64_t expanded_end = 0;
  std::vector<uint32_t> lengths;
};

struct RollingPathDecodePlan {
  gpu_codec::RollingDecodeSchedule schedule;
  std::vector<uint32_t> lengths;
  thrust::device_vector<int64_t> d_output_offsets;
  thrust::device_vector<int64_t> d_rule_offsets;
  thrust::device_vector<int64_t> d_rule_sizes;
  thrust::device_vector<uint64_t> d_offs_final;
  thrust::device_vector<int32_t> d_expanded_rules;
  uint32_t min_rule_id = 0;
  uint32_t max_rule_id = 0;
};

gpu_codec::RollingDecodeSchedule build_passthrough_decode_schedule(
    const thrust::device_vector<uint32_t> &d_lens_final, size_t encoded_size,
    uint32_t traversals_per_chunk, size_t rolling_output_chunk_bytes);

RollingPathDecodePlan prepare_rolling_path_decode_plan(
    const thrust::device_vector<int32_t> &d_encoded_path,
    const thrust::device_vector<int32_t> &d_rules_first,
    const thrust::device_vector<int32_t> &d_rules_second,
    uint32_t min_rule_id, size_t num_rules,
    const thrust::device_vector<uint32_t> &d_lens_final,
    uint32_t traversals_per_chunk, size_t rolling_output_chunk_bytes);

RollingPathChunkMetadata describe_rolling_path_chunk(
    const RollingPathDecodePlan &plan, size_t chunk_index);

} // namespace gpu_decompression

#endif // PATH_DECOMPRESSION_GPU_PLAN_HPP
