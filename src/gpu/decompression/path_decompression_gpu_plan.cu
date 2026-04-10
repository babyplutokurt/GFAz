#include "gpu/decompression/path_decompression_gpu_plan.hpp"

#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>
#include <thrust/transform_reduce.h>

namespace gpu_decompression {

namespace {

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

} // namespace

gpu_codec::RollingDecodeSchedule build_passthrough_decode_schedule(
    const thrust::device_vector<uint32_t> &d_lens_final, size_t encoded_size,
    uint32_t traversals_per_chunk, size_t max_expanded_chunk_bytes) {
  gpu_codec::RollingDecodeSchedule schedule;
  const uint32_t num_segs_final = static_cast<uint32_t>(d_lens_final.size());
  if (num_segs_final == 0) {
    schedule.output_size = static_cast<int64_t>(encoded_size);
    return schedule;
  }

  thrust::device_vector<uint64_t> d_offs_final(num_segs_final);
  thrust::exclusive_scan(d_lens_final.begin(), d_lens_final.end(),
                         d_offs_final.begin(), uint64_t(0));

  schedule.expanded_offsets.resize(num_segs_final);
  thrust::copy(d_offs_final.begin(), d_offs_final.end(),
               schedule.expanded_offsets.begin());
  schedule.encoded_offsets.assign(schedule.expanded_offsets.begin(),
                                  schedule.expanded_offsets.end());
  schedule.output_size = static_cast<int64_t>(encoded_size);

  uint32_t current_seg_begin = 0;
  for (uint32_t i = 0; i < num_segs_final; ++i) {
    const int64_t seg_expanded_start =
        static_cast<int64_t>(schedule.expanded_offsets[current_seg_begin]);
    const int64_t next_expanded_start =
        (i + 1 < num_segs_final)
            ? static_cast<int64_t>(schedule.expanded_offsets[i + 1])
            : static_cast<int64_t>(encoded_size);
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
    const int64_t chunk_begin =
        static_cast<int64_t>(schedule.expanded_offsets[current_seg_begin]);
    schedule.chunks.push_back({current_seg_begin, chunk_seg_end, chunk_begin,
                               next_expanded_start, chunk_begin,
                               next_expanded_start});
    current_seg_begin = chunk_seg_end;
  }

  return schedule;
}

RollingPathDecodePlan prepare_rolling_path_decode_plan(
    const thrust::device_vector<int32_t> &d_encoded_path,
    const thrust::device_vector<int32_t> &d_rules_first,
    const thrust::device_vector<int32_t> &d_rules_second,
    uint32_t min_rule_id, size_t num_rules,
    const thrust::device_vector<uint32_t> &d_lens_final,
    uint32_t traversals_per_chunk, size_t max_expanded_chunk_bytes) {
  RollingPathDecodePlan plan;
  plan.min_rule_id = min_rule_id;
  plan.max_rule_id = min_rule_id + static_cast<uint32_t>(num_rules);
  plan.lengths.resize(d_lens_final.size());
  thrust::copy(d_lens_final.begin(), d_lens_final.end(), plan.lengths.begin());

  if (num_rules == 0 || d_encoded_path.empty()) {
    plan.schedule = build_passthrough_decode_schedule(
        d_lens_final, d_encoded_path.size(), traversals_per_chunk,
        max_expanded_chunk_bytes);
    plan.d_offs_final = thrust::device_vector<uint64_t>(
        plan.schedule.expanded_offsets.begin(),
        plan.schedule.expanded_offsets.end());
    return plan;
  }

  plan.d_rule_sizes.resize(num_rules);
  gpu_codec::compute_rule_final_sizes_device_vec(
      d_rules_first, d_rules_second, plan.d_rule_sizes, min_rule_id);

  plan.d_rule_offsets.resize(num_rules);
  thrust::exclusive_scan(plan.d_rule_sizes.begin(), plan.d_rule_sizes.end(),
                         plan.d_rule_offsets.begin());

  const int64_t total_expanded_size =
      plan.d_rule_offsets.back() + plan.d_rule_sizes.back();
  plan.d_expanded_rules.resize(total_expanded_size);
  gpu_codec::expand_rules_to_buffer_device_vec(
      d_rules_first, d_rules_second, plan.d_rule_offsets,
      plan.d_expanded_rules, min_rule_id);

  plan.d_output_offsets.resize(d_encoded_path.size());
  FinalExpansionSizeOp size_op(
      thrust::raw_pointer_cast(plan.d_rule_sizes.data()), min_rule_id,
      plan.max_rule_id);
  auto size_iter =
      thrust::make_transform_iterator(d_encoded_path.begin(), size_op);
  const int64_t output_size = thrust::transform_reduce(
      d_encoded_path.begin(), d_encoded_path.end(), size_op, int64_t(0),
      thrust::plus<int64_t>());
  thrust::exclusive_scan(size_iter, size_iter + d_encoded_path.size(),
                         plan.d_output_offsets.begin());

  plan.schedule = gpu_codec::build_rolling_decode_schedule(
      plan.d_output_offsets, d_lens_final, d_encoded_path.size(), output_size,
      traversals_per_chunk, max_expanded_chunk_bytes);
  plan.d_offs_final = thrust::device_vector<uint64_t>(
      plan.schedule.expanded_offsets.begin(),
      plan.schedule.expanded_offsets.end());

  return plan;
}

RollingPathChunkMetadata describe_rolling_path_chunk(
    const RollingPathDecodePlan &plan, size_t chunk_index) {
  RollingPathChunkMetadata metadata;
  const auto &chunk = plan.schedule.chunks.at(chunk_index);
  metadata.node_count = static_cast<size_t>(chunk.expanded_count());
  metadata.segment_begin = chunk.segment_begin;
  metadata.segment_end = chunk.segment_end;
  metadata.expanded_begin = chunk.expanded_begin;
  metadata.expanded_end = chunk.expanded_end;
  metadata.lengths.assign(plan.lengths.begin() + chunk.segment_begin,
                          plan.lengths.begin() + chunk.segment_end);
  return metadata;
}

} // namespace gpu_decompression
