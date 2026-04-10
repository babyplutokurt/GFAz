#ifndef PATH_RULEBOOK_GPU_HPP
#define PATH_RULEBOOK_GPU_HPP

#include <cstdint>

#include <thrust/device_vector.h>

namespace gpu_codec {

void merge_counted_2mers_device_vec(
    thrust::device_vector<uint64_t>& d_keys,
    thrust::device_vector<uint32_t>& d_counts,
    thrust::device_vector<uint64_t>& d_unique_keys,
    thrust::device_vector<uint32_t>& d_total_counts);

void merge_reduced_counted_2mers_device_vec(
    const thrust::device_vector<uint64_t>& d_lhs_keys,
    const thrust::device_vector<uint32_t>& d_lhs_counts,
    const thrust::device_vector<uint64_t>& d_rhs_keys,
    const thrust::device_vector<uint32_t>& d_rhs_counts,
    thrust::device_vector<uint64_t>& d_unique_keys,
    thrust::device_vector<uint32_t>& d_total_counts);

void filter_rules_by_count_device_vec(
    const thrust::device_vector<uint64_t>& d_unique_keys,
    const thrust::device_vector<uint32_t>& d_total_counts,
    uint32_t min_count,
    thrust::device_vector<uint64_t>& d_round_rules);

void or_reduce_rules_used_device_vec(
    const thrust::device_vector<uint8_t>& d_chunk_used,
    thrust::device_vector<uint8_t>& d_round_used);

void build_rule_compaction_map_device_vec(
    const thrust::device_vector<uint8_t>& d_rules_used_round,
    thrust::device_vector<uint64_t>& d_new_indices);

void build_rule_reorder_map_device_vec(
    const thrust::device_vector<uint64_t>& d_rules,
    thrust::device_vector<uint32_t>& d_reorder_map);

} // namespace gpu_codec

#endif // PATH_RULEBOOK_GPU_HPP
