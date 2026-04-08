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
