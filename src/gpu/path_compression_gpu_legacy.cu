#include "gpu/compression_workflow_gpu_internal.hpp"
#include "gpu/codec_gpu.cuh"
#include "gpu/path_compression_gpu_legacy.hpp"

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

namespace gpu_compression {

CompressedData_gpu run_path_compression_gpu_full_device(
    const FlattenedPaths &paths, int num_rounds) {
  CompressedData_gpu result;

  if (paths.data.empty()) {
    // Even with empty data, store path_lengths so decompression can
    // reconstruct the correct number of (zero-length) paths/walks.
    if (!paths.lengths.empty()) {
      result.path_lengths_zstd_nvcomp =
          gpu_codec::nvcomp_zstd_compress_uint32(paths.lengths);
    }
    return result;
  }

  thrust::device_vector<int32_t> d_data(paths.data.begin(), paths.data.end());
  thrust::device_vector<uint32_t> d_lengths(paths.lengths.begin(),
                                            paths.lengths.end());
  uint32_t num_segments = static_cast<uint32_t>(paths.lengths.size());

  thrust::device_vector<uint64_t> d_offsets(num_segments);
  thrust::exclusive_scan(d_lengths.begin(), d_lengths.end(), d_offsets.begin(),
                         uint64_t(0));

  uint32_t start_id = gpu_codec::find_max_abs_device(d_data) + 1;

  {
    thrust::device_vector<uint8_t> d_is_first, d_is_last;
    gpu_codec::compute_boundary_masks(d_offsets, num_segments, d_data.size(),
                                      d_is_first, d_is_last);
    gpu_codec::segmented_delta_encode_device_vec(d_data, d_is_first);
  }

  uint32_t delta_max = gpu_codec::find_max_abs_device(d_data);
  if (delta_max >= start_id) {
    start_id = delta_max + 1;
  }

  thrust::device_vector<uint64_t> d_all_rules;
  uint32_t next_start_id = start_id;

  for (int round_idx = 0; round_idx < num_rounds; ++round_idx) {
    thrust::device_vector<uint8_t> d_is_first, d_is_last;
    gpu_codec::compute_boundary_masks(d_offsets, num_segments, d_data.size(),
                                      d_is_first, d_is_last);

    thrust::device_vector<uint64_t> d_round_rules =
        gpu_codec::find_repeated_2mers_segmented_device_vec(d_data, d_is_last);

    if (d_round_rules.empty()) {
      break;
    }

    uint32_t num_rules_found = d_round_rules.size();

    void *table_ptr = gpu_codec::create_rule_table_gpu_from_device(
        d_round_rules, next_start_id);

    thrust::device_vector<uint8_t> d_rules_used(num_rules_found, 0);
    d_lengths = gpu_codec::apply_2mer_rules_segmented_device_vec(
        d_data, table_ptr, d_rules_used, next_start_id, d_offsets,
        num_segments);

    gpu_codec::free_rule_table_gpu(table_ptr);

    gpu_codec::compact_rules_and_remap_device_vec(d_data, d_rules_used,
                                                  d_round_rules,
                                                  next_start_id);
    gpu_codec::sort_rules_and_remap_device_vec(d_data, d_round_rules,
                                               next_start_id);

    uint32_t num_used_rules = d_round_rules.size();
    result.layer_ranges.push_back({next_start_id, num_used_rules});

    size_t old_size = d_all_rules.size();
    d_all_rules.resize(old_size + num_used_rules);
    thrust::copy(d_round_rules.begin(), d_round_rules.end(),
                 d_all_rules.begin() + static_cast<std::ptrdiff_t>(old_size));

    next_start_id += num_used_rules;

    thrust::exclusive_scan(d_lengths.begin(), d_lengths.end(),
                           d_offsets.begin(), uint64_t(0));
  }

  result.encoded_path_zstd_nvcomp =
      compress_int32_device_gpu(d_data, "encoded_path");
  result.path_lengths_zstd_nvcomp =
      compress_uint32_gpu(paths.lengths, "path_lengths");

  if (!d_all_rules.empty()) {
    thrust::device_vector<int32_t> d_first, d_second;
    gpu_codec::split_and_delta_encode_rules_device_vec(d_all_rules, d_first,
                                                       d_second);

    result.rules_first_zstd_nvcomp =
        compress_int32_device_gpu(d_first, "rules_first");
    result.rules_second_zstd_nvcomp =
        compress_int32_device_gpu(d_second, "rules_second");
  }

  return result;
}

} // namespace gpu_compression
