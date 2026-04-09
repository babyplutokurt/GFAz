#include "gpu/compression_workflow_gpu_internal.hpp"
#include "gpu/codec_gpu.cuh"
#include "gpu/path_compression_gpu_legacy.hpp"

#include "codec/codec.hpp"
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

namespace gpu_compression {

namespace {

void finalize_traversal_columns(CompressedData &data,
                                const std::vector<int32_t> &encoded,
                                const std::vector<uint32_t> &lengths,
                                const std::vector<uint32_t> &original,
                                uint32_t num_paths) {
  const size_t path_count = std::min<size_t>(num_paths, lengths.size());
  data.sequence_lengths.assign(lengths.begin(), lengths.begin() + path_count);
  data.original_path_lengths.assign(original.begin(),
                                    original.begin() + path_count);
  data.walk_lengths.assign(lengths.begin() + path_count, lengths.end());
  data.original_walk_lengths.assign(original.begin() + path_count,
                                    original.end());

  size_t split_offset = 0;
  for (size_t i = 0; i < path_count; ++i)
    split_offset += lengths[i];

  std::vector<int32_t> path_data(encoded.begin(),
                                 encoded.begin() +
                                     static_cast<std::ptrdiff_t>(split_offset));
  std::vector<int32_t> walk_data(encoded.begin() +
                                     static_cast<std::ptrdiff_t>(split_offset),
                                 encoded.end());

  data.paths_zstd = Codec::zstd_compress_int32_vector(path_data);
  if (!walk_data.empty())
    data.walks_zstd = Codec::zstd_compress_int32_vector(walk_data);
}

} // namespace

CompressedData run_path_compression_gpu_full_device(const FlattenedPaths &paths,
                                                    uint32_t num_paths,
                                                    int num_rounds) {
  CompressedData result;

  if (paths.data.empty()) {
    result.original_path_lengths.assign(paths.lengths.begin(),
                                        paths.lengths.begin() +
                                            std::min<size_t>(num_paths,
                                                             paths.lengths.size()));
    result.sequence_lengths = result.original_path_lengths;
    result.original_walk_lengths.assign(
        paths.lengths.begin() +
            static_cast<std::ptrdiff_t>(std::min<size_t>(num_paths,
                                                         paths.lengths.size())),
        paths.lengths.end());
    result.walk_lengths = result.original_walk_lengths;
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
    size_t old_size = d_all_rules.size();
    result.layer_rule_ranges.push_back(
        LayerRuleRange{2, next_start_id, next_start_id + num_used_rules,
                       old_size * 2, num_used_rules * 2});
    d_all_rules.resize(old_size + num_used_rules);
    thrust::copy(d_round_rules.begin(), d_round_rules.end(),
                 d_all_rules.begin() + static_cast<std::ptrdiff_t>(old_size));

    next_start_id += num_used_rules;

    thrust::exclusive_scan(d_lengths.begin(), d_lengths.end(),
                           d_offsets.begin(), uint64_t(0));
  }

  std::vector<int32_t> encoded_host(d_data.size());
  thrust::copy(d_data.begin(), d_data.end(), encoded_host.begin());
  std::vector<uint32_t> encoded_lengths(d_lengths.size());
  thrust::copy(d_lengths.begin(), d_lengths.end(), encoded_lengths.begin());
  finalize_traversal_columns(result, encoded_host, encoded_lengths, paths.lengths,
                             num_paths);

  if (!d_all_rules.empty()) {
    thrust::device_vector<int32_t> d_first, d_second;
    gpu_codec::split_and_delta_encode_rules_device_vec(d_all_rules, d_first,
                                                       d_second);

    result.rules_first_zstd =
        compress_int32_device_gpu(d_first, "rules_first");
    result.rules_second_zstd =
        compress_int32_device_gpu(d_second, "rules_second");
  }

  return result;
}

} // namespace gpu_compression
