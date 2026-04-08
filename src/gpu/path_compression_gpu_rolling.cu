#include "gpu/compression_workflow_gpu_internal.hpp"
#include "gpu/codec_gpu.cuh"
#include "gpu/path_compression_gpu_rolling.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/system_error.h>

namespace gpu_compression {

namespace {

enum class ChunkOffsetBuildMode {
  kFallback,
  kGpuOnly,
  kCpuOnly,
};

ChunkOffsetBuildMode chunk_offset_build_mode() {
  const char *env = std::getenv("GFAZ_GPU_CHUNK_OFFSET_MODE");
  if (!env || *env == '\0') {
    return ChunkOffsetBuildMode::kFallback;
  }

  const std::string mode(env);
  if (mode == "gpu") {
    return ChunkOffsetBuildMode::kGpuOnly;
  }
  if (mode == "cpu") {
    return ChunkOffsetBuildMode::kCpuOnly;
  }
  return ChunkOffsetBuildMode::kFallback;
}

void build_chunk_offsets_cpu(const std::vector<uint32_t> &chunk_lengths,
                             thrust::device_vector<uint64_t> &d_chunk_offsets) {
  std::vector<uint64_t> h_chunk_offsets(chunk_lengths.size(), uint64_t(0));
  uint64_t running = 0;
  for (size_t i = 0; i < chunk_lengths.size(); ++i) {
    h_chunk_offsets[i] = running;
    running += chunk_lengths[i];
  }
  d_chunk_offsets.assign(h_chunk_offsets.begin(), h_chunk_offsets.end());
}

void build_chunk_offsets(const std::vector<uint32_t> &lengths,
                         const TraversalChunk &chunk, size_t chunk_idx,
                         const char *phase_label,
                         thrust::device_vector<uint32_t> &d_chunk_lengths,
                         thrust::device_vector<uint64_t> &d_chunk_offsets) {
  std::vector<uint32_t> h_chunk_lengths(
      lengths.begin() + static_cast<std::ptrdiff_t>(chunk.segment_begin),
      lengths.begin() + static_cast<std::ptrdiff_t>(chunk.segment_end));

  d_chunk_lengths.assign(h_chunk_lengths.begin(), h_chunk_lengths.end());
  d_chunk_offsets.resize(h_chunk_lengths.size());

  const ChunkOffsetBuildMode mode = chunk_offset_build_mode();
  if (scheduler_debug_enabled()) {
    std::cerr << "[GPU Scheduler] " << phase_label << " chunk[" << chunk_idx
              << "] offset_mode="
              << (mode == ChunkOffsetBuildMode::kGpuOnly
                      ? "gpu"
                      : (mode == ChunkOffsetBuildMode::kCpuOnly ? "cpu"
                                                                : "fallback"))
              << ", lengths_size=" << h_chunk_lengths.size() << std::endl;
  }

  if (mode == ChunkOffsetBuildMode::kCpuOnly) {
    // These per-chunk prefix sums are tiny, so a host-side fallback is cheap
    // and avoids depending on a flaky scan path for scheduler metadata.
    build_chunk_offsets_cpu(h_chunk_lengths, d_chunk_offsets);
    return;
  }

  try {
    thrust::exclusive_scan(d_chunk_lengths.begin(), d_chunk_lengths.end(),
                           d_chunk_offsets.begin(), uint64_t(0));
  } catch (const thrust::system_error &err) {
    if (mode == ChunkOffsetBuildMode::kGpuOnly) {
      throw;
    }
    if (scheduler_debug_enabled()) {
      std::cerr << "[GPU Scheduler] " << phase_label << " chunk[" << chunk_idx
                << "] GPU offset scan failed, falling back to CPU: "
                << err.what() << std::endl;
    }
    cudaGetLastError();
    build_chunk_offsets_cpu(h_chunk_lengths, d_chunk_offsets);
  }
}

void print_chunk_summary(const std::vector<TraversalChunk> &chunks,
                         size_t target_chunk_nodes, const char *label) {
  if (!scheduler_debug_enabled()) {
    return;
  }

  std::cerr << "[GPU Scheduler] " << label
            << ": target_chunk_nodes=" << target_chunk_nodes
            << ", num_chunks=" << chunks.size() << std::endl;

  const size_t limit = std::min<size_t>(chunks.size(), 4);
  for (size_t i = 0; i < limit; ++i) {
    const auto &chunk = chunks[i];
    std::cerr << "  chunk[" << i << "] segs=" << chunk.num_segments()
              << " nodes=" << chunk.num_nodes()
              << " segment_range=[" << chunk.segment_begin << ", "
              << chunk.segment_end << ")"
              << " node_range=[" << chunk.node_begin << ", "
              << chunk.node_end << ")" << std::endl;
  }

  if (chunks.size() > limit) {
    const auto &last = chunks.back();
    std::cerr << "  ..."
              << "\n  chunk[" << (chunks.size() - 1)
              << "] segs=" << last.num_segments()
              << " nodes=" << last.num_nodes()
              << " segment_range=[" << last.segment_begin << ", "
              << last.segment_end << ")"
              << " node_range=[" << last.node_begin << ", " << last.node_end
              << ")" << std::endl;
  }
}

void sync_rolling_stage(const char *stage_label, int round_idx,
                        size_t chunk_idx = static_cast<size_t>(-1)) {
  cudaError_t err = cudaDeviceSynchronize();
  if (err == cudaSuccess) {
    return;
  }

  std::string message = "[GPU Rolling] ";
  message += stage_label;
  message += " failed";
  if (round_idx >= 0) {
    message += " at round ";
    message += std::to_string(round_idx);
  }
  if (chunk_idx != static_cast<size_t>(-1)) {
    message += ", chunk ";
    message += std::to_string(chunk_idx);
  }
  message += ": ";
  message += cudaGetErrorString(err);
  throw std::runtime_error(message);
}

uint32_t find_max_abs_host(const std::vector<int32_t> &data) {
  uint32_t max_abs = 0;
  for (int32_t value : data) {
    uint32_t abs_value =
        (value >= 0) ? static_cast<uint32_t>(value)
                     : (0u - static_cast<uint32_t>(value));
    if (abs_value > max_abs) {
      max_abs = abs_value;
    }
  }
  return max_abs;
}

void delta_encode_chunks_on_gpu(std::vector<int32_t> &data,
                                const std::vector<uint32_t> &lengths,
                                const std::vector<TraversalChunk> &chunks) {
  size_t max_nodes = 0;
  size_t max_segs = 0;
  for (const auto &chunk : chunks) {
    max_nodes = std::max(max_nodes, chunk.num_nodes());
    max_segs = std::max(max_segs, chunk.num_segments());
  }

  thrust::device_vector<int32_t> d_chunk_data;
  d_chunk_data.reserve(max_nodes);
  thrust::device_vector<uint32_t> d_chunk_lengths;
  d_chunk_lengths.reserve(max_segs);
  thrust::device_vector<uint64_t> d_chunk_offsets;
  d_chunk_offsets.reserve(max_segs);
  thrust::device_vector<uint8_t> d_is_first;
  d_is_first.reserve(max_nodes);
  thrust::device_vector<uint8_t> d_is_last;
  d_is_last.reserve(max_nodes);

  for (size_t chunk_idx = 0; chunk_idx < chunks.size(); ++chunk_idx) {
    const auto &chunk = chunks[chunk_idx];
    if (chunk.num_nodes() == 0) {
      continue;
    }

    if (scheduler_debug_enabled()) {
      std::cerr << "[GPU Scheduler] delta-encode chunk[" << chunk_idx
                << "] segs=" << chunk.num_segments()
                << " nodes=" << chunk.num_nodes()
                << " segment_range=[" << chunk.segment_begin << ", "
                << chunk.segment_end << ")"
                << " node_range=[" << chunk.node_begin << ", "
                << chunk.node_end << ")" << std::endl;
    }

    d_chunk_data.resize(chunk.num_nodes());
    thrust::copy(data.begin() + static_cast<std::ptrdiff_t>(chunk.node_begin),
                 data.begin() + static_cast<std::ptrdiff_t>(chunk.node_end),
                 d_chunk_data.begin());

    if (scheduler_debug_enabled()) {
      cudaError_t copy_err = cudaPeekAtLastError();
      std::cerr << "[GPU Scheduler] delta-encode chunk[" << chunk_idx
                << "] after copies cuda_status=" << cudaGetErrorString(copy_err)
                << std::endl;
    }

    if (scheduler_debug_enabled()) {
      std::cerr << "[GPU Scheduler] delta-encode chunk[" << chunk_idx
                << "] before offset build" << std::endl;
    }
    build_chunk_offsets(lengths, chunk, chunk_idx, "delta-encode",
                        d_chunk_lengths,
                        d_chunk_offsets);
    if (scheduler_debug_enabled()) {
      cudaError_t scan_err = cudaPeekAtLastError();
      std::cerr << "[GPU Scheduler] delta-encode chunk[" << chunk_idx
                << "] after offset build cuda_status="
                << cudaGetErrorString(scan_err) << std::endl;
    }

    d_is_first.resize(d_chunk_data.size());
    d_is_last.resize(d_chunk_data.size());
    if (scheduler_debug_enabled()) {
      std::cerr << "[GPU Scheduler] delta-encode chunk[" << chunk_idx
                << "] before boundary mask build" << std::endl;
    }
    gpu_codec::compute_boundary_masks(d_chunk_offsets,
                                      static_cast<uint32_t>(chunk.num_segments()),
                                      d_chunk_data.size(), d_is_first,
                                      d_is_last);
    if (scheduler_debug_enabled()) {
      cudaError_t mask_err = cudaPeekAtLastError();
      std::cerr << "[GPU Scheduler] delta-encode chunk[" << chunk_idx
                << "] after boundary mask build cuda_status="
                << cudaGetErrorString(mask_err) << std::endl;
    }
    gpu_codec::segmented_delta_encode_device_vec(d_chunk_data, d_is_first);
    sync_rolling_stage("delta_encode", -1, chunk_idx);

    thrust::copy(d_chunk_data.begin(), d_chunk_data.end(),
                 data.begin() + static_cast<std::ptrdiff_t>(chunk.node_begin));
  }
}

} // namespace

CompressedData_gpu run_path_compression_gpu_rolling(const FlattenedPaths &paths,
                                                    int num_rounds,
                                                    size_t chunk_bytes) {
  CompressedData_gpu result;

  if (paths.data.empty()) {
    if (!paths.lengths.empty()) {
      result.path_lengths_zstd_nvcomp =
          gpu_codec::nvcomp_zstd_compress_uint32(paths.lengths);
    }
    return result;
  }

  std::vector<int32_t> current_data = paths.data;
  std::vector<uint32_t> current_lengths = paths.lengths;
  const uint32_t num_segments = static_cast<uint32_t>(current_lengths.size());
  const size_t target_chunk_nodes =
      std::max<size_t>(1, chunk_bytes / sizeof(int32_t));

  auto chunks = build_traversal_chunks(current_lengths, target_chunk_nodes);
  print_chunk_summary(chunks, target_chunk_nodes, "initial rolling plan");
  delta_encode_chunks_on_gpu(current_data, current_lengths, chunks);

  uint32_t start_id = find_max_abs_host(paths.data) + 1;
  uint32_t delta_max = find_max_abs_host(current_data);
  if (delta_max >= start_id) {
    start_id = delta_max + 1;
  }

  size_t max_nodes = 0;
  size_t max_segs = 0;
  for (const auto &chunk : chunks) {
    max_nodes = std::max(max_nodes, chunk.num_nodes());
    max_segs = std::max(max_segs, chunk.num_segments());
  }

  thrust::device_vector<int32_t> d_chunk_data;
  d_chunk_data.reserve(max_nodes);
  thrust::device_vector<uint32_t> d_chunk_lengths;
  d_chunk_lengths.reserve(max_segs);
  thrust::device_vector<uint64_t> d_chunk_offsets;
  d_chunk_offsets.reserve(max_segs);
  thrust::device_vector<uint8_t> d_is_first;
  d_is_first.reserve(max_nodes);
  thrust::device_vector<uint8_t> d_is_last;
  d_is_last.reserve(max_nodes);
  thrust::device_vector<uint64_t> d_unique_keys;
  d_unique_keys.reserve(max_nodes);
  thrust::device_vector<uint32_t> d_counts;
  d_counts.reserve(max_nodes);
  thrust::device_vector<uint8_t> d_rules_used_dev;
  thrust::device_vector<uint64_t> d_round_hist_keys;
  thrust::device_vector<uint32_t> d_round_hist_counts;
  d_round_hist_keys.reserve(max_nodes * chunks.size());
  d_round_hist_counts.reserve(max_nodes * chunks.size());
  thrust::device_vector<uint64_t> d_merged_keys;
  thrust::device_vector<uint32_t> d_merged_counts;
  thrust::device_vector<uint64_t> d_round_rules;
  thrust::device_vector<uint8_t> d_rules_used_round;
  thrust::device_vector<uint64_t> d_new_indices;
  thrust::device_vector<uint32_t> d_reorder_map;

  std::vector<uint64_t> all_rules;
  uint32_t next_start_id = start_id;

  for (int round_idx = 0; round_idx < num_rounds; ++round_idx) {
    chunks = build_traversal_chunks(current_lengths, target_chunk_nodes);
    if (round_idx == 0) {
      print_chunk_summary(chunks, target_chunk_nodes, "round 0 rolling plan");
    }

    d_round_hist_keys.clear();
    d_round_hist_counts.clear();

    for (size_t chunk_idx = 0; chunk_idx < chunks.size(); ++chunk_idx) {
      const auto &chunk = chunks[chunk_idx];
      if (chunk.num_nodes() < 2) {
        continue;
      }

      d_chunk_data.resize(chunk.num_nodes());
      thrust::copy(current_data.begin() +
                       static_cast<std::ptrdiff_t>(chunk.node_begin),
                   current_data.begin() +
                       static_cast<std::ptrdiff_t>(chunk.node_end),
                   d_chunk_data.begin());

      build_chunk_offsets(current_lengths, chunk, chunk_idx, "count",
                          d_chunk_lengths, d_chunk_offsets);

      d_is_first.resize(d_chunk_data.size());
      d_is_last.resize(d_chunk_data.size());
      gpu_codec::compute_boundary_masks(
          d_chunk_offsets, static_cast<uint32_t>(chunk.num_segments()),
          d_chunk_data.size(), d_is_first, d_is_last);
      sync_rolling_stage("count_boundary_masks", round_idx, chunk_idx);

      gpu_codec::count_2mers_segmented_device_vec(d_chunk_data, d_is_last,
                                                  d_unique_keys, d_counts);
      sync_rolling_stage("count_2mers", round_idx, chunk_idx);

      if (!d_unique_keys.empty()) {
        size_t old_size = d_round_hist_keys.size();
        d_round_hist_keys.resize(old_size + d_unique_keys.size());
        d_round_hist_counts.resize(old_size + d_counts.size());
        thrust::copy(d_unique_keys.begin(), d_unique_keys.end(),
                     d_round_hist_keys.begin() + old_size);
        thrust::copy(d_counts.begin(), d_counts.end(),
                     d_round_hist_counts.begin() + old_size);
      }
    }

    gpu_codec::merge_counted_2mers_device_vec(
        d_round_hist_keys, d_round_hist_counts, d_merged_keys,
        d_merged_counts);
    sync_rolling_stage("merge_counted_2mers", round_idx);
    gpu_codec::filter_rules_by_count_device_vec(d_merged_keys, d_merged_counts,
                                                2, d_round_rules);
    sync_rolling_stage("filter_rules_by_count", round_idx);

    if (d_round_rules.empty()) {
      break;
    }

    void *table_ptr =
        gpu_codec::create_rule_table_gpu_from_device(d_round_rules,
                                                     next_start_id);
    sync_rolling_stage("create_rule_table", round_idx);
    d_rules_used_round.resize(d_round_rules.size());
    thrust::fill(d_rules_used_round.begin(), d_rules_used_round.end(), 0);
    sync_rolling_stage("init_round_rule_usage", round_idx);

    std::vector<int32_t> next_data;
    next_data.reserve(current_data.size());
    std::vector<uint32_t> next_lengths;
    next_lengths.reserve(current_lengths.size());

    for (size_t chunk_idx = 0; chunk_idx < chunks.size(); ++chunk_idx) {
      const auto &chunk = chunks[chunk_idx];
      d_chunk_data.resize(chunk.num_nodes());
      thrust::copy(current_data.begin() +
                       static_cast<std::ptrdiff_t>(chunk.node_begin),
                   current_data.begin() +
                       static_cast<std::ptrdiff_t>(chunk.node_end),
                   d_chunk_data.begin());

      build_chunk_offsets(current_lengths, chunk, chunk_idx, "apply",
                          d_chunk_lengths, d_chunk_offsets);

      d_rules_used_dev.resize(d_round_rules.size());
      thrust::fill(d_rules_used_dev.begin(), d_rules_used_dev.end(), 0);
      thrust::device_vector<uint32_t> d_new_lengths =
          gpu_codec::apply_2mer_rules_segmented_device_vec(
              d_chunk_data, table_ptr, d_rules_used_dev, next_start_id,
              d_chunk_offsets, static_cast<uint32_t>(chunk.num_segments()));
      sync_rolling_stage("apply_rules", round_idx, chunk_idx);

      gpu_codec::or_reduce_rules_used_device_vec(d_rules_used_dev,
                                                 d_rules_used_round);
      sync_rolling_stage("reduce_rule_usage", round_idx, chunk_idx);

      size_t old_size = next_data.size();
      next_data.resize(old_size + d_chunk_data.size());
      thrust::copy(d_chunk_data.begin(), d_chunk_data.end(),
                   next_data.begin() + static_cast<std::ptrdiff_t>(old_size));

      size_t old_lengths_size = next_lengths.size();
      next_lengths.resize(old_lengths_size + d_new_lengths.size());
      thrust::copy(d_new_lengths.begin(), d_new_lengths.end(),
                   next_lengths.begin() +
                       static_cast<std::ptrdiff_t>(old_lengths_size));
    }

    gpu_codec::free_rule_table_gpu(table_ptr);
    sync_rolling_stage("free_rule_table", round_idx);

    current_data = std::move(next_data);
    current_lengths = std::move(next_lengths);

    uint32_t num_rules_before_compact =
        static_cast<uint32_t>(d_rules_used_round.size());
    gpu_codec::build_rule_compaction_map_device_vec(d_rules_used_round,
                                                    d_new_indices);
    sync_rolling_stage("build_rule_compaction_map", round_idx);

    auto end_it = thrust::copy_if(
        d_round_rules.begin(), d_round_rules.end(), d_rules_used_round.begin(),
        d_round_rules.begin(),
        [] __device__(uint8_t used) { return used != 0; });
    d_round_rules.resize(end_it - d_round_rules.begin());
    sync_rolling_stage("compact_rules", round_idx);

    uint32_t num_rules_after_compact =
        static_cast<uint32_t>(d_round_rules.size());
    if (num_rules_after_compact == 0) {
      break;
    }

    gpu_codec::build_rule_reorder_map_device_vec(d_round_rules, d_reorder_map);
    sync_rolling_stage("build_rule_reorder_map", round_idx);
    thrust::sort(d_round_rules.begin(), d_round_rules.end());
    sync_rolling_stage("sort_rules", round_idx);

    auto remap_chunks = build_traversal_chunks(current_lengths,
                                               target_chunk_nodes);
    for (const auto &chunk : remap_chunks) {
      if (chunk.num_nodes() == 0) {
        continue;
      }
      d_chunk_data.resize(chunk.num_nodes());
      thrust::copy(current_data.begin() +
                       static_cast<std::ptrdiff_t>(chunk.node_begin),
                   current_data.begin() +
                       static_cast<std::ptrdiff_t>(chunk.node_end),
                   d_chunk_data.begin());

      gpu_codec::remap_chunk_rule_ids_device_vec(
          d_chunk_data, d_new_indices, d_reorder_map, next_start_id,
          num_rules_before_compact, num_rules_after_compact);
      sync_rolling_stage("remap_rule_ids", round_idx);

      thrust::copy(d_chunk_data.begin(), d_chunk_data.end(),
                   current_data.begin() +
                       static_cast<std::ptrdiff_t>(chunk.node_begin));
    }

    result.layer_ranges.push_back({next_start_id, num_rules_after_compact});

    std::vector<uint64_t> h_round_rules(num_rules_after_compact);
    thrust::copy(d_round_rules.begin(), d_round_rules.end(),
                 h_round_rules.begin());
    all_rules.insert(all_rules.end(), h_round_rules.begin(),
                     h_round_rules.end());
    next_start_id += num_rules_after_compact;
  }

  result.encoded_path_zstd_nvcomp =
      compress_int32_gpu(current_data, "encoded_path");
  result.path_lengths_zstd_nvcomp =
      compress_uint32_gpu(paths.lengths, "path_lengths");

  if (!all_rules.empty()) {
    thrust::device_vector<uint64_t> d_all_rules(all_rules.begin(),
                                                all_rules.end());
    thrust::device_vector<int32_t> d_first, d_second;
    gpu_codec::split_and_delta_encode_rules_device_vec(d_all_rules, d_first,
                                                       d_second);
    result.rules_first_zstd_nvcomp =
        compress_int32_device_gpu(d_first, "rules_first");
    result.rules_second_zstd_nvcomp =
        compress_int32_device_gpu(d_second, "rules_second");
  }

  if (compression_debug_enabled()) {
    std::cout << "[GPU Compression] Rolling scheduler used for "
              << paths.total_nodes() * sizeof(int32_t) / (1024.0 * 1024.0)
              << " MB traversal payload across " << num_segments
              << " traversals (chunk budget "
              << chunk_bytes / (1024.0 * 1024.0) << " MB)" << std::endl;
  }

  return result;
}

} // namespace gpu_compression
