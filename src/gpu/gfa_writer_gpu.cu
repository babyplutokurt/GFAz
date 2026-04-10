#include "gpu/gfa_writer_gpu.hpp"
#include "gpu/decompression_workflow_gpu_internal.hpp"
#include "gpu/path_decompression_gpu_rolling.hpp"
#include "io/gfa_write_utils.hpp"
#include "io/gfa_writer.hpp"
#include "codec/codec.hpp"
#include "utils/runtime_utils.hpp"
#include "workflows/decompression_debug.hpp"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <thrust/device_vector.h>

#ifdef ENABLE_CUDA

namespace {

using Clock = std::chrono::high_resolution_clock;
using gfz::runtime_utils::elapsed_ms;
using namespace gfz::gfa_write_utils;
using namespace gfz::decompression_debug;

void stream_paths_gpu_rolling_to_writer(
    std::ofstream &out, const CompressedData &data,
    const thrust::device_vector<int32_t> &d_rules_first,
    const thrust::device_vector<int32_t> &d_rules_second,
    gpu_decompression::GpuDecompressionOptions options,
    std::vector<TimedDebugStage> &debug_stages) {
  if (data.paths_zstd.payload.empty() || data.original_path_lengths.empty()) {
    return;
  }

  auto t0 = Clock::now();
  std::vector<int32_t> encoded_host =
      Codec::zstd_decompress_int32_vector(data.paths_zstd);
  thrust::device_vector<int32_t> d_encoded(encoded_host.begin(),
                                           encoded_host.end());
  thrust::device_vector<uint32_t> d_final_lengths(
      data.original_path_lengths.begin(), data.original_path_lengths.end());
  std::vector<std::string> path_names =
      decompress_string_column(data.names_zstd, data.name_lengths_zstd);
  std::vector<std::string> path_overlaps =
      decompress_string_column(data.overlaps_zstd, data.overlap_lengths_zstd);
  auto t1 = Clock::now();
  debug_stages.push_back({"decode path payload+metadata", elapsed_ms(t0, t1)});

  t0 = Clock::now();
  gpu_decompression::stream_decompress_paths_gpu_rolling(
      d_encoded, d_rules_first, d_rules_second, data.min_rule_id(),
      data.total_rules(), d_final_lengths,
      std::max<uint32_t>(1, options.traversals_per_chunk),
      [&](const gpu_decompression::RollingPathPinnedHostBuffer &buffer) {
        size_t offset = 0;
        for (size_t local_index = 0; local_index < buffer.lengths.size();
             ++local_index) {
          const size_t global_index = buffer.segment_begin + local_index;
          const size_t length = buffer.lengths[local_index];
          const std::string &name =
              (global_index < path_names.size()) ? path_names[global_index]
                                                 : std::to_string(global_index);
          const std::string &overlap =
              (global_index < path_overlaps.size()) ? path_overlaps[global_index]
                                                    : "";
          const std::string line = format_path_line_numeric(
              name, buffer.host_nodes + offset, length, overlap);
          out.write(line.data(), static_cast<std::streamsize>(line.size()));
          offset += length;
        }
      },
      {.num_host_buffers = 2,
       .max_expanded_chunk_bytes =
           std::max<size_t>(1, options.max_expanded_chunk_bytes)});
  t1 = Clock::now();
  debug_stages.push_back({"expand+write paths (GPU rolling)",
                          elapsed_ms(t0, t1)});
}

void stream_walks_gpu_rolling_to_writer(
    std::ofstream &out, const CompressedData &data,
    const thrust::device_vector<int32_t> &d_rules_first,
    const thrust::device_vector<int32_t> &d_rules_second,
    gpu_decompression::GpuDecompressionOptions options,
    std::vector<TimedDebugStage> &debug_stages) {
  if (data.walks_zstd.payload.empty() || data.original_walk_lengths.empty()) {
    return;
  }

  auto t0 = Clock::now();
  std::vector<int32_t> encoded_host =
      Codec::zstd_decompress_int32_vector(data.walks_zstd);
  thrust::device_vector<int32_t> d_encoded(encoded_host.begin(),
                                           encoded_host.end());
  thrust::device_vector<uint32_t> d_final_lengths(
      data.original_walk_lengths.begin(), data.original_walk_lengths.end());
  std::vector<std::string> walk_sample_ids = decompress_string_column(
      data.walk_sample_ids_zstd, data.walk_sample_id_lengths_zstd);
  std::vector<uint32_t> walk_hap_indices =
      Codec::zstd_decompress_uint32_vector(data.walk_hap_indices_zstd);
  std::vector<std::string> walk_seq_ids = decompress_string_column(
      data.walk_seq_ids_zstd, data.walk_seq_id_lengths_zstd);
  std::vector<int64_t> walk_seq_starts = Codec::decompress_varint_int64(
      data.walk_seq_starts_zstd, data.walk_lengths.size());
  std::vector<int64_t> walk_seq_ends = Codec::decompress_varint_int64(
      data.walk_seq_ends_zstd, data.walk_lengths.size());
  auto t1 = Clock::now();
  debug_stages.push_back({"decode walk payload+metadata", elapsed_ms(t0, t1)});

  t0 = Clock::now();
  gpu_decompression::stream_decompress_paths_gpu_rolling(
      d_encoded, d_rules_first, d_rules_second, data.min_rule_id(),
      data.total_rules(), d_final_lengths,
      std::max<uint32_t>(1, options.traversals_per_chunk),
      [&](const gpu_decompression::RollingPathPinnedHostBuffer &buffer) {
        size_t offset = 0;
        for (size_t local_index = 0; local_index < buffer.lengths.size();
             ++local_index) {
          const size_t global_index = buffer.segment_begin + local_index;
          const size_t length = buffer.lengths[local_index];
          const std::string &sample_id =
              (global_index < walk_sample_ids.size())
                  ? walk_sample_ids[global_index]
                  : std::string("sample");
          const uint32_t hap_index =
              (global_index < walk_hap_indices.size())
                  ? walk_hap_indices[global_index]
                  : 0;
          const std::string &seq_id =
              (global_index < walk_seq_ids.size()) ? walk_seq_ids[global_index]
                                                   : std::string("unknown");
          const int64_t seq_start =
              (global_index < walk_seq_starts.size())
                  ? walk_seq_starts[global_index]
                  : -1;
          const int64_t seq_end =
              (global_index < walk_seq_ends.size()) ? walk_seq_ends[global_index]
                                                    : -1;
          const std::string line = format_walk_line_numeric(
              sample_id, hap_index, seq_id, seq_start, seq_end,
              buffer.host_nodes + offset, length);
          out.write(line.data(), static_cast<std::streamsize>(line.size()));
          offset += length;
        }
      },
      {.num_host_buffers = 2,
       .max_expanded_chunk_bytes =
           std::max<size_t>(1, options.max_expanded_chunk_bytes)});
  t1 = Clock::now();
  debug_stages.push_back({"expand+write walks (GPU rolling)",
                          elapsed_ms(t0, t1)});
}

} // namespace

void write_gfa_from_compressed_data_gpu(
    const CompressedData &data, const std::string &output_path,
    gpu_decompression::GpuDecompressionOptions options) {
  if (options.use_legacy_full_decompression) {
    GfaGraph graph = gpu_decompression::decompress_to_host_graph(data, options);
    write_gfa(graph, output_path);
    return;
  }

  std::vector<TimedDebugStage> debug_stages;
  const auto writer_total_start = Clock::now();
  std::ofstream out(output_path);
  if (!out) {
    throw std::runtime_error("GFA writer error: failed to open output file: " +
                             output_path);
  }

  std::vector<uint32_t> segment_lengths;
  std::string segment_sequences;
  std::vector<OptionalFieldColumn> segment_optional_fields;
  std::vector<uint32_t> link_from_ids;
  std::vector<uint32_t> link_to_ids;
  std::vector<char> link_from_orients;
  std::vector<char> link_to_orients;
  std::vector<uint32_t> link_overlap_nums;
  std::vector<char> link_overlap_ops;
  std::vector<OptionalFieldColumn> link_optional_fields;
  std::vector<uint32_t> jump_from_ids;
  std::vector<uint32_t> jump_to_ids;
  std::vector<char> jump_from_orients;
  std::vector<char> jump_to_orients;
  std::vector<std::string> jump_distances;
  std::vector<std::string> jump_rest_fields;
  std::vector<uint32_t> containment_container_ids;
  std::vector<uint32_t> containment_contained_ids;
  std::vector<char> containment_container_orients;
  std::vector<char> containment_contained_orients;
  std::vector<uint32_t> containment_positions;
  std::vector<std::string> containment_overlaps;
  std::vector<std::string> containment_rest_fields;

  auto t0 = Clock::now();
  auto decoded_rules = decode_rules(data);
  thrust::device_vector<int32_t> d_rules_first(decoded_rules.first.begin(),
                                               decoded_rules.first.end());
  thrust::device_vector<int32_t> d_rules_second(decoded_rules.second.begin(),
                                                decoded_rules.second.end());
  segment_sequences = Codec::zstd_decompress_string(data.segment_sequences_zstd);
  segment_lengths =
      Codec::zstd_decompress_uint32_vector(data.segment_seq_lengths_zstd);
  segment_optional_fields.reserve(data.segment_optional_fields_zstd.size());
  for (const auto &c : data.segment_optional_fields_zstd)
    segment_optional_fields.push_back(decompress_optional_column(c));
  link_from_ids =
      Codec::decompress_delta_varint_uint32(data.link_from_ids_zstd,
                                            data.num_links);
  link_to_ids =
      Codec::decompress_delta_varint_uint32(data.link_to_ids_zstd,
                                            data.num_links);
  link_from_orients =
      Codec::decompress_orientations(data.link_from_orients_zstd, data.num_links);
  link_to_orients =
      Codec::decompress_orientations(data.link_to_orients_zstd, data.num_links);
  link_overlap_nums =
      Codec::zstd_decompress_uint32_vector(data.link_overlap_nums_zstd);
  link_overlap_ops =
      Codec::zstd_decompress_char_vector(data.link_overlap_ops_zstd);
  link_optional_fields.reserve(data.link_optional_fields_zstd.size());
  for (const auto &c : data.link_optional_fields_zstd)
    link_optional_fields.push_back(decompress_optional_column(c));
  if (data.num_jumps > 0) {
    jump_from_ids = Codec::decompress_delta_varint_uint32(data.jump_from_ids_zstd,
                                                          data.num_jumps);
    jump_to_ids = Codec::decompress_delta_varint_uint32(data.jump_to_ids_zstd,
                                                        data.num_jumps);
    jump_from_orients = Codec::decompress_orientations(
        data.jump_from_orients_zstd, data.num_jumps);
    jump_to_orients =
        Codec::decompress_orientations(data.jump_to_orients_zstd, data.num_jumps);
    jump_distances = decompress_string_column(data.jump_distances_zstd,
                                              data.jump_distance_lengths_zstd);
    jump_rest_fields = decompress_string_column(data.jump_rest_fields_zstd,
                                                data.jump_rest_lengths_zstd);
  }
  if (data.num_containments > 0) {
    containment_container_ids = Codec::decompress_delta_varint_uint32(
        data.containment_container_ids_zstd, data.num_containments);
    containment_contained_ids = Codec::decompress_delta_varint_uint32(
        data.containment_contained_ids_zstd, data.num_containments);
    containment_container_orients = Codec::decompress_orientations(
        data.containment_container_orients_zstd, data.num_containments);
    containment_contained_orients = Codec::decompress_orientations(
        data.containment_contained_orients_zstd, data.num_containments);
    containment_positions =
        Codec::zstd_decompress_uint32_vector(data.containment_positions_zstd);
    containment_overlaps = decompress_string_column(
        data.containment_overlaps_zstd, data.containment_overlap_lengths_zstd);
    containment_rest_fields = decompress_string_column(
        data.containment_rest_fields_zstd, data.containment_rest_lengths_zstd);
  }
  auto t1 = Clock::now();
  debug_stages.push_back({"decode non-traversal fields+rules",
                          elapsed_ms(t0, t1)});

  const FieldOffsets segment_offsets = build_field_offsets(segment_optional_fields);
  const FieldOffsets link_offsets = build_field_offsets(link_optional_fields);

  t0 = Clock::now();
  if (!data.header_line.empty())
    out << data.header_line << '\n';
  write_segments_numeric(out, segment_sequences, segment_lengths,
                         segment_optional_fields, segment_offsets);
  write_links_numeric(out, link_from_ids, link_to_ids, link_from_orients,
                      link_to_orients, link_overlap_nums, link_overlap_ops,
                      link_optional_fields, link_offsets);
  write_jumps_numeric(out, jump_from_ids, jump_to_ids, jump_from_orients,
                      jump_to_orients, jump_distances, jump_rest_fields);
  write_containments_numeric(out, containment_container_ids,
                             containment_contained_ids,
                             containment_container_orients,
                             containment_contained_orients,
                             containment_positions, containment_overlaps,
                             containment_rest_fields);
  t1 = Clock::now();
  debug_stages.push_back({"write static graph fields", elapsed_ms(t0, t1)});

  stream_paths_gpu_rolling_to_writer(out, data, d_rules_first, d_rules_second,
                                     options, debug_stages);
  stream_walks_gpu_rolling_to_writer(out, data, d_rules_first, d_rules_second,
                                     options, debug_stages);

  out.close();

  if (gpu_decompression::decompression_debug_enabled()) {
    print_cpu_decompression_summary("GPU Direct Writer", data);
    print_cpu_decompression_timing(
        {"GPU Direct Writer", "rolling direct-writer path", debug_stages,
         elapsed_ms(writer_total_start, Clock::now())});
  }
}

#endif
