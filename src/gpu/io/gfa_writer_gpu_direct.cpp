#include "gpu/io/gfa_writer_gpu_direct.hpp"

#include "codec/codec.hpp"
#include "utils/runtime_utils.hpp"

#include <chrono>

namespace gpu_decompression {

namespace {

using Clock = std::chrono::high_resolution_clock;
using gfz::runtime_utils::elapsed_ms;
using namespace gfz::gfa_write_utils;

} // namespace

GpuDirectWriterStaticFields
decode_gpu_direct_writer_static_fields(const CompressedData &data) {
  const auto start = Clock::now();

  GpuDirectWriterStaticFields fields;
  fields.segment_sequences =
      Codec::zstd_decompress_string(data.segment_sequences_zstd);
  fields.segment_lengths =
      Codec::zstd_decompress_uint32_vector(data.segment_seq_lengths_zstd);
  fields.segment_optional_fields.reserve(data.segment_optional_fields_zstd.size());
  for (const auto &c : data.segment_optional_fields_zstd)
    fields.segment_optional_fields.push_back(decompress_optional_column(c));

  fields.link_from_ids =
      Codec::decompress_delta_varint_uint32(data.link_from_ids_zstd,
                                            data.num_links);
  fields.link_to_ids =
      Codec::decompress_delta_varint_uint32(data.link_to_ids_zstd,
                                            data.num_links);
  fields.link_from_orients =
      Codec::decompress_orientations(data.link_from_orients_zstd, data.num_links);
  fields.link_to_orients =
      Codec::decompress_orientations(data.link_to_orients_zstd, data.num_links);
  fields.link_overlap_nums =
      Codec::zstd_decompress_uint32_vector(data.link_overlap_nums_zstd);
  fields.link_overlap_ops =
      Codec::zstd_decompress_char_vector(data.link_overlap_ops_zstd);
  fields.link_optional_fields.reserve(data.link_optional_fields_zstd.size());
  for (const auto &c : data.link_optional_fields_zstd)
    fields.link_optional_fields.push_back(decompress_optional_column(c));

  if (data.num_jumps > 0) {
    fields.jump_from_ids = Codec::decompress_delta_varint_uint32(
        data.jump_from_ids_zstd, data.num_jumps);
    fields.jump_to_ids = Codec::decompress_delta_varint_uint32(
        data.jump_to_ids_zstd, data.num_jumps);
    fields.jump_from_orients = Codec::decompress_orientations(
        data.jump_from_orients_zstd, data.num_jumps);
    fields.jump_to_orients = Codec::decompress_orientations(
        data.jump_to_orients_zstd, data.num_jumps);
    fields.jump_distances = decompress_string_column(
        data.jump_distances_zstd, data.jump_distance_lengths_zstd);
    fields.jump_rest_fields = decompress_string_column(
        data.jump_rest_fields_zstd, data.jump_rest_lengths_zstd);
  }

  if (data.num_containments > 0) {
    fields.containment_container_ids = Codec::decompress_delta_varint_uint32(
        data.containment_container_ids_zstd, data.num_containments);
    fields.containment_contained_ids = Codec::decompress_delta_varint_uint32(
        data.containment_contained_ids_zstd, data.num_containments);
    fields.containment_container_orients = Codec::decompress_orientations(
        data.containment_container_orients_zstd, data.num_containments);
    fields.containment_contained_orients = Codec::decompress_orientations(
        data.containment_contained_orients_zstd, data.num_containments);
    fields.containment_positions =
        Codec::zstd_decompress_uint32_vector(data.containment_positions_zstd);
    fields.containment_overlaps = decompress_string_column(
        data.containment_overlaps_zstd, data.containment_overlap_lengths_zstd);
    fields.containment_rest_fields = decompress_string_column(
        data.containment_rest_fields_zstd, data.containment_rest_lengths_zstd);
  }

  fields.segment_offsets = build_field_offsets(fields.segment_optional_fields);
  fields.link_offsets = build_field_offsets(fields.link_optional_fields);
  fields.decode_ms = elapsed_ms(start, Clock::now());
  return fields;
}

GpuPathWriterMetadata decode_gpu_path_writer_metadata(const CompressedData &data) {
  const auto start = Clock::now();

  GpuPathWriterMetadata metadata;
  metadata.names =
      decompress_string_column(data.names_zstd, data.name_lengths_zstd);
  metadata.overlaps =
      decompress_string_column(data.overlaps_zstd, data.overlap_lengths_zstd);
  metadata.decode_ms = elapsed_ms(start, Clock::now());
  return metadata;
}

GpuWalkWriterMetadata decode_gpu_walk_writer_metadata(const CompressedData &data) {
  const auto start = Clock::now();

  GpuWalkWriterMetadata metadata;
  metadata.sample_ids = decompress_string_column(data.walk_sample_ids_zstd,
                                                 data.walk_sample_id_lengths_zstd);
  metadata.hap_indices =
      Codec::zstd_decompress_uint32_vector(data.walk_hap_indices_zstd);
  metadata.seq_ids = decompress_string_column(data.walk_seq_ids_zstd,
                                              data.walk_seq_id_lengths_zstd);
  metadata.seq_starts = Codec::decompress_varint_int64(
      data.walk_seq_starts_zstd, data.walk_lengths.size());
  metadata.seq_ends = Codec::decompress_varint_int64(data.walk_seq_ends_zstd,
                                                     data.walk_lengths.size());
  metadata.decode_ms = elapsed_ms(start, Clock::now());
  return metadata;
}

void write_gpu_direct_writer_static_fields(
    std::ofstream &out, const CompressedData &data,
    const GpuDirectWriterStaticFields &fields) {
  if (!data.header_line.empty())
    out << data.header_line << '\n';

  write_segments_numeric(out, fields.segment_sequences, fields.segment_lengths,
                         fields.segment_optional_fields, fields.segment_offsets);
  write_links_numeric(out, fields.link_from_ids, fields.link_to_ids,
                      fields.link_from_orients, fields.link_to_orients,
                      fields.link_overlap_nums, fields.link_overlap_ops,
                      fields.link_optional_fields, fields.link_offsets);
  write_jumps_numeric(out, fields.jump_from_ids, fields.jump_to_ids,
                      fields.jump_from_orients, fields.jump_to_orients,
                      fields.jump_distances, fields.jump_rest_fields);
  write_containments_numeric(out, fields.containment_container_ids,
                             fields.containment_contained_ids,
                             fields.containment_container_orients,
                             fields.containment_contained_orients,
                             fields.containment_positions,
                             fields.containment_overlaps,
                             fields.containment_rest_fields);
}

void write_gpu_path_chunk_lines(std::ofstream &out,
                                const RollingPathPinnedHostBuffer &buffer,
                                const GpuPathWriterMetadata &metadata) {
  size_t offset = 0;
  for (size_t local_index = 0; local_index < buffer.lengths.size();
       ++local_index) {
    const size_t global_index = buffer.segment_begin + local_index;
    const size_t length = buffer.lengths[local_index];
    const std::string &name =
        (global_index < metadata.names.size()) ? metadata.names[global_index]
                                               : std::to_string(global_index);
    const std::string &overlap =
        (global_index < metadata.overlaps.size()) ? metadata.overlaps[global_index]
                                                  : "";
    const std::string line =
        format_path_line_numeric(name, buffer.host_nodes + offset, length, overlap);
    out.write(line.data(), static_cast<std::streamsize>(line.size()));
    offset += length;
  }
}

void write_gpu_walk_chunk_lines(std::ofstream &out,
                                const RollingPathPinnedHostBuffer &buffer,
                                const GpuWalkWriterMetadata &metadata) {
  size_t offset = 0;
  for (size_t local_index = 0; local_index < buffer.lengths.size();
       ++local_index) {
    const size_t global_index = buffer.segment_begin + local_index;
    const size_t length = buffer.lengths[local_index];
    const std::string &sample_id =
        (global_index < metadata.sample_ids.size())
            ? metadata.sample_ids[global_index]
            : std::string("sample");
    const uint32_t hap_index =
        (global_index < metadata.hap_indices.size())
            ? metadata.hap_indices[global_index]
            : 0;
    const std::string &seq_id =
        (global_index < metadata.seq_ids.size()) ? metadata.seq_ids[global_index]
                                                 : std::string("unknown");
    const int64_t seq_start =
        (global_index < metadata.seq_starts.size())
            ? metadata.seq_starts[global_index]
            : -1;
    const int64_t seq_end = (global_index < metadata.seq_ends.size())
                                ? metadata.seq_ends[global_index]
                                : -1;
    const std::string line = format_walk_line_numeric(
        sample_id, hap_index, seq_id, seq_start, seq_end,
        buffer.host_nodes + offset, length);
    out.write(line.data(), static_cast<std::streamsize>(line.size()));
    offset += length;
  }
}

} // namespace gpu_decompression
