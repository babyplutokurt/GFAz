#include "gpu/codec_gpu.cuh"
#include "gpu/compression_workflow_gpu_internal.hpp"
#include "gpu/metadata_codec_gpu.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <thrust/device_ptr.h>
#include <vector>

namespace gpu_compression {

namespace {

std::string flattened_to_string(const FlattenedStrings &flat) {
  return std::string(flat.data.begin(), flat.data.end());
}

void print_compression_stats(const char *label, size_t original_size,
                             size_t compressed_size) {
  if (!compression_debug_enabled() || original_size == 0) {
    return;
  }

  double ratio = 100.0 * (1.0 - static_cast<double>(compressed_size) /
                                    static_cast<double>(original_size));
  std::cout << "  [Zstd] " << label << ": " << original_size << " -> "
            << compressed_size << " bytes (" << std::fixed
            << std::setprecision(1) << ratio << "% reduction)" << std::endl;
}

ZstdCompressedBlock compress_bytes_gpu(const std::vector<uint8_t> &input,
                                       const char *label = "bytes") {
  auto block =
      Codec::zstd_compress_string(std::string(input.begin(), input.end()));
  print_compression_stats(label, input.size(), block.payload.size());
  return block;
}

ZstdCompressedBlock compress_string_gpu(const std::string &input,
                                        const char *label = "string") {
  auto block = Codec::zstd_compress_string(input);
  print_compression_stats(label, input.size(), block.payload.size());
  return block;
}

ZstdCompressedBlock compress_float_gpu(const std::vector<float> &input,
                                       const char *label = "float_vec") {
  auto block = Codec::zstd_compress_float_vector(input);
  print_compression_stats(label, input.size() * sizeof(float),
                          block.payload.size());
  return block;
}

ZstdCompressedBlock compress_char_gpu(const std::vector<char> &input,
                                      const char *label = "char_vec") {
  auto block = Codec::zstd_compress_char_vector(input);
  print_compression_stats(label, input.size() * sizeof(char),
                          block.payload.size());
  return block;
}

ZstdCompressedBlock
compress_varint_int64_gpu(const std::vector<int64_t> &input,
                          const char *label = "varint_int64") {
  auto block = Codec::compress_varint_int64(input);
  print_compression_stats(label, input.size() * sizeof(int64_t),
                          block.payload.size());
  return block;
}

ZstdCompressedBlock
compress_delta_varint_uint32_gpu(const std::vector<uint32_t> &input,
                                 const char *label = "delta_varint_uint32") {
  auto block = Codec::compress_delta_varint_uint32(input);
  print_compression_stats(label, input.size() * sizeof(uint32_t),
                          block.payload.size());
  return block;
}

ZstdCompressedBlock
compress_orientations_gpu(const std::vector<char> &orients,
                          const char *label = "orientations") {
  auto block = Codec::compress_orientations(orients);
  print_compression_stats(label, orients.size() * sizeof(char),
                          block.payload.size());
  return block;
}

void compress_flattened_strings_gpu(const FlattenedStrings &flat,
                                    ZstdCompressedBlock &data_block,
                                    ZstdCompressedBlock &lengths_block,
                                    const char *data_label,
                                    const char *lengths_label) {
  data_block = compress_string_gpu(flattened_to_string(flat), data_label);
  lengths_block = compress_uint32_gpu(flat.lengths, lengths_label);
}

CompressedOptionalFieldColumn
compress_optional_column_gpu(const OptionalFieldColumn_gpu &col,
                             const char *prefix) {
  CompressedOptionalFieldColumn compressed_col;
  compressed_col.tag = col.tag;
  compressed_col.type = col.type;

  char label[64];

  switch (col.type) {
  case 'i':
    compressed_col.num_elements = col.int_values.size();
    std::snprintf(label, sizeof(label), "%s_optional_i", prefix);
    compressed_col.int_values_zstd =
        compress_varint_int64_gpu(col.int_values, label);
    break;
  case 'f':
    compressed_col.num_elements = col.float_values.size();
    std::snprintf(label, sizeof(label), "%s_optional_f", prefix);
    compressed_col.float_values_zstd =
        compress_float_gpu(col.float_values, label);
    break;
  case 'A':
    compressed_col.num_elements = col.char_values.size();
    std::snprintf(label, sizeof(label), "%s_optional_A", prefix);
    compressed_col.char_values_zstd = compress_char_gpu(col.char_values, label);
    break;
  case 'Z':
  case 'J':
  case 'H':
    compressed_col.num_elements = col.strings.lengths.size();
    std::snprintf(label, sizeof(label), "%s_optional_strings", prefix);
    compressed_col.strings_zstd = compress_string_gpu(
        std::string(col.strings.data.begin(), col.strings.data.end()), label);
    std::snprintf(label, sizeof(label), "%s_optional_string_lengths", prefix);
    compressed_col.string_lengths_zstd =
        compress_uint32_gpu(col.strings.lengths, label);
    break;
  case 'B':
    compressed_col.num_elements = col.b_subtypes.size();
    std::snprintf(label, sizeof(label), "%s_optional_b_subtypes", prefix);
    compressed_col.b_subtypes_zstd = compress_char_gpu(col.b_subtypes, label);
    std::snprintf(label, sizeof(label), "%s_optional_b_lengths", prefix);
    compressed_col.b_lengths_zstd = compress_uint32_gpu(col.b_lengths, label);
    std::snprintf(label, sizeof(label), "%s_optional_b_bytes", prefix);
    compressed_col.b_concat_bytes_zstd = compress_bytes_gpu(col.b_data, label);
    break;
  }
  return compressed_col;
}

void compress_optional_columns_gpu(
    const std::vector<OptionalFieldColumn_gpu> &columns,
    std::vector<CompressedOptionalFieldColumn> &out_columns,
    const char *prefix) {
  out_columns.reserve(out_columns.size() + columns.size());
  for (const auto &col : columns) {
    out_columns.push_back(compress_optional_column_gpu(col, prefix));
  }
}

} // namespace

ZstdCompressedBlock compress_uint32_gpu(const std::vector<uint32_t> &input,
                                        const char *label) {
  size_t original_bytes = input.size() * sizeof(uint32_t);
  auto block = Codec::zstd_compress_uint32_vector(input);
  print_compression_stats(label, original_bytes, block.payload.size());
  return block;
}

ZstdCompressedBlock compress_int32_gpu(const std::vector<int32_t> &input,
                                       const char *label) {
  size_t original_bytes = input.size() * sizeof(int32_t);
  auto block = Codec::zstd_compress_int32_vector(input);
  print_compression_stats(label, original_bytes, block.payload.size());
  return block;
}

ZstdCompressedBlock
compress_int32_device_gpu(const thrust::device_vector<int32_t> &d_input,
                          const char *label) {
  std::vector<int32_t> host(d_input.size());
  thrust::copy(d_input.begin(), d_input.end(), host.begin());
  auto block = Codec::zstd_compress_int32_vector(host);
  size_t original_bytes = d_input.size() * sizeof(int32_t);
  print_compression_stats(label, original_bytes, block.payload.size());
  return block;
}

void compress_graph_metadata_gpu(const GfaGraph_gpu &gpu_graph,
                                 CompressedData &data) {
  if (compression_debug_enabled()) {
    std::cout
        << "[GPU Compression] Using shared CPU Zstd for metadata compression"
        << std::endl;
  }

  compress_flattened_strings_gpu(gpu_graph.path_names, data.names_zstd,
                                 data.name_lengths_zstd, "path_names",
                                 "name_lengths");
  compress_flattened_strings_gpu(gpu_graph.path_overlaps, data.overlaps_zstd,
                                 data.overlap_lengths_zstd, "path_overlaps",
                                 "overlap_lengths");

  if (gpu_graph.num_walks > 0) {
    if (compression_debug_enabled()) {
      std::cout << "[GPU Compression] Compressing walk metadata ("
                << gpu_graph.num_walks << " walks)" << std::endl;
    }

    compress_flattened_strings_gpu(gpu_graph.walk_sample_ids,
                                   data.walk_sample_ids_zstd,
                                   data.walk_sample_id_lengths_zstd,
                                   "walk_sample_ids", "walk_sample_id_lengths");

    data.walk_hap_indices_zstd =
        compress_uint32_gpu(gpu_graph.walk_hap_indices, "walk_hap_indices");

    compress_flattened_strings_gpu(
        gpu_graph.walk_seq_ids, data.walk_seq_ids_zstd,
        data.walk_seq_id_lengths_zstd, "walk_seq_ids", "walk_seq_id_lengths");

    data.walk_seq_starts_zstd =
        compress_varint_int64_gpu(gpu_graph.walk_seq_starts, "walk_seq_starts");
    data.walk_seq_ends_zstd =
        compress_varint_int64_gpu(gpu_graph.walk_seq_ends, "walk_seq_ends");
  }

  compress_flattened_strings_gpu(gpu_graph.node_sequences,
                                 data.segment_sequences_zstd,
                                 data.segment_seq_lengths_zstd,
                                 "segment_sequences", "segment_seq_lengths");

  data.header_line = gpu_graph.header_line;

  compress_optional_columns_gpu(gpu_graph.segment_optional_fields,
                                data.segment_optional_fields_zstd, "segment");

  data.num_links = gpu_graph.link_from_ids.size();
  data.link_from_ids_zstd = compress_delta_varint_uint32_gpu(
      gpu_graph.link_from_ids, "link_from_ids");
  data.link_to_ids_zstd =
      compress_delta_varint_uint32_gpu(gpu_graph.link_to_ids, "link_to_ids");
  data.link_from_orients_zstd = compress_orientations_gpu(
      gpu_graph.link_from_orients, "link_from_orients");
  data.link_to_orients_zstd =
      compress_orientations_gpu(gpu_graph.link_to_orients, "link_to_orients");
  data.link_overlap_nums_zstd =
      compress_uint32_gpu(gpu_graph.link_overlap_nums, "link_overlap_nums");
  data.link_overlap_ops_zstd =
      compress_char_gpu(gpu_graph.link_overlap_ops, "link_overlap_ops");

  compress_optional_columns_gpu(gpu_graph.link_optional_fields,
                                data.link_optional_fields_zstd, "link");

  if (!gpu_graph.jump_from_ids.empty()) {
    data.num_jumps = gpu_graph.jump_from_ids.size();

    if (compression_debug_enabled()) {
      std::cout << "[GPU Compression] Compressing J-lines (" << data.num_jumps
                << " jumps)" << std::endl;
    }

    data.jump_from_ids_zstd = compress_delta_varint_uint32_gpu(
        gpu_graph.jump_from_ids, "jump_from_ids");
    data.jump_to_ids_zstd =
        compress_delta_varint_uint32_gpu(gpu_graph.jump_to_ids, "jump_to_ids");
    data.jump_from_orients_zstd = compress_orientations_gpu(
        gpu_graph.jump_from_orients, "jump_from_orients");
    data.jump_to_orients_zstd =
        compress_orientations_gpu(gpu_graph.jump_to_orients, "jump_to_orients");

    compress_flattened_strings_gpu(gpu_graph.jump_distances,
                                   data.jump_distances_zstd,
                                   data.jump_distance_lengths_zstd,
                                   "jump_distances", "jump_distance_lengths");
    compress_flattened_strings_gpu(
        gpu_graph.jump_rest_fields, data.jump_rest_fields_zstd,
        data.jump_rest_lengths_zstd, "jump_rest_fields", "jump_rest_lengths");
  }

  if (!gpu_graph.containment_container_ids.empty()) {
    data.num_containments = gpu_graph.containment_container_ids.size();

    if (compression_debug_enabled()) {
      std::cout << "[GPU Compression] Compressing C-lines ("
                << data.num_containments << " containments)" << std::endl;
    }

    data.containment_container_ids_zstd = compress_delta_varint_uint32_gpu(
        gpu_graph.containment_container_ids, "containment_container_ids");
    data.containment_contained_ids_zstd = compress_delta_varint_uint32_gpu(
        gpu_graph.containment_contained_ids, "containment_contained_ids");
    data.containment_container_orients_zstd =
        compress_orientations_gpu(gpu_graph.containment_container_orients,
                                  "containment_container_orients");
    data.containment_contained_orients_zstd =
        compress_orientations_gpu(gpu_graph.containment_contained_orients,
                                  "containment_contained_orients");
    data.containment_positions_zstd = compress_uint32_gpu(
        gpu_graph.containment_positions, "containment_positions");

    compress_flattened_strings_gpu(
        gpu_graph.containment_overlaps, data.containment_overlaps_zstd,
        data.containment_overlap_lengths_zstd, "containment_overlaps",
        "containment_overlap_lengths");
    compress_flattened_strings_gpu(
        gpu_graph.containment_rest_fields, data.containment_rest_fields_zstd,
        data.containment_rest_lengths_zstd, "containment_rest_fields",
        "containment_rest_lengths");
  }
}

} // namespace gpu_compression

namespace gpu_decompression {

namespace {

OptionalFieldColumn_gpu decompress_optional_field_column(
    const CompressedOptionalFieldColumn &compressed) {
  OptionalFieldColumn_gpu result;
  result.tag = compressed.tag;
  result.type = compressed.type;
  result.num_elements = compressed.num_elements;

  switch (compressed.type) {
  case 'i':
    result.int_values = Codec::decompress_varint_int64(
        compressed.int_values_zstd, compressed.num_elements);
    break;
  case 'f':
    result.float_values =
        Codec::zstd_decompress_float_vector(compressed.float_values_zstd);
    break;
  case 'A':
    result.char_values =
        Codec::zstd_decompress_char_vector(compressed.char_values_zstd);
    break;
  case 'Z':
  case 'J':
  case 'H': {
    std::string str_data =
        Codec::zstd_decompress_string(compressed.strings_zstd);
    result.strings.data = std::vector<char>(str_data.begin(), str_data.end());
    result.strings.lengths =
        Codec::zstd_decompress_uint32_vector(compressed.string_lengths_zstd);
    break;
  }
  case 'B': {
    result.b_subtypes =
        Codec::zstd_decompress_char_vector(compressed.b_subtypes_zstd);
    result.b_lengths =
        Codec::zstd_decompress_uint32_vector(compressed.b_lengths_zstd);
    std::string bytes =
        Codec::zstd_decompress_string(compressed.b_concat_bytes_zstd);
    result.b_data = std::vector<uint8_t>(bytes.begin(), bytes.end());
    break;
  }
  default:
    break;
  }

  return result;
}

FlattenedStrings
decompress_flattened_strings(const ZstdCompressedBlock &data_block,
                             const ZstdCompressedBlock &lengths_block) {
  FlattenedStrings result;
  std::string str_data = Codec::zstd_decompress_string(data_block);
  result.data = std::vector<char>(str_data.begin(), str_data.end());
  result.lengths = Codec::zstd_decompress_uint32_vector(lengths_block);
  return result;
}

void decompress_optional_columns(
    const std::vector<CompressedOptionalFieldColumn> &compressed_columns,
    std::vector<OptionalFieldColumn_gpu> &out_columns) {
  out_columns.reserve(out_columns.size() + compressed_columns.size());
  for (const auto &compressed_col : compressed_columns) {
    out_columns.push_back(decompress_optional_field_column(compressed_col));
  }
}

FlattenedStrings build_numeric_node_names(size_t num_segments) {
  FlattenedStrings result;
  result.lengths.reserve(num_segments);
  for (size_t i = 1; i <= num_segments; ++i) {
    const std::string name = std::to_string(i);
    result.lengths.push_back(static_cast<uint32_t>(name.size()));
    result.data.insert(result.data.end(), name.begin(), name.end());
  }
  return result;
}

} // namespace

void decompress_graph_metadata_gpu(const CompressedData &data,
                                   GfaGraph_gpu &result) {
  result.num_paths = static_cast<uint32_t>(data.sequence_lengths.size());
  result.num_walks = static_cast<uint32_t>(data.walk_lengths.size());

  result.path_names =
      decompress_flattened_strings(data.names_zstd, data.name_lengths_zstd);
  result.path_overlaps = decompress_flattened_strings(
      data.overlaps_zstd, data.overlap_lengths_zstd);

  if (result.num_walks > 0) {
    result.walk_sample_ids = decompress_flattened_strings(
        data.walk_sample_ids_zstd, data.walk_sample_id_lengths_zstd);
    result.walk_hap_indices =
        Codec::zstd_decompress_uint32_vector(data.walk_hap_indices_zstd);
    result.walk_seq_ids = decompress_flattened_strings(
        data.walk_seq_ids_zstd, data.walk_seq_id_lengths_zstd);
    result.walk_seq_starts = Codec::decompress_varint_int64(
        data.walk_seq_starts_zstd, result.num_walks);
    result.walk_seq_ends = Codec::decompress_varint_int64(
        data.walk_seq_ends_zstd, result.num_walks);
  }

  result.node_sequences = decompress_flattened_strings(
      data.segment_sequences_zstd, data.segment_seq_lengths_zstd);
  result.header_line = data.header_line;
  result.node_names =
      build_numeric_node_names(result.node_sequences.lengths.size());

  if (data.num_links > 0) {
    result.link_from_ids = Codec::decompress_delta_varint_uint32(
        data.link_from_ids_zstd, data.num_links);
    result.link_to_ids = Codec::decompress_delta_varint_uint32(
        data.link_to_ids_zstd, data.num_links);
    result.link_from_orients = Codec::decompress_orientations(
        data.link_from_orients_zstd, data.num_links);
    result.link_to_orients = Codec::decompress_orientations(
        data.link_to_orients_zstd, data.num_links);
    result.link_overlap_nums =
        Codec::zstd_decompress_uint32_vector(data.link_overlap_nums_zstd);
    result.link_overlap_ops =
        Codec::zstd_decompress_char_vector(data.link_overlap_ops_zstd);
  }

  if (!data.segment_optional_fields_zstd.empty()) {
    decompress_optional_columns(data.segment_optional_fields_zstd,
                                result.segment_optional_fields);
  }

  if (!data.link_optional_fields_zstd.empty()) {
    decompress_optional_columns(data.link_optional_fields_zstd,
                                result.link_optional_fields);
  }

  if (data.num_jumps > 0) {
    result.jump_from_ids = Codec::decompress_delta_varint_uint32(
        data.jump_from_ids_zstd, data.num_jumps);
    result.jump_to_ids = Codec::decompress_delta_varint_uint32(
        data.jump_to_ids_zstd, data.num_jumps);
    result.jump_from_orients = Codec::decompress_orientations(
        data.jump_from_orients_zstd, data.num_jumps);
    result.jump_to_orients = Codec::decompress_orientations(
        data.jump_to_orients_zstd, data.num_jumps);
    result.jump_distances = decompress_flattened_strings(
        data.jump_distances_zstd, data.jump_distance_lengths_zstd);
    result.jump_rest_fields = decompress_flattened_strings(
        data.jump_rest_fields_zstd, data.jump_rest_lengths_zstd);
  }

  if (data.num_containments > 0) {
    result.containment_container_ids = Codec::decompress_delta_varint_uint32(
        data.containment_container_ids_zstd, data.num_containments);
    result.containment_contained_ids = Codec::decompress_delta_varint_uint32(
        data.containment_contained_ids_zstd, data.num_containments);
    result.containment_container_orients = Codec::decompress_orientations(
        data.containment_container_orients_zstd, data.num_containments);
    result.containment_contained_orients = Codec::decompress_orientations(
        data.containment_contained_orients_zstd, data.num_containments);
    result.containment_positions =
        Codec::zstd_decompress_uint32_vector(data.containment_positions_zstd);
    result.containment_overlaps = decompress_flattened_strings(
        data.containment_overlaps_zstd, data.containment_overlap_lengths_zstd);
    result.containment_rest_fields = decompress_flattened_strings(
        data.containment_rest_fields_zstd, data.containment_rest_lengths_zstd);
  }

  result.num_segments =
      static_cast<uint32_t>(result.node_sequences.lengths.size());
  result.num_links = static_cast<uint32_t>(data.num_links);
}

} // namespace gpu_decompression
