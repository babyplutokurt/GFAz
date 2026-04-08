#include "gpu/compression_workflow_gpu_internal.hpp"
#include "gpu/codec_gpu.cuh"
#include "gpu/metadata_codec_gpu.hpp"

#include <cstdio>
#include <cstring>
#include <cstdint>
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
  std::cout << "  [nvComp] " << label << ": " << original_size << " -> "
            << compressed_size << " bytes (" << std::fixed
            << std::setprecision(1) << ratio << "% reduction)" << std::endl;
}

gpu_codec::NvcompCompressedBlock
compress_bytes_gpu(const std::vector<uint8_t> &input,
                   const char *label = "bytes") {
  auto block = gpu_codec::nvcomp_zstd_compress(input);
  print_compression_stats(label, input.size(), block.payload.size());
  return block;
}

gpu_codec::NvcompCompressedBlock
compress_string_gpu(const std::string &input, const char *label = "string") {
  auto block = gpu_codec::nvcomp_zstd_compress_string(input);
  print_compression_stats(label, input.size(), block.payload.size());
  return block;
}

gpu_codec::NvcompCompressedBlock
compress_float_gpu(const std::vector<float> &input,
                   const char *label = "float_vec") {
  const uint8_t *bytes = reinterpret_cast<const uint8_t *>(input.data());
  std::vector<uint8_t> payload(bytes, bytes + input.size() * sizeof(float));
  return compress_bytes_gpu(payload, label);
}

gpu_codec::NvcompCompressedBlock
compress_char_gpu(const std::vector<char> &input,
                  const char *label = "char_vec") {
  const uint8_t *bytes = reinterpret_cast<const uint8_t *>(input.data());
  std::vector<uint8_t> payload(bytes, bytes + input.size() * sizeof(char));
  return compress_bytes_gpu(payload, label);
}

uint64_t zigzag_encode_64(int64_t value) {
  return (static_cast<uint64_t>(value) << 1) ^
         static_cast<uint64_t>(value >> 63);
}

void append_varint_64(uint64_t value, std::vector<uint8_t> &out) {
  while (value >= 0x80) {
    out.push_back(static_cast<uint8_t>((value & 0x7F) | 0x80));
    value >>= 7;
  }
  out.push_back(static_cast<uint8_t>(value));
}

gpu_codec::NvcompCompressedBlock
compress_varint_int64_gpu(const std::vector<int64_t> &input,
                          const char *label = "varint_int64") {
  std::vector<uint8_t> varint_bytes;
  varint_bytes.reserve(input.size() * 4);
  for (int64_t val : input) {
    append_varint_64(zigzag_encode_64(val), varint_bytes);
  }
  return compress_bytes_gpu(varint_bytes, label);
}

void append_varint_32(uint32_t value, std::vector<uint8_t> &out) {
  while (value >= 0x80) {
    out.push_back(static_cast<uint8_t>((value & 0x7F) | 0x80));
    value >>= 7;
  }
  out.push_back(static_cast<uint8_t>(value));
}

gpu_codec::NvcompCompressedBlock
compress_delta_varint_uint32_gpu(const std::vector<uint32_t> &input,
                                 const char *label = "delta_varint_uint32") {
  if (input.empty()) {
    return gpu_codec::NvcompCompressedBlock{};
  }

  std::vector<int32_t> deltas(input.size());
  deltas[0] = static_cast<int32_t>(input[0]);
  for (size_t i = 1; i < input.size(); ++i) {
    deltas[i] =
        static_cast<int32_t>(input[i]) - static_cast<int32_t>(input[i - 1]);
  }

  std::vector<uint8_t> varint_bytes;
  varint_bytes.reserve(input.size() * 2);
  for (int32_t val : deltas) {
    uint32_t zigzag =
        (static_cast<uint32_t>(val) << 1) ^ static_cast<uint32_t>(val >> 31);
    append_varint_32(zigzag, varint_bytes);
  }

  return compress_bytes_gpu(varint_bytes, label);
}

gpu_codec::NvcompCompressedBlock
compress_orientations_gpu(const std::vector<char> &orients,
                          const char *label = "orientations") {
  std::vector<uint8_t> packed = gpu_codec::pack_orientations_gpu(orients);
  return compress_bytes_gpu(packed, label);
}

void compress_flattened_strings_gpu(
    const FlattenedStrings &flat, gpu_codec::NvcompCompressedBlock &data_block,
    gpu_codec::NvcompCompressedBlock &lengths_block, const char *data_label,
    const char *lengths_label) {
  data_block = compress_string_gpu(flattened_to_string(flat), data_label);
  lengths_block = compress_uint32_gpu(flat.lengths, lengths_label);
}

CompressedOptionalFieldColumn_gpu
compress_optional_column_gpu(const OptionalFieldColumn_gpu &col,
                             const char *prefix) {
  CompressedOptionalFieldColumn_gpu compressed_col;
  compressed_col.tag = col.tag;
  compressed_col.type = col.type;

  char label[64];

  switch (col.type) {
  case 'i':
    compressed_col.num_elements = col.int_values.size();
    std::snprintf(label, sizeof(label), "%s_optional_i", prefix);
    compressed_col.int_values_zstd_nvcomp =
        compress_varint_int64_gpu(col.int_values, label);
    break;
  case 'f':
    compressed_col.num_elements = col.float_values.size();
    std::snprintf(label, sizeof(label), "%s_optional_f", prefix);
    compressed_col.float_values_zstd_nvcomp =
        compress_float_gpu(col.float_values, label);
    break;
  case 'A':
    compressed_col.num_elements = col.char_values.size();
    std::snprintf(label, sizeof(label), "%s_optional_A", prefix);
    compressed_col.char_values_zstd_nvcomp =
        compress_char_gpu(col.char_values, label);
    break;
  case 'Z':
  case 'J':
  case 'H':
    compressed_col.num_elements = col.strings.lengths.size();
    std::snprintf(label, sizeof(label), "%s_optional_strings", prefix);
    compressed_col.strings_zstd_nvcomp = compress_string_gpu(
        std::string(col.strings.data.begin(), col.strings.data.end()), label);
    std::snprintf(label, sizeof(label), "%s_optional_string_lengths", prefix);
    compressed_col.string_lengths_zstd_nvcomp =
        compress_uint32_gpu(col.strings.lengths, label);
    break;
  case 'B':
    compressed_col.num_elements = col.b_subtypes.size();
    std::snprintf(label, sizeof(label), "%s_optional_b_subtypes", prefix);
    compressed_col.b_subtypes_zstd_nvcomp =
        compress_char_gpu(col.b_subtypes, label);
    std::snprintf(label, sizeof(label), "%s_optional_b_lengths", prefix);
    compressed_col.b_lengths_zstd_nvcomp =
        compress_uint32_gpu(col.b_lengths, label);
    std::snprintf(label, sizeof(label), "%s_optional_b_bytes", prefix);
    compressed_col.b_concat_bytes_zstd_nvcomp =
        compress_bytes_gpu(col.b_data, label);
    break;
  }
  return compressed_col;
}

void compress_optional_columns_gpu(
    const std::vector<OptionalFieldColumn_gpu> &columns,
    std::vector<CompressedOptionalFieldColumn_gpu> &out_columns,
    const char *prefix) {
  out_columns.reserve(out_columns.size() + columns.size());
  for (const auto &col : columns) {
    out_columns.push_back(compress_optional_column_gpu(col, prefix));
  }
}

} // namespace

gpu_codec::NvcompCompressedBlock
compress_uint32_gpu(const std::vector<uint32_t> &input, const char *label) {
  size_t original_bytes = input.size() * sizeof(uint32_t);
  auto block = gpu_codec::nvcomp_zstd_compress_uint32(input);
  print_compression_stats(label, original_bytes, block.payload.size());
  return block;
}

gpu_codec::NvcompCompressedBlock
compress_int32_gpu(const std::vector<int32_t> &input, const char *label) {
  size_t original_bytes = input.size() * sizeof(int32_t);
  auto block = gpu_codec::nvcomp_zstd_compress_int32(input);
  print_compression_stats(label, original_bytes, block.payload.size());
  return block;
}

gpu_codec::NvcompCompressedBlock
compress_int32_device_gpu(const thrust::device_vector<int32_t> &d_input,
                          const char *label) {
  auto block = gpu_codec::nvcomp_zstd_compress_int32_device(
      thrust::raw_pointer_cast(d_input.data()), d_input.size());
  size_t original_bytes = d_input.size() * sizeof(int32_t);
  print_compression_stats(label, original_bytes, block.payload.size());
  return block;
}

void compress_graph_metadata_gpu(const GfaGraph_gpu &gpu_graph,
                                 CompressedData_gpu &data) {
  data.num_paths = gpu_graph.num_paths;
  data.num_walks = gpu_graph.num_walks;

  if (compression_debug_enabled()) {
    std::cout
        << "[GPU Compression] Using nvComp GPU ZSTD for metadata compression"
        << std::endl;
  }

  compress_flattened_strings_gpu(gpu_graph.path_names, data.names_zstd_nvcomp,
                                 data.name_lengths_zstd_nvcomp, "path_names",
                                 "name_lengths");
  compress_flattened_strings_gpu(
      gpu_graph.path_overlaps, data.overlaps_zstd_nvcomp,
      data.overlap_lengths_zstd_nvcomp, "path_overlaps", "overlap_lengths");

  if (gpu_graph.num_walks > 0) {
    if (compression_debug_enabled()) {
      std::cout << "[GPU Compression] Compressing walk metadata ("
                << gpu_graph.num_walks << " walks)" << std::endl;
    }

    compress_flattened_strings_gpu(gpu_graph.walk_sample_ids,
                                   data.walk_sample_ids_zstd_nvcomp,
                                   data.walk_sample_id_lengths_zstd_nvcomp,
                                   "walk_sample_ids", "walk_sample_id_lengths");

    data.walk_hap_indices_zstd_nvcomp =
        compress_uint32_gpu(gpu_graph.walk_hap_indices, "walk_hap_indices");

    compress_flattened_strings_gpu(gpu_graph.walk_seq_ids,
                                   data.walk_seq_ids_zstd_nvcomp,
                                   data.walk_seq_id_lengths_zstd_nvcomp,
                                   "walk_seq_ids", "walk_seq_id_lengths");

    data.walk_seq_starts_zstd_nvcomp =
        compress_varint_int64_gpu(gpu_graph.walk_seq_starts, "walk_seq_starts");
    data.walk_seq_ends_zstd_nvcomp =
        compress_varint_int64_gpu(gpu_graph.walk_seq_ends, "walk_seq_ends");
  }

  compress_flattened_strings_gpu(gpu_graph.node_sequences,
                                 data.segment_sequences_zstd_nvcomp,
                                 data.segment_seq_lengths_zstd_nvcomp,
                                 "segment_sequences", "segment_seq_lengths");

  data.header_line = gpu_graph.header_line;

  compress_flattened_strings_gpu(
      gpu_graph.node_names, data.node_names_zstd_nvcomp,
      data.node_name_lengths_zstd_nvcomp, "node_names", "node_name_lengths");

  compress_optional_columns_gpu(gpu_graph.segment_optional_fields,
                                data.segment_optional_fields_zstd_nvcomp,
                                "segment");

  data.num_links = gpu_graph.link_from_ids.size();
  data.link_from_ids_zstd_nvcomp = compress_delta_varint_uint32_gpu(
      gpu_graph.link_from_ids, "link_from_ids");
  data.link_to_ids_zstd_nvcomp =
      compress_delta_varint_uint32_gpu(gpu_graph.link_to_ids, "link_to_ids");
  data.link_from_orients_zstd_nvcomp = compress_orientations_gpu(
      gpu_graph.link_from_orients, "link_from_orients");
  data.link_to_orients_zstd_nvcomp =
      compress_orientations_gpu(gpu_graph.link_to_orients, "link_to_orients");
  data.link_overlap_nums_zstd_nvcomp =
      compress_uint32_gpu(gpu_graph.link_overlap_nums, "link_overlap_nums");
  data.link_overlap_ops_zstd_nvcomp =
      compress_char_gpu(gpu_graph.link_overlap_ops, "link_overlap_ops");

  compress_optional_columns_gpu(gpu_graph.link_optional_fields,
                                data.link_optional_fields_zstd_nvcomp, "link");

  if (!gpu_graph.jump_from_ids.empty()) {
    data.num_jumps_stored = gpu_graph.jump_from_ids.size();

    if (compression_debug_enabled()) {
      std::cout << "[GPU Compression] Compressing J-lines ("
                << data.num_jumps_stored << " jumps)" << std::endl;
    }

    data.jump_from_ids_zstd_nvcomp = compress_delta_varint_uint32_gpu(
        gpu_graph.jump_from_ids, "jump_from_ids");
    data.jump_to_ids_zstd_nvcomp =
        compress_delta_varint_uint32_gpu(gpu_graph.jump_to_ids, "jump_to_ids");
    data.jump_from_orients_zstd_nvcomp = compress_orientations_gpu(
        gpu_graph.jump_from_orients, "jump_from_orients");
    data.jump_to_orients_zstd_nvcomp =
        compress_orientations_gpu(gpu_graph.jump_to_orients, "jump_to_orients");

    compress_flattened_strings_gpu(gpu_graph.jump_distances,
                                   data.jump_distances_zstd_nvcomp,
                                   data.jump_distance_lengths_zstd_nvcomp,
                                   "jump_distances", "jump_distance_lengths");
    compress_flattened_strings_gpu(gpu_graph.jump_rest_fields,
                                   data.jump_rest_fields_zstd_nvcomp,
                                   data.jump_rest_lengths_zstd_nvcomp,
                                   "jump_rest_fields", "jump_rest_lengths");
  }

  if (!gpu_graph.containment_container_ids.empty()) {
    data.num_containments_stored = gpu_graph.containment_container_ids.size();

    if (compression_debug_enabled()) {
      std::cout << "[GPU Compression] Compressing C-lines ("
                << data.num_containments_stored << " containments)"
                << std::endl;
    }

    data.containment_container_ids_zstd_nvcomp =
        compress_delta_varint_uint32_gpu(gpu_graph.containment_container_ids,
                                         "containment_container_ids");
    data.containment_contained_ids_zstd_nvcomp =
        compress_delta_varint_uint32_gpu(gpu_graph.containment_contained_ids,
                                         "containment_contained_ids");
    data.containment_container_orients_zstd_nvcomp =
        compress_orientations_gpu(gpu_graph.containment_container_orients,
                                  "containment_container_orients");
    data.containment_contained_orients_zstd_nvcomp =
        compress_orientations_gpu(gpu_graph.containment_contained_orients,
                                  "containment_contained_orients");
    data.containment_positions_zstd_nvcomp =
        compress_uint32_gpu(gpu_graph.containment_positions,
                            "containment_positions");

    compress_flattened_strings_gpu(
        gpu_graph.containment_overlaps, data.containment_overlaps_zstd_nvcomp,
        data.containment_overlap_lengths_zstd_nvcomp, "containment_overlaps",
        "containment_overlap_lengths");
    compress_flattened_strings_gpu(gpu_graph.containment_rest_fields,
                                   data.containment_rest_fields_zstd_nvcomp,
                                   data.containment_rest_lengths_zstd_nvcomp,
                                   "containment_rest_fields",
                                   "containment_rest_lengths");
  }
}

} // namespace gpu_compression

namespace gpu_decompression {

namespace {

// Decode a single varint from a byte stream and return the bytes consumed.
size_t decode_varint_32(const uint8_t *data, size_t max_len, uint32_t &out) {
  out = 0;
  size_t shift = 0;
  size_t i = 0;
  while (i < max_len) {
    uint8_t byte = data[i++];
    out |= static_cast<uint32_t>(byte & 0x7F) << shift;
    if ((byte & 0x80) == 0) {
      break;
    }
    shift += 7;
  }
  return i;
}

int32_t zigzag_decode_32(uint32_t n) {
  return static_cast<int32_t>((n >> 1) ^ -(static_cast<int32_t>(n & 1)));
}

size_t decode_varint_64(const uint8_t *data, size_t max_len, uint64_t &out) {
  out = 0;
  size_t shift = 0;
  size_t i = 0;
  while (i < max_len) {
    uint8_t byte = data[i++];
    out |= static_cast<uint64_t>(byte & 0x7F) << shift;
    if ((byte & 0x80) == 0) {
      break;
    }
    shift += 7;
  }
  return i;
}

int64_t zigzag_decode_64(uint64_t n) {
  return static_cast<int64_t>((n >> 1) ^ -(static_cast<int64_t>(n & 1)));
}

std::vector<uint32_t>
decompress_delta_varint_uint32(const gpu_codec::NvcompCompressedBlock &block,
                               size_t expected_count) {
  if (block.payload.empty()) {
    return {};
  }

  std::vector<uint8_t> varint_bytes = gpu_codec::nvcomp_zstd_decompress(block);
  std::vector<uint32_t> result;
  result.reserve(expected_count);

  size_t offset = 0;
  int32_t prev = 0;
  while (offset < varint_bytes.size() && result.size() < expected_count) {
    uint32_t zigzag_val;
    offset += decode_varint_32(varint_bytes.data() + offset,
                               varint_bytes.size() - offset, zigzag_val);
    int32_t delta = zigzag_decode_32(zigzag_val);
    int32_t current = prev + delta;
    result.push_back(static_cast<uint32_t>(current));
    prev = current;
  }

  return result;
}

std::vector<char>
decompress_orientations(const gpu_codec::NvcompCompressedBlock &block,
                        size_t expected_count) {
  if (block.payload.empty()) {
    return {};
  }

  std::vector<uint8_t> packed = gpu_codec::nvcomp_zstd_decompress(block);
  return gpu_codec::unpack_orientations_gpu(packed, expected_count);
}

std::vector<int64_t>
decompress_varint_int64(const gpu_codec::NvcompCompressedBlock &block,
                        size_t expected_count) {
  if (block.payload.empty()) {
    return {};
  }

  std::vector<uint8_t> varint_bytes = gpu_codec::nvcomp_zstd_decompress(block);
  std::vector<int64_t> result;
  result.reserve(expected_count);

  size_t offset = 0;
  while (offset < varint_bytes.size() && result.size() < expected_count) {
    uint64_t zigzag_val;
    offset += decode_varint_64(varint_bytes.data() + offset,
                               varint_bytes.size() - offset, zigzag_val);
    result.push_back(zigzag_decode_64(zigzag_val));
  }

  return result;
}

std::vector<float> decompress_float(
    const gpu_codec::NvcompCompressedBlock &block) {
  if (block.payload.empty()) {
    return {};
  }

  std::vector<uint8_t> bytes = gpu_codec::nvcomp_zstd_decompress(block);
  size_t count = bytes.size() / sizeof(float);
  std::vector<float> result(count);
  std::memcpy(result.data(), bytes.data(), count * sizeof(float));
  return result;
}

std::vector<char> decompress_char(
    const gpu_codec::NvcompCompressedBlock &block) {
  if (block.payload.empty()) {
    return {};
  }

  std::vector<uint8_t> bytes = gpu_codec::nvcomp_zstd_decompress(block);
  return std::vector<char>(bytes.begin(), bytes.end());
}

OptionalFieldColumn_gpu decompress_optional_field_column(
    const gpu_compression::CompressedOptionalFieldColumn_gpu &compressed) {
  OptionalFieldColumn_gpu result;
  result.tag = compressed.tag;
  result.type = compressed.type;
  result.num_elements = compressed.num_elements;

  switch (compressed.type) {
  case 'i':
    result.int_values = decompress_varint_int64(
        compressed.int_values_zstd_nvcomp, compressed.num_elements);
    break;
  case 'f':
    result.float_values = decompress_float(compressed.float_values_zstd_nvcomp);
    break;
  case 'A':
    result.char_values = decompress_char(compressed.char_values_zstd_nvcomp);
    break;
  case 'Z':
  case 'J':
  case 'H': {
    std::string str_data =
        gpu_codec::nvcomp_zstd_decompress_string(compressed.strings_zstd_nvcomp);
    result.strings.data = std::vector<char>(str_data.begin(), str_data.end());
    result.strings.lengths = gpu_codec::nvcomp_zstd_decompress_uint32(
        compressed.string_lengths_zstd_nvcomp);
    break;
  }
  case 'B': {
    std::vector<uint8_t> subtypes_bytes =
        gpu_codec::nvcomp_zstd_decompress(compressed.b_subtypes_zstd_nvcomp);
    result.b_subtypes =
        std::vector<char>(subtypes_bytes.begin(), subtypes_bytes.end());
    result.b_lengths = gpu_codec::nvcomp_zstd_decompress_uint32(
        compressed.b_lengths_zstd_nvcomp);
    result.b_data = gpu_codec::nvcomp_zstd_decompress(
        compressed.b_concat_bytes_zstd_nvcomp);
    break;
  }
  default:
    break;
  }

  return result;
}

FlattenedStrings decompress_flattened_strings(
    const gpu_codec::NvcompCompressedBlock &data_block,
    const gpu_codec::NvcompCompressedBlock &lengths_block) {
  FlattenedStrings result;
  std::string str_data = gpu_codec::nvcomp_zstd_decompress_string(data_block);
  result.data = std::vector<char>(str_data.begin(), str_data.end());
  result.lengths = gpu_codec::nvcomp_zstd_decompress_uint32(lengths_block);
  return result;
}

void decompress_optional_columns(
    const std::vector<gpu_compression::CompressedOptionalFieldColumn_gpu>
        &compressed_columns,
    std::vector<OptionalFieldColumn_gpu> &out_columns) {
  out_columns.reserve(out_columns.size() + compressed_columns.size());
  for (const auto &compressed_col : compressed_columns) {
    out_columns.push_back(decompress_optional_field_column(compressed_col));
  }
}

} // namespace

void decompress_graph_metadata_gpu(
    const gpu_compression::CompressedData_gpu &data, GfaGraph_gpu &result) {
  result.num_paths = data.num_paths;
  result.num_walks = data.num_walks;

  result.path_names = decompress_flattened_strings(
      data.names_zstd_nvcomp, data.name_lengths_zstd_nvcomp);
  result.path_overlaps = decompress_flattened_strings(
      data.overlaps_zstd_nvcomp, data.overlap_lengths_zstd_nvcomp);

  if (data.num_walks > 0) {
    result.walk_sample_ids = decompress_flattened_strings(
        data.walk_sample_ids_zstd_nvcomp,
        data.walk_sample_id_lengths_zstd_nvcomp);
    result.walk_hap_indices = gpu_codec::nvcomp_zstd_decompress_uint32(
        data.walk_hap_indices_zstd_nvcomp);
    result.walk_seq_ids = decompress_flattened_strings(
        data.walk_seq_ids_zstd_nvcomp, data.walk_seq_id_lengths_zstd_nvcomp);
    result.walk_seq_starts = decompress_varint_int64(
        data.walk_seq_starts_zstd_nvcomp, data.num_walks);
    result.walk_seq_ends =
        decompress_varint_int64(data.walk_seq_ends_zstd_nvcomp, data.num_walks);
  }

  result.node_sequences = decompress_flattened_strings(
      data.segment_sequences_zstd_nvcomp, data.segment_seq_lengths_zstd_nvcomp);
  result.header_line = data.header_line;
  result.node_names = decompress_flattened_strings(
      data.node_names_zstd_nvcomp, data.node_name_lengths_zstd_nvcomp);

  if (data.num_links > 0) {
    result.link_from_ids =
        decompress_delta_varint_uint32(data.link_from_ids_zstd_nvcomp,
                                       data.num_links);
    result.link_to_ids = decompress_delta_varint_uint32(
        data.link_to_ids_zstd_nvcomp, data.num_links);
    result.link_from_orients = decompress_orientations(
        data.link_from_orients_zstd_nvcomp, data.num_links);
    result.link_to_orients = decompress_orientations(
        data.link_to_orients_zstd_nvcomp, data.num_links);
    result.link_overlap_nums =
        gpu_codec::nvcomp_zstd_decompress_uint32(data.link_overlap_nums_zstd_nvcomp);
    result.link_overlap_ops =
        decompress_char(data.link_overlap_ops_zstd_nvcomp);
  }

  if (!data.segment_optional_fields_zstd_nvcomp.empty()) {
    decompress_optional_columns(data.segment_optional_fields_zstd_nvcomp,
                                result.segment_optional_fields);
  }

  if (!data.link_optional_fields_zstd_nvcomp.empty()) {
    decompress_optional_columns(data.link_optional_fields_zstd_nvcomp,
                                result.link_optional_fields);
  }

  if (data.num_jumps_stored > 0) {
    result.jump_from_ids = decompress_delta_varint_uint32(
        data.jump_from_ids_zstd_nvcomp, data.num_jumps_stored);
    result.jump_to_ids = decompress_delta_varint_uint32(
        data.jump_to_ids_zstd_nvcomp, data.num_jumps_stored);
    result.jump_from_orients = decompress_orientations(
        data.jump_from_orients_zstd_nvcomp, data.num_jumps_stored);
    result.jump_to_orients = decompress_orientations(
        data.jump_to_orients_zstd_nvcomp, data.num_jumps_stored);
    result.jump_distances = decompress_flattened_strings(
        data.jump_distances_zstd_nvcomp, data.jump_distance_lengths_zstd_nvcomp);
    result.jump_rest_fields = decompress_flattened_strings(
        data.jump_rest_fields_zstd_nvcomp, data.jump_rest_lengths_zstd_nvcomp);
  }

  if (data.num_containments_stored > 0) {
    result.containment_container_ids = decompress_delta_varint_uint32(
        data.containment_container_ids_zstd_nvcomp,
        data.num_containments_stored);
    result.containment_contained_ids = decompress_delta_varint_uint32(
        data.containment_contained_ids_zstd_nvcomp,
        data.num_containments_stored);
    result.containment_container_orients = decompress_orientations(
        data.containment_container_orients_zstd_nvcomp,
        data.num_containments_stored);
    result.containment_contained_orients = decompress_orientations(
        data.containment_contained_orients_zstd_nvcomp,
        data.num_containments_stored);
    result.containment_positions = gpu_codec::nvcomp_zstd_decompress_uint32(
        data.containment_positions_zstd_nvcomp);
    result.containment_overlaps = decompress_flattened_strings(
        data.containment_overlaps_zstd_nvcomp,
        data.containment_overlap_lengths_zstd_nvcomp);
    result.containment_rest_fields = decompress_flattened_strings(
        data.containment_rest_fields_zstd_nvcomp,
        data.containment_rest_lengths_zstd_nvcomp);
  }

  result.num_segments =
      static_cast<uint32_t>(result.node_sequences.lengths.size());
  result.num_links = static_cast<uint32_t>(data.num_links);
}

} // namespace gpu_decompression
