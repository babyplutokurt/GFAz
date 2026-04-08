#include "gpu/compression_workflow_gpu_internal.hpp"
#include "gpu/codec_gpu.cuh"
#include "gpu/metadata_codec_gpu.hpp"

#include <cstdio>
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
