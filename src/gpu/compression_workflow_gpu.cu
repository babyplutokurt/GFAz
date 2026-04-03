#include "gfa_parser.hpp"
#include "gpu/codec_gpu.cuh"
#include "gpu/compression_workflow_gpu.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include "gpu/codec_gpu_nvcomp.cuh"

namespace gpu_compression {

// Debug flag for compression stats (can be controlled via environment or
// compile flag)
static bool g_debug_compression = false;
using Clock = std::chrono::high_resolution_clock;

static double elapsed_ms(const Clock::time_point &start,
                         const Clock::time_point &end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

static std::string flattened_to_string(const FlattenedStrings &flat) {
  return std::string(flat.data.begin(), flat.data.end());
}

static void print_compression_stats(const char *label, size_t original_size,
                                    size_t compressed_size) {
  if (!g_debug_compression || original_size == 0)
    return;

  double ratio = 100.0 * (1.0 - static_cast<double>(compressed_size) /
                                    static_cast<double>(original_size));
  std::cout << "  [nvComp] " << label << ": " << original_size << " -> "
            << compressed_size << " bytes (" << std::fixed
            << std::setprecision(1) << ratio << "% reduction)" << std::endl;
}

// =============================================================================
// GPU ZSTD compression wrappers using nvComp
// All functions follow the pattern: compress_<type>_gpu(input, label)
// =============================================================================

static gpu_codec::NvcompCompressedBlock
compress_bytes_gpu(const std::vector<uint8_t> &input,
                   const char *label = "bytes") {
  auto block = gpu_codec::nvcomp_zstd_compress(input);
  print_compression_stats(label, input.size(), block.payload.size());
  return block;
}

static gpu_codec::NvcompCompressedBlock
compress_string_gpu(const std::string &input, const char *label = "string") {
  auto block = gpu_codec::nvcomp_zstd_compress_string(input);
  print_compression_stats(label, input.size(), block.payload.size());
  return block;
}

static gpu_codec::NvcompCompressedBlock
compress_uint32_gpu(const std::vector<uint32_t> &input,
                    const char *label = "uint32_vec") {
  size_t original_bytes = input.size() * sizeof(uint32_t);
  auto block = gpu_codec::nvcomp_zstd_compress_uint32(input);
  print_compression_stats(label, original_bytes, block.payload.size());
  return block;
}

static gpu_codec::NvcompCompressedBlock
compress_float_gpu(const std::vector<float> &input,
                   const char *label = "float_vec") {
  const uint8_t *bytes = reinterpret_cast<const uint8_t *>(input.data());
  std::vector<uint8_t> payload(bytes, bytes + input.size() * sizeof(float));
  return compress_bytes_gpu(payload, label);
}

static gpu_codec::NvcompCompressedBlock
compress_char_gpu(const std::vector<char> &input,
                  const char *label = "char_vec") {
  const uint8_t *bytes = reinterpret_cast<const uint8_t *>(input.data());
  std::vector<uint8_t> payload(bytes, bytes + input.size() * sizeof(char));
  return compress_bytes_gpu(payload, label);
}

void set_gpu_compression_debug(bool enabled) {
  g_debug_compression = enabled;
}

// Device-resident compression (avoids D->H->D round-trip)
static gpu_codec::NvcompCompressedBlock
compress_int32_device_gpu(const thrust::device_vector<int32_t> &d_input,
                          const char *label = "int32_device") {
  auto block = gpu_codec::nvcomp_zstd_compress_int32_device(
      thrust::raw_pointer_cast(d_input.data()), d_input.size());
  size_t original_bytes = d_input.size() * sizeof(int32_t);
  print_compression_stats(label, original_bytes, block.payload.size());
  return block;
}

static uint64_t zigzag_encode_64(int64_t value) {
  return (static_cast<uint64_t>(value) << 1) ^
         static_cast<uint64_t>(value >> 63);
}

static void append_varint_64(uint64_t value, std::vector<uint8_t> &out) {
  while (value >= 0x80) {
    out.push_back(static_cast<uint8_t>((value & 0x7F) | 0x80));
    value >>= 7;
  }
  out.push_back(static_cast<uint8_t>(value));
}

static gpu_codec::NvcompCompressedBlock
compress_varint_int64_gpu(const std::vector<int64_t> &input,
                          const char *label = "varint_int64") {
  std::vector<uint8_t> varint_bytes;
  varint_bytes.reserve(input.size() * 4);
  for (int64_t val : input) {
    append_varint_64(zigzag_encode_64(val), varint_bytes);
  }
  return compress_bytes_gpu(varint_bytes, label);
}

static void append_varint_32(uint32_t value, std::vector<uint8_t> &out) {
  while (value >= 0x80) {
    out.push_back(static_cast<uint8_t>((value & 0x7F) | 0x80));
    value >>= 7;
  }
  out.push_back(static_cast<uint8_t>(value));
}

static gpu_codec::NvcompCompressedBlock
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
    // Zigzag encode: cast to unsigned first to avoid UB on negative left-shift
    uint32_t zigzag =
        (static_cast<uint32_t>(val) << 1) ^ static_cast<uint32_t>(val >> 31);
    append_varint_32(zigzag, varint_bytes);
  }

  return compress_bytes_gpu(varint_bytes, label);
}

static std::vector<int32_t>
inverse_delta_decode_host(const std::vector<int32_t> &delta_encoded) {
  if (delta_encoded.empty()) {
    return {};
  }
  std::vector<int32_t> decoded(delta_encoded.size());
  decoded[0] = delta_encoded[0];
  for (size_t i = 1; i < delta_encoded.size(); ++i) {
    decoded[i] = decoded[i - 1] + delta_encoded[i];
  }
  return decoded;
}

static gpu_codec::NvcompCompressedBlock
compress_orientations_gpu(const std::vector<char> &orients,
                          const char *label = "orientations") {

  // Use GPU kernel for bit-packing
  std::vector<uint8_t> packed = gpu_codec::pack_orientations_gpu(orients);

  return compress_bytes_gpu(packed, label);
}

// Helper: compress a FlattenedStrings into data + lengths blocks
static void compress_flattened_strings_gpu(
    const FlattenedStrings &flat, gpu_codec::NvcompCompressedBlock &data_block,
    gpu_codec::NvcompCompressedBlock &lengths_block, const char *data_label,
    const char *lengths_label) {
  data_block = compress_string_gpu(flattened_to_string(flat), data_label);
  lengths_block = compress_uint32_gpu(flat.lengths, lengths_label);
}

// Helper: compress an optional field column
static CompressedOptionalFieldColumn_gpu
compress_optional_column_gpu(const OptionalFieldColumn_gpu &col,
                             const char *prefix) {
  CompressedOptionalFieldColumn_gpu compressed_col;
  compressed_col.tag = col.tag;
  compressed_col.type = col.type;

  // Build label strings (using static buffers for simplicity)
  char label[64];

  switch (col.type) {
  case 'i':
    compressed_col.num_elements = col.int_values.size();
    snprintf(label, sizeof(label), "%s_optional_i", prefix);
    compressed_col.int_values_zstd_nvcomp =
        compress_varint_int64_gpu(col.int_values, label);
    break;
  case 'f':
    compressed_col.num_elements = col.float_values.size();
    snprintf(label, sizeof(label), "%s_optional_f", prefix);
    compressed_col.float_values_zstd_nvcomp =
        compress_float_gpu(col.float_values, label);
    break;
  case 'A':
    compressed_col.num_elements = col.char_values.size();
    snprintf(label, sizeof(label), "%s_optional_A", prefix);
    compressed_col.char_values_zstd_nvcomp =
        compress_char_gpu(col.char_values, label);
    break;
  case 'Z':
  case 'J':
  case 'H':
    compressed_col.num_elements = col.strings.lengths.size();
    snprintf(label, sizeof(label), "%s_optional_strings", prefix);
    compressed_col.strings_zstd_nvcomp = compress_string_gpu(
        std::string(col.strings.data.begin(), col.strings.data.end()), label);
    snprintf(label, sizeof(label), "%s_optional_string_lengths", prefix);
    compressed_col.string_lengths_zstd_nvcomp =
        compress_uint32_gpu(col.strings.lengths, label);
    break;
  case 'B':
    compressed_col.num_elements = col.b_subtypes.size();
    snprintf(label, sizeof(label), "%s_optional_b_subtypes", prefix);
    compressed_col.b_subtypes_zstd_nvcomp =
        compress_char_gpu(col.b_subtypes, label);
    snprintf(label, sizeof(label), "%s_optional_b_lengths", prefix);
    compressed_col.b_lengths_zstd_nvcomp =
        compress_uint32_gpu(col.b_lengths, label);
    snprintf(label, sizeof(label), "%s_optional_b_bytes", prefix);
    compressed_col.b_concat_bytes_zstd_nvcomp =
        compress_bytes_gpu(col.b_data, label);
    break;
  }
  return compressed_col;
}

static void compress_optional_columns_gpu(
    const std::vector<OptionalFieldColumn_gpu> &columns,
    std::vector<CompressedOptionalFieldColumn_gpu> &out_columns,
    const char *prefix) {
  out_columns.reserve(out_columns.size() + columns.size());
  for (const auto &col : columns) {
    out_columns.push_back(compress_optional_column_gpu(col, prefix));
  }
}

CompressedData_gpu run_path_compression_gpu(const FlattenedPaths &paths,
                                            int num_rounds) {
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

  // 1. Copy paths to device once
  thrust::device_vector<int32_t> d_data(paths.data.begin(), paths.data.end());

  // 2. Compute start_id on GPU (max abs value + 1)
  uint32_t start_id = gpu_codec::find_max_abs_device(d_data) + 1;

  // 3. Delta encode on GPU
  gpu_codec::delta_encode_device_vec(d_data);

  // 4. Recompute start_id after delta encoding (delta values may be larger)
  uint32_t delta_max = gpu_codec::find_max_abs_device(d_data);
  if (delta_max >= start_id) {
    start_id = delta_max + 1;
  }

  // 5. Accumulator for all rules (device vector)
  thrust::device_vector<uint64_t> d_all_rules;
  uint32_t next_start_id = start_id;

  // 6. Compression rounds
  for (int round_idx = 0; round_idx < num_rounds; ++round_idx) {
    // Find repeated 2-mers (stays on device)
    thrust::device_vector<uint64_t> d_round_rules =
        gpu_codec::find_repeated_2mers_device_vec(d_data);

    if (d_round_rules.empty()) {
      break; // No more rules found
    }

    uint32_t num_rules_found = d_round_rules.size();

    // Create hash table from device memory
    void *table_ptr = gpu_codec::create_rule_table_gpu_from_device(
        d_round_rules, next_start_id);

    // Apply rules (all on device)
    thrust::device_vector<uint8_t> d_rules_used(num_rules_found, 0);
    gpu_codec::apply_2mer_rules_device_vec(d_data, table_ptr, d_rules_used,
                                           next_start_id);

    // Cleanup table
    gpu_codec::free_rule_table_gpu(table_ptr);

    // Compact rules (all on device)
    gpu_codec::compact_rules_and_remap_device_vec(d_data, d_rules_used,
                                                  d_round_rules, next_start_id);

    // Sort rules (all on device)
    gpu_codec::sort_rules_and_remap_device_vec(d_data, d_round_rules,
                                               next_start_id);

    // Record layer range
    uint32_t num_used_rules = d_round_rules.size();
    result.layer_ranges.push_back({next_start_id, num_used_rules});

    // Append rules to accumulator (device-to-device copy)
    size_t old_size = d_all_rules.size();
    d_all_rules.resize(old_size + num_used_rules);
    thrust::copy(d_round_rules.begin(), d_round_rules.end(),
                 d_all_rules.begin() + old_size);

    // Update next start ID
    next_start_id += num_used_rules;
  }

  // 7. Compress encoded path directly from device (no D->H copy needed)
  result.encoded_path_zstd_nvcomp =
      compress_int32_device_gpu(d_data, "encoded_path");

  // 8. Compress path_lengths with nvComp (host data, small so copy is OK)
  result.path_lengths_zstd_nvcomp =
      compress_uint32_gpu(paths.lengths, "path_lengths");

  // 9. Split rules into first/second, delta-encode on GPU, then compress
  // directly from device
  if (!d_all_rules.empty()) {
    thrust::device_vector<int32_t> d_first, d_second;
    gpu_codec::split_and_delta_encode_rules_device_vec(d_all_rules, d_first,
                                                       d_second);

    // Compress directly from device (no D->H copy needed)
    result.rules_first_zstd_nvcomp =
        compress_int32_device_gpu(d_first, "rules_first");
    result.rules_second_zstd_nvcomp =
        compress_int32_device_gpu(d_second, "rules_second");
  }

  return result;
}

std::map<uint32_t, uint64_t> build_rulebook(const CompressedData_gpu &data) {
  std::map<uint32_t, uint64_t> rulebook;

  // Decompress rules_first and rules_second
  std::vector<int32_t> first_delta =
      gpu_codec::nvcomp_zstd_decompress_int32(data.rules_first_zstd_nvcomp);
  std::vector<int32_t> second_delta =
      gpu_codec::nvcomp_zstd_decompress_int32(data.rules_second_zstd_nvcomp);

  if (first_delta.empty() || second_delta.empty()) {
    return rulebook;
  }

  // Inverse delta-encode to get original rule endpoints.
  std::vector<int32_t> first = inverse_delta_decode_host(first_delta);
  std::vector<int32_t> second = inverse_delta_decode_host(second_delta);

  // Build rulebook from unpacked rules
  size_t offset = 0;
  for (const auto &range : data.layer_ranges) {
    for (uint32_t i = 0; i < range.count; ++i) {
      uint32_t rule_id = range.start_id + i;
      // Pack first and second back into uint64_t
      uint64_t packed =
          (static_cast<uint64_t>(static_cast<uint32_t>(first[offset + i]))
           << 32) |
          static_cast<uint64_t>(static_cast<uint32_t>(second[offset + i]));
      rulebook[rule_id] = packed;
    }
    offset += range.count;
  }

  return rulebook;
}

CompressedData_gpu compress_gfa_gpu(const std::string &gfa_file_path,
                                    int num_rounds) {
  GfaParser parser;
  GfaGraph graph = parser.parse(gfa_file_path);

  GfaGraph_gpu gpu_graph = convert_to_gpu_layout(graph);

  return compress_gpu_graph(gpu_graph, num_rounds);
}

CompressedData_gpu compress_gpu_graph(const GfaGraph_gpu &gpu_graph,
                                      int num_rounds) {
  // Start timer for compression (GfaGraph_gpu -> CompressedData_gpu)
  auto compress_start = Clock::now();

  CompressedData_gpu data =
      run_path_compression_gpu(gpu_graph.paths, num_rounds);

  // Store path/walk split info
  data.num_paths = gpu_graph.num_paths;
  data.num_walks = gpu_graph.num_walks;

  // ====== PATH METADATA (P-lines only) ======
  if (g_debug_compression) {
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

  // ====== WALK METADATA (W-lines only) ======
  if (gpu_graph.num_walks > 0) {
    if (g_debug_compression) {
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

  // ====== SEGMENT DATA ======
  compress_flattened_strings_gpu(gpu_graph.node_sequences,
                                 data.segment_sequences_zstd_nvcomp,
                                 data.segment_seq_lengths_zstd_nvcomp,
                                 "segment_sequences", "segment_seq_lengths");

  // ====== HEADER + NODE NAMES (for full-fidelity round-trip) ======
  data.header_line = gpu_graph.header_line;

  compress_flattened_strings_gpu(
      gpu_graph.node_names, data.node_names_zstd_nvcomp,
      data.node_name_lengths_zstd_nvcomp, "node_names", "node_name_lengths");

  // ====== SEGMENT OPTIONAL FIELDS ======
  compress_optional_columns_gpu(gpu_graph.segment_optional_fields,
                                data.segment_optional_fields_zstd_nvcomp,
                                "segment");

  // ====== LINK DATA ======
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

  // ====== LINK OPTIONAL FIELDS ======
  compress_optional_columns_gpu(gpu_graph.link_optional_fields,
                                data.link_optional_fields_zstd_nvcomp, "link");

  // ====== JUMP DATA (J-lines) - structured columnar ======
  if (!gpu_graph.jump_from_ids.empty()) {
    data.num_jumps_stored = gpu_graph.jump_from_ids.size();

    if (g_debug_compression) {
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

  // ====== CONTAINMENT DATA (C-lines) - structured columnar ======
  if (!gpu_graph.containment_container_ids.empty()) {
    data.num_containments_stored = gpu_graph.containment_container_ids.size();

    if (g_debug_compression) {
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
    data.containment_positions_zstd_nvcomp = compress_uint32_gpu(
        gpu_graph.containment_positions, "containment_positions");

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

  // End timer and print compression time
  auto compress_end = Clock::now();
  double compress_time_ms = elapsed_ms(compress_start, compress_end);

  if (g_debug_compression) {
    std::cout << "[GPU Compression] Total compression time (GfaGraph_gpu -> "
                 "CompressedData_gpu): "
              << std::fixed << std::setprecision(2) << compress_time_ms
              << " ms (" << compress_time_ms / 1000.0 << " s)" << std::endl;
  }

  return data;
}

} // namespace gpu_compression
