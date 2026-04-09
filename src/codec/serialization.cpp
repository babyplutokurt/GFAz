#include "codec/serialization.hpp"
#include "utils/debug_log.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>


namespace {

constexpr const char *kSerializationErrorPrefix = "GFAZ serialization error: ";

// Binary I/O helpers

template <typename T> void write_val(std::ofstream &out, const T &v) {
  out.write(reinterpret_cast<const char *>(&v), sizeof(T));
}

template <typename T> T read_val(std::ifstream &in) {
  T v;
  in.read(reinterpret_cast<char *>(&v), sizeof(T));
  return v;
}

void write_bytes(std::ofstream &out, const std::vector<uint8_t> &vec) {
  uint64_t size = vec.size();
  write_val(out, size);
  if (size > 0)
    out.write(reinterpret_cast<const char *>(vec.data()), size);
}

std::vector<uint8_t> read_bytes(std::ifstream &in) {
  uint64_t size = read_val<uint64_t>(in);
  std::vector<uint8_t> vec(size);
  if (size > 0)
    in.read(reinterpret_cast<char *>(vec.data()), size);
  return vec;
}

void write_u32_vec(std::ofstream &out, const std::vector<uint32_t> &vec) {
  uint64_t size = vec.size();
  write_val(out, size);
  if (size > 0)
    out.write(reinterpret_cast<const char *>(vec.data()),
              size * sizeof(uint32_t));
}

std::vector<uint32_t> read_u32_vec(std::ifstream &in) {
  uint64_t size = read_val<uint64_t>(in);
  std::vector<uint32_t> vec(size);
  if (size > 0)
    in.read(reinterpret_cast<char *>(vec.data()), size * sizeof(uint32_t));
  return vec;
}

void write_str(std::ofstream &out, const std::string &s) {
  uint64_t size = s.size();
  write_val(out, size);
  if (size > 0)
    out.write(s.data(), size);
}

std::string read_str(std::ifstream &in) {
  uint64_t size = read_val<uint64_t>(in);
  std::string s(size, '\0');
  if (size > 0)
    in.read(&s[0], size);
  return s;
}

void write_block(std::ofstream &out, const ZstdCompressedBlock &b) {
  write_val(out, b.original_size);
  write_bytes(out, b.payload);
}

ZstdCompressedBlock read_block(std::ifstream &in) {
  ZstdCompressedBlock b;
  b.original_size = read_val<size_t>(in);
  b.payload = read_bytes(in);
  return b;
}

void write_range(std::ofstream &out, const LayerRuleRange &r) {
  write_val(out, r.k);
  write_val(out, r.start_id);
  write_val(out, r.end_id);
  write_val(out, r.flattened_offset);
  write_val(out, r.element_count);
}

LayerRuleRange read_range(std::ifstream &in) {
  LayerRuleRange r;
  r.k = read_val<int>(in);
  r.start_id = read_val<uint32_t>(in);
  r.end_id = read_val<uint32_t>(in);
  r.flattened_offset = read_val<size_t>(in);
  r.element_count = read_val<size_t>(in);
  return r;
}

void write_opt_col(std::ofstream &out, const CompressedOptionalFieldColumn &c) {
  write_str(out, c.tag);
  write_val(out, c.type);
  write_val(out, c.num_elements);
  write_block(out, c.int_values_zstd);
  write_block(out, c.float_values_zstd);
  write_block(out, c.char_values_zstd);
  write_block(out, c.strings_zstd);
  write_block(out, c.string_lengths_zstd);
  write_block(out, c.b_subtypes_zstd);
  write_block(out, c.b_lengths_zstd);
  write_block(out, c.b_concat_bytes_zstd);
}

CompressedOptionalFieldColumn read_opt_col(std::ifstream &in) {
  CompressedOptionalFieldColumn c;
  c.tag = read_str(in);
  c.type = read_val<char>(in);
  c.num_elements = read_val<size_t>(in);
  c.int_values_zstd = read_block(in);
  c.float_values_zstd = read_block(in);
  c.char_values_zstd = read_block(in);
  c.strings_zstd = read_block(in);
  c.string_lengths_zstd = read_block(in);
  c.b_subtypes_zstd = read_block(in);
  c.b_lengths_zstd = read_block(in);
  c.b_concat_bytes_zstd = read_block(in);
  return c;
}

} // namespace

void serialize_compressed_data(const CompressedData &data,
                               const std::string &output_path) {
  std::ofstream out(output_path, std::ios::binary);
  if (!out)
    throw std::runtime_error(std::string(kSerializationErrorPrefix) +
                             "failed to open output file: " + output_path);

  // Magic and version
  write_val(out, GFAZ_MAGIC);
  write_val(out, GFAZ_VERSION);

  // Header
  write_str(out, data.header_line);

  // Rules and paths
  uint64_t layer_count = data.layer_rule_ranges.size();
  write_val(out, layer_count);
  for (const auto &r : data.layer_rule_ranges)
    write_range(out, r);

  write_u32_vec(out, data.sequence_lengths);
  write_u32_vec(out, data.original_path_lengths);
  write_block(out, data.rules_first_zstd);
  write_block(out, data.rules_second_zstd);
  write_block(out, data.paths_zstd);
  write_val(out, data.delta_round);

  // Path names and overlaps
  write_block(out, data.names_zstd);
  write_block(out, data.name_lengths_zstd);
  write_block(out, data.overlaps_zstd);
  write_block(out, data.overlap_lengths_zstd);

  // Segments
  write_block(out, data.segment_sequences_zstd);
  write_block(out, data.segment_seq_lengths_zstd);

  uint64_t seg_opt_count = data.segment_optional_fields_zstd.size();
  write_val(out, seg_opt_count);
  for (const auto &c : data.segment_optional_fields_zstd)
    write_opt_col(out, c);

  // Links
  write_block(out, data.link_from_ids_zstd);
  write_block(out, data.link_to_ids_zstd);
  write_block(out, data.link_from_orients_zstd);
  write_block(out, data.link_to_orients_zstd);
  write_block(out, data.link_overlap_nums_zstd);
  write_block(out, data.link_overlap_ops_zstd);
  write_val(out, data.num_links);

  uint64_t link_opt_count = data.link_optional_fields_zstd.size();
  write_val(out, link_opt_count);
  for (const auto &c : data.link_optional_fields_zstd)
    write_opt_col(out, c);

  // J-lines (jumps)
  write_val(out, data.num_jumps);
  write_block(out, data.jump_from_ids_zstd);
  write_block(out, data.jump_from_orients_zstd);
  write_block(out, data.jump_to_ids_zstd);
  write_block(out, data.jump_to_orients_zstd);
  write_block(out, data.jump_distances_zstd);
  write_block(out, data.jump_distance_lengths_zstd);
  write_block(out, data.jump_rest_fields_zstd);
  write_block(out, data.jump_rest_lengths_zstd);

  // C-lines (containments)
  write_val(out, data.num_containments);
  write_block(out, data.containment_container_ids_zstd);
  write_block(out, data.containment_container_orients_zstd);
  write_block(out, data.containment_contained_ids_zstd);
  write_block(out, data.containment_contained_orients_zstd);
  write_block(out, data.containment_positions_zstd);
  write_block(out, data.containment_overlaps_zstd);
  write_block(out, data.containment_overlap_lengths_zstd);
  write_block(out, data.containment_rest_fields_zstd);
  write_block(out, data.containment_rest_lengths_zstd);

  // Walks
  write_u32_vec(out, data.walk_lengths);
  write_u32_vec(out, data.original_walk_lengths);
  write_block(out, data.walks_zstd);
  write_block(out, data.walk_sample_ids_zstd);
  write_block(out, data.walk_sample_id_lengths_zstd);
  write_block(out, data.walk_hap_indices_zstd);
  write_block(out, data.walk_seq_ids_zstd);
  write_block(out, data.walk_seq_id_lengths_zstd);
  write_block(out, data.walk_seq_starts_zstd);
  write_block(out, data.walk_seq_ends_zstd);

  out.close();

  // Report file size
  std::ifstream check(output_path, std::ios::binary | std::ios::ate);
  size_t file_size = check.tellg();
  GFAZ_LOG("Serialized to " << output_path << " (" << file_size << " bytes, "
                            << std::fixed << std::setprecision(2)
                            << (file_size / 1024.0 / 1024.0) << " MB)");
}

CompressedData deserialize_compressed_data(const std::string &input_path) {
  std::ifstream in(input_path, std::ios::binary);
  if (!in)
    throw std::runtime_error(std::string(kSerializationErrorPrefix) +
                             "failed to open input file: " + input_path);

  // Verify magic and version
  uint32_t magic = read_val<uint32_t>(in);
  if (magic != GFAZ_MAGIC)
    throw std::runtime_error(std::string(kSerializationErrorPrefix) +
                             "invalid file format (magic mismatch)");

  uint32_t version = read_val<uint32_t>(in);
  if (version != GFAZ_VERSION)
    throw std::runtime_error(std::string(kSerializationErrorPrefix) +
                             "unsupported file version " +
                             std::to_string(version) + " (expected " +
                             std::to_string(GFAZ_VERSION) + ")");

  CompressedData data;

  // Header
  data.header_line = read_str(in);

  // Rules and paths
  uint64_t layer_count = read_val<uint64_t>(in);
  data.layer_rule_ranges.reserve(layer_count);
  for (uint64_t i = 0; i < layer_count; ++i)
    data.layer_rule_ranges.push_back(read_range(in));

  data.sequence_lengths = read_u32_vec(in);
  data.original_path_lengths = read_u32_vec(in);
  data.rules_first_zstd = read_block(in);
  data.rules_second_zstd = read_block(in);
  data.paths_zstd = read_block(in);
  data.delta_round = read_val<int>(in);

  // Path names and overlaps
  data.names_zstd = read_block(in);
  data.name_lengths_zstd = read_block(in);
  data.overlaps_zstd = read_block(in);
  data.overlap_lengths_zstd = read_block(in);

  // Segments
  data.segment_sequences_zstd = read_block(in);
  data.segment_seq_lengths_zstd = read_block(in);

  uint64_t seg_opt_count = read_val<uint64_t>(in);
  data.segment_optional_fields_zstd.reserve(seg_opt_count);
  for (uint64_t i = 0; i < seg_opt_count; ++i)
    data.segment_optional_fields_zstd.push_back(read_opt_col(in));

  // Links
  data.link_from_ids_zstd = read_block(in);
  data.link_to_ids_zstd = read_block(in);
  data.link_from_orients_zstd = read_block(in);
  data.link_to_orients_zstd = read_block(in);
  data.link_overlap_nums_zstd = read_block(in);
  data.link_overlap_ops_zstd = read_block(in);
  data.num_links = read_val<size_t>(in);

  uint64_t link_opt_count = read_val<uint64_t>(in);
  data.link_optional_fields_zstd.reserve(link_opt_count);
  for (uint64_t i = 0; i < link_opt_count; ++i)
    data.link_optional_fields_zstd.push_back(read_opt_col(in));

  // J-lines (jumps)
  data.num_jumps = read_val<size_t>(in);
  data.jump_from_ids_zstd = read_block(in);
  data.jump_from_orients_zstd = read_block(in);
  data.jump_to_ids_zstd = read_block(in);
  data.jump_to_orients_zstd = read_block(in);
  data.jump_distances_zstd = read_block(in);
  data.jump_distance_lengths_zstd = read_block(in);
  data.jump_rest_fields_zstd = read_block(in);
  data.jump_rest_lengths_zstd = read_block(in);

  // C-lines (containments)
  data.num_containments = read_val<size_t>(in);
  data.containment_container_ids_zstd = read_block(in);
  data.containment_container_orients_zstd = read_block(in);
  data.containment_contained_ids_zstd = read_block(in);
  data.containment_contained_orients_zstd = read_block(in);
  data.containment_positions_zstd = read_block(in);
  data.containment_overlaps_zstd = read_block(in);
  data.containment_overlap_lengths_zstd = read_block(in);
  data.containment_rest_fields_zstd = read_block(in);
  data.containment_rest_lengths_zstd = read_block(in);

  // Walks
  data.walk_lengths = read_u32_vec(in);
  data.original_walk_lengths = read_u32_vec(in);
  data.walks_zstd = read_block(in);
  data.walk_sample_ids_zstd = read_block(in);
  data.walk_sample_id_lengths_zstd = read_block(in);
  data.walk_hap_indices_zstd = read_block(in);
  data.walk_seq_ids_zstd = read_block(in);
  data.walk_seq_id_lengths_zstd = read_block(in);
  data.walk_seq_starts_zstd = read_block(in);
  data.walk_seq_ends_zstd = read_block(in);

  GFAZ_LOG("Deserialized from " << input_path);
  return data;
}

