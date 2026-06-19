#include "core/codec/serialization.hpp"
#include "core/utils/debug_log.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>


namespace {

constexpr const char *kSerializationErrorPrefix = "GFAZ serialization error: ";

[[noreturn]] void throw_corrupt(const char *what) {
  throw std::runtime_error(std::string(kSerializationErrorPrefix) +
                           "corrupt or truncated .gfaz (" + what + ")");
}

// Bounded reader: validates every declared length against the bytes actually
// remaining in the file and that each read completed, so a corrupt/truncated
// container fails with a clear error instead of a giant allocation, a silent
// zero-filled buffer, or out-of-bounds decoding downstream.
struct BinReader {
  std::ifstream &in;
  uint64_t file_size;

  uint64_t remaining() {
    const std::streampos pos = in.tellg();
    if (pos < 0)
      return 0;
    const uint64_t p = static_cast<uint64_t>(pos);
    return (p <= file_size) ? (file_size - p) : 0;
  }

  void need(uint64_t bytes, const char *what) {
    if (bytes > remaining())
      throw_corrupt(what);
  }

  // Overflow-safe variant for element counts: rejects a count whose minimum
  // on-disk footprint already exceeds the remaining bytes, so a corrupt count
  // can't drive a giant reserve() before the per-element reads run.
  void need_count(uint64_t count, uint64_t min_elem_bytes, const char *what) {
    if (min_elem_bytes != 0 && count > remaining() / min_elem_bytes)
      throw_corrupt(what);
  }

  void after_read(const char *what) {
    if (!in.good())
      throw_corrupt(what);
  }
};

// Binary I/O helpers

template <typename T> void write_val(std::ofstream &out, const T &v) {
  out.write(reinterpret_cast<const char *>(&v), sizeof(T));
}

template <typename T> T read_val(BinReader &r) {
  T v;
  r.need(sizeof(T), "scalar field");
  r.in.read(reinterpret_cast<char *>(&v), sizeof(T));
  r.after_read("scalar field");
  return v;
}

void write_bytes(std::ofstream &out, const std::vector<uint8_t> &vec) {
  uint64_t size = vec.size();
  write_val(out, size);
  if (size > 0)
    out.write(reinterpret_cast<const char *>(vec.data()), size);
}

std::vector<uint8_t> read_bytes(BinReader &r) {
  uint64_t size = read_val<uint64_t>(r);
  r.need(size, "byte block");
  std::vector<uint8_t> vec(size);
  if (size > 0) {
    r.in.read(reinterpret_cast<char *>(vec.data()), size);
    r.after_read("byte block");
  }
  return vec;
}

void write_u32_vec(std::ofstream &out, const std::vector<uint32_t> &vec) {
  uint64_t size = vec.size();
  write_val(out, size);
  if (size > 0)
    out.write(reinterpret_cast<const char *>(vec.data()),
              size * sizeof(uint32_t));
}

std::vector<uint32_t> read_u32_vec(BinReader &r) {
  uint64_t size = read_val<uint64_t>(r);
  r.need(size * sizeof(uint32_t), "uint32 vector");
  std::vector<uint32_t> vec(size);
  if (size > 0) {
    r.in.read(reinterpret_cast<char *>(vec.data()), size * sizeof(uint32_t));
    r.after_read("uint32 vector");
  }
  return vec;
}

void write_str(std::ofstream &out, const std::string &s) {
  uint64_t size = s.size();
  write_val(out, size);
  if (size > 0)
    out.write(s.data(), size);
}

std::string read_str(BinReader &r) {
  uint64_t size = read_val<uint64_t>(r);
  r.need(size, "string field");
  std::string s(size, '\0');
  if (size > 0) {
    r.in.read(&s[0], size);
    r.after_read("string field");
  }
  return s;
}

void write_block(std::ofstream &out, const gfaz::ZstdCompressedBlock &b) {
  write_val(out, static_cast<uint64_t>(b.original_size));
  write_bytes(out, b.payload);
}

gfaz::ZstdCompressedBlock read_block(BinReader &r) {
  gfaz::ZstdCompressedBlock b;
  b.original_size = static_cast<size_t>(read_val<uint64_t>(r));
  b.payload = read_bytes(r);
  return b;
}

void write_range(std::ofstream &out, const gfaz::LayerRuleRange &r) {
  write_val(out, r.k);
  write_val(out, r.start_id);
  write_val(out, r.end_id);
  write_val(out, static_cast<uint64_t>(r.flattened_offset));
  write_val(out, static_cast<uint64_t>(r.element_count));
}

gfaz::LayerRuleRange read_range(BinReader &r) {
  gfaz::LayerRuleRange range;
  range.k = read_val<int>(r);
  range.start_id = read_val<uint32_t>(r);
  range.end_id = read_val<uint32_t>(r);
  range.flattened_offset = static_cast<size_t>(read_val<uint64_t>(r));
  range.element_count = static_cast<size_t>(read_val<uint64_t>(r));
  return range;
}

void write_opt_col(std::ofstream &out,
                   const gfaz::CompressedOptionalFieldColumn &c) {
  write_str(out, c.tag);
  write_val(out, c.type);
  write_val(out, static_cast<uint64_t>(c.num_elements));
  write_block(out, c.int_values_zstd);
  write_block(out, c.float_values_zstd);
  write_block(out, c.char_values_zstd);
  write_block(out, c.strings_zstd);
  write_block(out, c.string_lengths_zstd);
  write_block(out, c.b_subtypes_zstd);
  write_block(out, c.b_lengths_zstd);
  write_block(out, c.b_concat_bytes_zstd);
}

gfaz::CompressedOptionalFieldColumn read_opt_col(BinReader &r) {
  gfaz::CompressedOptionalFieldColumn c;
  c.tag = read_str(r);
  c.type = read_val<char>(r);
  c.num_elements = static_cast<size_t>(read_val<uint64_t>(r));
  c.int_values_zstd = read_block(r);
  c.float_values_zstd = read_block(r);
  c.char_values_zstd = read_block(r);
  c.strings_zstd = read_block(r);
  c.string_lengths_zstd = read_block(r);
  c.b_subtypes_zstd = read_block(r);
  c.b_lengths_zstd = read_block(r);
  c.b_concat_bytes_zstd = read_block(r);
  return c;
}

} // namespace

namespace gfaz {

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
  write_val(out, static_cast<uint64_t>(data.num_links));

  uint64_t link_opt_count = data.link_optional_fields_zstd.size();
  write_val(out, link_opt_count);
  for (const auto &c : data.link_optional_fields_zstd)
    write_opt_col(out, c);

  // J-lines (jumps)
  write_val(out, static_cast<uint64_t>(data.num_jumps));
  write_block(out, data.jump_from_ids_zstd);
  write_block(out, data.jump_from_orients_zstd);
  write_block(out, data.jump_to_ids_zstd);
  write_block(out, data.jump_to_orients_zstd);
  write_block(out, data.jump_distances_zstd);
  write_block(out, data.jump_distance_lengths_zstd);
  write_block(out, data.jump_rest_fields_zstd);
  write_block(out, data.jump_rest_lengths_zstd);

  // C-lines (containments)
  write_val(out, static_cast<uint64_t>(data.num_containments));
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

  in.seekg(0, std::ios::end);
  const std::streampos end_pos = in.tellg();
  in.seekg(0, std::ios::beg);
  if (end_pos < 0)
    throw std::runtime_error(std::string(kSerializationErrorPrefix) +
                             "failed to determine file size: " + input_path);
  BinReader r{in, static_cast<uint64_t>(end_pos)};

  // Verify magic and version
  uint32_t magic = read_val<uint32_t>(r);
  if (magic != GFAZ_MAGIC)
    throw std::runtime_error(std::string(kSerializationErrorPrefix) +
                             "invalid file format (magic mismatch)");

  uint32_t version = read_val<uint32_t>(r);
  if (version != GFAZ_VERSION)
    throw std::runtime_error(std::string(kSerializationErrorPrefix) +
                             "unsupported file version " +
                             std::to_string(version) + " (expected " +
                             std::to_string(GFAZ_VERSION) + ")");

  CompressedData data;

  // Header
  data.header_line = read_str(r);

  // Rules and paths
  uint64_t layer_count = read_val<uint64_t>(r);
  // Each LayerRuleRange is 28 bytes on disk (int + 2*u32 + 2*u64).
  r.need_count(layer_count, 28, "layer range count");
  data.layer_rule_ranges.reserve(layer_count);
  for (uint64_t i = 0; i < layer_count; ++i)
    data.layer_rule_ranges.push_back(read_range(r));

  data.sequence_lengths = read_u32_vec(r);
  data.original_path_lengths = read_u32_vec(r);
  data.rules_first_zstd = read_block(r);
  data.rules_second_zstd = read_block(r);
  data.paths_zstd = read_block(r);
  data.delta_round = read_val<int>(r);
  if (data.delta_round < 0) {
    std::cerr << "Warning: serialized delta_round=" << data.delta_round
              << " is invalid, clamping to 0" << std::endl;
    data.delta_round = 0;
  }

  // Path names and overlaps
  data.names_zstd = read_block(r);
  data.name_lengths_zstd = read_block(r);
  data.overlaps_zstd = read_block(r);
  data.overlap_lengths_zstd = read_block(r);

  // Segments
  data.segment_sequences_zstd = read_block(r);
  data.segment_seq_lengths_zstd = read_block(r);

  uint64_t seg_opt_count = read_val<uint64_t>(r);
  // Min on-disk footprint per column: tag len (8) + type (1) + num_elements (8).
  r.need_count(seg_opt_count, 17, "segment optional column count");
  data.segment_optional_fields_zstd.reserve(seg_opt_count);
  for (uint64_t i = 0; i < seg_opt_count; ++i)
    data.segment_optional_fields_zstd.push_back(read_opt_col(r));

  // Links
  data.link_from_ids_zstd = read_block(r);
  data.link_to_ids_zstd = read_block(r);
  data.link_from_orients_zstd = read_block(r);
  data.link_to_orients_zstd = read_block(r);
  data.link_overlap_nums_zstd = read_block(r);
  data.link_overlap_ops_zstd = read_block(r);
  data.num_links = static_cast<size_t>(read_val<uint64_t>(r));

  uint64_t link_opt_count = read_val<uint64_t>(r);
  r.need_count(link_opt_count, 17, "link optional column count");
  data.link_optional_fields_zstd.reserve(link_opt_count);
  for (uint64_t i = 0; i < link_opt_count; ++i)
    data.link_optional_fields_zstd.push_back(read_opt_col(r));

  // J-lines (jumps)
  data.num_jumps = static_cast<size_t>(read_val<uint64_t>(r));
  data.jump_from_ids_zstd = read_block(r);
  data.jump_from_orients_zstd = read_block(r);
  data.jump_to_ids_zstd = read_block(r);
  data.jump_to_orients_zstd = read_block(r);
  data.jump_distances_zstd = read_block(r);
  data.jump_distance_lengths_zstd = read_block(r);
  data.jump_rest_fields_zstd = read_block(r);
  data.jump_rest_lengths_zstd = read_block(r);

  // C-lines (containments)
  data.num_containments = static_cast<size_t>(read_val<uint64_t>(r));
  data.containment_container_ids_zstd = read_block(r);
  data.containment_container_orients_zstd = read_block(r);
  data.containment_contained_ids_zstd = read_block(r);
  data.containment_contained_orients_zstd = read_block(r);
  data.containment_positions_zstd = read_block(r);
  data.containment_overlaps_zstd = read_block(r);
  data.containment_overlap_lengths_zstd = read_block(r);
  data.containment_rest_fields_zstd = read_block(r);
  data.containment_rest_lengths_zstd = read_block(r);

  // Walks
  data.walk_lengths = read_u32_vec(r);
  data.original_walk_lengths = read_u32_vec(r);
  data.walks_zstd = read_block(r);
  data.walk_sample_ids_zstd = read_block(r);
  data.walk_sample_id_lengths_zstd = read_block(r);
  data.walk_hap_indices_zstd = read_block(r);
  data.walk_seq_ids_zstd = read_block(r);
  data.walk_seq_id_lengths_zstd = read_block(r);
  data.walk_seq_starts_zstd = read_block(r);
  data.walk_seq_ends_zstd = read_block(r);

  GFAZ_LOG("Deserialized from " << input_path);
  return data;
}

} // namespace gfaz
