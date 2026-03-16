#include "gpu/serialization_gpu.hpp"

#include <fstream>
#include <stdexcept>

namespace {

constexpr const char *kGpuSerializationErrorPrefix =
    "GFAZ GPU serialization error: ";

template <typename T>
void write_val(std::ofstream &out, const T &v) {
  out.write(reinterpret_cast<const char *>(&v), sizeof(T));
}

template <typename T>
T read_val(std::ifstream &in) {
  T v{};
  in.read(reinterpret_cast<char *>(&v), sizeof(T));
  return v;
}

void write_bytes(std::ofstream &out, const std::vector<uint8_t> &vec) {
  uint64_t size = vec.size();
  write_val(out, size);
  if (size > 0) {
    out.write(reinterpret_cast<const char *>(vec.data()), size);
  }
}

std::vector<uint8_t> read_bytes(std::ifstream &in) {
  uint64_t size = read_val<uint64_t>(in);
  std::vector<uint8_t> vec(size);
  if (size > 0) {
    in.read(reinterpret_cast<char *>(vec.data()), size);
  }
  return vec;
}

void write_str(std::ofstream &out, const std::string &s) {
  uint64_t size = s.size();
  write_val(out, size);
  if (size > 0) {
    out.write(s.data(), size);
  }
}

std::string read_str(std::ifstream &in) {
  uint64_t size = read_val<uint64_t>(in);
  std::string s(size, '\0');
  if (size > 0) {
    in.read(&s[0], size);
  }
  return s;
}

void write_block(std::ofstream &out, const gpu_codec::NvcompCompressedBlock &b) {
  write_val(out, b.original_size);
  write_bytes(out, b.payload);
}

gpu_codec::NvcompCompressedBlock read_block(std::ifstream &in) {
  gpu_codec::NvcompCompressedBlock b;
  b.original_size = read_val<size_t>(in);
  b.payload = read_bytes(in);
  return b;
}

void write_rule_range(std::ofstream &out,
                      const gpu_compression::GPURuleRange &r) {
  write_val(out, r.start_id);
  write_val(out, r.count);
}

gpu_compression::GPURuleRange read_rule_range(std::ifstream &in) {
  gpu_compression::GPURuleRange r{};
  r.start_id = read_val<uint32_t>(in);
  r.count = read_val<uint32_t>(in);
  return r;
}

void write_opt_col(
    std::ofstream &out,
    const gpu_compression::CompressedOptionalFieldColumn_gpu &c) {
  write_str(out, c.tag);
  write_val(out, c.type);
  write_val(out, c.num_elements);
  write_block(out, c.int_values_zstd_nvcomp);
  write_block(out, c.float_values_zstd_nvcomp);
  write_block(out, c.char_values_zstd_nvcomp);
  write_block(out, c.strings_zstd_nvcomp);
  write_block(out, c.string_lengths_zstd_nvcomp);
  write_block(out, c.b_subtypes_zstd_nvcomp);
  write_block(out, c.b_lengths_zstd_nvcomp);
  write_block(out, c.b_concat_bytes_zstd_nvcomp);
}

gpu_compression::CompressedOptionalFieldColumn_gpu read_opt_col(
    std::ifstream &in) {
  gpu_compression::CompressedOptionalFieldColumn_gpu c;
  c.tag = read_str(in);
  c.type = read_val<char>(in);
  c.num_elements = read_val<size_t>(in);
  c.int_values_zstd_nvcomp = read_block(in);
  c.float_values_zstd_nvcomp = read_block(in);
  c.char_values_zstd_nvcomp = read_block(in);
  c.strings_zstd_nvcomp = read_block(in);
  c.string_lengths_zstd_nvcomp = read_block(in);
  c.b_subtypes_zstd_nvcomp = read_block(in);
  c.b_lengths_zstd_nvcomp = read_block(in);
  c.b_concat_bytes_zstd_nvcomp = read_block(in);
  return c;
}

} // namespace

void serialize_compressed_data_gpu(
    const gpu_compression::CompressedData_gpu &data,
    const std::string &output_path) {
  std::ofstream out(output_path, std::ios::binary);
  if (!out) {
    throw std::runtime_error(std::string(kGpuSerializationErrorPrefix) +
                             "failed to open output file: " + output_path);
  }

  write_val(out, GFAZ_GPU_MAGIC);
  write_val(out, GFAZ_GPU_VERSION);

  write_block(out, data.encoded_path_zstd_nvcomp);
  write_block(out, data.rules_first_zstd_nvcomp);
  write_block(out, data.rules_second_zstd_nvcomp);

  uint64_t layer_count = data.layer_ranges.size();
  write_val(out, layer_count);
  for (const auto &r : data.layer_ranges) {
    write_rule_range(out, r);
  }

  write_val(out, data.num_paths);
  write_val(out, data.num_walks);
  write_block(out, data.path_lengths_zstd_nvcomp);

  write_block(out, data.names_zstd_nvcomp);
  write_block(out, data.name_lengths_zstd_nvcomp);
  write_block(out, data.overlaps_zstd_nvcomp);
  write_block(out, data.overlap_lengths_zstd_nvcomp);

  write_block(out, data.walk_sample_ids_zstd_nvcomp);
  write_block(out, data.walk_sample_id_lengths_zstd_nvcomp);
  write_block(out, data.walk_hap_indices_zstd_nvcomp);
  write_block(out, data.walk_seq_ids_zstd_nvcomp);
  write_block(out, data.walk_seq_id_lengths_zstd_nvcomp);
  write_block(out, data.walk_seq_starts_zstd_nvcomp);
  write_block(out, data.walk_seq_ends_zstd_nvcomp);

  write_str(out, data.header_line);

  write_block(out, data.segment_sequences_zstd_nvcomp);
  write_block(out, data.segment_seq_lengths_zstd_nvcomp);
  write_block(out, data.node_names_zstd_nvcomp);
  write_block(out, data.node_name_lengths_zstd_nvcomp);

  uint64_t seg_opt_count = data.segment_optional_fields_zstd_nvcomp.size();
  write_val(out, seg_opt_count);
  for (const auto &c : data.segment_optional_fields_zstd_nvcomp) {
    write_opt_col(out, c);
  }

  write_block(out, data.link_from_ids_zstd_nvcomp);
  write_block(out, data.link_to_ids_zstd_nvcomp);
  write_block(out, data.link_from_orients_zstd_nvcomp);
  write_block(out, data.link_to_orients_zstd_nvcomp);
  write_block(out, data.link_overlap_nums_zstd_nvcomp);
  write_block(out, data.link_overlap_ops_zstd_nvcomp);
  write_val(out, data.num_links);

  uint64_t link_opt_count = data.link_optional_fields_zstd_nvcomp.size();
  write_val(out, link_opt_count);
  for (const auto &c : data.link_optional_fields_zstd_nvcomp) {
    write_opt_col(out, c);
  }

  write_val(out, data.num_jumps_stored);
  write_block(out, data.jump_from_ids_zstd_nvcomp);
  write_block(out, data.jump_to_ids_zstd_nvcomp);
  write_block(out, data.jump_from_orients_zstd_nvcomp);
  write_block(out, data.jump_to_orients_zstd_nvcomp);
  write_block(out, data.jump_distances_zstd_nvcomp);
  write_block(out, data.jump_distance_lengths_zstd_nvcomp);
  write_block(out, data.jump_rest_fields_zstd_nvcomp);
  write_block(out, data.jump_rest_lengths_zstd_nvcomp);

  write_val(out, data.num_containments_stored);
  write_block(out, data.containment_container_ids_zstd_nvcomp);
  write_block(out, data.containment_contained_ids_zstd_nvcomp);
  write_block(out, data.containment_container_orients_zstd_nvcomp);
  write_block(out, data.containment_contained_orients_zstd_nvcomp);
  write_block(out, data.containment_positions_zstd_nvcomp);
  write_block(out, data.containment_overlaps_zstd_nvcomp);
  write_block(out, data.containment_overlap_lengths_zstd_nvcomp);
  write_block(out, data.containment_rest_fields_zstd_nvcomp);
  write_block(out, data.containment_rest_lengths_zstd_nvcomp);
}

gpu_compression::CompressedData_gpu
deserialize_compressed_data_gpu(const std::string &input_path) {
  std::ifstream in(input_path, std::ios::binary);
  if (!in) {
    throw std::runtime_error(std::string(kGpuSerializationErrorPrefix) +
                             "failed to open input file: " + input_path);
  }

  uint32_t magic = read_val<uint32_t>(in);
  if (magic != GFAZ_GPU_MAGIC) {
    throw std::runtime_error(std::string(kGpuSerializationErrorPrefix) +
                             "invalid file format (magic mismatch)");
  }

  uint32_t version = read_val<uint32_t>(in);
  if (version != GFAZ_GPU_VERSION) {
    throw std::runtime_error(std::string(kGpuSerializationErrorPrefix) +
                             "unsupported file version " +
                             std::to_string(version) + " (expected " +
                             std::to_string(GFAZ_GPU_VERSION) + ")");
  }

  gpu_compression::CompressedData_gpu data;

  data.encoded_path_zstd_nvcomp = read_block(in);
  data.rules_first_zstd_nvcomp = read_block(in);
  data.rules_second_zstd_nvcomp = read_block(in);

  uint64_t layer_count = read_val<uint64_t>(in);
  data.layer_ranges.reserve(layer_count);
  for (uint64_t i = 0; i < layer_count; ++i) {
    data.layer_ranges.push_back(read_rule_range(in));
  }

  data.num_paths = read_val<uint32_t>(in);
  data.num_walks = read_val<uint32_t>(in);
  data.path_lengths_zstd_nvcomp = read_block(in);

  data.names_zstd_nvcomp = read_block(in);
  data.name_lengths_zstd_nvcomp = read_block(in);
  data.overlaps_zstd_nvcomp = read_block(in);
  data.overlap_lengths_zstd_nvcomp = read_block(in);

  data.walk_sample_ids_zstd_nvcomp = read_block(in);
  data.walk_sample_id_lengths_zstd_nvcomp = read_block(in);
  data.walk_hap_indices_zstd_nvcomp = read_block(in);
  data.walk_seq_ids_zstd_nvcomp = read_block(in);
  data.walk_seq_id_lengths_zstd_nvcomp = read_block(in);
  data.walk_seq_starts_zstd_nvcomp = read_block(in);
  data.walk_seq_ends_zstd_nvcomp = read_block(in);

  data.header_line = read_str(in);

  data.segment_sequences_zstd_nvcomp = read_block(in);
  data.segment_seq_lengths_zstd_nvcomp = read_block(in);
  data.node_names_zstd_nvcomp = read_block(in);
  data.node_name_lengths_zstd_nvcomp = read_block(in);

  uint64_t seg_opt_count = read_val<uint64_t>(in);
  data.segment_optional_fields_zstd_nvcomp.reserve(seg_opt_count);
  for (uint64_t i = 0; i < seg_opt_count; ++i) {
    data.segment_optional_fields_zstd_nvcomp.push_back(read_opt_col(in));
  }

  data.link_from_ids_zstd_nvcomp = read_block(in);
  data.link_to_ids_zstd_nvcomp = read_block(in);
  data.link_from_orients_zstd_nvcomp = read_block(in);
  data.link_to_orients_zstd_nvcomp = read_block(in);
  data.link_overlap_nums_zstd_nvcomp = read_block(in);
  data.link_overlap_ops_zstd_nvcomp = read_block(in);
  data.num_links = read_val<size_t>(in);

  uint64_t link_opt_count = read_val<uint64_t>(in);
  data.link_optional_fields_zstd_nvcomp.reserve(link_opt_count);
  for (uint64_t i = 0; i < link_opt_count; ++i) {
    data.link_optional_fields_zstd_nvcomp.push_back(read_opt_col(in));
  }

  data.num_jumps_stored = read_val<size_t>(in);
  data.jump_from_ids_zstd_nvcomp = read_block(in);
  data.jump_to_ids_zstd_nvcomp = read_block(in);
  data.jump_from_orients_zstd_nvcomp = read_block(in);
  data.jump_to_orients_zstd_nvcomp = read_block(in);
  data.jump_distances_zstd_nvcomp = read_block(in);
  data.jump_distance_lengths_zstd_nvcomp = read_block(in);
  data.jump_rest_fields_zstd_nvcomp = read_block(in);
  data.jump_rest_lengths_zstd_nvcomp = read_block(in);

  data.num_containments_stored = read_val<size_t>(in);
  data.containment_container_ids_zstd_nvcomp = read_block(in);
  data.containment_contained_ids_zstd_nvcomp = read_block(in);
  data.containment_container_orients_zstd_nvcomp = read_block(in);
  data.containment_contained_orients_zstd_nvcomp = read_block(in);
  data.containment_positions_zstd_nvcomp = read_block(in);
  data.containment_overlaps_zstd_nvcomp = read_block(in);
  data.containment_overlap_lengths_zstd_nvcomp = read_block(in);
  data.containment_rest_fields_zstd_nvcomp = read_block(in);
  data.containment_rest_lengths_zstd_nvcomp = read_block(in);

  return data;
}
