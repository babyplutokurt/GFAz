#include "io/gfa_write_utils.hpp"

#include "codec/codec.hpp"

#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace {

constexpr const char *kWriterErrorPrefix = "GFA writer error: ";

std::string format_float(float val) {
  std::ostringstream oss;
  oss << val;
  return oss.str();
}

inline size_t b_elem_size(char subtype) {
  switch (subtype) {
  case 'c':
  case 'C':
    return 1;
  case 's':
  case 'S':
    return 2;
  case 'i':
  case 'I':
  case 'f':
    return 4;
  default:
    return 0;
  }
}

void reconstruct_strings(const std::string &concat,
                         const std::vector<uint32_t> &lengths,
                         std::vector<std::string> &out) {
  out.clear();
  out.reserve(lengths.size());

  size_t offset = 0;
  for (uint32_t len : lengths) {
    if (offset + len > concat.size()) {
      throw std::runtime_error(std::string(kWriterErrorPrefix) +
                               "string column is truncated");
    }
    out.push_back(concat.substr(offset, len));
    offset += len;
  }
}

} // namespace

namespace gfaz::gfa_write_utils {

FieldOffsets build_field_offsets(const std::vector<gfaz::OptionalFieldColumn> &cols) {
  FieldOffsets offsets(cols.size());
  for (size_t c = 0; c < cols.size(); ++c) {
    const auto &col = cols[c];
    if (col.type == 'Z' || col.type == 'J' || col.type == 'H') {
      const size_t n = col.string_lengths.size();
      offsets[c].resize(n + 1, 0);
      for (size_t i = 0; i < n; ++i)
        offsets[c][i + 1] = offsets[c][i] + col.string_lengths[i];
    } else if (col.type == 'B') {
      const size_t n = col.b_lengths.size();
      offsets[c].resize(n + 1, 0);
      for (size_t i = 0; i < n; ++i) {
        size_t elem_sz = b_elem_size(col.b_subtypes[i]);
        offsets[c][i + 1] = offsets[c][i] + col.b_lengths[i] * elem_sz;
      }
    }
  }
  return offsets;
}

std::string format_optional_fields(const std::vector<gfaz::OptionalFieldColumn> &cols,
                                   const FieldOffsets &offsets, size_t index) {
  std::string result;
  for (size_t c = 0; c < cols.size(); ++c) {
    const auto &col = cols[c];
    switch (col.type) {
    case 'i':
      if (index < col.int_values.size()) {
        int64_t val = col.int_values[index];
        if (val != std::numeric_limits<int64_t>::min())
          result += "\t" + col.tag + ":i:" + std::to_string(val);
      }
      break;
    case 'f':
      if (index < col.float_values.size()) {
        float val = col.float_values[index];
        if (val != std::numeric_limits<float>::lowest())
          result += "\t" + col.tag + ":f:" + format_float(val);
      }
      break;
    case 'A':
      if (index < col.char_values.size()) {
        char val = col.char_values[index];
        if (val != '\0')
          result += "\t" + col.tag + ":A:" + std::string(1, val);
      }
      break;
    case 'Z':
    case 'J':
    case 'H':
      if (index < col.string_lengths.size()) {
        uint32_t len = col.string_lengths[index];
        if (len > 0) {
          size_t off = offsets[c][index];
          result += "\t" + col.tag + ":" + std::string(1, col.type) + ":";
          result.append(col.concatenated_strings, off, len);
        }
      }
      break;
    case 'B':
      if (index < col.b_lengths.size()) {
        char subtype = col.b_subtypes[index];
        uint32_t count = col.b_lengths[index];
        if (subtype != '\0' && count > 0) {
          size_t byte_off = offsets[c][index];
          size_t elem_sz = b_elem_size(subtype);
          const uint8_t *ptr = col.b_concat_bytes.data() + byte_off;

          result += "\t" + col.tag + ":B:" + std::string(1, subtype);
          for (uint32_t i = 0; i < count; ++i) {
            result += ',';
            if (subtype == 'c') {
              int8_t v;
              std::memcpy(&v, ptr + i * elem_sz, 1);
              result += std::to_string(v);
            } else if (subtype == 'C') {
              uint8_t v = ptr[i];
              result += std::to_string(v);
            } else if (subtype == 's') {
              int16_t v;
              std::memcpy(&v, ptr + i * elem_sz, 2);
              result += std::to_string(v);
            } else if (subtype == 'S') {
              uint16_t v;
              std::memcpy(&v, ptr + i * elem_sz, 2);
              result += std::to_string(v);
            } else if (subtype == 'i') {
              int32_t v;
              std::memcpy(&v, ptr + i * elem_sz, 4);
              result += std::to_string(v);
            } else if (subtype == 'I') {
              uint32_t v;
              std::memcpy(&v, ptr + i * elem_sz, 4);
              result += std::to_string(v);
            } else if (subtype == 'f') {
              float v;
              std::memcpy(&v, ptr + i * elem_sz, 4);
              result += format_float(v);
            }
          }
        }
      }
      break;
    }
  }
  return result;
}

void append_numeric_node_name(std::string &out, uint32_t node_id) {
  out += std::to_string(node_id);
}

std::vector<std::string>
decompress_string_column(const gfaz::ZstdCompressedBlock &strings_zstd,
                         const gfaz::ZstdCompressedBlock &lengths_zstd) {
  std::vector<std::string> out;
  reconstruct_strings(gfaz::Codec::zstd_decompress_string(strings_zstd),
                      gfaz::Codec::zstd_decompress_uint32_vector(lengths_zstd), out);
  return out;
}

gfaz::OptionalFieldColumn
decompress_optional_column(const gfaz::CompressedOptionalFieldColumn &c) {
  gfaz::OptionalFieldColumn col;
  col.tag = c.tag;
  col.type = c.type;

  switch (c.type) {
  case 'i':
    col.int_values =
        gfaz::Codec::decompress_varint_int64(c.int_values_zstd, c.num_elements);
    break;
  case 'f':
    col.float_values = gfaz::Codec::zstd_decompress_float_vector(c.float_values_zstd);
    break;
  case 'A':
    col.char_values = gfaz::Codec::zstd_decompress_char_vector(c.char_values_zstd);
    break;
  case 'Z':
  case 'J':
  case 'H':
    col.concatenated_strings = gfaz::Codec::zstd_decompress_string(c.strings_zstd);
    col.string_lengths =
        gfaz::Codec::zstd_decompress_uint32_vector(c.string_lengths_zstd);
    break;
  case 'B': {
    col.b_subtypes = gfaz::Codec::zstd_decompress_char_vector(c.b_subtypes_zstd);
    col.b_lengths = gfaz::Codec::zstd_decompress_uint32_vector(c.b_lengths_zstd);
    const std::string bytes =
        gfaz::Codec::zstd_decompress_string(c.b_concat_bytes_zstd);
    col.b_concat_bytes = std::vector<uint8_t>(bytes.begin(), bytes.end());
    break;
  }
  default:
    throw std::runtime_error(std::string(kWriterErrorPrefix) +
                             "unsupported optional field type");
  }

  return col;
}

SequenceOffsets build_offsets(const std::vector<uint32_t> &lengths) {
  SequenceOffsets offsets(lengths.size() + 1, 0);
  for (size_t i = 0; i < lengths.size(); ++i)
    offsets[i + 1] = offsets[i] + lengths[i];
  return offsets;
}

std::pair<std::vector<int32_t>, std::vector<int32_t>>
decode_rules(const gfaz::CompressedData &data) {
  std::vector<int32_t> first =
      gfaz::Codec::zstd_decompress_int32_vector(data.rules_first_zstd);
  std::vector<int32_t> second =
      gfaz::Codec::zstd_decompress_int32_vector(data.rules_second_zstd);
  gfaz::Codec::delta_decode_int32(first);
  gfaz::Codec::delta_decode_int32(second);
  return {std::move(first), std::move(second)};
}

std::string format_path_line_numeric(const std::string &path_name,
                                     const int32_t *path_data, size_t path_size,
                                     const std::string &overlap) {
  std::string line = "P\t";
  line += path_name;
  line += '\t';

  for (size_t i = 0; i < path_size; ++i) {
    if (i > 0)
      line += ',';

    const int32_t node = path_data[i];
    const bool reverse = node < 0;
    append_numeric_node_name(
        line, static_cast<uint32_t>(reverse ? -node : node));
    line += (reverse ? '-' : '+');
  }

  line += '\t';
  line += overlap.empty() ? "*" : overlap;
  line += '\n';
  return line;
}

std::string format_walk_line_numeric(const std::string &sample_id,
                                     uint32_t hap_index,
                                     const std::string &seq_id,
                                     int64_t seq_start, int64_t seq_end,
                                     const int32_t *walk_data,
                                     size_t walk_size) {
  std::string line = "W\t";
  line += sample_id;
  line += '\t';
  line += std::to_string(hap_index);
  line += '\t';
  line += seq_id;
  line += '\t';
  line += (seq_start >= 0) ? std::to_string(seq_start) : "*";
  line += '\t';
  line += (seq_end >= 0) ? std::to_string(seq_end) : "*";
  line += '\t';

  for (size_t i = 0; i < walk_size; ++i) {
    const int32_t node = walk_data[i];
    const bool reverse = node < 0;
    line += (reverse ? '<' : '>');
    append_numeric_node_name(
        line, static_cast<uint32_t>(reverse ? -node : node));
  }

  line += '\n';
  return line;
}

void write_segments_numeric(std::ofstream &out,
                            const std::string &segment_sequences,
                            const std::vector<uint32_t> &segment_lengths,
                            const std::vector<gfaz::OptionalFieldColumn>
                                &segment_optional_fields,
                            const FieldOffsets &segment_offsets) {
  std::string line;
  line.reserve(4096);
  size_t segment_seq_offset = 0;
  for (size_t i = 0; i < segment_lengths.size(); ++i) {
    line.clear();
    line += "S\t";
    append_numeric_node_name(line, static_cast<uint32_t>(i + 1));
    line += '\t';

    const uint32_t len = segment_lengths[i];
    if (segment_seq_offset + len > segment_sequences.size()) {
      throw std::runtime_error(std::string(kWriterErrorPrefix) +
                               "segment sequence column is truncated");
    }
    line.append(segment_sequences, segment_seq_offset, len);
    segment_seq_offset += len;

    line += format_optional_fields(segment_optional_fields, segment_offsets, i);
    line += '\n';
    out.write(line.data(), static_cast<std::streamsize>(line.size()));
  }
}

void write_links_numeric(
    std::ofstream &out, const std::vector<uint32_t> &link_from_ids,
    const std::vector<uint32_t> &link_to_ids,
    const std::vector<char> &link_from_orients,
    const std::vector<char> &link_to_orients,
    const std::vector<uint32_t> &link_overlap_nums,
    const std::vector<char> &link_overlap_ops,
    const std::vector<gfaz::OptionalFieldColumn> &link_optional_fields,
    const FieldOffsets &link_offsets) {
  std::string line;
  line.reserve(4096);
  for (size_t i = 0; i < link_from_ids.size(); ++i) {
    line.clear();
    line += "L\t";
    append_numeric_node_name(line, link_from_ids[i]);
    line += '\t';
    line += link_from_orients[i];
    line += '\t';
    append_numeric_node_name(line, link_to_ids[i]);
    line += '\t';
    line += link_to_orients[i];
    line += '\t';
    if (i < link_overlap_ops.size() && link_overlap_ops[i] != '\0') {
      line += std::to_string(link_overlap_nums[i]);
      line += link_overlap_ops[i];
    } else {
      line += '*';
    }
    line += format_optional_fields(link_optional_fields, link_offsets, i);
    line += '\n';
    out.write(line.data(), static_cast<std::streamsize>(line.size()));
  }
}

void write_jumps_numeric(std::ofstream &out,
                         const std::vector<uint32_t> &jump_from_ids,
                         const std::vector<uint32_t> &jump_to_ids,
                         const std::vector<char> &jump_from_orients,
                         const std::vector<char> &jump_to_orients,
                         const std::vector<std::string> &jump_distances,
                         const std::vector<std::string> &jump_rest_fields) {
  std::string line;
  line.reserve(4096);
  for (size_t i = 0; i < jump_from_ids.size(); ++i) {
    line.clear();
    line += "J\t";
    append_numeric_node_name(line, jump_from_ids[i]);
    line += '\t';
    line += jump_from_orients[i];
    line += '\t';
    append_numeric_node_name(line, jump_to_ids[i]);
    line += '\t';
    line += jump_to_orients[i];
    line += '\t';
    line += (i < jump_distances.size()) ? jump_distances[i] : "*";
    if (i < jump_rest_fields.size() && !jump_rest_fields[i].empty()) {
      line += '\t';
      line += jump_rest_fields[i];
    }
    line += '\n';
    out.write(line.data(), static_cast<std::streamsize>(line.size()));
  }
}

void write_containments_numeric(
    std::ofstream &out,
    const std::vector<uint32_t> &containment_container_ids,
    const std::vector<uint32_t> &containment_contained_ids,
    const std::vector<char> &containment_container_orients,
    const std::vector<char> &containment_contained_orients,
    const std::vector<uint32_t> &containment_positions,
    const std::vector<std::string> &containment_overlaps,
    const std::vector<std::string> &containment_rest_fields) {
  std::string line;
  line.reserve(4096);
  for (size_t i = 0; i < containment_container_ids.size(); ++i) {
    line.clear();
    line += "C\t";
    append_numeric_node_name(line, containment_container_ids[i]);
    line += '\t';
    line += containment_container_orients[i];
    line += '\t';
    append_numeric_node_name(line, containment_contained_ids[i]);
    line += '\t';
    line += containment_contained_orients[i];
    line += '\t';
    line += std::to_string(containment_positions[i]);
    line += '\t';
    line += (i < containment_overlaps.size()) ? containment_overlaps[i] : "*";
    if (i < containment_rest_fields.size() &&
        !containment_rest_fields[i].empty()) {
      line += '\t';
      line += containment_rest_fields[i];
    }
    line += '\n';
    out.write(line.data(), static_cast<std::streamsize>(line.size()));
  }
}

} // namespace gfaz::gfa_write_utils
