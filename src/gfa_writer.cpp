#include "gfa_writer.hpp"
#include "debug_log.hpp"

#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>

namespace {

constexpr const char *kWriterErrorPrefix = "GFA writer error: ";

// Format float without unnecessary trailing zeros
std::string format_float(float val) {
  std::ostringstream oss;
  oss << val;
  return oss.str();
}

// Precomputed offsets for string and B-type fields in optional columns
using FieldOffsets = std::vector<std::vector<size_t>>;

// Get byte size per element for B-type subtype
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

FieldOffsets build_field_offsets(const std::vector<OptionalFieldColumn> &cols) {
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

// Format optional fields for a given index
std::string format_optional_fields(const std::vector<OptionalFieldColumn> &cols,
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

// Get node name or numeric ID as string
inline void append_node_name(std::string &out, uint32_t node_id,
                             const std::vector<std::string> &names) {
  if (node_id < names.size() && !names[node_id].empty())
    out += names[node_id];
  else
    out += std::to_string(node_id);
}

} // namespace

void write_gfa(const GfaGraph &graph, const std::string &output_path) {
  std::ofstream out(output_path);
  if (!out)
    throw std::runtime_error(std::string(kWriterErrorPrefix) +
                             "failed to open output file: " + output_path);

  GFAZ_LOG("Writing GFA to " << output_path << "...");

  // Precompute string offsets for optional fields
  const FieldOffsets seg_offsets =
      build_field_offsets(graph.segment_optional_fields);
  const FieldOffsets link_offsets =
      build_field_offsets(graph.link_optional_fields);

  std::string line;
  line.reserve(4096);

  // H (Header)
  if (!graph.header_line.empty())
    out << graph.header_line << "\n";

  // S (Segments) - index 0 is placeholder, segments start at 1
  size_t num_segments = graph.node_sequences.size();
  for (size_t i = 1; i < num_segments; ++i) {
    line.clear();
    line += "S\t";
    append_node_name(line, i, graph.node_id_to_name);
    line += '\t';
    line += graph.node_sequences[i];
    line += format_optional_fields(graph.segment_optional_fields, seg_offsets,
                                   i - 1);
    line += '\n';
    out.write(line.data(), line.size());

    if (i % 10000000 == 0) {
      GFAZ_LOG("  Segments: " << i << "/" << (num_segments - 1));
      out.flush();
    }
  }

  // L (Links)
  size_t num_links = graph.links.from_ids.size();
  for (size_t i = 0; i < num_links; ++i) {
    line.clear();
    line += "L\t";

    append_node_name(line, graph.links.from_ids[i], graph.node_id_to_name);
    line += '\t';
    line += graph.links.from_orients[i];
    line += '\t';

    append_node_name(line, graph.links.to_ids[i], graph.node_id_to_name);
    line += '\t';
    line += graph.links.to_orients[i];
    line += '\t';

    if (graph.links.overlap_ops[i] != '\0') {
      line += std::to_string(graph.links.overlap_nums[i]);
      line += graph.links.overlap_ops[i];
    } else {
      line += '*';
    }

    line += format_optional_fields(graph.link_optional_fields, link_offsets, i);
    line += '\n';
    out.write(line.data(), line.size());

    if ((i + 1) % 10000000 == 0) {
      GFAZ_LOG("  Links: " << (i + 1) << "/" << num_links);
      out.flush();
    }
  }

  // P (Paths)
  for (size_t p = 0; p < graph.paths.size(); ++p) {
    line.clear();
    line += "P\t";
    line += graph.path_names[p];
    line += '\t';

    const auto &path = graph.paths[p];
    for (size_t n = 0; n < path.size(); ++n) {
      if (n > 0)
        line += ',';

      NodeId node = path[n];
      bool is_reverse = (node < 0);
      uint32_t node_id = is_reverse ? static_cast<uint32_t>(-node)
                                    : static_cast<uint32_t>(node);
      append_node_name(line, node_id, graph.node_id_to_name);
      line += (is_reverse ? '-' : '+');
    }

    line += '\t';
    if (p < graph.path_overlaps.size() && !graph.path_overlaps[p].empty())
      line += graph.path_overlaps[p];
    else
      line += '*';
    line += '\n';

    out.write(line.data(), line.size());
  }

  // W (Walks)
  for (size_t w = 0; w < graph.walks.walks.size(); ++w) {
    line.clear();
    line += "W\t";

    // Sample ID
    if (w < graph.walks.sample_ids.size())
      line += graph.walks.sample_ids[w];
    else
      line += "sample";
    line += '\t';

    // Haplotype index
    if (w < graph.walks.hap_indices.size())
      line += std::to_string(graph.walks.hap_indices[w]);
    else
      line += "0";
    line += '\t';

    // Sequence ID
    if (w < graph.walks.seq_ids.size())
      line += graph.walks.seq_ids[w];
    else
      line += "unknown";
    line += '\t';

    // Sequence start
    if (w < graph.walks.seq_starts.size() && graph.walks.seq_starts[w] >= 0)
      line += std::to_string(graph.walks.seq_starts[w]);
    else
      line += '*';
    line += '\t';

    // Sequence end
    if (w < graph.walks.seq_ends.size() && graph.walks.seq_ends[w] >= 0)
      line += std::to_string(graph.walks.seq_ends[w]);
    else
      line += '*';
    line += '\t';

    // Walk nodes (>node or <node format)
    const auto &walk = graph.walks.walks[w];
    for (size_t n = 0; n < walk.size(); ++n) {
      NodeId node = walk[n];
      bool is_reverse = (node < 0);
      uint32_t node_id = is_reverse ? static_cast<uint32_t>(-node)
                                    : static_cast<uint32_t>(node);
      line += (is_reverse ? '<' : '>');
      append_node_name(line, node_id, graph.node_id_to_name);
    }
    line += '\n';

    out.write(line.data(), line.size());

    if ((w + 1) % 10000 == 0) {
      GFAZ_LOG("  Walks: " << (w + 1) << "/" << graph.walks.walks.size());
      out.flush();
    }
  }

  // J (Jump) lines
  for (size_t i = 0; i < graph.jumps.size(); ++i) {
    line.clear();
    line += "J\t";

    append_node_name(line, graph.jumps.from_ids[i], graph.node_id_to_name);
    line += '\t';
    line += graph.jumps.from_orients[i];
    line += '\t';

    append_node_name(line, graph.jumps.to_ids[i], graph.node_id_to_name);
    line += '\t';
    line += graph.jumps.to_orients[i];
    line += '\t';

    line += graph.jumps.distances[i];

    if (i < graph.jumps.rest_fields.size() &&
        !graph.jumps.rest_fields[i].empty()) {
      line += '\t';
      line += graph.jumps.rest_fields[i];
    }
    line += '\n';
    out.write(line.data(), line.size());
  }

  // C (Containment) lines
  for (size_t i = 0; i < graph.containments.size(); ++i) {
    line.clear();
    line += "C\t";

    append_node_name(line, graph.containments.container_ids[i],
                     graph.node_id_to_name);
    line += '\t';
    line += graph.containments.container_orients[i];
    line += '\t';

    append_node_name(line, graph.containments.contained_ids[i],
                     graph.node_id_to_name);
    line += '\t';
    line += graph.containments.contained_orients[i];
    line += '\t';

    line += std::to_string(graph.containments.positions[i]);
    line += '\t';
    line += graph.containments.overlaps[i];

    if (i < graph.containments.rest_fields.size() &&
        !graph.containments.rest_fields[i].empty()) {
      line += '\t';
      line += graph.containments.rest_fields[i];
    }
    line += '\n';
    out.write(line.data(), line.size());
  }

  out.close();

  // Report file size
  std::ifstream check(output_path, std::ios::binary | std::ios::ate);
  size_t file_size = check.tellg();
  GFAZ_LOG("Wrote GFA file: " << output_path << " (" << file_size << " bytes)");
}
