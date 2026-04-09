#include "workflows/compression_utils.hpp"
#include "codec/codec.hpp"
#include "grammar/packed_2mer.hpp"

#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace gfz::compression_utils {

using gfz::runtime_utils::format_size;

size_t total_node_count(const std::vector<std::vector<NodeId>> &sequences) {
  size_t total = 0;
  for (const auto &seq : sequences)
    total += seq.size();
  return total;
}

void append_string_column(const std::vector<std::string> &values,
                          std::string &concatenated,
                          std::vector<uint32_t> &lengths) {
  concatenated.clear();
  lengths.clear();
  for (const auto &value : values) {
    concatenated += value;
    lengths.push_back(static_cast<uint32_t>(value.size()));
  }
}

CompressedOptionalFieldColumn
compress_optional_column(const OptionalFieldColumn &col) {
  CompressedOptionalFieldColumn out;
  out.tag = col.tag;
  out.type = col.type;

  switch (col.type) {
  case 'i':
    out.num_elements = col.int_values.size();
    out.int_values_zstd = Codec::compress_varint_int64(col.int_values);
    break;
  case 'f':
    out.num_elements = col.float_values.size();
    out.float_values_zstd = Codec::zstd_compress_float_vector(col.float_values);
    break;
  case 'A':
    out.num_elements = col.char_values.size();
    out.char_values_zstd = Codec::zstd_compress_char_vector(col.char_values);
    break;
  case 'Z':
  case 'J':
  case 'H':
    out.num_elements = col.string_lengths.size();
    out.strings_zstd = Codec::zstd_compress_string(col.concatenated_strings);
    out.string_lengths_zstd =
        Codec::zstd_compress_uint32_vector(col.string_lengths);
    break;
  case 'B':
    out.num_elements = col.b_subtypes.size();
    out.b_subtypes_zstd = Codec::zstd_compress_char_vector(col.b_subtypes);
    out.b_lengths_zstd = Codec::zstd_compress_uint32_vector(col.b_lengths);
    out.b_concat_bytes_zstd = Codec::zstd_compress_string(
        std::string(col.b_concat_bytes.begin(), col.b_concat_bytes.end()));
    break;
  default:
    throw std::invalid_argument(std::string(kCompressionErrorPrefix) +
                                "unsupported optional field type '" +
                                std::string(1, col.type) + "' for tag '" +
                                col.tag + "'");
  }
  return out;
}

void flatten_paths(const std::vector<std::vector<NodeId>> &paths,
                   const std::vector<std::string> &path_names,
                   const std::vector<std::string> &path_overlaps,
                   std::vector<int32_t> &flattened,
                   std::vector<uint32_t> &lengths, std::string &names_concat,
                   std::vector<uint32_t> &name_lengths,
                   std::string &overlaps_concat,
                   std::vector<uint32_t> &overlap_lengths) {
  if (path_names.size() != paths.size() || path_overlaps.size() != paths.size())
    throw std::runtime_error(
        std::string(kCompressionErrorPrefix) +
        "invalid path metadata (names/overlaps count does not match paths)");

  const size_t total = total_node_count(paths);

  flattened.clear();
  flattened.resize(total);
  lengths.clear();

  append_string_column(path_names, names_concat, name_lengths);
  append_string_column(path_overlaps, overlaps_concat, overlap_lengths);

  size_t offset = 0;
  for (size_t i = 0; i < paths.size(); ++i) {
    const auto &path = paths[i];
    lengths.push_back(static_cast<uint32_t>(path.size()));

    for (NodeId node : path)
      flattened[offset++] = node;
  }
}

void flatten_segments(const std::vector<std::string> &sequences,
                      std::string &concat, std::vector<uint32_t> &lengths,
                      uint32_t max_id) {
  concat.clear();
  lengths.clear();
  for (size_t i = 1; i < sequences.size() && i < max_id; ++i) {
    concat += sequences[i];
    lengths.push_back(static_cast<uint32_t>(sequences[i].size()));
  }
}

void process_rules(const std::vector<Packed2mer> &rulebook,
                   uint32_t layer_start_id,
                   const std::vector<LayerRuleRange> &ranges,
                   std::vector<int32_t> &first, std::vector<int32_t> &second) {
  size_t total = 0;
  for (const auto &r : ranges)
    total += r.end_id - r.start_id;

  first.clear();
  second.clear();
  first.reserve(total);
  second.reserve(total);

  for (const auto &range : ranges) {
    for (uint32_t id = range.start_id; id < range.end_id; ++id) {
      size_t idx = id - layer_start_id;
      if (idx < rulebook.size()) {
        first.push_back(unpack_first(rulebook[idx]));
        second.push_back(unpack_second(rulebook[idx]));
      } else {
        first.push_back(0);
        second.push_back(0);
      }
    }
  }

  Codec::delta_encode_int32(first);
  Codec::delta_encode_int32(second);
}

void flatten_walks(const std::vector<std::vector<NodeId>> &walks,
                   std::vector<int32_t> &flattened,
                   std::vector<uint32_t> &lengths) {
  flattened.clear();
  flattened.reserve(total_node_count(walks));
  lengths.clear();
  lengths.reserve(walks.size());

  for (const auto &walk : walks) {
    lengths.push_back(static_cast<uint32_t>(walk.size()));
    for (NodeId node : walk)
      flattened.push_back(node);
  }
}

void remap_rule_ids(std::vector<std::vector<NodeId>> &sequences,
                    uint32_t rules_start_id,
                    const std::vector<uint32_t> &id_map) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (size_t i = 0; i < sequences.size(); ++i) {
    for (auto &node : sequences[i]) {
      int32_t abs_id = std::abs(node);
      if (static_cast<uint32_t>(abs_id) < rules_start_id)
        continue;

      const uint32_t offset = static_cast<uint32_t>(abs_id) - rules_start_id;
      if (offset >= id_map.size() || id_map[offset] == 0)
        continue;

      node = (node > 0) ? static_cast<int32_t>(id_map[offset])
                        : -static_cast<int32_t>(id_map[offset]);
    }
  }
}

void print_compression_stats(const CompressedData &d, size_t num_segments,
                             bool show_stats) {
  auto sum_optional =
      [](const std::vector<CompressedOptionalFieldColumn> &cols) {
        size_t total = 0;
        for (const auto &c : cols) {
          total += c.int_values_zstd.payload.size();
          total += c.float_values_zstd.payload.size();
          total += c.char_values_zstd.payload.size();
          total += c.strings_zstd.payload.size();
          total += c.string_lengths_zstd.payload.size();
          total += c.b_subtypes_zstd.payload.size();
          total += c.b_lengths_zstd.payload.size();
          total += c.b_concat_bytes_zstd.payload.size();
        }
        return total;
      };

  size_t seg_opt = sum_optional(d.segment_optional_fields_zstd);
  size_t link_opt = sum_optional(d.link_optional_fields_zstd);

  size_t jump_bytes = d.jump_from_ids_zstd.payload.size() +
                      d.jump_to_ids_zstd.payload.size() +
                      d.jump_from_orients_zstd.payload.size() +
                      d.jump_to_orients_zstd.payload.size() +
                      d.jump_distances_zstd.payload.size() +
                      d.jump_distance_lengths_zstd.payload.size() +
                      d.jump_rest_fields_zstd.payload.size() +
                      d.jump_rest_lengths_zstd.payload.size();

  size_t containment_bytes =
      d.containment_container_ids_zstd.payload.size() +
      d.containment_contained_ids_zstd.payload.size() +
      d.containment_container_orients_zstd.payload.size() +
      d.containment_contained_orients_zstd.payload.size() +
      d.containment_positions_zstd.payload.size() +
      d.containment_overlaps_zstd.payload.size() +
      d.containment_overlap_lengths_zstd.payload.size() +
      d.containment_rest_fields_zstd.payload.size() +
      d.containment_rest_lengths_zstd.payload.size();

  size_t total = 0;
  total += d.rules_first_zstd.payload.size();
  total += d.rules_second_zstd.payload.size();
  total += d.paths_zstd.payload.size();
  total += d.sequence_lengths.size() * sizeof(uint32_t);
  total += d.layer_rule_ranges.size() * sizeof(LayerRuleRange);
  total += d.names_zstd.payload.size();
  total += d.name_lengths_zstd.payload.size();
  total += d.overlaps_zstd.payload.size();
  total += d.overlap_lengths_zstd.payload.size();
  total += d.segment_sequences_zstd.payload.size();
  total += d.segment_seq_lengths_zstd.payload.size();
  total += seg_opt;
  total += d.link_from_ids_zstd.payload.size();
  total += d.link_to_ids_zstd.payload.size();
  total += d.link_from_orients_zstd.payload.size();
  total += d.link_to_orients_zstd.payload.size();
  total += d.link_overlap_nums_zstd.payload.size();
  total += d.link_overlap_ops_zstd.payload.size();
  total += link_opt;
  total += jump_bytes;
  total += containment_bytes;
  total += d.walks_zstd.payload.size();
  total += d.walk_lengths.size() * sizeof(uint32_t);
  total += d.walk_sample_ids_zstd.payload.size();
  total += d.walk_sample_id_lengths_zstd.payload.size();
  total += d.walk_hap_indices_zstd.payload.size();
  total += d.walk_seq_ids_zstd.payload.size();
  total += d.walk_seq_id_lengths_zstd.payload.size();
  total += d.walk_seq_starts_zstd.payload.size();
  total += d.walk_seq_ends_zstd.payload.size();

  if (!show_stats)
    return;

  std::cerr << "\n=== Compressed Data Breakdown ===" << std::endl;

  // Rules
  std::cerr << "Rules (2-mer grammar):" << std::endl;
  std::cerr << "  layer_rule_ranges:      " << std::setw(12)
            << format_size(d.layer_rule_ranges.size() * sizeof(LayerRuleRange))
            << std::endl;
  std::cerr << "  rules_first_zstd:       " << std::setw(12)
            << format_size(d.rules_first_zstd.payload.size()) << std::endl;
  std::cerr << "  rules_second_zstd:      " << std::setw(12)
            << format_size(d.rules_second_zstd.payload.size()) << std::endl;

  // Paths
  std::cerr << "Paths (P-lines): " << d.sequence_lengths.size() << std::endl;
  std::cerr << "  sequence_lengths:       " << std::setw(12)
            << format_size(d.sequence_lengths.size() * sizeof(uint32_t))
            << std::endl;
  std::cerr << "  paths_zstd:             " << std::setw(12)
            << format_size(d.paths_zstd.payload.size()) << std::endl;
  std::cerr << "  names_zstd:             " << std::setw(12)
            << format_size(d.names_zstd.payload.size()) << std::endl;
  std::cerr << "  name_lengths_zstd:      " << std::setw(12)
            << format_size(d.name_lengths_zstd.payload.size()) << std::endl;
  std::cerr << "  overlaps_zstd:          " << std::setw(12)
            << format_size(d.overlaps_zstd.payload.size()) << std::endl;
  std::cerr << "  overlap_lengths_zstd:   " << std::setw(12)
            << format_size(d.overlap_lengths_zstd.payload.size()) << std::endl;

  // Segments
  std::cerr << "Segments (S-lines): " << num_segments << std::endl;
  std::cerr << "  segment_sequences_zstd: " << std::setw(12)
            << format_size(d.segment_sequences_zstd.payload.size())
            << std::endl;
  std::cerr << "  segment_seq_lengths:    " << std::setw(12)
            << format_size(d.segment_seq_lengths_zstd.payload.size())
            << std::endl;
  std::cerr << "  segment_optional_fields:" << std::setw(12)
            << format_size(seg_opt) << " ("
            << d.segment_optional_fields_zstd.size() << " columns)"
            << std::endl;

  // Links
  std::cerr << "Links (L-lines): " << d.num_links << std::endl;
  std::cerr << "  link_from_ids_zstd:     " << std::setw(12)
            << format_size(d.link_from_ids_zstd.payload.size()) << std::endl;
  std::cerr << "  link_to_ids_zstd:       " << std::setw(12)
            << format_size(d.link_to_ids_zstd.payload.size()) << std::endl;
  std::cerr << "  link_from_orients_zstd: " << std::setw(12)
            << format_size(d.link_from_orients_zstd.payload.size())
            << std::endl;
  std::cerr << "  link_to_orients_zstd:   " << std::setw(12)
            << format_size(d.link_to_orients_zstd.payload.size()) << std::endl;
  std::cerr << "  link_overlap_nums_zstd: " << std::setw(12)
            << format_size(d.link_overlap_nums_zstd.payload.size())
            << std::endl;
  std::cerr << "  link_overlap_ops_zstd:  " << std::setw(12)
            << format_size(d.link_overlap_ops_zstd.payload.size()) << std::endl;
  std::cerr << "  link_optional_fields:   " << std::setw(12)
            << format_size(link_opt) << " ("
            << d.link_optional_fields_zstd.size() << " columns)" << std::endl;

  // J/C lines
  if (d.num_jumps > 0) {
    std::cerr << "Jumps (J-lines): " << d.num_jumps << std::endl;
    std::cerr << "  total:                  " << std::setw(12)
              << format_size(jump_bytes) << std::endl;
  }
  if (d.num_containments > 0) {
    std::cerr << "Containments (C-lines): " << d.num_containments << std::endl;
    std::cerr << "  total:                  " << std::setw(12)
              << format_size(containment_bytes) << std::endl;
  }

  // Walks
  if (!d.walk_lengths.empty()) {
    std::cerr << "Walks (W-lines): " << d.walk_lengths.size() << std::endl;
    std::cerr << "  walk_lengths:           " << std::setw(12)
              << format_size(d.walk_lengths.size() * sizeof(uint32_t))
              << std::endl;
    std::cerr << "  walks_zstd:             " << std::setw(12)
              << format_size(d.walks_zstd.payload.size()) << std::endl;
    std::cerr << "  walk_sample_ids_zstd:   " << std::setw(12)
              << format_size(d.walk_sample_ids_zstd.payload.size())
              << std::endl;
    std::cerr << "  walk_sample_id_lengths: " << std::setw(12)
              << format_size(d.walk_sample_id_lengths_zstd.payload.size())
              << std::endl;
    std::cerr << "  walk_hap_indices_zstd:  " << std::setw(12)
              << format_size(d.walk_hap_indices_zstd.payload.size())
              << std::endl;
    std::cerr << "  walk_seq_ids_zstd:      " << std::setw(12)
              << format_size(d.walk_seq_ids_zstd.payload.size()) << std::endl;
    std::cerr << "  walk_seq_id_lengths:    " << std::setw(12)
              << format_size(d.walk_seq_id_lengths_zstd.payload.size())
              << std::endl;
    std::cerr << "  walk_seq_starts_zstd:   " << std::setw(12)
              << format_size(d.walk_seq_starts_zstd.payload.size())
              << std::endl;
    std::cerr << "  walk_seq_ends_zstd:     " << std::setw(12)
              << format_size(d.walk_seq_ends_zstd.payload.size()) << std::endl;
  }

  // Total
  std::cerr << "----------------------------------------" << std::endl;
  std::cerr << "Total: " << format_size(total) << std::endl;
}

} // namespace gfz::compression_utils
