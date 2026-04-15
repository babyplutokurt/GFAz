#ifndef MODEL_COMPRESSED_DATA_HPP
#define MODEL_COMPRESSED_DATA_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace gfaz {

struct LayerRuleRange {
  int k;                   // k-mer size (always 2 for 2-mer grammar)
  uint32_t start_id;       // First rule ID in this layer
  uint32_t end_id;         // One past last rule ID
  size_t flattened_offset; // Offset in flattened rule array
  size_t element_count;    // Number of elements (rule_count * k)
};

struct ZstdCompressedBlock {
  std::vector<uint8_t> payload;
  size_t original_size = 0;
};

struct CompressedOptionalFieldColumn {
  std::string tag;
  char type = '\0';
  size_t num_elements = 0;

  ZstdCompressedBlock int_values_zstd;
  ZstdCompressedBlock float_values_zstd;
  ZstdCompressedBlock char_values_zstd;
  ZstdCompressedBlock strings_zstd;
  ZstdCompressedBlock string_lengths_zstd;

  // Type 'B' (byte array)
  ZstdCompressedBlock b_subtypes_zstd;
  ZstdCompressedBlock b_lengths_zstd;
  ZstdCompressedBlock b_concat_bytes_zstd;
};

struct CompressedData {
  std::string header_line;

  // Grammar rules (2-mer)
  std::vector<LayerRuleRange> layer_rule_ranges;
  ZstdCompressedBlock rules_first_zstd;  // First elements of each 2-mer
  ZstdCompressedBlock rules_second_zstd; // Second elements of each 2-mer
  int delta_round = 1;

  uint32_t min_rule_id() const {
    return layer_rule_ranges.empty() ? 0 : layer_rule_ranges[0].start_id;
  }

  size_t total_rules() const {
    size_t count = 0;
    for (const auto &range : layer_rule_ranges)
      count += (range.end_id - range.start_id);
    return count;
  }

  // Paths (P-lines)
  std::vector<uint32_t> sequence_lengths; // Compressed lengths (after grammar)
  std::vector<uint32_t>
      original_path_lengths; // Original lengths (before grammar)
  ZstdCompressedBlock paths_zstd;
  ZstdCompressedBlock names_zstd;
  ZstdCompressedBlock name_lengths_zstd;
  ZstdCompressedBlock overlaps_zstd;
  ZstdCompressedBlock overlap_lengths_zstd;

  // Segments (S-lines)
  ZstdCompressedBlock segment_sequences_zstd;
  ZstdCompressedBlock segment_seq_lengths_zstd;
  std::vector<CompressedOptionalFieldColumn> segment_optional_fields_zstd;

  // Links (L-lines)
  size_t num_links = 0;
  ZstdCompressedBlock link_from_ids_zstd;
  ZstdCompressedBlock link_to_ids_zstd;
  ZstdCompressedBlock link_from_orients_zstd;
  ZstdCompressedBlock link_to_orients_zstd;
  ZstdCompressedBlock link_overlap_nums_zstd;
  ZstdCompressedBlock link_overlap_ops_zstd;
  std::vector<CompressedOptionalFieldColumn> link_optional_fields_zstd;

  // J-lines (Jump) - columnar storage
  size_t num_jumps = 0;
  ZstdCompressedBlock jump_from_ids_zstd;
  ZstdCompressedBlock jump_from_orients_zstd;
  ZstdCompressedBlock jump_to_ids_zstd;
  ZstdCompressedBlock jump_to_orients_zstd;
  ZstdCompressedBlock jump_distances_zstd;
  ZstdCompressedBlock jump_distance_lengths_zstd;
  ZstdCompressedBlock jump_rest_fields_zstd;
  ZstdCompressedBlock jump_rest_lengths_zstd;

  // C-lines (Containment) - columnar storage
  size_t num_containments = 0;
  ZstdCompressedBlock containment_container_ids_zstd;
  ZstdCompressedBlock containment_container_orients_zstd;
  ZstdCompressedBlock containment_contained_ids_zstd;
  ZstdCompressedBlock containment_contained_orients_zstd;
  ZstdCompressedBlock containment_positions_zstd;
  ZstdCompressedBlock containment_overlaps_zstd;
  ZstdCompressedBlock containment_overlap_lengths_zstd;
  ZstdCompressedBlock containment_rest_fields_zstd;
  ZstdCompressedBlock containment_rest_lengths_zstd;

  // Walks (W-lines)
  std::vector<uint32_t> walk_lengths; // Compressed lengths (after grammar)
  std::vector<uint32_t>
      original_walk_lengths; // Original lengths (before grammar)
  ZstdCompressedBlock walks_zstd;
  ZstdCompressedBlock walk_sample_ids_zstd;
  ZstdCompressedBlock walk_sample_id_lengths_zstd;
  ZstdCompressedBlock walk_hap_indices_zstd;
  ZstdCompressedBlock walk_seq_ids_zstd;
  ZstdCompressedBlock walk_seq_id_lengths_zstd;
  ZstdCompressedBlock walk_seq_starts_zstd;
  ZstdCompressedBlock walk_seq_ends_zstd;
};

} // namespace gfaz

#endif // MODEL_COMPRESSED_DATA_HPP
