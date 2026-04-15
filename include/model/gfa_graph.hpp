#ifndef MODEL_GFA_GRAPH_HPP
#define MODEL_GFA_GRAPH_HPP

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

// Signed node ID where sign encodes orientation: positive = forward, negative = reverse.
// Uses 1-based indexing to allow sign to encode orientation (0 would be ambiguous).
using NodeId = int32_t;

struct OptionalFieldColumn {
  std::string tag;  // 2-char tag (e.g., "LN", "RC")
  char type;        // SAM-style type: 'A', 'i', 'f', 'Z', 'J', 'H', 'B'

  // Type-specific storage (only one populated based on 'type')
  std::vector<int64_t> int_values;
  std::vector<float> float_values;
  std::vector<char> char_values;
  std::string concatenated_strings;
  std::vector<uint32_t> string_lengths;

  // Type 'B' (byte array) storage
  std::vector<char> b_subtypes;
  std::vector<uint32_t> b_lengths;
  std::vector<uint8_t> b_concat_bytes;
};

struct LinkData {
  std::vector<uint32_t> from_ids;
  std::vector<uint32_t> to_ids;
  std::vector<char> from_orients;
  std::vector<char> to_orients;
  std::vector<uint32_t> overlap_nums;
  std::vector<char> overlap_ops;
};

struct WalkData {
  std::vector<std::vector<NodeId>> walks;
  std::vector<std::string> sample_ids;
  std::vector<uint32_t> hap_indices;
  std::vector<std::string> seq_ids;
  std::vector<int64_t> seq_starts;
  std::vector<int64_t> seq_ends;

  size_t size() const { return walks.size(); }
};

// J-line columnar storage
struct JumpData {
  std::vector<uint32_t> from_ids;
  std::vector<char> from_orients;
  std::vector<uint32_t> to_ids;
  std::vector<char> to_orients;
  std::vector<std::string> distances;     // "*" or numeric string
  std::vector<std::string> rest_fields;   // Optional fields after distance

  size_t size() const { return from_ids.size(); }
};

// C-line columnar storage
struct ContainmentData {
  std::vector<uint32_t> container_ids;
  std::vector<char> container_orients;
  std::vector<uint32_t> contained_ids;
  std::vector<char> contained_orients;
  std::vector<uint32_t> positions;
  std::vector<std::string> overlaps;      // CIGAR string
  std::vector<std::string> rest_fields;   // Optional fields after overlap

  size_t size() const { return container_ids.size(); }
};

struct SegmentData {
  // Index 0 is a placeholder for 1-based node IDs.
  std::vector<std::string> node_id_to_name;
  std::vector<std::string> node_sequences;
  std::vector<OptionalFieldColumn> optional_fields;

  size_t size() const {
    return node_sequences.empty() ? 0 : node_sequences.size() - 1;
  }

  bool valid() const { return node_id_to_name.size() == node_sequences.size(); }
};

struct PathData {
  std::vector<std::string> names;
  std::vector<std::vector<NodeId>> traversals;
  std::vector<std::string> overlaps;

  size_t size() const { return traversals.size(); }

  bool valid() const {
    return names.size() == traversals.size() &&
           overlaps.size() == traversals.size();
  }
};

struct GfaGraph {
  std::string header_line;

  // Segments (S-lines)
  std::unordered_map<std::string, uint32_t> node_name_to_id;
  SegmentData segments;

  // Paths (P-lines)
  PathData paths_data;

  // Walks (W-lines)
  WalkData walks;

  // Links (L-lines)
  LinkData links;
  std::vector<OptionalFieldColumn> link_optional_fields;

  // Jumps (J-lines)
  JumpData jumps;

  // Containments (C-lines)
  ContainmentData containments;
};

struct LineOffset {
  size_t offset;
  size_t length;
};

#endif // MODEL_GFA_GRAPH_HPP
