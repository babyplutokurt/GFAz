#ifndef GFA_PARSER_HPP
#define GFA_PARSER_HPP

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

struct GfaGraph {
  std::string header_line;

  // Segments (S-lines) - index 0 is placeholder for 1-based node IDs.
  // In the CPU .gfaz format, decompression intentionally reconstructs
  // segments with dense numeric names "1", "2", ..., so original segment
  // names are not preserved there. The graph model still stores names because
  // the parser and writer operate on full-fidelity in-memory GFAs.
  std::unordered_map<std::string, uint32_t> node_name_to_id;
  std::vector<std::string> node_id_to_name;
  std::vector<std::string> node_sequences;
  std::vector<OptionalFieldColumn> segment_optional_fields;

  // Paths (P-lines)
  std::vector<std::string> path_names;
  std::vector<std::vector<NodeId>> paths;
  std::vector<std::string> path_overlaps;

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

class GfaParser {
public:
  GfaParser();
  GfaGraph parse(const std::string &gfa_file_path, int num_threads = 0);

private:
  std::unordered_map<std::string, std::pair<char, size_t>> segment_field_meta_;
  std::unordered_map<std::string, std::pair<char, size_t>> link_field_meta_;
  std::unordered_map<std::string, std::pair<size_t, size_t>> string_field_stats_;

  // Fast-path optimization: when all segment names are sequential integers (1,2,3...),
  // skip hash map lookups and convert names directly to IDs.
  bool all_segment_names_numeric_ = true;

  void prescan_optional_fields(const std::string &gfa_file_path, GfaGraph &graph);

  void parse_s_line(std::string_view line, GfaGraph &graph);
  void parse_l_line(std::string_view line, GfaGraph &graph);
  void parse_p_line(std::string_view line, GfaGraph &graph, size_t index);
  void parse_w_line(std::string_view line, GfaGraph &graph, size_t index);

  void parse_segment_field(std::string_view field, size_t segment_index, GfaGraph &graph);
  void parse_link_field(std::string_view field, size_t link_index, GfaGraph &graph);
  void parse_j_line(std::string_view line, GfaGraph &graph);
  void parse_c_line(std::string_view line, GfaGraph &graph);

  static bool is_numeric(std::string_view s);
};

#endif
