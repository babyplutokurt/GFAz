#ifndef GFA_PARSER_HPP
#define GFA_PARSER_HPP

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>


#include "model/gfa_graph.hpp"

class GfaParser {
public:
  GfaParser();
  GfaGraph parse(const std::string &gfa_file_path, int num_threads = 0);

private:
  std::unordered_map<uint16_t, std::pair<char, size_t>> segment_field_meta_;
  std::unordered_map<uint16_t, std::pair<char, size_t>> link_field_meta_;
  std::unordered_map<std::string_view, uint32_t> node_name_lookup_;
  size_t num_segments_hint_ = 0;
  size_t num_links_hint_ = 0;

  // Fast-path optimization: when all segment names are sequential integers (1,2,3...),
  // skip hash map lookups and convert names directly to IDs.
  bool all_segment_names_numeric_ = true;

  void parse_s_line(std::string_view line, GfaGraph &graph);
  void parse_l_line(std::string_view line, GfaGraph &graph);
  void parse_p_line(std::string_view line, GfaGraph &graph, size_t index);
  void parse_w_line(std::string_view line, GfaGraph &graph, size_t index);

  void parse_segment_field(std::string_view field, size_t segment_index, GfaGraph &graph);
  void parse_link_field(std::string_view field, size_t link_index, GfaGraph &graph);
  void parse_j_line(std::string_view line, GfaGraph &graph);
  void parse_c_line(std::string_view line, GfaGraph &graph);
  uint32_t resolve_node_id(std::string_view node_name_view) const;
  static uint16_t field_tag_key(std::string_view field);

  static bool is_numeric(std::string_view s);
};

#endif

