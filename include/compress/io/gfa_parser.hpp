#ifndef GFA_PARSER_HPP
#define GFA_PARSER_HPP

#include <cstdint>
#include <memory_resource>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>


#include "compress/io/traversal_sink.hpp"
#include "core/model/gfa_graph.hpp"

// Allocator that leaves trivially constructible elements uninitialized, so a
// large vector can be sized once and then filled (and first touched) by
// several threads.
template <typename T> struct DefaultInitAllocator : std::allocator<T> {
  template <typename U> struct rebind {
    using other = DefaultInitAllocator<U>;
  };
  DefaultInitAllocator() = default;
  template <typename U>
  DefaultInitAllocator(const DefaultInitAllocator<U> &) noexcept {}
  template <typename U> void construct(U *p) { ::new (static_cast<void *>(p)) U; }
  template <typename U, typename... Args> void construct(U *p, Args &&...args) {
    ::new (static_cast<void *>(p)) U(std::forward<Args>(args)...);
  }
};

using LineOffsets =
    std::vector<gfaz::LineOffset, DefaultInitAllocator<gfaz::LineOffset>>;

class GfaParser {
public:
  GfaParser();
  // With a `traversal_sink`, path/walk traversals are handed to the sink
  // instead of being stored in the returned graph (names and other path/walk
  // metadata are still stored).
  gfaz::GfaGraph parse(const std::string &gfa_file_path, int num_threads = 0,
                       bool direct_segment_columns = false,
                       gfaz::TraversalSink *traversal_sink = nullptr);

private:
  std::unordered_map<uint16_t, std::pair<char, size_t>> segment_field_meta_;
  std::unordered_map<uint16_t, std::pair<char, size_t>> link_field_meta_;
  // The name lookup is only needed while parsing. Its nodes come from a pool
  // so they are not interleaved on the heap with segment strings that outlive
  // the parser; otherwise freeing the map leaves that memory fragmented and
  // resident for the rest of the compression.
  std::pmr::monotonic_buffer_resource name_lookup_pool_;
  std::pmr::unordered_map<std::string_view, uint32_t> node_name_lookup_{
      &name_lookup_pool_};
  // Set when every segment name is a canonical decimal number but the names
  // are not simply 1..N in file order: maps the number to the segment ID, so
  // resolving a name is an array read instead of a string hash lookup.
  std::vector<uint32_t> numeric_name_table_;
  // Segment names of the chunked parser when they are not all numbers;
  // node_name_lookup_ keys point into it.
  std::string segment_name_arena_;
  size_t num_segments_hint_ = 0;
  size_t num_links_hint_ = 0;
  gfaz::TraversalSink *traversal_sink_ = nullptr;

  // Fast-path optimization: when all segment names are sequential integers (1,2,3...),
  // skip hash map lookups and convert names directly to IDs.
  bool all_segment_names_numeric_ = true;

  void parse_s_line(std::string_view line, gfaz::GfaGraph &graph);
  void parse_l_line(std::string_view line, gfaz::GfaGraph &graph);
  void parse_p_line(std::string_view line, gfaz::GfaGraph &graph, size_t index);
  void parse_w_line(std::string_view line, gfaz::GfaGraph &graph, size_t index);

  // Parallel parsers for the general (non-direct) mode. Each returns false,
  // leaving the graph as it found it, for any input the serial line parsers
  // treat specially; the caller then runs those.
  bool parse_segments_in_chunks(const char *data,
                                const LineOffsets &offsets,
                                size_t page_size, bool direct_segment_columns,
                                gfaz::GfaGraph &graph);
  bool parse_links_in_chunks(const char *data,
                             const LineOffsets &offsets,
                             size_t page_size, gfaz::GfaGraph &graph);

  void parse_segment_field(std::string_view field, size_t segment_index, gfaz::GfaGraph &graph);
  void parse_link_field(std::string_view field, size_t link_index, gfaz::GfaGraph &graph);
  void parse_j_line(std::string_view line, gfaz::GfaGraph &graph);
  void parse_c_line(std::string_view line, gfaz::GfaGraph &graph);
  uint32_t resolve_node_id(std::string_view node_name_view) const;
  static uint16_t field_tag_key(std::string_view field);

  static bool is_numeric(std::string_view s);
};

#endif
