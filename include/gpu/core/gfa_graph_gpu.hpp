#ifndef GFA_GRAPH_GPU_HPP
#define GFA_GRAPH_GPU_HPP

#include "model/gfa_graph.hpp"
#include <cstdint>
#include <string>
#include <vector>

// Reusable flattened string pattern (lengths-only, compute offsets via prefix
// sum on GPU)
struct FlattenedStrings {
  std::vector<char> data;        // Concatenated chars
  std::vector<uint32_t> lengths; // Length per string

  size_t count() const { return lengths.size(); }
  size_t total_chars() const { return data.size(); }
};

// Critical for compression - flattened paths (lengths-only style)
struct FlattenedPaths {
  std::vector<int32_t> data;     // All NodeIds concatenated
  std::vector<uint32_t> lengths; // Nodes per path

  size_t num_paths() const { return lengths.size(); }
  size_t total_nodes() const { return data.size(); }
};

// GPU-friendly version of gfaz::OptionalFieldColumn
struct OptionalFieldColumn_gpu {
  std::string tag;     // Two-character tag (e.g., "LN", "RC")
  char type;           // GFA type: 'A', 'i', 'f', 'Z', 'J', 'H', 'B'
  size_t num_elements; // Number of segments/links this applies to

  // Type-specific storage (same as CPU, already contiguous)
  std::vector<int64_t> int_values; // type 'i'
  std::vector<float> float_values; // type 'f'
  std::vector<char> char_values;   // type 'A'

  // String types - flattened (lengths-only)
  FlattenedStrings strings; // types 'Z', 'J', 'H'

  // Type 'B' (byte array) storage (lengths-only)
  std::vector<char> b_subtypes;    // c/C/s/S/i/I/f per segment
  std::vector<uint32_t> b_lengths; // Element count per segment
  std::vector<uint8_t> b_data;     // All array data concatenated
};

// GPU-friendly gfaz::GfaGraph structure
// All nested containers are flattened for GPU memory access patterns
struct GfaGraph_gpu {
  // ====== METADATA ======
  std::string header_line;   // Header (small, kept as-is)
  uint32_t num_segments = 0; // Number of segments (nodes)
  uint32_t num_paths = 0;    // Number of P-line paths
  uint32_t num_walks = 0;    // Number of W-line walks
  uint32_t num_links = 0;    // Number of links

  // ====== SEGMENT (NODE) DATA ======
  FlattenedStrings node_names;     // Flattened node_id_to_name
  FlattenedStrings node_sequences; // Flattened node_sequences

  // ====== PATH + WALK DATA (concatenated: paths first, then walks) ======
  // The first num_paths entries in paths.lengths are P-line paths,
  // the remaining num_walks entries are W-line walks.
  // GPU grammar compression operates on the entire concatenated data vector.
  FlattenedPaths paths;           // All paths+walks concatenated
  FlattenedStrings path_names;    // Flattened path names (P-lines only)
  FlattenedStrings path_overlaps; // Flattened path overlaps (P-lines only)

  // ====== WALK METADATA (W-lines only) ======
  FlattenedStrings walk_sample_ids;       // sample_id per walk
  std::vector<uint32_t> walk_hap_indices; // hap_index per walk
  FlattenedStrings walk_seq_ids;          // seq_id per walk
  std::vector<int64_t> walk_seq_starts;   // seq_start per walk
  std::vector<int64_t> walk_seq_ends;     // seq_end per walk

  // ====== LINK DATA (already columnar - direct copy from gfaz::LinkData) ======
  std::vector<uint32_t> link_from_ids;
  std::vector<uint32_t> link_to_ids;
  std::vector<char> link_from_orients;
  std::vector<char> link_to_orients;
  std::vector<uint32_t> link_overlap_nums;
  std::vector<char> link_overlap_ops;

  // ====== OPTIONAL FIELDS ======
  std::vector<OptionalFieldColumn_gpu> segment_optional_fields;
  std::vector<OptionalFieldColumn_gpu> link_optional_fields;

  // ====== JUMP DATA (J-lines) - structured columnar, matching CPU backend
  // ======
  std::vector<uint32_t> jump_from_ids;
  std::vector<char> jump_from_orients;
  std::vector<uint32_t> jump_to_ids;
  std::vector<char> jump_to_orients;
  FlattenedStrings jump_distances;   // "*" or numeric string per jump
  FlattenedStrings jump_rest_fields; // Optional fields after distance

  // ====== CONTAINMENT DATA (C-lines) - structured columnar, matching CPU
  // backend ======
  std::vector<uint32_t> containment_container_ids;
  std::vector<char> containment_container_orients;
  std::vector<uint32_t> containment_contained_ids;
  std::vector<char> containment_contained_orients;
  std::vector<uint32_t> containment_positions;
  FlattenedStrings containment_overlaps;    // CIGAR string per containment
  FlattenedStrings containment_rest_fields; // Optional fields after overlap

  // Convenience methods
  size_t paths_total_elements() const { return paths.total_nodes(); }
  size_t segments_total_chars() const { return node_sequences.total_chars(); }
  size_t num_jumps() const { return jump_from_ids.size(); }
  size_t num_containments() const { return containment_container_ids.size(); }
};

// Conversion functions
GfaGraph_gpu convert_to_gpu_layout(const gfaz::GfaGraph &graph);
gfaz::GfaGraph convert_from_gpu_layout(const GfaGraph_gpu &gpu_graph);

#endif // GFA_GRAPH_GPU_HPP
