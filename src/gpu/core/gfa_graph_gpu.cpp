#include "gpu/core/gfa_graph_gpu.hpp"

namespace {

// Helper: Flatten vector<string> into FlattenedStrings (lengths-only)
FlattenedStrings
flatten_string_vector(const std::vector<std::string> &strings) {
  FlattenedStrings result;

  // Pre-calculate total size for single allocation
  size_t total_chars = 0;
  for (const auto &s : strings) {
    total_chars += s.size();
  }

  result.data.reserve(total_chars);
  result.lengths.reserve(strings.size());

  for (const auto &s : strings) {
    result.lengths.push_back(static_cast<uint32_t>(s.size()));
    result.data.insert(result.data.end(), s.begin(), s.end());
  }

  return result;
}

// Helper: Convert OptionalFieldColumn to GPU-friendly version
OptionalFieldColumn_gpu convert_optional_field(const OptionalFieldColumn &col) {
  OptionalFieldColumn_gpu result;
  result.tag = col.tag;
  result.type = col.type;

  switch (col.type) {
  case 'i':
    result.num_elements = col.int_values.size();
    result.int_values = col.int_values; // Direct copy - already contiguous
    break;
  case 'f':
    result.num_elements = col.float_values.size();
    result.float_values = col.float_values;
    break;
  case 'A':
    result.num_elements = col.char_values.size();
    result.char_values = col.char_values;
    break;
  case 'Z':
  case 'J':
  case 'H':
    // String types - flatten with lengths only
    result.num_elements = col.string_lengths.size();
    result.strings.data.assign(col.concatenated_strings.begin(),
                               col.concatenated_strings.end());
    result.strings.lengths.reserve(col.string_lengths.size());
    for (uint32_t len : col.string_lengths) {
      result.strings.lengths.push_back(len);
    }
    break;
  case 'B':
    result.num_elements = col.b_subtypes.size();
    result.b_subtypes = col.b_subtypes;
    result.b_lengths = col.b_lengths;
    result.b_data = col.b_concat_bytes;
    break;
  default:
    result.num_elements = 0;
    break;
  }

  return result;
}

} // anonymous namespace

GfaGraph_gpu convert_to_gpu_layout(const GfaGraph &graph) {
  GfaGraph_gpu gpu;

  // ====== METADATA ======
  gpu.header_line = graph.header_line;
  gpu.num_segments = static_cast<uint32_t>(graph.node_id_to_name.size());
  gpu.num_paths = static_cast<uint32_t>(graph.paths.size());
  gpu.num_walks = static_cast<uint32_t>(graph.walks.walks.size());
  gpu.num_links = static_cast<uint32_t>(graph.links.from_ids.size());

  // ====== SEGMENT DATA ======
  gpu.node_names = flatten_string_vector(graph.node_id_to_name);
  gpu.node_sequences = flatten_string_vector(graph.node_sequences);

  // ====== PATH + WALK DATA (concatenated: paths first, then walks) ======
  {
    size_t total_path_nodes = 0;
    for (const auto &p : graph.paths)
      total_path_nodes += p.size();
    size_t total_walk_nodes = 0;
    for (const auto &w : graph.walks.walks)
      total_walk_nodes += w.size();

    gpu.paths.data.reserve(total_path_nodes + total_walk_nodes);
    gpu.paths.lengths.reserve(graph.paths.size() + graph.walks.walks.size());

    // Paths first
    for (const auto &p : graph.paths) {
      gpu.paths.lengths.push_back(static_cast<uint32_t>(p.size()));
      gpu.paths.data.insert(gpu.paths.data.end(), p.begin(), p.end());
    }
    // Walks second
    for (const auto &w : graph.walks.walks) {
      gpu.paths.lengths.push_back(static_cast<uint32_t>(w.size()));
      gpu.paths.data.insert(gpu.paths.data.end(), w.begin(), w.end());
    }
  }

  // ====== PATH METADATA (P-lines only) ======
  gpu.path_names = flatten_string_vector(graph.path_names);
  gpu.path_overlaps = flatten_string_vector(graph.path_overlaps);

  // ====== WALK METADATA (W-lines only) ======
  gpu.walk_sample_ids = flatten_string_vector(graph.walks.sample_ids);
  gpu.walk_hap_indices = graph.walks.hap_indices;
  gpu.walk_seq_ids = flatten_string_vector(graph.walks.seq_ids);
  gpu.walk_seq_starts = graph.walks.seq_starts;
  gpu.walk_seq_ends = graph.walks.seq_ends;

  // ====== LINK DATA ======
  gpu.link_from_ids = graph.links.from_ids;
  gpu.link_to_ids = graph.links.to_ids;
  gpu.link_from_orients = graph.links.from_orients;
  gpu.link_to_orients = graph.links.to_orients;
  gpu.link_overlap_nums = graph.links.overlap_nums;
  gpu.link_overlap_ops = graph.links.overlap_ops;

  // ====== OPTIONAL FIELDS ======
  gpu.segment_optional_fields.reserve(graph.segment_optional_fields.size());
  for (const auto &col : graph.segment_optional_fields) {
    gpu.segment_optional_fields.push_back(convert_optional_field(col));
  }

  gpu.link_optional_fields.reserve(graph.link_optional_fields.size());
  for (const auto &col : graph.link_optional_fields) {
    gpu.link_optional_fields.push_back(convert_optional_field(col));
  }

  // ====== JUMP DATA (J-lines) - structured columnar ======
  gpu.jump_from_ids = graph.jumps.from_ids;
  gpu.jump_from_orients = graph.jumps.from_orients;
  gpu.jump_to_ids = graph.jumps.to_ids;
  gpu.jump_to_orients = graph.jumps.to_orients;
  gpu.jump_distances = flatten_string_vector(graph.jumps.distances);
  gpu.jump_rest_fields = flatten_string_vector(graph.jumps.rest_fields);

  // ====== CONTAINMENT DATA (C-lines) - structured columnar ======
  gpu.containment_container_ids = graph.containments.container_ids;
  gpu.containment_container_orients = graph.containments.container_orients;
  gpu.containment_contained_ids = graph.containments.contained_ids;
  gpu.containment_contained_orients = graph.containments.contained_orients;
  gpu.containment_positions = graph.containments.positions;
  gpu.containment_overlaps = flatten_string_vector(graph.containments.overlaps);
  gpu.containment_rest_fields =
      flatten_string_vector(graph.containments.rest_fields);

  return gpu;
}

// ====== REVERSE CONVERSION: GfaGraph_gpu -> GfaGraph ======

namespace {

// Helper: Unflatten FlattenedStrings back to vector<string>
std::vector<std::string> unflatten_strings(const FlattenedStrings &flat) {
  std::vector<std::string> result;
  result.reserve(flat.lengths.size());

  size_t offset = 0;
  for (uint32_t len : flat.lengths) {
    result.emplace_back(flat.data.begin() + offset,
                        flat.data.begin() + offset + len);
    offset += len;
  }

  return result;
}

// Helper: Convert GPU optional field back to CPU format
OptionalFieldColumn
convert_optional_field_from_gpu(const OptionalFieldColumn_gpu &gpu_col) {
  OptionalFieldColumn result;
  result.tag = gpu_col.tag;
  result.type = gpu_col.type;

  switch (gpu_col.type) {
  case 'i':
    result.int_values = gpu_col.int_values;
    break;
  case 'f':
    result.float_values = gpu_col.float_values;
    break;
  case 'A':
    result.char_values = gpu_col.char_values;
    break;
  case 'Z':
  case 'J':
  case 'H':
    // String types - unflatten
    result.concatenated_strings.assign(gpu_col.strings.data.begin(),
                                       gpu_col.strings.data.end());
    result.string_lengths = gpu_col.strings.lengths;
    break;
  case 'B':
    result.b_subtypes = gpu_col.b_subtypes;
    result.b_lengths = gpu_col.b_lengths;
    result.b_concat_bytes = gpu_col.b_data;
    break;
  default:
    break;
  }

  return result;
}

} // anonymous namespace

GfaGraph convert_from_gpu_layout(const GfaGraph_gpu &gpu) {
  GfaGraph graph;

  // ====== METADATA ======
  graph.header_line = gpu.header_line;

  // ====== SEGMENT DATA ======
  graph.node_id_to_name = unflatten_strings(gpu.node_names);
  graph.node_sequences = unflatten_strings(gpu.node_sequences);

  // Rebuild node_name_to_id map
  graph.node_name_to_id.reserve(graph.node_id_to_name.size());
  for (size_t i = 0; i < graph.node_id_to_name.size(); ++i) {
    graph.node_name_to_id[graph.node_id_to_name[i]] = static_cast<uint32_t>(i);
  }

  // ====== SPLIT PATHS AND WALKS from concatenated data ======
  {
    size_t expected_entries =
        static_cast<size_t>(gpu.num_paths) + gpu.num_walks;
    if (gpu.paths.lengths.size() < expected_entries) {
      // Defensive: if lengths is shorter, only reconstruct what we have
      expected_entries = gpu.paths.lengths.size();
    }

    size_t offset = 0;

    // First num_paths entries are P-line paths
    uint32_t actual_paths = std::min(
        gpu.num_paths, static_cast<uint32_t>(gpu.paths.lengths.size()));
    graph.paths.reserve(actual_paths);
    for (uint32_t i = 0; i < actual_paths; ++i) {
      uint32_t len = gpu.paths.lengths[i];
      graph.paths.emplace_back(gpu.paths.data.begin() + offset,
                               gpu.paths.data.begin() + offset + len);
      offset += len;
    }

    // Remaining num_walks entries are W-line walks
    uint32_t actual_walks = 0;
    if (gpu.paths.lengths.size() > gpu.num_paths) {
      actual_walks = std::min(
          gpu.num_walks,
          static_cast<uint32_t>(gpu.paths.lengths.size() - gpu.num_paths));
    }
    graph.walks.walks.reserve(actual_walks);
    for (uint32_t i = 0; i < actual_walks; ++i) {
      uint32_t len = gpu.paths.lengths[gpu.num_paths + i];
      graph.walks.walks.emplace_back(gpu.paths.data.begin() + offset,
                                     gpu.paths.data.begin() + offset + len);
      offset += len;
    }
  }

  // ====== PATH METADATA (P-lines only) ======
  graph.path_names = unflatten_strings(gpu.path_names);
  graph.path_overlaps = unflatten_strings(gpu.path_overlaps);

  // ====== WALK METADATA (W-lines only) ======
  graph.walks.sample_ids = unflatten_strings(gpu.walk_sample_ids);
  graph.walks.hap_indices = gpu.walk_hap_indices;
  graph.walks.seq_ids = unflatten_strings(gpu.walk_seq_ids);
  graph.walks.seq_starts = gpu.walk_seq_starts;
  graph.walks.seq_ends = gpu.walk_seq_ends;

  // ====== LINK DATA ======
  graph.links.from_ids = gpu.link_from_ids;
  graph.links.to_ids = gpu.link_to_ids;
  graph.links.from_orients = gpu.link_from_orients;
  graph.links.to_orients = gpu.link_to_orients;
  graph.links.overlap_nums = gpu.link_overlap_nums;
  graph.links.overlap_ops = gpu.link_overlap_ops;

  // ====== OPTIONAL FIELDS ======
  graph.segment_optional_fields.reserve(gpu.segment_optional_fields.size());
  for (const auto &gpu_col : gpu.segment_optional_fields) {
    graph.segment_optional_fields.push_back(
        convert_optional_field_from_gpu(gpu_col));
  }

  graph.link_optional_fields.reserve(gpu.link_optional_fields.size());
  for (const auto &gpu_col : gpu.link_optional_fields) {
    graph.link_optional_fields.push_back(
        convert_optional_field_from_gpu(gpu_col));
  }

  // ====== JUMP DATA (J-lines) - structured columnar ======
  graph.jumps.from_ids = gpu.jump_from_ids;
  graph.jumps.from_orients = gpu.jump_from_orients;
  graph.jumps.to_ids = gpu.jump_to_ids;
  graph.jumps.to_orients = gpu.jump_to_orients;
  graph.jumps.distances = unflatten_strings(gpu.jump_distances);
  graph.jumps.rest_fields = unflatten_strings(gpu.jump_rest_fields);

  // ====== CONTAINMENT DATA (C-lines) - structured columnar ======
  graph.containments.container_ids = gpu.containment_container_ids;
  graph.containments.container_orients = gpu.containment_container_orients;
  graph.containments.contained_ids = gpu.containment_contained_ids;
  graph.containments.contained_orients = gpu.containment_contained_orients;
  graph.containments.positions = gpu.containment_positions;
  graph.containments.overlaps = unflatten_strings(gpu.containment_overlaps);
  graph.containments.rest_fields =
      unflatten_strings(gpu.containment_rest_fields);

  return graph;
}
