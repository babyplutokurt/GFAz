#include "workflows/decompression_workflow.hpp"
#include "codec/codec.hpp"
#include "utils/debug_log.hpp"
#include "utils/runtime_utils.hpp"
#include "utils/threading_utils.hpp"
#include "workflows/decompression_debug.hpp"

#include <chrono>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

constexpr const char *kDecompressionWarningPrefix =
    "Decompression workflow warning: ";

using Clock = std::chrono::high_resolution_clock;
using gfaz::runtime_utils::elapsed_ms;
using namespace gfaz::decompression_debug;

// Reconstruct 2D sequences from flattened array + lengths
void reconstruct_sequences(const std::vector<int32_t> &flat,
                           const std::vector<uint32_t> &lengths,
                           std::vector<std::vector<gfaz::NodeId>> &out) {
  out.clear();
  out.reserve(lengths.size());
  size_t offset = 0;

  for (uint32_t len : lengths) {
    std::vector<gfaz::NodeId> seq;
    seq.reserve(len);
    for (size_t j = 0; j < len && offset < flat.size(); ++j)
      seq.push_back(flat[offset++]);
    out.push_back(std::move(seq));
  }
}

// Reconstruct string array from concatenated string + lengths
void reconstruct_strings(const std::string &concat,
                         const std::vector<uint32_t> &lengths,
                         std::vector<std::string> &out) {
  out.clear();
  out.reserve(lengths.size());
  size_t offset = 0;

  for (uint32_t len : lengths) {
    if (offset + len <= concat.size()) {
      out.push_back(concat.substr(offset, len));
      offset += len;
    } else {
      out.emplace_back();
    }
  }
}

void decompress_string_column(const gfaz::ZstdCompressedBlock &strings_zstd,
                              const gfaz::ZstdCompressedBlock &lengths_zstd,
                              std::vector<std::string> &out) {
  std::string concatenated = gfaz::Codec::zstd_decompress_string(strings_zstd);
  std::vector<uint32_t> lengths =
      gfaz::Codec::zstd_decompress_uint32_vector(lengths_zstd);
  reconstruct_strings(concatenated, lengths, out);
}

// Recursively expand a rule ID into constituent nodes.
// Uses vector-based O(1) lookup (rule_idx = rule_id - min_rule_id).
void expand_rule(uint32_t rule_id, bool reverse,
                 const std::vector<int32_t> &first,
                 const std::vector<int32_t> &second, uint32_t min_id,
                 uint32_t max_id, std::vector<gfaz::NodeId> &out) {
  uint32_t idx = rule_id - min_id;
  int32_t a = first[idx];
  int32_t b = second[idx];

  if (!reverse) {
    // Forward: expand first, then second
    uint32_t abs_a = static_cast<uint32_t>(std::abs(a));
    if (abs_a >= min_id && abs_a < max_id)
      expand_rule(abs_a, a < 0, first, second, min_id, max_id, out);
    else
      out.push_back(a);

    uint32_t abs_b = static_cast<uint32_t>(std::abs(b));
    if (abs_b >= min_id && abs_b < max_id)
      expand_rule(abs_b, b < 0, first, second, min_id, max_id, out);
    else
      out.push_back(b);
  } else {
    // Reverse: expand second (negated) then first (negated)
    // Sign flip: positive becomes reverse, negative becomes forward
    uint32_t abs_b = static_cast<uint32_t>(std::abs(b));
    if (abs_b >= min_id && abs_b < max_id)
      expand_rule(abs_b, b >= 0, first, second, min_id, max_id, out);
    else
      out.push_back(-b);

    uint32_t abs_a = static_cast<uint32_t>(std::abs(a));
    if (abs_a >= min_id && abs_a < max_id)
      expand_rule(abs_a, a >= 0, first, second, min_id, max_id, out);
    else
      out.push_back(-a);
  }
}

// Expand all rule references in sequences with exact allocation and
// verification
void expand_sequences(std::vector<std::vector<gfaz::NodeId>> &seqs,
                      const std::vector<int32_t> &first,
                      const std::vector<int32_t> &second, uint32_t min_id,
                      size_t num_rules,
                      const std::vector<uint32_t> &original_lengths) {
  if (num_rules == 0)
    return;

  uint32_t max_id = min_id + static_cast<uint32_t>(num_rules);
  bool have_original = !original_lengths.empty();

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (size_t i = 0; i < seqs.size(); ++i) {
    auto &encoded = seqs[i];
    std::vector<gfaz::NodeId> expanded;

    // Use exact allocation if original length available, otherwise 40x
    // heuristic
    if (have_original && i < original_lengths.size())
      expanded.reserve(original_lengths[i]);
    else
      expanded.reserve(encoded.size() * 40);

    for (gfaz::NodeId node : encoded) {
      uint32_t abs_id = static_cast<uint32_t>(std::abs(node));
      if (abs_id >= min_id && abs_id < max_id)
        expand_rule(abs_id, node < 0, first, second, min_id, max_id, expanded);
      else
        expanded.push_back(node);
    }

    // Verify decompression integrity
    if (have_original && i < original_lengths.size()) {
      if (expanded.size() != original_lengths[i]) {
        std::cerr << kDecompressionWarningPrefix << "sequence " << i
                  << " expected length " << original_lengths[i] << ", got "
                  << expanded.size() << std::endl;
      }
    }

    encoded = std::move(expanded);
  }
}

// Decompress, reconstruct, expand rules, and inverse delta transform
void decompress_expand_sequences(const gfaz::ZstdCompressedBlock &block,
                                 const std::vector<uint32_t> &lengths,
                                 const std::vector<uint32_t> &original_lengths,
                                 const std::vector<int32_t> &first,
                                 const std::vector<int32_t> &second,
                                 uint32_t min_id, size_t num_rules,
                                 int delta_round,
                                 std::vector<std::vector<gfaz::NodeId>> &out) {
  {
    std::vector<int32_t> flat = gfaz::Codec::zstd_decompress_int32_vector(block);
    reconstruct_sequences(flat, lengths, out);
  }

  expand_sequences(out, first, second, min_id, num_rules, original_lengths);

  for (int i = 0; i < delta_round; ++i)
    gfaz::Codec::inverse_delta_transform(out);
}

// Decompress an optional field column
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
    std::string bytes = gfaz::Codec::zstd_decompress_string(c.b_concat_bytes_zstd);
    col.b_concat_bytes = std::vector<uint8_t>(bytes.begin(), bytes.end());
    break;
  }
  }
  return col;
}

void decompress_walk_metadata(const gfaz::CompressedData &data, gfaz::GfaGraph &graph) {
#ifdef _OPENMP
#pragma omp parallel sections
  {
#pragma omp section
    decompress_string_column(data.walk_sample_ids_zstd,
                             data.walk_sample_id_lengths_zstd,
                             graph.walks.sample_ids);
#pragma omp section
    graph.walks.hap_indices =
        gfaz::Codec::zstd_decompress_uint32_vector(data.walk_hap_indices_zstd);
#pragma omp section
    decompress_string_column(data.walk_seq_ids_zstd,
                             data.walk_seq_id_lengths_zstd,
                             graph.walks.seq_ids);
#pragma omp section
    graph.walks.seq_starts = gfaz::Codec::decompress_varint_int64(
        data.walk_seq_starts_zstd, data.walk_lengths.size());
#pragma omp section
    graph.walks.seq_ends = gfaz::Codec::decompress_varint_int64(
        data.walk_seq_ends_zstd, data.walk_lengths.size());
  }
#else
  decompress_string_column(data.walk_sample_ids_zstd,
                           data.walk_sample_id_lengths_zstd,
                           graph.walks.sample_ids);
  graph.walks.hap_indices =
      gfaz::Codec::zstd_decompress_uint32_vector(data.walk_hap_indices_zstd);
  decompress_string_column(data.walk_seq_ids_zstd,
                           data.walk_seq_id_lengths_zstd, graph.walks.seq_ids);
  graph.walks.seq_starts = gfaz::Codec::decompress_varint_int64(
      data.walk_seq_starts_zstd, data.walk_lengths.size());
  graph.walks.seq_ends = gfaz::Codec::decompress_varint_int64(
      data.walk_seq_ends_zstd, data.walk_lengths.size());
#endif
}

} // namespace

// ---------------------------------------------------------------------------
// Public workflow entry point
// ---------------------------------------------------------------------------

void decompress_gfa(const gfaz::CompressedData &data, gfaz::GfaGraph &graph,
                    int num_threads) {
  ScopedOMPThreads omp_scope(num_threads);

  auto decomp_total_start = Clock::now();
  std::vector<TimedDebugStage> debug_stages;

  graph = gfaz::GfaGraph();
  graph.header_line = data.header_line;

  // =========================================================================
  // Step 1: ZSTD decompress (rules + paths)
  // =========================================================================
  auto t0 = Clock::now();
  std::vector<int32_t> rules_first, rules_second;

#ifdef _OPENMP
#pragma omp parallel sections
  {
#pragma omp section
    rules_first = gfaz::Codec::zstd_decompress_int32_vector(data.rules_first_zstd);
#pragma omp section
    rules_second = gfaz::Codec::zstd_decompress_int32_vector(data.rules_second_zstd);
  }
#else
  rules_first = gfaz::Codec::zstd_decompress_int32_vector(data.rules_first_zstd);
  rules_second = gfaz::Codec::zstd_decompress_int32_vector(data.rules_second_zstd);
#endif

  std::vector<int32_t> paths_flat =
      gfaz::Codec::zstd_decompress_int32_vector(data.paths_zstd);

  auto t1 = Clock::now();
  debug_stages.push_back(
      {"decode compressed rule/path fields", elapsed_ms(t0, t1)});

  // =========================================================================
  // Step 2: Decode rules (prefix sum / delta decode)
  // =========================================================================
  t0 = Clock::now();
#ifdef _OPENMP
#pragma omp parallel sections
  {
#pragma omp section
    gfaz::Codec::delta_decode_int32(rules_first);
#pragma omp section
    gfaz::Codec::delta_decode_int32(rules_second);
  }
#else
  gfaz::Codec::delta_decode_int32(rules_first);
  gfaz::Codec::delta_decode_int32(rules_second);
#endif
  t1 = Clock::now();
  debug_stages.push_back({"delta decode rule columns", elapsed_ms(t0, t1)});

  uint32_t min_rule_id = data.min_rule_id();
  size_t num_rules = rules_first.size();

  // Reconstruct paths from flat array
  reconstruct_sequences(paths_flat, data.sequence_lengths, graph.paths_data.traversals);
  paths_flat.clear();
  paths_flat.shrink_to_fit();

  // =========================================================================
  // Step 3: Expand paths (rule expansion)
  // =========================================================================
  t0 = Clock::now();
  expand_sequences(graph.paths_data.traversals, rules_first, rules_second, min_rule_id,
                   num_rules, data.original_path_lengths);
  t1 = Clock::now();
  debug_stages.push_back({"expand path traversals", elapsed_ms(t0, t1)});

  // =========================================================================
  // Step 4: Decode paths (prefix sum / inverse delta)
  // =========================================================================
  t0 = Clock::now();
  for (int i = 0; i < data.delta_round; ++i)
    gfaz::Codec::inverse_delta_transform(graph.paths_data.traversals);
  t1 = Clock::now();
  debug_stages.push_back({"inverse delta traversals", elapsed_ms(t0, t1)});
  log_cpu_decompression_memory_checkpoint("CPU Legacy",
                                          "after traversal reconstruction");

  // =========================================================================
  // Rest: metadata, walks, segments, links, etc.
  // =========================================================================
  t0 = Clock::now();

  // Path metadata
#ifdef _OPENMP
#pragma omp parallel sections
  {
#pragma omp section
    decompress_string_column(data.names_zstd, data.name_lengths_zstd,
                             graph.paths_data.names);
#pragma omp section
    decompress_string_column(data.overlaps_zstd, data.overlap_lengths_zstd,
                             graph.paths_data.overlaps);
  }
#else
  decompress_string_column(data.names_zstd, data.name_lengths_zstd,
                           graph.paths_data.names);
  decompress_string_column(data.overlaps_zstd, data.overlap_lengths_zstd,
                           graph.paths_data.overlaps);
#endif
  t1 = Clock::now();
  debug_stages.push_back({"decode path metadata", elapsed_ms(t0, t1)});

  // Decompress walks
  if (!data.walks_zstd.payload.empty() && !data.walk_lengths.empty()) {
    t0 = Clock::now();
    decompress_expand_sequences(data.walks_zstd, data.walk_lengths,
                                data.original_walk_lengths, rules_first,
                                rules_second, min_rule_id, num_rules,
                                data.delta_round, graph.walks.walks);
    t1 = Clock::now();
    debug_stages.push_back({"expand walk traversals", elapsed_ms(t0, t1)});

    t0 = Clock::now();
    decompress_walk_metadata(data, graph);
    t1 = Clock::now();
    debug_stages.push_back({"decode walk metadata", elapsed_ms(t0, t1)});
    log_cpu_decompression_memory_checkpoint("CPU Legacy",
                                            "after walk reconstruction");
  }

  // Release rules after expansion
  rules_first.clear();
  rules_first.shrink_to_fit();
  rules_second.clear();
  rules_second.shrink_to_fit();

  // Decompress segments and links
  std::string seg_seqs;
  std::vector<uint32_t> seg_lens;
  t0 = Clock::now();

#ifdef _OPENMP
#pragma omp parallel sections
  {
#pragma omp section
    {
      seg_seqs = gfaz::Codec::zstd_decompress_string(data.segment_sequences_zstd);
      seg_lens =
          gfaz::Codec::zstd_decompress_uint32_vector(data.segment_seq_lengths_zstd);
    }
#pragma omp section
    {
      graph.links.from_ids = gfaz::Codec::decompress_delta_varint_uint32(
          data.link_from_ids_zstd, data.num_links);
      graph.links.to_ids = gfaz::Codec::decompress_delta_varint_uint32(
          data.link_to_ids_zstd, data.num_links);
    }
#pragma omp section
    {
      graph.links.from_orients = gfaz::Codec::decompress_orientations(
          data.link_from_orients_zstd, data.num_links);
      graph.links.to_orients = gfaz::Codec::decompress_orientations(
          data.link_to_orients_zstd, data.num_links);
    }
#pragma omp section

    {
      graph.links.overlap_nums =
          gfaz::Codec::zstd_decompress_uint32_vector(data.link_overlap_nums_zstd);
      graph.links.overlap_ops =
          gfaz::Codec::zstd_decompress_char_vector(data.link_overlap_ops_zstd);
    }
  }
#else
  seg_seqs = gfaz::Codec::zstd_decompress_string(data.segment_sequences_zstd);
  seg_lens =
      gfaz::Codec::zstd_decompress_uint32_vector(data.segment_seq_lengths_zstd);
  graph.links.from_ids = gfaz::Codec::decompress_delta_varint_uint32(
      data.link_from_ids_zstd, data.num_links);
  graph.links.to_ids = gfaz::Codec::decompress_delta_varint_uint32(
      data.link_to_ids_zstd, data.num_links);
  graph.links.from_orients = gfaz::Codec::decompress_orientations(
      data.link_from_orients_zstd, data.num_links);
  graph.links.to_orients =
      gfaz::Codec::decompress_orientations(data.link_to_orients_zstd, data.num_links);
  graph.links.overlap_nums =
      gfaz::Codec::zstd_decompress_uint32_vector(data.link_overlap_nums_zstd);
  graph.links.overlap_ops =
      gfaz::Codec::zstd_decompress_char_vector(data.link_overlap_ops_zstd);
#endif
  t1 = Clock::now();
  debug_stages.push_back({"decode segment/link columns", elapsed_ms(t0, t1)});

  // Reconstruct segments (index 0 is placeholder for 1-based IDs).
  // CPU .gfaz intentionally restores dense numeric names instead of original
  // segment names, because the serialized CPU representation stores segments
  // by implicit 1-based ID order only.
  t0 = Clock::now();
  graph.segments.node_id_to_name.push_back("");
  graph.segments.node_sequences.push_back("");

  size_t offset = 0;
  for (size_t i = 0; i < seg_lens.size(); ++i) {
    uint32_t id = static_cast<uint32_t>(i + 1);
    std::string name = std::to_string(id);
    uint32_t len = seg_lens[i];

    std::string seq;
    if (offset + len <= seg_seqs.size()) {
      seq = seg_seqs.substr(offset, len);
      offset += len;
    }

    graph.segments.node_id_to_name.push_back(name);
    graph.segments.node_sequences.push_back(seq);
    graph.node_name_to_id[name] = id;
  }
  t1 = Clock::now();
  debug_stages.push_back({"reconstruct in-memory segments", elapsed_ms(t0, t1)});

  // Decompress optional fields
  t0 = Clock::now();
  for (const auto &c : data.segment_optional_fields_zstd)
    graph.segments.optional_fields.push_back(decompress_optional_column(c));

  for (const auto &c : data.link_optional_fields_zstd)
    graph.link_optional_fields.push_back(decompress_optional_column(c));
  t1 = Clock::now();
  debug_stages.push_back({"decode optional fields", elapsed_ms(t0, t1)});

  // Decompress J-lines (jumps)
  if (data.num_jumps > 0) {
    t0 = Clock::now();
    graph.jumps.from_ids = gfaz::Codec::decompress_delta_varint_uint32(
        data.jump_from_ids_zstd, data.num_jumps);
    graph.jumps.to_ids = gfaz::Codec::decompress_delta_varint_uint32(
        data.jump_to_ids_zstd, data.num_jumps);
    graph.jumps.from_orients = gfaz::Codec::decompress_orientations(
        data.jump_from_orients_zstd, data.num_jumps);
    graph.jumps.to_orients = gfaz::Codec::decompress_orientations(
        data.jump_to_orients_zstd, data.num_jumps);

    std::string dists = gfaz::Codec::zstd_decompress_string(data.jump_distances_zstd);
    std::vector<uint32_t> dist_lens =
        gfaz::Codec::zstd_decompress_uint32_vector(data.jump_distance_lengths_zstd);
    reconstruct_strings(dists, dist_lens, graph.jumps.distances);

    std::string rest =
        gfaz::Codec::zstd_decompress_string(data.jump_rest_fields_zstd);
    std::vector<uint32_t> rest_lens =
        gfaz::Codec::zstd_decompress_uint32_vector(data.jump_rest_lengths_zstd);
    reconstruct_strings(rest, rest_lens, graph.jumps.rest_fields);
    t1 = Clock::now();
    debug_stages.push_back({"decode jump columns", elapsed_ms(t0, t1)});
  }

  // Decompress C-lines (containments)
  if (data.num_containments > 0) {
    t0 = Clock::now();
    graph.containments.container_ids = gfaz::Codec::decompress_delta_varint_uint32(
        data.containment_container_ids_zstd, data.num_containments);
    graph.containments.contained_ids = gfaz::Codec::decompress_delta_varint_uint32(
        data.containment_contained_ids_zstd, data.num_containments);
    graph.containments.container_orients = gfaz::Codec::decompress_orientations(
        data.containment_container_orients_zstd, data.num_containments);
    graph.containments.contained_orients = gfaz::Codec::decompress_orientations(
        data.containment_contained_orients_zstd, data.num_containments);
    graph.containments.positions =
        gfaz::Codec::zstd_decompress_uint32_vector(data.containment_positions_zstd);

    std::string overlaps =
        gfaz::Codec::zstd_decompress_string(data.containment_overlaps_zstd);
    std::vector<uint32_t> overlap_lens = gfaz::Codec::zstd_decompress_uint32_vector(
        data.containment_overlap_lengths_zstd);
    reconstruct_strings(overlaps, overlap_lens, graph.containments.overlaps);

    std::string rest =
        gfaz::Codec::zstd_decompress_string(data.containment_rest_fields_zstd);
    std::vector<uint32_t> rest_lens = gfaz::Codec::zstd_decompress_uint32_vector(
        data.containment_rest_lengths_zstd);
    reconstruct_strings(rest, rest_lens, graph.containments.rest_fields);
    t1 = Clock::now();
    debug_stages.push_back({"decode containment columns", elapsed_ms(t0, t1)});
  }
  log_cpu_decompression_memory_checkpoint("CPU Legacy",
                                          "after full graph reconstruction");

  auto decomp_total_end = Clock::now();
  double decomp_total_ms = elapsed_ms(decomp_total_start, decomp_total_end);

  if (gfaz_debug_enabled()) {
    print_cpu_decompression_summary("CPU Legacy", graph, num_rules,
                                    data.delta_round);
    print_cpu_decompression_timing({"CPU Legacy",
                                    "legacy in-memory reconstruction",
                                    debug_stages, decomp_total_ms});
  }
}
