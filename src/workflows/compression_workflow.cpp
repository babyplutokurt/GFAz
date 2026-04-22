#include "workflows/compression_workflow.hpp"
#include "codec/codec.hpp"
#include "grammar/packed_2mer.hpp"
#include "grammar/path_encoder.hpp"
#include "grammar/rule_generator.hpp"
#include "io/gfa_parser.hpp"
#include "utils/debug_log.hpp"
#include "utils/threading_utils.hpp"
#include "workflows/compression_debug.hpp"
#include "workflows/compression_utils.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(_OPENMP) && defined(__GLIBCXX__)
#include <parallel/algorithm>
#endif

using namespace gfaz::compression_utils;
using namespace gfaz::compression_debug;
using namespace gfaz::runtime_utils;

static void run_grammar_compression(std::vector<std::vector<gfaz::NodeId>> &paths,
                                    std::vector<std::vector<gfaz::NodeId>> &walks,
                                    uint32_t &next_id, int num_rounds,
                                    size_t freq_threshold, uint32_t layer_start,
                                    std::vector<Packed2mer> &rulebook,
                                    int num_threads);

namespace {

struct CompressionContext {
  gfaz::GfaGraph graph;
  gfaz::CompressedData out;

  int num_rounds = 0;
  size_t freq_threshold = 0;
  int delta_round = 1;
  int num_threads = 0;
  bool show_stats = false;

  uint32_t next_id = 0;
  uint32_t layer_start = 0;
  uint32_t max_abs = 0;
  size_t num_segments = 0;
  size_t total_elements = 0;
  double data_size_mb = 0;

  std::vector<Packed2mer> rulebook;
};

struct PathCompressionInput {
  std::vector<int32_t> flat;
  std::string names_concat;
  std::vector<uint32_t> name_lengths;
  std::string overlaps_concat;
  std::vector<uint32_t> overlap_lengths;
};

struct WalkCompressionInput {
  std::vector<int32_t> flat;
  std::string sample_ids_concat;
  std::vector<uint32_t> sample_id_lengths;
  std::string seq_ids_concat;
  std::vector<uint32_t> seq_id_lengths;
};

struct SegmentCompressionInput {
  std::string segment_concat;
  std::vector<uint32_t> segment_lengths;
};

struct LinkCompressionInput {
  const gfaz::LinkData *links = nullptr;
};

struct JumpCompressionInput {
  std::string distances_concat;
  std::vector<uint32_t> distance_lengths;
  std::string rest_fields_concat;
  std::vector<uint32_t> rest_lengths;
  const gfaz::JumpData *jumps = nullptr;
};

struct ContainmentCompressionInput {
  std::string overlaps_concat;
  std::vector<uint32_t> overlap_lengths;
  std::string rest_fields_concat;
  std::vector<uint32_t> rest_lengths;
  const gfaz::ContainmentData *containments = nullptr;
};

void release_path_fields(CompressionContext &ctx) {
  ctx.graph.paths_data.traversals.clear();
  ctx.graph.paths_data.traversals.shrink_to_fit();
  ctx.graph.paths_data.names.clear();
  ctx.graph.paths_data.names.shrink_to_fit();
  ctx.graph.paths_data.overlaps.clear();
  ctx.graph.paths_data.overlaps.shrink_to_fit();
}

void release_walk_fields(CompressionContext &ctx) {
  ctx.graph.walks.walks.clear();
  ctx.graph.walks.walks.shrink_to_fit();
  ctx.graph.walks.sample_ids.clear();
  ctx.graph.walks.sample_ids.shrink_to_fit();
  ctx.graph.walks.hap_indices.clear();
  ctx.graph.walks.hap_indices.shrink_to_fit();
  ctx.graph.walks.seq_ids.clear();
  ctx.graph.walks.seq_ids.shrink_to_fit();
  ctx.graph.walks.seq_starts.clear();
  ctx.graph.walks.seq_starts.shrink_to_fit();
  ctx.graph.walks.seq_ends.clear();
  ctx.graph.walks.seq_ends.shrink_to_fit();
}

void release_segment_link_fields(CompressionContext &ctx) {
  ctx.graph.segments.node_sequences.clear();
  ctx.graph.segments.node_sequences.shrink_to_fit();
  ctx.graph.links.from_ids.clear();
  ctx.graph.links.from_ids.shrink_to_fit();
  ctx.graph.links.to_ids.clear();
  ctx.graph.links.to_ids.shrink_to_fit();
  ctx.graph.links.from_orients.clear();
  ctx.graph.links.from_orients.shrink_to_fit();
  ctx.graph.links.to_orients.clear();
  ctx.graph.links.to_orients.shrink_to_fit();
  ctx.graph.links.overlap_nums.clear();
  ctx.graph.links.overlap_nums.shrink_to_fit();
  ctx.graph.links.overlap_ops.clear();
  ctx.graph.links.overlap_ops.shrink_to_fit();
}

void release_optional_fields(CompressionContext &ctx) {
  ctx.graph.segments.optional_fields.clear();
  ctx.graph.segments.optional_fields.shrink_to_fit();
  ctx.graph.link_optional_fields.clear();
  ctx.graph.link_optional_fields.shrink_to_fit();
}

void release_jump_fields(CompressionContext &ctx) {
  ctx.graph.jumps.from_ids.clear();
  ctx.graph.jumps.from_ids.shrink_to_fit();
  ctx.graph.jumps.from_orients.clear();
  ctx.graph.jumps.from_orients.shrink_to_fit();
  ctx.graph.jumps.to_ids.clear();
  ctx.graph.jumps.to_ids.shrink_to_fit();
  ctx.graph.jumps.to_orients.clear();
  ctx.graph.jumps.to_orients.shrink_to_fit();
  ctx.graph.jumps.distances.clear();
  ctx.graph.jumps.distances.shrink_to_fit();
  ctx.graph.jumps.rest_fields.clear();
  ctx.graph.jumps.rest_fields.shrink_to_fit();
}

void release_containment_fields(CompressionContext &ctx) {
  ctx.graph.containments.container_ids.clear();
  ctx.graph.containments.container_ids.shrink_to_fit();
  ctx.graph.containments.container_orients.clear();
  ctx.graph.containments.container_orients.shrink_to_fit();
  ctx.graph.containments.contained_ids.clear();
  ctx.graph.containments.contained_ids.shrink_to_fit();
  ctx.graph.containments.contained_orients.clear();
  ctx.graph.containments.contained_orients.shrink_to_fit();
  ctx.graph.containments.positions.clear();
  ctx.graph.containments.positions.shrink_to_fit();
  ctx.graph.containments.overlaps.clear();
  ctx.graph.containments.overlaps.shrink_to_fit();
  ctx.graph.containments.rest_fields.clear();
  ctx.graph.containments.rest_fields.shrink_to_fit();
}

void initialize_output_metadata(CompressionContext &ctx) {
  ctx.out.header_line = ctx.graph.header_line;

  // Negative delta rounds are invalid. Zero rounds are supported, but require
  // seeding rule IDs from the raw traversal symbol space.
  if (ctx.delta_round < 0) {
    std::cerr << "Warning: delta_round=" << ctx.delta_round
              << " is invalid, clamping to 0" << std::endl;
    ctx.delta_round = 0;
  }
  ctx.out.delta_round = ctx.delta_round;

  for (const auto &p : ctx.graph.paths_data.traversals)
    ctx.out.original_path_lengths.push_back(static_cast<uint32_t>(p.size()));
  for (const auto &w : ctx.graph.walks.walks)
    ctx.out.original_walk_lengths.push_back(static_cast<uint32_t>(w.size()));

  ctx.total_elements = total_node_count(ctx.graph.paths_data.traversals) +
                       total_node_count(ctx.graph.walks.walks);
  ctx.data_size_mb = ctx.total_elements * sizeof(int32_t) / (1024.0 * 1024.0);
}

void prepare_id_space_for_traversal_transform(CompressionContext &ctx) {
  // Rule IDs must be disjoint from the post-delta traversal symbol space.
  // Seed next_id from the maximum absolute delta-domain symbol instead of the
  // parser's name metadata, which is not needed beyond parsing.
  ctx.graph.node_name_to_id.clear();
  ctx.graph.node_name_to_id.rehash(0);
  ctx.graph.segments.node_id_to_name.clear();
  ctx.graph.segments.node_id_to_name.shrink_to_fit();
  ctx.next_id = 0;
}

uint32_t max_abs_symbol(
    const std::vector<std::vector<gfaz::NodeId>> &sequences) {
  uint32_t max_abs = 0;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) reduction(max : max_abs)
#endif
  for (size_t i = 0; i < sequences.size(); ++i) {
    const auto &sequence = sequences[i];
    for (gfaz::NodeId node : sequence) {
      const uint32_t abs_id = static_cast<uint32_t>(std::abs(node));
      if (abs_id > max_abs)
        max_abs = abs_id;
    }
  }
  return max_abs;
}

double apply_delta_transform(CompressionContext &ctx) {
  const auto start = std::chrono::high_resolution_clock::now();
  // With delta_round == 0 the grammar sees raw node IDs, so reserve rule IDs
  // above that raw symbol space. For delta_round > 0 we keep the current fast
  // path and let the fused delta transform compute the relevant bound.
  ctx.max_abs = 0;
  if (ctx.delta_round == 0) {
    const uint32_t path_max = max_abs_symbol(ctx.graph.paths_data.traversals);
    const uint32_t walk_max = max_abs_symbol(ctx.graph.walks.walks);
    ctx.max_abs = std::max(path_max, walk_max);
  }
  for (int i = 0; i < ctx.delta_round; ++i) {
    const uint32_t path_max =
        gfaz::Codec::delta_transform_and_max_abs(ctx.graph.paths_data.traversals);
    const uint32_t walk_max =
        gfaz::Codec::delta_transform_and_max_abs(ctx.graph.walks.walks);
    ctx.max_abs = std::max(ctx.max_abs, std::max(path_max, walk_max));
  }
  const auto end = std::chrono::high_resolution_clock::now();

  if (ctx.max_abs >= ctx.next_id) {
    if (ctx.max_abs == UINT32_MAX) {
      throw std::overflow_error(
          std::string(kCompressionErrorPrefix) +
          "delta values too large for rule ID assignment");
    }
    ctx.next_id = ctx.max_abs + 1;
  }

  return elapsed_ms(start, end);
}

double run_grammar_stage(CompressionContext &ctx) {
  ctx.layer_start = ctx.next_id;
  const auto start = std::chrono::high_resolution_clock::now();
  run_grammar_compression(ctx.graph.paths_data.traversals, ctx.graph.walks.walks, ctx.next_id,
                          ctx.num_rounds, ctx.freq_threshold, ctx.layer_start,
                          ctx.rulebook, ctx.num_threads);
  const auto end = std::chrono::high_resolution_clock::now();

  const uint32_t rule_count = ctx.next_id - ctx.layer_start;
  ctx.out.layer_rule_ranges.push_back(
      gfaz::LayerRuleRange{2, ctx.layer_start, ctx.next_id, 0, rule_count * 2});
  return elapsed_ms(start, end);
}

double compress_rule_columns(CompressionContext &ctx) {
  const auto start = std::chrono::high_resolution_clock::now();
  std::vector<int32_t> first, second;
  process_rules(ctx.rulebook, ctx.layer_start, ctx.out.layer_rule_ranges, first,
                second);
  ctx.rulebook.clear();
  ctx.rulebook.shrink_to_fit();

#ifdef _OPENMP
#pragma omp parallel sections
  {
#pragma omp section
    ctx.out.rules_first_zstd = gfaz::Codec::zstd_compress_int32_vector(first);
#pragma omp section
    ctx.out.rules_second_zstd = gfaz::Codec::zstd_compress_int32_vector(second);
  }
#else
  ctx.out.rules_first_zstd = gfaz::Codec::zstd_compress_int32_vector(first);
  ctx.out.rules_second_zstd = gfaz::Codec::zstd_compress_int32_vector(second);
#endif

  const auto end = std::chrono::high_resolution_clock::now();
  return elapsed_ms(start, end);
}

double compress_path_fields(CompressionContext &ctx) {
  const auto start = std::chrono::high_resolution_clock::now();
  PathCompressionInput input;
  flatten_traversal_sequences(ctx.graph.paths_data.traversals, input.flat,
                              ctx.out.sequence_lengths);
  flatten_path_metadata(ctx.graph.paths_data.traversals, ctx.graph.paths_data.names,
                        ctx.graph.paths_data.overlaps, input.names_concat,
                        input.name_lengths, input.overlaps_concat,
                        input.overlap_lengths);

#ifdef _OPENMP
#pragma omp parallel sections
  {
#pragma omp section
    ctx.out.paths_zstd = gfaz::Codec::zstd_compress_int32_vector(input.flat);
#pragma omp section
    {
      ctx.out.names_zstd = gfaz::Codec::zstd_compress_string(input.names_concat);
      ctx.out.name_lengths_zstd =
          gfaz::Codec::zstd_compress_uint32_vector(input.name_lengths);
    }
#pragma omp section
    {
      ctx.out.overlaps_zstd =
          gfaz::Codec::zstd_compress_string(input.overlaps_concat);
      ctx.out.overlap_lengths_zstd =
          gfaz::Codec::zstd_compress_uint32_vector(input.overlap_lengths);
    }
  }
#else
  ctx.out.paths_zstd = gfaz::Codec::zstd_compress_int32_vector(input.flat);
  ctx.out.names_zstd = gfaz::Codec::zstd_compress_string(input.names_concat);
  ctx.out.name_lengths_zstd =
      gfaz::Codec::zstd_compress_uint32_vector(input.name_lengths);
  ctx.out.overlaps_zstd = gfaz::Codec::zstd_compress_string(input.overlaps_concat);
  ctx.out.overlap_lengths_zstd =
      gfaz::Codec::zstd_compress_uint32_vector(input.overlap_lengths);
#endif

  release_path_fields(ctx);

  const auto end = std::chrono::high_resolution_clock::now();
  return elapsed_ms(start, end);
}

double compress_walk_fields(CompressionContext &ctx) {
  const auto start = std::chrono::high_resolution_clock::now();
  if (!ctx.graph.walks.walks.empty()) {
    WalkCompressionInput input;
    flatten_traversal_sequences(ctx.graph.walks.walks, input.flat,
                                ctx.out.walk_lengths);
    ctx.out.walks_zstd = gfaz::Codec::zstd_compress_int32_vector(input.flat);

    flatten_walk_string_metadata(ctx.graph.walks, input.sample_ids_concat,
                                 input.sample_id_lengths, input.seq_ids_concat,
                                 input.seq_id_lengths);

    ctx.out.walk_sample_ids_zstd =
        gfaz::Codec::zstd_compress_string(input.sample_ids_concat);
    ctx.out.walk_sample_id_lengths_zstd =
        gfaz::Codec::zstd_compress_uint32_vector(input.sample_id_lengths);
    ctx.out.walk_hap_indices_zstd =
        gfaz::Codec::zstd_compress_uint32_vector(ctx.graph.walks.hap_indices);
    ctx.out.walk_seq_ids_zstd =
        gfaz::Codec::zstd_compress_string(input.seq_ids_concat);
    ctx.out.walk_seq_id_lengths_zstd =
        gfaz::Codec::zstd_compress_uint32_vector(input.seq_id_lengths);
    ctx.out.walk_seq_starts_zstd =
        gfaz::Codec::compress_varint_int64(ctx.graph.walks.seq_starts);
    ctx.out.walk_seq_ends_zstd =
        gfaz::Codec::compress_varint_int64(ctx.graph.walks.seq_ends);
  }

  release_walk_fields(ctx);

  const auto end = std::chrono::high_resolution_clock::now();
  return elapsed_ms(start, end);
}

double compress_segment_link_fields(CompressionContext &ctx) {
  SegmentCompressionInput segment_input;
  LinkCompressionInput link_input;
  flatten_segment_sequences(ctx.graph.segments.node_sequences,
                            segment_input.segment_concat,
                            segment_input.segment_lengths, ctx.next_id);
  ctx.num_segments = segment_input.segment_lengths.size();
  link_input.links = &ctx.graph.links;
  ctx.out.num_links = link_input.links->from_ids.size();

  const auto start = std::chrono::high_resolution_clock::now();
#ifdef _OPENMP
#pragma omp parallel sections
  {
#pragma omp section
    {
      ctx.out.segment_sequences_zstd =
          gfaz::Codec::zstd_compress_string(segment_input.segment_concat);
      ctx.out.segment_seq_lengths_zstd =
          gfaz::Codec::zstd_compress_uint32_vector(segment_input.segment_lengths);
    }
#pragma omp section
    {
      ctx.out.link_from_ids_zstd =
          gfaz::Codec::compress_delta_varint_uint32(link_input.links->from_ids);
      ctx.out.link_to_ids_zstd =
          gfaz::Codec::compress_delta_varint_uint32(link_input.links->to_ids);
    }
#pragma omp section
    {
      ctx.out.link_from_orients_zstd =
          gfaz::Codec::compress_orientations(link_input.links->from_orients);
      ctx.out.link_to_orients_zstd =
          gfaz::Codec::compress_orientations(link_input.links->to_orients);
    }
#pragma omp section
    {
      ctx.out.link_overlap_nums_zstd =
          gfaz::Codec::zstd_compress_uint32_vector(link_input.links->overlap_nums);
      ctx.out.link_overlap_ops_zstd =
          gfaz::Codec::zstd_compress_char_vector(link_input.links->overlap_ops);
    }
  }
#else
  ctx.out.segment_sequences_zstd =
      gfaz::Codec::zstd_compress_string(segment_input.segment_concat);
  ctx.out.segment_seq_lengths_zstd =
      gfaz::Codec::zstd_compress_uint32_vector(segment_input.segment_lengths);
  ctx.out.link_from_ids_zstd =
      gfaz::Codec::compress_delta_varint_uint32(link_input.links->from_ids);
  ctx.out.link_to_ids_zstd =
      gfaz::Codec::compress_delta_varint_uint32(link_input.links->to_ids);
  ctx.out.link_from_orients_zstd =
      gfaz::Codec::compress_orientations(link_input.links->from_orients);
  ctx.out.link_to_orients_zstd =
      gfaz::Codec::compress_orientations(link_input.links->to_orients);
  ctx.out.link_overlap_nums_zstd =
      gfaz::Codec::zstd_compress_uint32_vector(link_input.links->overlap_nums);
  ctx.out.link_overlap_ops_zstd =
      gfaz::Codec::zstd_compress_char_vector(link_input.links->overlap_ops);
#endif

  release_segment_link_fields(ctx);

  const auto end = std::chrono::high_resolution_clock::now();
  return elapsed_ms(start, end);
}

double compress_optional_fields(CompressionContext &ctx) {
  const auto start = std::chrono::high_resolution_clock::now();
  for (const auto &col : ctx.graph.segments.optional_fields)
    ctx.out.segment_optional_fields_zstd.push_back(
        compress_optional_column(col));
  for (const auto &col : ctx.graph.link_optional_fields)
    ctx.out.link_optional_fields_zstd.push_back(compress_optional_column(col));

  release_optional_fields(ctx);

  const auto end = std::chrono::high_resolution_clock::now();
  return elapsed_ms(start, end);
}

double compress_jump_fields(CompressionContext &ctx) {
  const auto start = std::chrono::high_resolution_clock::now();
  if (!ctx.graph.jumps.from_ids.empty()) {
    JumpCompressionInput input;
    input.jumps = &ctx.graph.jumps;
    append_string_column(input.jumps->distances, input.distances_concat,
                         input.distance_lengths);
    append_string_column(input.jumps->rest_fields, input.rest_fields_concat,
                         input.rest_lengths);

    ctx.out.num_jumps = ctx.graph.jumps.size();

    ctx.out.jump_from_ids_zstd =
        gfaz::Codec::compress_delta_varint_uint32(input.jumps->from_ids);
    ctx.out.jump_to_ids_zstd =
        gfaz::Codec::compress_delta_varint_uint32(input.jumps->to_ids);
    ctx.out.jump_from_orients_zstd =
        gfaz::Codec::compress_orientations(input.jumps->from_orients);
    ctx.out.jump_to_orients_zstd =
        gfaz::Codec::compress_orientations(input.jumps->to_orients);
    ctx.out.jump_distances_zstd =
        gfaz::Codec::zstd_compress_string(input.distances_concat);
    ctx.out.jump_distance_lengths_zstd =
        gfaz::Codec::zstd_compress_uint32_vector(input.distance_lengths);
    ctx.out.jump_rest_fields_zstd =
        gfaz::Codec::zstd_compress_string(input.rest_fields_concat);
    ctx.out.jump_rest_lengths_zstd =
        gfaz::Codec::zstd_compress_uint32_vector(input.rest_lengths);
  }

  release_jump_fields(ctx);

  const auto end = std::chrono::high_resolution_clock::now();
  return elapsed_ms(start, end);
}

double compress_containment_fields(CompressionContext &ctx) {
  const auto start = std::chrono::high_resolution_clock::now();
  if (!ctx.graph.containments.container_ids.empty()) {
    ContainmentCompressionInput input;
    input.containments = &ctx.graph.containments;
    append_string_column(input.containments->overlaps, input.overlaps_concat,
                         input.overlap_lengths);
    append_string_column(input.containments->rest_fields,
                         input.rest_fields_concat, input.rest_lengths);

    ctx.out.num_containments = ctx.graph.containments.size();

    ctx.out.containment_container_ids_zstd =
        gfaz::Codec::compress_delta_varint_uint32(input.containments->container_ids);
    ctx.out.containment_contained_ids_zstd =
        gfaz::Codec::compress_delta_varint_uint32(input.containments->contained_ids);
    ctx.out.containment_container_orients_zstd =
        gfaz::Codec::compress_orientations(input.containments->container_orients);
    ctx.out.containment_contained_orients_zstd =
        gfaz::Codec::compress_orientations(input.containments->contained_orients);
    ctx.out.containment_positions_zstd =
        gfaz::Codec::zstd_compress_uint32_vector(input.containments->positions);
    ctx.out.containment_overlaps_zstd =
        gfaz::Codec::zstd_compress_string(input.overlaps_concat);
    ctx.out.containment_overlap_lengths_zstd =
        gfaz::Codec::zstd_compress_uint32_vector(input.overlap_lengths);
    ctx.out.containment_rest_fields_zstd =
        gfaz::Codec::zstd_compress_string(input.rest_fields_concat);
    ctx.out.containment_rest_lengths_zstd =
        gfaz::Codec::zstd_compress_uint32_vector(input.rest_lengths);
  }

  release_containment_fields(ctx);

  const auto end = std::chrono::high_resolution_clock::now();
  return elapsed_ms(start, end);
}

} // namespace

// ---------------------------------------------------------------------------
// Grammar compression core
// ---------------------------------------------------------------------------

// Run multi-round 2-mer grammar compression on paths and walks
static void run_grammar_compression(std::vector<std::vector<gfaz::NodeId>> &paths,
                                    std::vector<std::vector<gfaz::NodeId>> &walks,
                                    uint32_t &next_id, int num_rounds,
                                    size_t freq_threshold, uint32_t layer_start,
                                    std::vector<Packed2mer> &rulebook,
                                    int num_threads) {
  // Compute total traversal size for throughput calculation
  const size_t total_elements =
      total_node_count(paths) + total_node_count(walks);
  double data_size_mb = total_elements * sizeof(int32_t) / (1024.0 * 1024.0);

  // Timing accumulators
  double time_generate_rules_ms = 0;
  double time_encode_paths_ms = 0;
  double time_compact_sort_remap_ms = 0;
  int actual_rounds = 0;

  auto total_start = std::chrono::high_resolution_clock::now();

  for (int round = 0; round < num_rounds; ++round) {
    auto t0 = std::chrono::high_resolution_clock::now();
    RuleGenerator gen;
    CompressionRules2Mer rules = gen.generate_rules_2mer_combined(
        paths, walks, next_id, freq_threshold, num_threads);
    auto t1 = std::chrono::high_resolution_clock::now();
    const double round_generate_ms = elapsed_ms(t0, t1);
    time_generate_rules_ms += round_generate_ms;

    if (rules.rule_id_to_kmer.empty())
      break;

    actual_rounds++;

    t0 = std::chrono::high_resolution_clock::now();
    PathEncoder encoder;
    std::vector<uint8_t> rules_used;
    encoder.encode_paths_2mer(paths, rules, rules_used);
    encoder.encode_paths_2mer(walks, rules, rules_used);
    t1 = std::chrono::high_resolution_clock::now();
    const double round_encode_ms = elapsed_ms(t0, t1);
    time_encode_paths_ms += round_encode_ms;

    t0 = std::chrono::high_resolution_clock::now();

    rules.kmer_to_rule_id.clear();

    // =========================================================================
    // FUSED Compact + Sort + Remap (single-pass optimization)
    // Instead of: compact → remap → sort → remap (2 passes over paths/walks)
    // We do: extract used → sort by value → build unified map → remap (1 pass)
    // This improves cache locality and reduces total work.
    // =========================================================================

    const size_t num_rules = rules.rule_id_to_kmer.size();

    // Build prefix sum for parallel extraction
    std::vector<uint32_t> prefix(num_rules + 1, 0);
    for (size_t i = 0; i < num_rules; ++i)
      prefix[i + 1] =
          prefix[i] + (i < rules_used.size() && rules_used[i] ? 1 : 0);

    uint32_t total_used = prefix.back();

    if (total_used == 0) {
      rules.rule_id_to_kmer.clear();
      rules.rule_id_to_kmer.shrink_to_fit();
      t1 = std::chrono::high_resolution_clock::now();
      const double round_remap_ms = elapsed_ms(t0, t1);
      time_compact_sort_remap_ms += round_remap_ms;
      if (gfaz_debug_enabled()) {
        print_grammar_round({round + 1,
                             round_generate_ms,
                             round_encode_ms,
                             round_remap_ms,
                             total_used,
                             num_rules,
                             {}});
      }
      break;
    }

    // Extract used rules with original IDs
    std::vector<std::pair<Packed2mer, uint32_t>> used(total_used);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t idx = 0; idx < num_rules; ++idx) {
      if (idx < rules_used.size() && rules_used[idx]) {
        uint32_t pos = prefix[idx];
        uint32_t old_id = rules.rules_start_id + static_cast<uint32_t>(idx);
        used[pos] = {rules.rule_id_to_kmer[idx], old_id};
      }
    }

    rules.rule_id_to_kmer.clear();
    rules.rule_id_to_kmer.shrink_to_fit();
    prefix.clear();
    prefix.shrink_to_fit();

    // Sort by 2-mer value for better ZSTD compression
#if defined(_OPENMP) && defined(__GLIBCXX__)
    __gnu_parallel::sort(
        used.begin(), used.end(),
        [](const auto &a, const auto &b) { return a.first < b.first; });
#else
    std::sort(used.begin(), used.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });
#endif

    // Build unified ID mapping: old_id → new_sorted_id
    std::vector<uint32_t> id_map(rules_used.size(), 0);
    uint32_t new_id = next_id;
    for (const auto &[kmer, old_id] : used) {
      uint32_t idx = old_id - rules.rules_start_id;
      id_map[idx] = new_id++;
    }
    rules_used.clear();
    remap_rule_ids(paths, rules.rules_start_id, id_map);
    remap_rule_ids(walks, rules.rules_start_id, id_map);

    id_map.clear();
    id_map.shrink_to_fit();

    // Add sorted rules to rulebook
    for (size_t i = 0; i < used.size(); ++i) {
      uint32_t final_id = next_id + static_cast<uint32_t>(i);
      size_t idx = final_id - layer_start;
      if (idx >= rulebook.size())
        rulebook.resize(idx + 1);
      rulebook[idx] = used[i].first;
    }

    next_id = new_id;

    t1 = std::chrono::high_resolution_clock::now();
    const double round_remap_ms = elapsed_ms(t0, t1);
    time_compact_sort_remap_ms += round_remap_ms;
    if (gfaz_debug_enabled()) {
      print_grammar_round({round + 1, round_generate_ms, round_encode_ms,
                           round_remap_ms, total_used, num_rules,
                           read_process_memory_snapshot()});
    }
  }

  auto total_end = std::chrono::high_resolution_clock::now();
  double total_ms = elapsed_ms(total_start, total_end);

  if (gfaz_debug_enabled()) {
    print_grammar_timing_breakdown(actual_rounds, data_size_mb, total_elements,
                                   time_generate_rules_ms, time_encode_paths_ms,
                                   time_compact_sort_remap_ms, total_ms);
  }
}

// ---------------------------------------------------------------------------
// Public workflow entry point
// ---------------------------------------------------------------------------

gfaz::CompressedData compress_gfa(const std::string &gfa_file_path, int num_rounds,
                            size_t freq_threshold, int delta_round,
                            int num_threads, bool show_stats) {
  // CPU compression pipeline:
  // 1. Parse the input GFA into gfaz::GfaGraph.
  // 2. Initialize output metadata and establish the traversal ID space.
  // 3. Apply delta encoding to paths and walks.
  // 4. Run multi-round 2-mer grammar compression over traversals.
  // 5. Encode the rulebook into compressed rule columns.
  // 6. Compress each record group into gfaz::CompressedData field blocks:
  //    P, W, S, L, optional fields, J, and C.
  // 7. Emit compression stats and return the finalized container.
  ScopedOMPThreads omp_scope(num_threads);
  CompressionContext ctx;
  ctx.num_rounds = num_rounds;
  ctx.freq_threshold = freq_threshold;
  ctx.delta_round = delta_round;
  ctx.num_threads = num_threads;
  ctx.show_stats = show_stats;

  {
    GfaParser parser;
    ctx.graph = parser.parse(gfa_file_path, num_threads);
  }
  log_cpu_memory_checkpoint("[CPU Workflow][Memory] after parse");

  // Stage 1: initialize shared output metadata and traversal symbol space.
  initialize_output_metadata(ctx);
  prepare_id_space_for_traversal_transform(ctx);

  auto compress_total_start = std::chrono::high_resolution_clock::now();

  // Stage 2: transform traversals and build the grammar rulebook.
  const double time_delta_ms = apply_delta_transform(ctx);
  log_cpu_memory_checkpoint("[CPU Workflow][Memory] after delta transform");
  const double time_grammar_ms = run_grammar_stage(ctx);
  log_cpu_memory_checkpoint("[CPU Workflow][Memory] after grammar compression");

  // Debug-only check of traversal reduction after grammar encoding.
  if (gfaz_debug_enabled()) {
    size_t original_path_len = 0, original_walk_len = 0;
    size_t encoded_path_len = 0, encoded_walk_len = 0;

    for (const auto &len : ctx.out.original_path_lengths)
      original_path_len += len;
    for (const auto &len : ctx.out.original_walk_lengths)
      original_walk_len += len;
    for (const auto &p : ctx.graph.paths_data.traversals)
      encoded_path_len += p.size();
    for (const auto &w : ctx.graph.walks.walks)
      encoded_walk_len += w.size();

    print_traversal_reduction({original_path_len, encoded_path_len,
                               original_walk_len, encoded_walk_len});
  }

  const uint32_t rule_count = ctx.next_id - ctx.layer_start;

  // Stage 3: compress the grammar rulebook into compressed rule columns.
  const double time_rule_columns_ms = compress_rule_columns(ctx);
  log_cpu_memory_checkpoint(
      "[CPU Entropy][Memory] After rule columns compression");

  // Stage 4: compress record-group payloads and metadata columns.
  const double time_paths_ms = compress_path_fields(ctx);
  const double time_walks_ms = compress_walk_fields(ctx);
  log_cpu_memory_checkpoint(
      "[CPU Entropy][Memory] After Path/Walk compression");
  const double time_segments_links_ms = compress_segment_link_fields(ctx);
  const double time_optional_fields_ms = compress_optional_fields(ctx);
  log_cpu_memory_checkpoint(
      "[CPU Entropy][Memory] After Segment/Link/Optional compression");
  const double time_jumps_ms = compress_jump_fields(ctx);
  const double time_containments_ms = compress_containment_fields(ctx);
  const double time_entropy_ms = time_rule_columns_ms + time_paths_ms +
                                 time_walks_ms + time_segments_links_ms +
                                 time_optional_fields_ms + time_jumps_ms +
                                 time_containments_ms;
  log_cpu_memory_checkpoint("[CPU Entropy][Memory] After total entropy coding");

  auto compress_total_end = std::chrono::high_resolution_clock::now();
  double compress_total_ms =
      elapsed_ms(compress_total_start, compress_total_end);

  // Stage 5: report timing/ratio breakdowns and finalize the output container.
  if (gfaz_debug_enabled()) {
    const double rules_size_mb =
        rule_count * sizeof(int32_t) * 2 / (1024.0 * 1024.0);

    std::vector<EntropyStepDebugInfo> entropy_steps = {
        {"compress rule columns", "delta+ZSTD", time_rule_columns_ms,
         collect_rules_ratio(ctx.out)},
        {"compress path fields", "ZSTD", time_paths_ms,
         collect_path_ratio(ctx.out)},
        {"compress walk fields", "ZSTD+varint", time_walks_ms,
         collect_walk_ratio(ctx.out)},
        {"compress segment/link fields", "mixed", time_segments_links_ms,
         collect_segment_link_ratio(ctx.out)},
        {"compress optional fields", "mixed", time_optional_fields_ms,
         collect_optional_field_ratio(ctx.out)}};
    if (ctx.out.num_jumps > 0) {
      entropy_steps.push_back({"compress jump fields", "mixed", time_jumps_ms,
                               collect_jump_ratio(ctx.out)});
    }
    if (ctx.out.num_containments > 0) {
      entropy_steps.push_back({"compress containment fields", "mixed",
                               time_containments_ms,
                               collect_containment_ratio(ctx.out)});
    }

    print_cpu_compression_timing(
        {ctx.data_size_mb, ctx.total_elements, ctx.delta_round, time_delta_ms,
         time_grammar_ms, rules_size_mb, std::move(entropy_steps),
         time_entropy_ms, compress_total_ms});
  }

  print_compression_stats(ctx.out, ctx.num_segments, ctx.show_stats);

  return ctx.out;
}
