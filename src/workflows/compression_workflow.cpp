#include "workflows/compression_workflow.hpp"
#include "codec/codec.hpp"
#include "utils/debug_log.hpp"
#include "io/gfa_parser.hpp"
#include "grammar/packed_2mer.hpp"
#include "grammar/path_encoder.hpp"
#include "grammar/rule_generator.hpp"
#include "workflows/compression_utils.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>

#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(_OPENMP) && defined(__GLIBCXX__)
#include <parallel/algorithm>
#endif

using namespace gfz::compression_utils;

// ---------------------------------------------------------------------------
// Grammar compression core
// ---------------------------------------------------------------------------

// Run multi-round 2-mer grammar compression on paths and walks
static void run_grammar_compression(std::vector<std::vector<NodeId>> &paths,
                                    std::vector<std::vector<NodeId>> &walks,
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
        std::cerr << "  round " << (round + 1) << ": generate="
                  << std::fixed << std::setprecision(2) << round_generate_ms
                  << " ms, encode=" << round_encode_ms
                  << " ms, remap=" << round_remap_ms
                  << " ms, rules=" << total_used << "/" << num_rules
                  << std::endl;
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
      const auto snapshot = read_process_memory_snapshot();
      std::cerr << "  round " << (round + 1)
                << " | RssAnon="
                << format_size(snapshot.rss_anon_kb * 1024)
                << " | VmRSS=" << format_size(snapshot.vm_rss_kb * 1024)
                << " | VmHWM=" << format_size(snapshot.vm_hwm_kb * 1024)
                << " | generate=" << std::fixed << std::setprecision(2)
                << round_generate_ms
                << " ms, encode=" << round_encode_ms
                << " ms, remap=" << round_remap_ms
                << " ms, rules=" << total_used << "/" << num_rules
                << std::endl;
    }
  }

  auto total_end = std::chrono::high_resolution_clock::now();
  double total_ms = elapsed_ms(total_start, total_end);
  double round_total_ms = time_generate_rules_ms + time_encode_paths_ms +
                          time_compact_sort_remap_ms;

  if (gfaz_debug_enabled()) {
    std::cerr << "[CPU Grammar Compress] === TIMING BREAKDOWN ("
              << actual_rounds << " rounds, " << std::fixed
              << std::setprecision(1) << data_size_mb << " MB, "
              << total_elements << " elements) ===" << std::endl;
    std::cerr << "    1. generate_rules_2mer:          " << std::fixed
              << std::setprecision(2) << time_generate_rules_ms << " ms"
              << std::endl;
    std::cerr << "    2. encode_paths_2mer:            " << std::fixed
              << std::setprecision(2) << time_encode_paths_ms << " ms"
              << std::endl;
    std::cerr << "    3. compact_sort_remap:           " << std::fixed
              << std::setprecision(2) << time_compact_sort_remap_ms << " ms"
              << std::endl;
    std::cerr << "    ─────────────────────────────" << std::endl;
    std::cerr << "    Rounds subtotal:                 " << std::fixed
              << std::setprecision(2) << round_total_ms << " ms" << std::endl;
    std::cerr << "    TOTAL (grammar compression):     " << std::fixed
              << std::setprecision(2) << total_ms << " ms" << std::endl;
  }
}

// ---------------------------------------------------------------------------
// Public workflow entry point
// ---------------------------------------------------------------------------

CompressedData compress_gfa(const std::string &gfa_file_path, int num_rounds,
                            size_t freq_threshold, int delta_round,
                            int num_threads, bool show_stats) {
  ScopedOMPThreads omp_scope(num_threads);
  CompressedData out;

  GfaGraph graph;
  {
    GfaParser parser;
    graph = parser.parse(gfa_file_path, num_threads);
  }
  log_memory_checkpoint("after parse");

  out.header_line = graph.header_line;
  out.delta_round = delta_round;

  graph.node_name_to_id.clear();
  graph.node_name_to_id.rehash(0);
  graph.node_id_to_name.clear();
  graph.node_id_to_name.shrink_to_fit();

  // Store original lengths before any transformation (for exact allocation on
  // decompression)
  for (const auto &p : graph.paths)
    out.original_path_lengths.push_back(static_cast<uint32_t>(p.size()));
  for (const auto &w : graph.walks.walks)
    out.original_walk_lengths.push_back(static_cast<uint32_t>(w.size()));

  uint32_t next_id = graph.node_id_to_name.size();

  // Compute total traversal size for throughput calculation
  const size_t total_elements =
      total_node_count(graph.paths) + total_node_count(graph.walks.walks);
  double data_size_mb = total_elements * sizeof(int32_t) / (1024.0 * 1024.0);

  auto compress_total_start = std::chrono::high_resolution_clock::now();

  // Delta encoding with max value tracking to prevent ID collision
  auto t_delta_start = std::chrono::high_resolution_clock::now();
  uint32_t max_abs = 0;
  for (int i = 0; i < delta_round; ++i) {
    uint32_t path_max = Codec::delta_transform_and_max_abs(graph.paths);
    uint32_t walk_max = Codec::delta_transform_and_max_abs(graph.walks.walks);
    max_abs = std::max(max_abs, std::max(path_max, walk_max));
  }
  auto t_delta_end = std::chrono::high_resolution_clock::now();
  double time_delta_ms = elapsed_ms(t_delta_start, t_delta_end);

  if (max_abs >= next_id) {
    if (max_abs == UINT32_MAX)
      throw std::overflow_error(
          std::string(kCompressionErrorPrefix) +
          "delta values too large for rule ID assignment");
    next_id = max_abs + 1;
  }

  // Grammar compression
  uint32_t layer_start = next_id;
  std::vector<Packed2mer> rulebook;

  auto t_grammar_start = std::chrono::high_resolution_clock::now();
  run_grammar_compression(graph.paths, graph.walks.walks, next_id, num_rounds,
                          freq_threshold, layer_start, rulebook, num_threads);
  auto t_grammar_end = std::chrono::high_resolution_clock::now();
  double time_grammar_ms = elapsed_ms(t_grammar_start, t_grammar_end);
  log_memory_checkpoint("after grammar compression");

  // Print path/walk length comparison: original vs encoded
  {
    size_t original_path_len = 0, original_walk_len = 0;
    size_t encoded_path_len = 0, encoded_walk_len = 0;

    for (const auto &len : out.original_path_lengths)
      original_path_len += len;
    for (const auto &len : out.original_walk_lengths)
      original_walk_len += len;
    for (const auto &p : graph.paths)
      encoded_path_len += p.size();
    for (const auto &w : graph.walks.walks)
      encoded_walk_len += w.size();

    size_t original_total = original_path_len + original_walk_len;
    size_t encoded_total = encoded_path_len + encoded_walk_len;

    if (gfaz_debug_enabled()) {
      std::cerr << "[CPU Compress] traversal reduction:" << std::endl;
      std::cerr << "  paths: " << original_path_len << " -> "
                << encoded_path_len;
      if (original_path_len > 0) {
        std::cerr << " (" << std::fixed << std::setprecision(2)
                  << 100.0 * encoded_path_len / original_path_len << "%)";
      }
      std::cerr << std::endl;
      std::cerr << "  walks: " << original_walk_len << " -> "
                << encoded_walk_len;
      if (original_walk_len > 0) {
        std::cerr << " (" << std::fixed << std::setprecision(2)
                  << 100.0 * encoded_walk_len / original_walk_len << "%)";
      }
      std::cerr << std::endl;
      std::cerr << "  total: " << original_total << " -> " << encoded_total;
      if (original_total > 0) {
        std::cerr << " (" << std::fixed << std::setprecision(2)
                  << 100.0 * encoded_total / original_total << "%)";
      }
      std::cerr << std::endl;
    }
  }

  uint32_t rule_count = next_id - layer_start;
  out.layer_rule_ranges.push_back(
      LayerRuleRange{2, layer_start, next_id, 0, rule_count * 2});

  // Split rules and delta encode
  auto t_process_rules_start = std::chrono::high_resolution_clock::now();
  std::vector<int32_t> first, second;
  process_rules(rulebook, layer_start, out.layer_rule_ranges, first, second);
  rulebook.clear();
  rulebook.shrink_to_fit();
  auto t_process_rules_end = std::chrono::high_resolution_clock::now();
  double time_process_rules_ms =
      elapsed_ms(t_process_rules_start, t_process_rules_end);

  // ZSTD compress rules
  auto t_zstd_rules_start = std::chrono::high_resolution_clock::now();
  {
#ifdef _OPENMP
#pragma omp parallel sections
    {
#pragma omp section
      out.rules_first_zstd = Codec::zstd_compress_int32_vector(first);
#pragma omp section
      out.rules_second_zstd = Codec::zstd_compress_int32_vector(second);
    }
#else
    out.rules_first_zstd = Codec::zstd_compress_int32_vector(first);
    out.rules_second_zstd = Codec::zstd_compress_int32_vector(second);
#endif
  }
  auto t_zstd_rules_end = std::chrono::high_resolution_clock::now();
  double time_zstd_rules_ms = elapsed_ms(t_zstd_rules_start, t_zstd_rules_end);

  // Compress paths
  auto t_zstd_start = std::chrono::high_resolution_clock::now();
  double time_paths_ms = 0;
  double time_walks_ms = 0;
  double time_segments_links_ms = 0;
  double time_optional_fields_ms = 0;
  double time_jumps_ms = 0;
  double time_containments_ms = 0;
  {
    const auto t_paths_start = std::chrono::high_resolution_clock::now();
    std::vector<int32_t> flat;
    std::string names, overlaps;
    std::vector<uint32_t> name_lens, overlap_lens;

    flatten_paths(graph.paths, graph.path_names, graph.path_overlaps, flat,
                  out.sequence_lengths, names, name_lens, overlaps,
                  overlap_lens);

#ifdef _OPENMP
#pragma omp parallel sections
    {
#pragma omp section
      out.paths_zstd = Codec::zstd_compress_int32_vector(flat);
#pragma omp section
      {
        out.names_zstd = Codec::zstd_compress_string(names);
        out.name_lengths_zstd = Codec::zstd_compress_uint32_vector(name_lens);
      }
#pragma omp section
      {
        out.overlaps_zstd = Codec::zstd_compress_string(overlaps);
        out.overlap_lengths_zstd =
            Codec::zstd_compress_uint32_vector(overlap_lens);
      }
    }
#else
    out.paths_zstd = Codec::zstd_compress_int32_vector(flat);
    out.names_zstd = Codec::zstd_compress_string(names);
    out.name_lengths_zstd = Codec::zstd_compress_uint32_vector(name_lens);
    out.overlaps_zstd = Codec::zstd_compress_string(overlaps);
    out.overlap_lengths_zstd = Codec::zstd_compress_uint32_vector(overlap_lens);
#endif
    const auto t_paths_end = std::chrono::high_resolution_clock::now();
    time_paths_ms = elapsed_ms(t_paths_start, t_paths_end);
  }
  graph.paths.clear();
  graph.paths.shrink_to_fit();
  graph.path_names.clear();
  graph.path_names.shrink_to_fit();
  graph.path_overlaps.clear();
  graph.path_overlaps.shrink_to_fit();

  // Compress walks
  const auto t_walks_start = std::chrono::high_resolution_clock::now();
  if (!graph.walks.walks.empty()) {
    std::vector<int32_t> flat;
    flatten_walks(graph.walks.walks, flat, out.walk_lengths);
    out.walks_zstd = Codec::zstd_compress_int32_vector(flat);

    std::string sample_ids, seq_ids;
    std::vector<uint32_t> sample_lens, seq_lens;
    append_string_column(graph.walks.sample_ids, sample_ids, sample_lens);
    append_string_column(graph.walks.seq_ids, seq_ids, seq_lens);

    out.walk_sample_ids_zstd = Codec::zstd_compress_string(sample_ids);
    out.walk_sample_id_lengths_zstd =
        Codec::zstd_compress_uint32_vector(sample_lens);
    out.walk_hap_indices_zstd =
        Codec::zstd_compress_uint32_vector(graph.walks.hap_indices);
    out.walk_seq_ids_zstd = Codec::zstd_compress_string(seq_ids);
    out.walk_seq_id_lengths_zstd = Codec::zstd_compress_uint32_vector(seq_lens);
    out.walk_seq_starts_zstd =
        Codec::compress_varint_int64(graph.walks.seq_starts);
    out.walk_seq_ends_zstd = Codec::compress_varint_int64(graph.walks.seq_ends);
  }
  const auto t_walks_end = std::chrono::high_resolution_clock::now();
  time_walks_ms = elapsed_ms(t_walks_start, t_walks_end);
  graph.walks.walks.clear();
  graph.walks.walks.shrink_to_fit();
  graph.walks.sample_ids.clear();
  graph.walks.sample_ids.shrink_to_fit();
  graph.walks.hap_indices.clear();
  graph.walks.hap_indices.shrink_to_fit();
  graph.walks.seq_ids.clear();
  graph.walks.seq_ids.shrink_to_fit();
  graph.walks.seq_starts.clear();
  graph.walks.seq_starts.shrink_to_fit();
  graph.walks.seq_ends.clear();
  graph.walks.seq_ends.shrink_to_fit();

  // Compress segments
  std::string seg_concat;
  std::vector<uint32_t> seg_lens;
  flatten_segments(graph.node_sequences, seg_concat, seg_lens, next_id);
  size_t num_segments = seg_lens.size();

  // Compress links
  const auto &links = graph.links;
  out.num_links = links.from_ids.size();

  const auto t_segments_links_start = std::chrono::high_resolution_clock::now();
#ifdef _OPENMP
#pragma omp parallel sections
  {
#pragma omp section
    {
      out.segment_sequences_zstd = Codec::zstd_compress_string(seg_concat);
      out.segment_seq_lengths_zstd =
          Codec::zstd_compress_uint32_vector(seg_lens);
    }
#pragma omp section
    {
      out.link_from_ids_zstd =
          Codec::compress_delta_varint_uint32(links.from_ids);
      out.link_to_ids_zstd = Codec::compress_delta_varint_uint32(links.to_ids);
    }
#pragma omp section
    {
      out.link_from_orients_zstd =
          Codec::compress_orientations(links.from_orients);
      out.link_to_orients_zstd = Codec::compress_orientations(links.to_orients);
    }
#pragma omp section
    {
      out.link_overlap_nums_zstd =
          Codec::zstd_compress_uint32_vector(links.overlap_nums);
      out.link_overlap_ops_zstd =
          Codec::zstd_compress_char_vector(links.overlap_ops);
    }
  }
#else
  out.segment_sequences_zstd = Codec::zstd_compress_string(seg_concat);
  out.segment_seq_lengths_zstd = Codec::zstd_compress_uint32_vector(seg_lens);
  out.link_from_ids_zstd = Codec::compress_delta_varint_uint32(links.from_ids);
  out.link_to_ids_zstd = Codec::compress_delta_varint_uint32(links.to_ids);
  out.link_from_orients_zstd = Codec::compress_orientations(links.from_orients);
  out.link_to_orients_zstd = Codec::compress_orientations(links.to_orients);
  out.link_overlap_nums_zstd =
      Codec::zstd_compress_uint32_vector(links.overlap_nums);
  out.link_overlap_ops_zstd =
      Codec::zstd_compress_char_vector(links.overlap_ops);
#endif
  const auto t_segments_links_end = std::chrono::high_resolution_clock::now();
  time_segments_links_ms =
      elapsed_ms(t_segments_links_start, t_segments_links_end);
  graph.node_sequences.clear();
  graph.node_sequences.shrink_to_fit();
  graph.links.from_ids.clear();
  graph.links.from_ids.shrink_to_fit();
  graph.links.to_ids.clear();
  graph.links.to_ids.shrink_to_fit();
  graph.links.from_orients.clear();
  graph.links.from_orients.shrink_to_fit();
  graph.links.to_orients.clear();
  graph.links.to_orients.shrink_to_fit();
  graph.links.overlap_nums.clear();
  graph.links.overlap_nums.shrink_to_fit();
  graph.links.overlap_ops.clear();
  graph.links.overlap_ops.shrink_to_fit();

  // Compress optional fields
  const auto t_optional_fields_start = std::chrono::high_resolution_clock::now();
  for (const auto &col : graph.segment_optional_fields)
    out.segment_optional_fields_zstd.push_back(compress_optional_column(col));

  for (const auto &col : graph.link_optional_fields)
    out.link_optional_fields_zstd.push_back(compress_optional_column(col));
  const auto t_optional_fields_end = std::chrono::high_resolution_clock::now();
  time_optional_fields_ms =
      elapsed_ms(t_optional_fields_start, t_optional_fields_end);
  graph.segment_optional_fields.clear();
  graph.segment_optional_fields.shrink_to_fit();
  graph.link_optional_fields.clear();
  graph.link_optional_fields.shrink_to_fit();

  // Compress J-lines (jumps)
  const auto t_jumps_start = std::chrono::high_resolution_clock::now();
  if (!graph.jumps.from_ids.empty()) {
    out.num_jumps = graph.jumps.size();

    std::string dists_concat, rest_concat;
    std::vector<uint32_t> dist_lens, rest_lens;
    for (const auto &d : graph.jumps.distances) {
      dists_concat += d;
      dist_lens.push_back(static_cast<uint32_t>(d.size()));
    }
    for (const auto &r : graph.jumps.rest_fields) {
      rest_concat += r;
      rest_lens.push_back(static_cast<uint32_t>(r.size()));
    }

    out.jump_from_ids_zstd =
        Codec::compress_delta_varint_uint32(graph.jumps.from_ids);
    out.jump_to_ids_zstd =
        Codec::compress_delta_varint_uint32(graph.jumps.to_ids);
    out.jump_from_orients_zstd =
        Codec::compress_orientations(graph.jumps.from_orients);
    out.jump_to_orients_zstd =
        Codec::compress_orientations(graph.jumps.to_orients);
    out.jump_distances_zstd = Codec::zstd_compress_string(dists_concat);
    out.jump_distance_lengths_zstd =
        Codec::zstd_compress_uint32_vector(dist_lens);
    out.jump_rest_fields_zstd = Codec::zstd_compress_string(rest_concat);
    out.jump_rest_lengths_zstd = Codec::zstd_compress_uint32_vector(rest_lens);
  }
  const auto t_jumps_end = std::chrono::high_resolution_clock::now();
  time_jumps_ms = elapsed_ms(t_jumps_start, t_jumps_end);
  graph.jumps.from_ids.clear();
  graph.jumps.from_ids.shrink_to_fit();
  graph.jumps.from_orients.clear();
  graph.jumps.from_orients.shrink_to_fit();
  graph.jumps.to_ids.clear();
  graph.jumps.to_ids.shrink_to_fit();
  graph.jumps.to_orients.clear();
  graph.jumps.to_orients.shrink_to_fit();
  graph.jumps.distances.clear();
  graph.jumps.distances.shrink_to_fit();
  graph.jumps.rest_fields.clear();
  graph.jumps.rest_fields.shrink_to_fit();

  // Compress C-lines (containments)
  const auto t_containments_start = std::chrono::high_resolution_clock::now();
  if (!graph.containments.container_ids.empty()) {
    out.num_containments = graph.containments.size();

    std::string overlaps_concat, rest_concat;
    std::vector<uint32_t> overlap_lens, rest_lens;
    for (const auto &o : graph.containments.overlaps) {
      overlaps_concat += o;
      overlap_lens.push_back(static_cast<uint32_t>(o.size()));
    }
    for (const auto &r : graph.containments.rest_fields) {
      rest_concat += r;
      rest_lens.push_back(static_cast<uint32_t>(r.size()));
    }

    out.containment_container_ids_zstd =
        Codec::compress_delta_varint_uint32(graph.containments.container_ids);
    out.containment_contained_ids_zstd =
        Codec::compress_delta_varint_uint32(graph.containments.contained_ids);
    out.containment_container_orients_zstd =
        Codec::compress_orientations(graph.containments.container_orients);
    out.containment_contained_orients_zstd =
        Codec::compress_orientations(graph.containments.contained_orients);
    out.containment_positions_zstd =
        Codec::zstd_compress_uint32_vector(graph.containments.positions);
    out.containment_overlaps_zstd =
        Codec::zstd_compress_string(overlaps_concat);
    out.containment_overlap_lengths_zstd =
        Codec::zstd_compress_uint32_vector(overlap_lens);
    out.containment_rest_fields_zstd = Codec::zstd_compress_string(rest_concat);
    out.containment_rest_lengths_zstd =
        Codec::zstd_compress_uint32_vector(rest_lens);
  }
  const auto t_containments_end = std::chrono::high_resolution_clock::now();
  time_containments_ms = elapsed_ms(t_containments_start, t_containments_end);
  graph.containments.container_ids.clear();
  graph.containments.container_ids.shrink_to_fit();
  graph.containments.container_orients.clear();
  graph.containments.container_orients.shrink_to_fit();
  graph.containments.contained_ids.clear();
  graph.containments.contained_ids.shrink_to_fit();
  graph.containments.contained_orients.clear();
  graph.containments.contained_orients.shrink_to_fit();
  graph.containments.positions.clear();
  graph.containments.positions.shrink_to_fit();
  graph.containments.overlaps.clear();
  graph.containments.overlaps.shrink_to_fit();
  graph.containments.rest_fields.clear();
  graph.containments.rest_fields.shrink_to_fit();

  auto t_zstd_end = std::chrono::high_resolution_clock::now();
  double time_zstd_ms = elapsed_ms(t_zstd_start, t_zstd_end);
  log_memory_checkpoint("after all field compression");

  auto compress_total_end = std::chrono::high_resolution_clock::now();
  double compress_total_ms =
      elapsed_ms(compress_total_start, compress_total_end);

  if (gfaz_debug_enabled()) {
    auto block_original_size = [](const ZstdCompressedBlock &block) {
      return block.original_size;
    };
    auto block_payload_size = [](const ZstdCompressedBlock &block) {
      return block.payload.size();
    };
    auto sum_optional_sizes =
        [&](const std::vector<CompressedOptionalFieldColumn> &cols) {
          size_t original = 0;
          size_t compressed = 0;
          for (const auto &c : cols) {
            const ZstdCompressedBlock *blocks[] = {
                &c.int_values_zstd,      &c.float_values_zstd,
                &c.char_values_zstd,     &c.strings_zstd,
                &c.string_lengths_zstd,  &c.b_subtypes_zstd,
                &c.b_lengths_zstd,       &c.b_concat_bytes_zstd};
            for (const auto *block : blocks) {
              original += block_original_size(*block);
              compressed += block_payload_size(*block);
            }
          }
          return std::pair<size_t, size_t>{original, compressed};
        };
    auto format_ratio = [&](size_t original, size_t compressed) {
      std::ostringstream oss;
      oss << format_size(original) << " -> " << format_size(compressed);
      if (compressed > 0) {
        oss << " (" << std::fixed << std::setprecision(2)
            << static_cast<double>(original) / static_cast<double>(compressed)
            << "x)";
      }
      return oss.str();
    };
    auto emit_post_step = [&](int step, const std::string &label,
                              const std::string &codec_label, double time_ms,
                              size_t original, size_t compressed) {
      std::cerr << "    " << step << ". " << label;
      if (!codec_label.empty())
        std::cerr << " (" << codec_label << ")";
      std::cerr << ": " << std::fixed << std::setprecision(2) << time_ms
                << " ms";
      if (original > 0 || compressed > 0)
        std::cerr << " | " << format_ratio(original, compressed);
      std::cerr << std::endl;
    };

    const size_t rules_original =
        block_original_size(out.rules_first_zstd) +
        block_original_size(out.rules_second_zstd);
    const size_t rules_compressed =
        block_payload_size(out.rules_first_zstd) +
        block_payload_size(out.rules_second_zstd);
    const size_t path_original =
        block_original_size(out.paths_zstd) + block_original_size(out.names_zstd) +
        block_original_size(out.name_lengths_zstd) +
        block_original_size(out.overlaps_zstd) +
        block_original_size(out.overlap_lengths_zstd);
    const size_t path_compressed =
        block_payload_size(out.paths_zstd) + block_payload_size(out.names_zstd) +
        block_payload_size(out.name_lengths_zstd) +
        block_payload_size(out.overlaps_zstd) +
        block_payload_size(out.overlap_lengths_zstd);
    const size_t walk_original =
        block_original_size(out.walks_zstd) +
        block_original_size(out.walk_sample_ids_zstd) +
        block_original_size(out.walk_sample_id_lengths_zstd) +
        block_original_size(out.walk_hap_indices_zstd) +
        block_original_size(out.walk_seq_ids_zstd) +
        block_original_size(out.walk_seq_id_lengths_zstd) +
        block_original_size(out.walk_seq_starts_zstd) +
        block_original_size(out.walk_seq_ends_zstd);
    const size_t walk_compressed =
        block_payload_size(out.walks_zstd) +
        block_payload_size(out.walk_sample_ids_zstd) +
        block_payload_size(out.walk_sample_id_lengths_zstd) +
        block_payload_size(out.walk_hap_indices_zstd) +
        block_payload_size(out.walk_seq_ids_zstd) +
        block_payload_size(out.walk_seq_id_lengths_zstd) +
        block_payload_size(out.walk_seq_starts_zstd) +
        block_payload_size(out.walk_seq_ends_zstd);
    const size_t segments_links_original =
        block_original_size(out.segment_sequences_zstd) +
        block_original_size(out.segment_seq_lengths_zstd) +
        block_original_size(out.link_from_ids_zstd) +
        block_original_size(out.link_to_ids_zstd) +
        block_original_size(out.link_from_orients_zstd) +
        block_original_size(out.link_to_orients_zstd) +
        block_original_size(out.link_overlap_nums_zstd) +
        block_original_size(out.link_overlap_ops_zstd);
    const size_t segments_links_compressed =
        block_payload_size(out.segment_sequences_zstd) +
        block_payload_size(out.segment_seq_lengths_zstd) +
        block_payload_size(out.link_from_ids_zstd) +
        block_payload_size(out.link_to_ids_zstd) +
        block_payload_size(out.link_from_orients_zstd) +
        block_payload_size(out.link_to_orients_zstd) +
        block_payload_size(out.link_overlap_nums_zstd) +
        block_payload_size(out.link_overlap_ops_zstd);
    const auto [optional_original, optional_compressed] =
        sum_optional_sizes(out.segment_optional_fields_zstd);
    const auto [link_optional_original, link_optional_compressed] =
        sum_optional_sizes(out.link_optional_fields_zstd);
    const size_t optional_fields_original =
        optional_original + link_optional_original;
    const size_t optional_fields_compressed =
        optional_compressed + link_optional_compressed;
    const size_t jumps_original =
        block_original_size(out.jump_from_ids_zstd) +
        block_original_size(out.jump_from_orients_zstd) +
        block_original_size(out.jump_to_ids_zstd) +
        block_original_size(out.jump_to_orients_zstd) +
        block_original_size(out.jump_distances_zstd) +
        block_original_size(out.jump_distance_lengths_zstd) +
        block_original_size(out.jump_rest_fields_zstd) +
        block_original_size(out.jump_rest_lengths_zstd);
    const size_t jumps_compressed =
        block_payload_size(out.jump_from_ids_zstd) +
        block_payload_size(out.jump_from_orients_zstd) +
        block_payload_size(out.jump_to_ids_zstd) +
        block_payload_size(out.jump_to_orients_zstd) +
        block_payload_size(out.jump_distances_zstd) +
        block_payload_size(out.jump_distance_lengths_zstd) +
        block_payload_size(out.jump_rest_fields_zstd) +
        block_payload_size(out.jump_rest_lengths_zstd);
    const size_t containments_original =
        block_original_size(out.containment_container_ids_zstd) +
        block_original_size(out.containment_container_orients_zstd) +
        block_original_size(out.containment_contained_ids_zstd) +
        block_original_size(out.containment_contained_orients_zstd) +
        block_original_size(out.containment_positions_zstd) +
        block_original_size(out.containment_overlaps_zstd) +
        block_original_size(out.containment_overlap_lengths_zstd) +
        block_original_size(out.containment_rest_fields_zstd) +
        block_original_size(out.containment_rest_lengths_zstd);
    const size_t containments_compressed =
        block_payload_size(out.containment_container_ids_zstd) +
        block_payload_size(out.containment_container_orients_zstd) +
        block_payload_size(out.containment_contained_ids_zstd) +
        block_payload_size(out.containment_contained_orients_zstd) +
        block_payload_size(out.containment_positions_zstd) +
        block_payload_size(out.containment_overlaps_zstd) +
        block_payload_size(out.containment_overlap_lengths_zstd) +
        block_payload_size(out.containment_rest_fields_zstd) +
        block_payload_size(out.containment_rest_lengths_zstd);

    int step = 3;
    std::cerr << "\n[CPU Compress] === TIMING BREAKDOWN (" << std::fixed
              << std::setprecision(1) << data_size_mb << " MB, "
              << total_elements << " path+walk elements) ===" << std::endl;
    std::cerr << "  Pre-processing:" << std::endl;
    std::cerr << "    1. delta_transform (x" << delta_round
              << "):         " << std::fixed << std::setprecision(2)
              << time_delta_ms << " ms" << std::endl;
    std::cerr << "  Grammar compression:" << std::endl;
    std::cerr << "    2. run_grammar_compression:      " << std::fixed
              << std::setprecision(2) << time_grammar_ms << " ms"
              << std::endl;
    std::cerr << "  Post-processing:" << std::endl;
    double rules_size_mb = rule_count * sizeof(int32_t) * 2 / (1024.0 * 1024.0);
    std::cerr << "    " << step++
              << ". encode rules (" << std::fixed
              << std::setprecision(1) << rules_size_mb
              << " MB): " << std::setprecision(2) << time_process_rules_ms
              << " ms" << std::endl;
    emit_post_step(step++, "compress rule fields", "ZSTD", time_zstd_rules_ms,
                   rules_original, rules_compressed);
    emit_post_step(step++, "compress path fields", "ZSTD", time_paths_ms,
                   path_original, path_compressed);
    emit_post_step(step++, "compress walk fields", "ZSTD+varint",
                   time_walks_ms, walk_original, walk_compressed);
    emit_post_step(step++, "compress segment/link fields", "mixed",
                   time_segments_links_ms, segments_links_original,
                   segments_links_compressed);
    emit_post_step(step++, "compress optional fields", "mixed",
                   time_optional_fields_ms, optional_fields_original,
                   optional_fields_compressed);
    if (out.num_jumps > 0) {
      emit_post_step(step++, "compress jump fields", "mixed", time_jumps_ms,
                     jumps_original, jumps_compressed);
    }
    if (out.num_containments > 0) {
      emit_post_step(step++, "compress containment fields", "mixed",
                     time_containments_ms, containments_original,
                     containments_compressed);
    }
    std::cerr << "    field compression subtotal:      " << std::fixed
              << std::setprecision(2) << time_zstd_ms << " ms" << std::endl;
    std::cerr << "  ─────────────────────────────────" << std::endl;
    std::cerr << "  TOTAL (compress_gfa):              " << std::fixed
              << std::setprecision(2) << compress_total_ms << " ms"
              << std::endl;
  }

  print_compression_stats(out, num_segments, show_stats);

  return out;
}
