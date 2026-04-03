#include "compression_workflow.hpp"
#include "codec.hpp"
#include "debug_log.hpp"
#include "gfa_parser.hpp"
#include "packed_2mer.hpp"
#include "path_encoder.hpp"
#include "rule_generator.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(_OPENMP) && defined(__GLIBCXX__)
#include <parallel/algorithm>
#endif

namespace {

constexpr const char *kCompressionErrorPrefix = "Compression workflow error: ";

using Clock = std::chrono::high_resolution_clock;

double elapsed_ms(const Clock::time_point &start,
                  const Clock::time_point &end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

double gbps_from_mb(double size_mb, double time_ms) {
  return (time_ms > 0) ? (size_mb / 1024.0) / (time_ms / 1000.0) : 0;
}

size_t total_node_count(const std::vector<std::vector<NodeId>> &sequences) {
  size_t total = 0;
  for (const auto &seq : sequences)
    total += seq.size();
  return total;
}

std::string format_size(size_t bytes);

struct ProcessMemorySnapshot {
  size_t vm_rss_kb = 0;
  size_t vm_hwm_kb = 0;
};

ProcessMemorySnapshot read_process_memory_snapshot() {
  ProcessMemorySnapshot snapshot;

  std::ifstream status("/proc/self/status");
  if (!status)
    return snapshot;

  std::string line;
  while (std::getline(status, line)) {
    auto parse_kb_field = [&](const char *prefix, size_t &out_value) {
      const std::string_view view(line);
      const std::string_view key(prefix);
      if (view.size() < key.size() || view.substr(0, key.size()) != key)
        return false;

      std::istringstream iss(std::string(view.substr(key.size())));
      size_t value = 0;
      std::string unit;
      if (iss >> value >> unit) {
        out_value = value;
        return true;
      }
      return false;
    };

    if (parse_kb_field("VmRSS:", snapshot.vm_rss_kb))
      continue;
    parse_kb_field("VmHWM:", snapshot.vm_hwm_kb);
  }

  return snapshot;
}

void log_memory_checkpoint(const std::string &label) {
  if (!gfaz_debug_enabled())
    return;

  const ProcessMemorySnapshot snapshot = read_process_memory_snapshot();
  std::cerr << "[CPU Compress][Memory] " << label
            << " | VmRSS=" << format_size(snapshot.vm_rss_kb * 1024)
            << " | VmHWM=" << format_size(snapshot.vm_hwm_kb * 1024)
            << std::endl;
}

void append_string_column(const std::vector<std::string> &values,
                          std::string &concatenated,
                          std::vector<uint32_t> &lengths) {
  concatenated.clear();
  lengths.clear();
  for (const auto &value : values) {
    concatenated += value;
    lengths.push_back(static_cast<uint32_t>(value.size()));
  }
}

// Compress an optional field column based on its type
CompressedOptionalFieldColumn
compress_optional_column(const OptionalFieldColumn &col) {
  CompressedOptionalFieldColumn out;
  out.tag = col.tag;
  out.type = col.type;

  switch (col.type) {
  case 'i':
    out.num_elements = col.int_values.size();
    out.int_values_zstd = Codec::compress_varint_int64(col.int_values);
    break;
  case 'f':
    out.num_elements = col.float_values.size();
    out.float_values_zstd = Codec::zstd_compress_float_vector(col.float_values);
    break;
  case 'A':
    out.num_elements = col.char_values.size();
    out.char_values_zstd = Codec::zstd_compress_char_vector(col.char_values);
    break;
  case 'Z':
  case 'J':
  case 'H':
    out.num_elements = col.string_lengths.size();
    out.strings_zstd = Codec::zstd_compress_string(col.concatenated_strings);
    out.string_lengths_zstd =
        Codec::zstd_compress_uint32_vector(col.string_lengths);
    break;
  case 'B':
    out.num_elements = col.b_subtypes.size();
    out.b_subtypes_zstd = Codec::zstd_compress_char_vector(col.b_subtypes);
    out.b_lengths_zstd = Codec::zstd_compress_uint32_vector(col.b_lengths);
    out.b_concat_bytes_zstd = Codec::zstd_compress_string(
        std::string(col.b_concat_bytes.begin(), col.b_concat_bytes.end()));
    break;
  default:
    throw std::invalid_argument(std::string(kCompressionErrorPrefix) +
                                "unsupported optional field type '" +
                                std::string(1, col.type) + "' for tag '" +
                                col.tag + "'");
  }
  return out;
}

// Flatten paths into a single array with metadata
void flatten_paths(const std::vector<std::vector<NodeId>> &paths,
                   const std::vector<std::string> &path_names,
                   const std::vector<std::string> &path_overlaps,
                   std::vector<int32_t> &flattened,
                   std::vector<uint32_t> &lengths, std::string &names_concat,
                   std::vector<uint32_t> &name_lengths,
                   std::string &overlaps_concat,
                   std::vector<uint32_t> &overlap_lengths) {
  if (path_names.size() != paths.size() || path_overlaps.size() != paths.size())
    throw std::runtime_error(
        std::string(kCompressionErrorPrefix) +
        "invalid path metadata (names/overlaps count does not match paths)");

  const size_t total = total_node_count(paths);

  flattened.clear();
  flattened.resize(total);
  lengths.clear();

  append_string_column(path_names, names_concat, name_lengths);
  append_string_column(path_overlaps, overlaps_concat, overlap_lengths);

  size_t offset = 0;
  for (size_t i = 0; i < paths.size(); ++i) {
    const auto &path = paths[i];
    lengths.push_back(static_cast<uint32_t>(path.size()));

    for (NodeId node : path)
      flattened[offset++] = node;
  }
}

// Flatten segment sequences (skip index 0 placeholder)
void flatten_segments(const std::vector<std::string> &sequences,
                      std::string &concat, std::vector<uint32_t> &lengths,
                      uint32_t max_id) {
  concat.clear();
  lengths.clear();
  for (size_t i = 1; i < sequences.size() && i < max_id; ++i) {
    concat += sequences[i];
    lengths.push_back(static_cast<uint32_t>(sequences[i].size()));
  }
}

// Split rules into first/second arrays and delta encode
void process_rules(const std::vector<Packed2mer> &rulebook,
                   uint32_t layer_start_id,
                   const std::vector<LayerRuleRange> &ranges,
                   std::vector<int32_t> &first, std::vector<int32_t> &second) {
  size_t total = 0;
  for (const auto &r : ranges)
    total += r.end_id - r.start_id;

  first.clear();
  second.clear();
  first.reserve(total);
  second.reserve(total);

  for (const auto &range : ranges) {
    for (uint32_t id = range.start_id; id < range.end_id; ++id) {
      size_t idx = id - layer_start_id;
      if (idx < rulebook.size()) {
        first.push_back(unpack_first(rulebook[idx]));
        second.push_back(unpack_second(rulebook[idx]));
      } else {
        first.push_back(0);
        second.push_back(0);
      }
    }
  }

  Codec::delta_encode_int32(first);
  Codec::delta_encode_int32(second);
}

void flatten_walks(const std::vector<std::vector<NodeId>> &walks,
                   std::vector<int32_t> &flattened,
                   std::vector<uint32_t> &lengths) {
  flattened.clear();
  flattened.reserve(total_node_count(walks));
  lengths.clear();
  lengths.reserve(walks.size());

  for (const auto &walk : walks) {
    lengths.push_back(static_cast<uint32_t>(walk.size()));
    for (NodeId node : walk)
      flattened.push_back(node);
  }
}

void remap_rule_ids(std::vector<std::vector<NodeId>> &sequences,
                    uint32_t rules_start_id,
                    const std::vector<uint32_t> &id_map) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (size_t i = 0; i < sequences.size(); ++i) {
    for (auto &node : sequences[i]) {
      int32_t abs_id = std::abs(node);
      if (static_cast<uint32_t>(abs_id) < rules_start_id)
        continue;

      const uint32_t offset = static_cast<uint32_t>(abs_id) - rules_start_id;
      if (offset >= id_map.size() || id_map[offset] == 0)
        continue;

      node = (node > 0) ? static_cast<int32_t>(id_map[offset])
                        : -static_cast<int32_t>(id_map[offset]);
    }
  }
}

std::string format_size(size_t bytes) {
  std::ostringstream oss;
  if (bytes >= 1024 * 1024)
    oss << std::fixed << std::setprecision(2) << (bytes / (1024.0 * 1024.0))
        << " MB";
  else if (bytes >= 1024)
    oss << std::fixed << std::setprecision(1) << (bytes / 1024.0) << " KB";
  else
    oss << bytes << " Bytes";
  return oss.str();
}

void print_compression_stats(const CompressedData &d, size_t num_segments,
                             int num_rounds, int delta_round) {
  auto sum_optional =
      [](const std::vector<CompressedOptionalFieldColumn> &cols) {
        size_t total = 0;
        for (const auto &c : cols) {
          total += c.int_values_zstd.payload.size();
          total += c.float_values_zstd.payload.size();
          total += c.char_values_zstd.payload.size();
          total += c.strings_zstd.payload.size();
          total += c.string_lengths_zstd.payload.size();
          total += c.b_subtypes_zstd.payload.size();
          total += c.b_lengths_zstd.payload.size();
          total += c.b_concat_bytes_zstd.payload.size();
        }
        return total;
      };

  size_t seg_opt = sum_optional(d.segment_optional_fields_zstd);
  size_t link_opt = sum_optional(d.link_optional_fields_zstd);

  size_t jump_bytes = d.jump_from_ids_zstd.payload.size() +
                      d.jump_to_ids_zstd.payload.size() +
                      d.jump_from_orients_zstd.payload.size() +
                      d.jump_to_orients_zstd.payload.size() +
                      d.jump_distances_zstd.payload.size() +
                      d.jump_distance_lengths_zstd.payload.size() +
                      d.jump_rest_fields_zstd.payload.size() +
                      d.jump_rest_lengths_zstd.payload.size();

  size_t containment_bytes =
      d.containment_container_ids_zstd.payload.size() +
      d.containment_contained_ids_zstd.payload.size() +
      d.containment_container_orients_zstd.payload.size() +
      d.containment_contained_orients_zstd.payload.size() +
      d.containment_positions_zstd.payload.size() +
      d.containment_overlaps_zstd.payload.size() +
      d.containment_overlap_lengths_zstd.payload.size() +
      d.containment_rest_fields_zstd.payload.size() +
      d.containment_rest_lengths_zstd.payload.size();

  size_t total = 0;
  total += d.rules_first_zstd.payload.size();
  total += d.rules_second_zstd.payload.size();
  total += d.paths_zstd.payload.size();
  total += d.sequence_lengths.size() * sizeof(uint32_t);
  total += d.layer_rule_ranges.size() * sizeof(LayerRuleRange);
  total += d.names_zstd.payload.size();
  total += d.name_lengths_zstd.payload.size();
  total += d.overlaps_zstd.payload.size();
  total += d.overlap_lengths_zstd.payload.size();
  total += d.segment_sequences_zstd.payload.size();
  total += d.segment_seq_lengths_zstd.payload.size();
  total += seg_opt;
  total += d.link_from_ids_zstd.payload.size();
  total += d.link_to_ids_zstd.payload.size();
  total += d.link_from_orients_zstd.payload.size();
  total += d.link_to_orients_zstd.payload.size();
  total += d.link_overlap_nums_zstd.payload.size();
  total += d.link_overlap_ops_zstd.payload.size();
  total += link_opt;
  total += jump_bytes;
  total += containment_bytes;
  total += d.walks_zstd.payload.size();
  total += d.walk_lengths.size() * sizeof(uint32_t);
  total += d.walk_sample_ids_zstd.payload.size();
  total += d.walk_sample_id_lengths_zstd.payload.size();
  total += d.walk_hap_indices_zstd.payload.size();
  total += d.walk_seq_ids_zstd.payload.size();
  total += d.walk_seq_id_lengths_zstd.payload.size();
  total += d.walk_seq_starts_zstd.payload.size();
  total += d.walk_seq_ends_zstd.payload.size();

  if (!gfaz_debug_enabled())
    return;

  std::cerr << "\n=== Compressed Data Breakdown ===" << std::endl;

  // Rules
  std::cerr << "Rules (2-mer grammar):" << std::endl;
  std::cerr << "  layer_rule_ranges:      " << std::setw(12)
            << format_size(d.layer_rule_ranges.size() * sizeof(LayerRuleRange))
            << std::endl;
  std::cerr << "  rules_first_zstd:       " << std::setw(12)
            << format_size(d.rules_first_zstd.payload.size()) << std::endl;
  std::cerr << "  rules_second_zstd:      " << std::setw(12)
            << format_size(d.rules_second_zstd.payload.size()) << std::endl;

  // Paths
  std::cerr << "Paths (P-lines): " << d.sequence_lengths.size() << std::endl;
  std::cerr << "  sequence_lengths:       " << std::setw(12)
            << format_size(d.sequence_lengths.size() * sizeof(uint32_t))
            << std::endl;
  std::cerr << "  paths_zstd:             " << std::setw(12)
            << format_size(d.paths_zstd.payload.size()) << std::endl;
  std::cerr << "  names_zstd:             " << std::setw(12)
            << format_size(d.names_zstd.payload.size()) << std::endl;
  std::cerr << "  name_lengths_zstd:      " << std::setw(12)
            << format_size(d.name_lengths_zstd.payload.size()) << std::endl;
  std::cerr << "  overlaps_zstd:          " << std::setw(12)
            << format_size(d.overlaps_zstd.payload.size()) << std::endl;
  std::cerr << "  overlap_lengths_zstd:   " << std::setw(12)
            << format_size(d.overlap_lengths_zstd.payload.size()) << std::endl;

  // Segments
  std::cerr << "Segments (S-lines): " << num_segments << std::endl;
  std::cerr << "  segment_sequences_zstd: " << std::setw(12)
            << format_size(d.segment_sequences_zstd.payload.size())
            << std::endl;
  std::cerr << "  segment_seq_lengths:    " << std::setw(12)
            << format_size(d.segment_seq_lengths_zstd.payload.size())
            << std::endl;
  std::cerr << "  segment_optional_fields:" << std::setw(12)
            << format_size(seg_opt) << " ("
            << d.segment_optional_fields_zstd.size() << " columns)"
            << std::endl;

  // Links
  std::cerr << "Links (L-lines): " << d.num_links << std::endl;
  std::cerr << "  link_from_ids_zstd:     " << std::setw(12)
            << format_size(d.link_from_ids_zstd.payload.size()) << std::endl;
  std::cerr << "  link_to_ids_zstd:       " << std::setw(12)
            << format_size(d.link_to_ids_zstd.payload.size()) << std::endl;
  std::cerr << "  link_from_orients_zstd: " << std::setw(12)
            << format_size(d.link_from_orients_zstd.payload.size())
            << std::endl;
  std::cerr << "  link_to_orients_zstd:   " << std::setw(12)
            << format_size(d.link_to_orients_zstd.payload.size()) << std::endl;
  std::cerr << "  link_overlap_nums_zstd: " << std::setw(12)
            << format_size(d.link_overlap_nums_zstd.payload.size())
            << std::endl;
  std::cerr << "  link_overlap_ops_zstd:  " << std::setw(12)
            << format_size(d.link_overlap_ops_zstd.payload.size()) << std::endl;
  std::cerr << "  link_optional_fields:   " << std::setw(12)
            << format_size(link_opt) << " ("
            << d.link_optional_fields_zstd.size() << " columns)" << std::endl;

  // J/C lines
  if (d.num_jumps > 0) {
    std::cerr << "Jumps (J-lines): " << d.num_jumps << std::endl;
    std::cerr << "  total:                  " << std::setw(12)
              << format_size(jump_bytes) << std::endl;
  }
  if (d.num_containments > 0) {
    std::cerr << "Containments (C-lines): " << d.num_containments << std::endl;
    std::cerr << "  total:                  " << std::setw(12)
              << format_size(containment_bytes) << std::endl;
  }

  // Walks
  if (!d.walk_lengths.empty()) {
    std::cerr << "Walks (W-lines): " << d.walk_lengths.size() << std::endl;
    std::cerr << "  walk_lengths:           " << std::setw(12)
              << format_size(d.walk_lengths.size() * sizeof(uint32_t))
              << std::endl;
    std::cerr << "  walks_zstd:             " << std::setw(12)
              << format_size(d.walks_zstd.payload.size()) << std::endl;
    std::cerr << "  walk_sample_ids_zstd:   " << std::setw(12)
              << format_size(d.walk_sample_ids_zstd.payload.size())
              << std::endl;
    std::cerr << "  walk_sample_id_lengths: " << std::setw(12)
              << format_size(d.walk_sample_id_lengths_zstd.payload.size())
              << std::endl;
    std::cerr << "  walk_hap_indices_zstd:  " << std::setw(12)
              << format_size(d.walk_hap_indices_zstd.payload.size())
              << std::endl;
    std::cerr << "  walk_seq_ids_zstd:      " << std::setw(12)
              << format_size(d.walk_seq_ids_zstd.payload.size()) << std::endl;
    std::cerr << "  walk_seq_id_lengths:    " << std::setw(12)
              << format_size(d.walk_seq_id_lengths_zstd.payload.size())
              << std::endl;
    std::cerr << "  walk_seq_starts_zstd:   " << std::setw(12)
              << format_size(d.walk_seq_starts_zstd.payload.size())
              << std::endl;
    std::cerr << "  walk_seq_ends_zstd:     " << std::setw(12)
              << format_size(d.walk_seq_ends_zstd.payload.size()) << std::endl;
  }

  // Total
  std::cerr << "----------------------------------------" << std::endl;
  std::cerr << "Total: " << format_size(total) << " (" << num_rounds
            << " rounds, delta=" << delta_round << ")" << std::endl;
}

} // namespace

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
    time_generate_rules_ms += elapsed_ms(t0, t1);
    if (gfaz_debug_enabled()) {
      std::ostringstream label;
      label << "round " << (round + 1) << " after generate_rules";
      log_memory_checkpoint(label.str());
    }

    if (rules.rule_id_to_kmer.empty())
      break;

    actual_rounds++;

    t0 = std::chrono::high_resolution_clock::now();
    PathEncoder encoder;
    std::vector<uint8_t> rules_used;
    encoder.encode_paths_2mer(paths, rules, rules_used);
    encoder.encode_paths_2mer(walks, rules, rules_used);
    t1 = std::chrono::high_resolution_clock::now();
    time_encode_paths_ms += elapsed_ms(t0, t1);
    if (gfaz_debug_enabled()) {
      std::ostringstream label;
      label << "round " << (round + 1) << " after encode_paths";
      log_memory_checkpoint(label.str());
    }

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

    GFAZ_LOG("  Round " << (round + 1) << ": 2-mers found = " << num_rules
                        << ", 2-mers used = " << total_used);

    if (total_used == 0) {
      rules.rule_id_to_kmer.clear();
      rules.rule_id_to_kmer.shrink_to_fit();
      t1 = std::chrono::high_resolution_clock::now();
      time_compact_sort_remap_ms += elapsed_ms(t0, t1);
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
    time_compact_sort_remap_ms += elapsed_ms(t0, t1);
    if (gfaz_debug_enabled()) {
      std::ostringstream label;
      label << "round " << (round + 1) << " after compact_sort_remap";
      log_memory_checkpoint(label.str());
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
              << std::setprecision(2) << time_generate_rules_ms << " ms  ("
              << std::setprecision(2)
              << gbps_from_mb(data_size_mb, time_generate_rules_ms) << " GB/s)"
              << std::endl;
    std::cerr << "    2. encode_paths_2mer:            " << std::fixed
              << std::setprecision(2) << time_encode_paths_ms << " ms  ("
              << std::setprecision(2)
              << gbps_from_mb(data_size_mb, time_encode_paths_ms) << " GB/s)"
              << std::endl;
    std::cerr << "    3. compact_sort_remap:           " << std::fixed
              << std::setprecision(2) << time_compact_sort_remap_ms << " ms  ("
              << std::setprecision(2)
              << gbps_from_mb(data_size_mb, time_compact_sort_remap_ms)
              << " GB/s)" << std::endl;
    std::cerr << "    ─────────────────────────────" << std::endl;
    std::cerr << "    Rounds subtotal:                 " << std::fixed
              << std::setprecision(2) << round_total_ms << " ms  ("
              << std::setprecision(2)
              << gbps_from_mb(data_size_mb, round_total_ms) << " GB/s)"
              << std::endl;
    std::cerr << "    TOTAL (grammar compression):     " << std::fixed
              << std::setprecision(2) << total_ms << " ms  ("
              << std::setprecision(2) << gbps_from_mb(data_size_mb, total_ms)
              << " GB/s)" << std::endl;
  }
}

// ---------------------------------------------------------------------------
// Public workflow entry point
// ---------------------------------------------------------------------------

CompressedData compress_gfa(const std::string &gfa_file_path, int num_rounds,
                            size_t freq_threshold, int delta_round,
                            int num_threads) {
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
  log_memory_checkpoint("after dropping segment name maps");

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
  GFAZ_LOG("Delta encoding rounds: " << delta_round);
  auto t_delta_start = std::chrono::high_resolution_clock::now();
  uint32_t max_abs = 0;
  for (int i = 0; i < delta_round; ++i) {
    uint32_t path_max = Codec::delta_transform_and_max_abs(graph.paths);
    uint32_t walk_max = Codec::delta_transform_and_max_abs(graph.walks.walks);
    max_abs = std::max(max_abs, std::max(path_max, walk_max));
  }
  auto t_delta_end = std::chrono::high_resolution_clock::now();
  double time_delta_ms = elapsed_ms(t_delta_start, t_delta_end);
  log_memory_checkpoint("after delta transform");

  if (max_abs >= next_id) {
    if (max_abs == UINT32_MAX)
      throw std::overflow_error(
          std::string(kCompressionErrorPrefix) +
          "delta values too large for rule ID assignment");
    next_id = max_abs + 1;
  }

  // Grammar compression
  GFAZ_LOG("Grammar compression rounds: " << num_rounds);
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
      std::cerr << "Path/Walk compression:" << std::endl;
      std::cerr << "  Paths: " << original_path_len << " -> "
                << encoded_path_len << " (" << std::fixed
                << std::setprecision(2)
                << (original_path_len > 0
                        ? 100.0 * encoded_path_len / original_path_len
                        : 0)
                << "%)" << std::endl;
      if (original_walk_len > 0) {
        std::cerr << "  Walks: " << original_walk_len << " -> "
                  << encoded_walk_len << " (" << std::fixed
                  << std::setprecision(2)
                  << 100.0 * encoded_walk_len / original_walk_len << "%)"
                  << std::endl;
      }
      std::cerr << "  Total: " << original_total << " -> " << encoded_total
                << " (" << std::fixed << std::setprecision(2)
                << (original_total > 0 ? 100.0 * encoded_total / original_total
                                       : 0)
                << "%)" << std::endl;
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
  log_memory_checkpoint("after process_rules");

  if (gfaz_debug_enabled()) {
    std::cerr << "Raw rule vectors before ZSTD:" << std::endl;
    std::cerr << "  first:  " << first.size() << " elements, "
              << format_size(first.size() * sizeof(int32_t)) << std::endl;
    std::cerr << "  second: " << second.size() << " elements, "
              << format_size(second.size() * sizeof(int32_t)) << std::endl;
  }

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
  log_memory_checkpoint("after rule zstd");

  // Compress paths
  auto t_zstd_start = std::chrono::high_resolution_clock::now();
  {
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
  }
  graph.paths.clear();
  graph.paths.shrink_to_fit();
  graph.path_names.clear();
  graph.path_names.shrink_to_fit();
  graph.path_overlaps.clear();
  graph.path_overlaps.shrink_to_fit();
  log_memory_checkpoint("after path flatten+compress");

  // Compress walks
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
  log_memory_checkpoint("after walk flatten+compress");

  // Compress segments
  std::string seg_concat;
  std::vector<uint32_t> seg_lens;
  flatten_segments(graph.node_sequences, seg_concat, seg_lens, next_id);
  size_t num_segments = seg_lens.size();

  // Compress links
  const auto &links = graph.links;
  out.num_links = links.from_ids.size();

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
  log_memory_checkpoint("after segment/link compress");

  // Compress optional fields
  for (const auto &col : graph.segment_optional_fields)
    out.segment_optional_fields_zstd.push_back(compress_optional_column(col));

  for (const auto &col : graph.link_optional_fields)
    out.link_optional_fields_zstd.push_back(compress_optional_column(col));
  graph.segment_optional_fields.clear();
  graph.segment_optional_fields.shrink_to_fit();
  graph.link_optional_fields.clear();
  graph.link_optional_fields.shrink_to_fit();
  log_memory_checkpoint("after optional-field compress");

  // Compress J-lines (jumps)
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
  log_memory_checkpoint("after jump compress");

  // Compress C-lines (containments)
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
  log_memory_checkpoint("after containment compress");

  auto t_zstd_end = std::chrono::high_resolution_clock::now();
  double time_zstd_ms = elapsed_ms(t_zstd_start, t_zstd_end);
  log_memory_checkpoint("after all field compression");

  auto compress_total_end = std::chrono::high_resolution_clock::now();
  double compress_total_ms =
      elapsed_ms(compress_total_start, compress_total_end);

  if (gfaz_debug_enabled()) {
    std::cerr << "\n[CPU Compress] === TIMING BREAKDOWN (" << std::fixed
              << std::setprecision(1) << data_size_mb << " MB, "
              << total_elements << " path+walk elements) ===" << std::endl;
    std::cerr << "  Pre-processing:" << std::endl;
    std::cerr << "    1. delta_transform (x" << delta_round
              << "):         " << std::fixed << std::setprecision(2)
              << time_delta_ms << " ms  (" << std::setprecision(2)
              << gbps_from_mb(data_size_mb, time_delta_ms) << " GB/s)"
              << std::endl;
    std::cerr << "  Grammar compression:" << std::endl;
    std::cerr << "    2. run_grammar_compression:      " << std::fixed
              << std::setprecision(2) << time_grammar_ms << " ms  ("
              << std::setprecision(2)
              << gbps_from_mb(data_size_mb, time_grammar_ms) << " GB/s)"
              << std::endl;
    std::cerr << "  Post-processing:" << std::endl;
    double rules_size_mb = rule_count * sizeof(int32_t) * 2 / (1024.0 * 1024.0);
    std::cerr << "    3. split+delta_encode rules (" << std::fixed
              << std::setprecision(1) << rules_size_mb
              << " MB): " << std::setprecision(2) << time_process_rules_ms
              << " ms  (" << std::setprecision(2)
              << gbps_from_mb(rules_size_mb, time_process_rules_ms) << " GB/s)"
              << std::endl;
    std::cerr << "    4. ZSTD compress rules:          " << std::fixed
              << std::setprecision(2) << time_zstd_rules_ms << " ms"
              << std::endl;
    std::cerr << "    5. ZSTD compress (all fields):   " << std::fixed
              << std::setprecision(2) << time_zstd_ms << " ms" << std::endl;
    std::cerr << "  ─────────────────────────────────" << std::endl;
    std::cerr << "  TOTAL (compress_gfa):              " << std::fixed
              << std::setprecision(2) << compress_total_ms << " ms  ("
              << std::setprecision(2)
              << gbps_from_mb(data_size_mb, compress_total_ms) << " GB/s)"
              << std::endl;
  }

  print_compression_stats(out, num_segments, num_rounds, delta_round);

  return out;
}
