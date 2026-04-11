#include "workflows/compression_debug.hpp"
#include "utils/debug_log.hpp"

#include <iomanip>
#include <iostream>
#include <sstream>

namespace gfz::compression_debug {

namespace {

using gfz::runtime_utils::format_memory_snapshot;
using gfz::runtime_utils::format_size;

std::string format_reduction_percent(size_t original, size_t encoded) {
  if (original == 0)
    return {};

  std::ostringstream oss;
  oss << " (" << std::fixed << std::setprecision(2)
      << 100.0 * static_cast<double>(encoded) / static_cast<double>(original)
      << "%)";
  return oss.str();
}

std::string format_ratio(const CompressionRatio &ratio) {
  std::ostringstream oss;
  oss << format_size(ratio.original_bytes) << " -> "
      << format_size(ratio.compressed_bytes);
  if (ratio.compressed_bytes > 0) {
    oss << " (" << std::fixed << std::setprecision(2)
        << static_cast<double>(ratio.original_bytes) /
               static_cast<double>(ratio.compressed_bytes)
        << "x)";
  }
  return oss.str();
}

} // namespace

void print_grammar_round(const GrammarRoundDebugInfo &info) {
  std::cerr << "  round " << info.round
            << " | " << format_memory_snapshot(info.memory)
            << " | generate=" << std::fixed << std::setprecision(2)
            << info.generate_ms << " ms, encode=" << info.encode_ms
            << " ms, remap=" << info.remap_ms
            << " ms, rules=" << info.rules_used << "/" << info.num_rules
            << std::endl;
}

void print_grammar_timing_breakdown(int actual_rounds, double data_size_mb,
                                    size_t total_elements,
                                    double time_generate_rules_ms,
                                    double time_encode_paths_ms,
                                    double time_compact_sort_remap_ms,
                                    double total_ms) {
  const double round_total_ms = time_generate_rules_ms + time_encode_paths_ms +
                                time_compact_sort_remap_ms;

  std::cerr << "[CPU Grammar Compress] === TIMING BREAKDOWN (" << actual_rounds
            << " rounds, " << std::fixed << std::setprecision(1)
            << data_size_mb << " MB, " << total_elements << " elements) ==="
            << std::endl;
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

void print_traversal_reduction(const TraversalReductionDebugInfo &info) {
  const size_t original_total = info.original_paths + info.original_walks;
  const size_t encoded_total = info.encoded_paths + info.encoded_walks;

  std::cerr << "[CPU Compress] traversal reduction:" << std::endl;
  std::cerr << "  paths: " << info.original_paths << " -> " << info.encoded_paths
            << format_reduction_percent(info.original_paths, info.encoded_paths)
            << std::endl;
  std::cerr << "  walks: " << info.original_walks << " -> " << info.encoded_walks
            << format_reduction_percent(info.original_walks, info.encoded_walks)
            << std::endl;
  std::cerr << "  total: " << original_total << " -> " << encoded_total
            << format_reduction_percent(original_total, encoded_total)
            << std::endl;
}

void log_cpu_memory_checkpoint(const std::string &label) {
  if (!gfaz_debug_enabled())
    return;

  const auto snapshot = runtime_utils::read_process_memory_snapshot();
  std::cerr << label
            << " | " << format_memory_snapshot(snapshot) << std::endl;
}

CompressionRatio sum_optional_field_ratio(
    const std::vector<CompressedOptionalFieldColumn> &cols) {
  CompressionRatio total;
  for (const auto &c : cols) {
    const ZstdCompressedBlock *blocks[] = {
        &c.int_values_zstd,     &c.float_values_zstd,    &c.char_values_zstd,
        &c.strings_zstd,        &c.string_lengths_zstd,  &c.b_subtypes_zstd,
        &c.b_lengths_zstd,      &c.b_concat_bytes_zstd};
    for (const auto *block : blocks) {
      total.original_bytes += block->original_size;
      total.compressed_bytes += block->payload.size();
    }
  }
  return total;
}

CompressionRatio block_ratio(const ZstdCompressedBlock &block) {
  return CompressionRatio{block.original_size, block.payload.size()};
}

CompressionRatio sum_ratios(std::initializer_list<CompressionRatio> ratios) {
  CompressionRatio total;
  for (const auto &ratio : ratios) {
    total.original_bytes += ratio.original_bytes;
    total.compressed_bytes += ratio.compressed_bytes;
  }
  return total;
}

CompressionRatio collect_rules_ratio(const CompressedData &data) {
  return sum_ratios({block_ratio(data.rules_first_zstd),
                     block_ratio(data.rules_second_zstd)});
}

CompressionRatio collect_path_ratio(const CompressedData &data) {
  return sum_ratios({block_ratio(data.paths_zstd), block_ratio(data.names_zstd),
                     block_ratio(data.name_lengths_zstd),
                     block_ratio(data.overlaps_zstd),
                     block_ratio(data.overlap_lengths_zstd)});
}

CompressionRatio collect_walk_ratio(const CompressedData &data) {
  return sum_ratios({block_ratio(data.walks_zstd),
                     block_ratio(data.walk_sample_ids_zstd),
                     block_ratio(data.walk_sample_id_lengths_zstd),
                     block_ratio(data.walk_hap_indices_zstd),
                     block_ratio(data.walk_seq_ids_zstd),
                     block_ratio(data.walk_seq_id_lengths_zstd),
                     block_ratio(data.walk_seq_starts_zstd),
                     block_ratio(data.walk_seq_ends_zstd)});
}

CompressionRatio collect_segment_link_ratio(const CompressedData &data) {
  return sum_ratios({block_ratio(data.segment_sequences_zstd),
                     block_ratio(data.segment_seq_lengths_zstd),
                     block_ratio(data.link_from_ids_zstd),
                     block_ratio(data.link_to_ids_zstd),
                     block_ratio(data.link_from_orients_zstd),
                     block_ratio(data.link_to_orients_zstd),
                     block_ratio(data.link_overlap_nums_zstd),
                     block_ratio(data.link_overlap_ops_zstd)});
}

CompressionRatio collect_optional_field_ratio(const CompressedData &data) {
  return sum_ratios({sum_optional_field_ratio(data.segment_optional_fields_zstd),
                     sum_optional_field_ratio(data.link_optional_fields_zstd)});
}

CompressionRatio collect_jump_ratio(const CompressedData &data) {
  return sum_ratios({block_ratio(data.jump_from_ids_zstd),
                     block_ratio(data.jump_from_orients_zstd),
                     block_ratio(data.jump_to_ids_zstd),
                     block_ratio(data.jump_to_orients_zstd),
                     block_ratio(data.jump_distances_zstd),
                     block_ratio(data.jump_distance_lengths_zstd),
                     block_ratio(data.jump_rest_fields_zstd),
                     block_ratio(data.jump_rest_lengths_zstd)});
}

CompressionRatio collect_containment_ratio(const CompressedData &data) {
  return sum_ratios({block_ratio(data.containment_container_ids_zstd),
                     block_ratio(data.containment_container_orients_zstd),
                     block_ratio(data.containment_contained_ids_zstd),
                     block_ratio(data.containment_contained_orients_zstd),
                     block_ratio(data.containment_positions_zstd),
                     block_ratio(data.containment_overlaps_zstd),
                     block_ratio(data.containment_overlap_lengths_zstd),
                     block_ratio(data.containment_rest_fields_zstd),
                     block_ratio(data.containment_rest_lengths_zstd)});
}

void print_cpu_compression_timing(const CpuCompressionTimingDebugInfo &info) {
  int step = 1;

  std::cerr << "\n[CPU Compress] === TIMING BREAKDOWN (" << std::fixed
            << std::setprecision(1) << info.data_size_mb << " MB, "
            << info.total_elements << " path+walk elements) ===" << std::endl;
  std::cerr << "  Traversal transform:" << std::endl;
  std::cerr << "    " << step++ << ". delta_transform (x" << info.delta_round
            << "):         " << std::fixed << std::setprecision(2)
            << info.time_delta_ms << " ms" << std::endl;
  std::cerr << "  Grammar compression:" << std::endl;
  std::cerr << "    " << step++ << ". run_grammar_compression:      "
            << std::fixed
            << std::setprecision(2) << info.time_grammar_ms << " ms"
            << std::endl;
  std::cerr << "  Entropy coding:" << std::endl;

  for (const auto &entropy_step : info.entropy_steps) {
    std::cerr << "    " << step++ << ". " << entropy_step.label;
    if (!entropy_step.codec_label.empty())
      std::cerr << " (" << entropy_step.codec_label << ")";
    std::cerr << ": " << std::fixed << std::setprecision(2)
              << entropy_step.time_ms << " ms";
    if (entropy_step.ratio.original_bytes > 0 ||
        entropy_step.ratio.compressed_bytes > 0) {
      std::cerr << " | " << format_ratio(entropy_step.ratio);
    }
    std::cerr << std::endl;
  }

  std::cerr << "    entropy coding subtotal:         " << std::fixed
            << std::setprecision(2) << info.time_entropy_ms
            << " ms" << std::endl;
  std::cerr << "  ─────────────────────────────────" << std::endl;
  std::cerr << "  TOTAL (compress_gfa):              " << std::fixed
            << std::setprecision(2) << info.total_ms << " ms" << std::endl;
}

} // namespace gfz::compression_debug
