#pragma once

#include "model/compressed_data.hpp"
#include "utils/runtime_utils.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace gfz::compression_debug {

struct GrammarRoundDebugInfo {
  int round = 0;
  double generate_ms = 0;
  double encode_ms = 0;
  double remap_ms = 0;
  size_t rules_used = 0;
  size_t num_rules = 0;
  runtime_utils::ProcessMemorySnapshot memory;
};

struct TraversalReductionDebugInfo {
  size_t original_paths = 0;
  size_t encoded_paths = 0;
  size_t original_walks = 0;
  size_t encoded_walks = 0;
};

struct CompressionRatio {
  size_t original_bytes = 0;
  size_t compressed_bytes = 0;
};

struct CompressionPostStepDebugInfo {
  std::string label;
  std::string codec_label;
  double time_ms = 0;
  CompressionRatio ratio;
};

struct CpuCompressionTimingDebugInfo {
  double data_size_mb = 0;
  size_t total_elements = 0;
  int delta_round = 0;
  double time_delta_ms = 0;
  double time_grammar_ms = 0;
  double rules_size_mb = 0;
  double time_process_rules_ms = 0;
  std::vector<CompressionPostStepDebugInfo> post_steps;
  double field_compression_subtotal_ms = 0;
  double total_ms = 0;
};

void print_grammar_round(const GrammarRoundDebugInfo &info);

void print_grammar_timing_breakdown(int actual_rounds, double data_size_mb,
                                    size_t total_elements,
                                    double time_generate_rules_ms,
                                    double time_encode_paths_ms,
                                    double time_compact_sort_remap_ms,
                                    double total_ms);

void print_traversal_reduction(const TraversalReductionDebugInfo &info);

void log_cpu_memory_checkpoint(const std::string &label);

CompressionRatio sum_optional_field_ratio(
    const std::vector<CompressedOptionalFieldColumn> &cols);

CompressionRatio block_ratio(const ZstdCompressedBlock &block);

CompressionRatio sum_ratios(std::initializer_list<CompressionRatio> ratios);

CompressionRatio collect_rules_ratio(const CompressedData &data);
CompressionRatio collect_path_ratio(const CompressedData &data);
CompressionRatio collect_walk_ratio(const CompressedData &data);
CompressionRatio collect_segment_link_ratio(const CompressedData &data);
CompressionRatio collect_optional_field_ratio(const CompressedData &data);
CompressionRatio collect_jump_ratio(const CompressedData &data);
CompressionRatio collect_containment_ratio(const CompressedData &data);

void print_cpu_compression_timing(
    const CpuCompressionTimingDebugInfo &info);

} // namespace gfz::compression_debug
