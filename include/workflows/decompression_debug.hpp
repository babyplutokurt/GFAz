#pragma once

#include "model/compressed_data.hpp"
#include "model/gfa_graph.hpp"

#include <string>
#include <vector>

namespace gfaz::decompression_debug {

struct TimedDebugStage {
  std::string label;
  double time_ms = 0;
};

struct CpuDecompressionTimingDebugInfo {
  std::string scope_label;
  std::string header_suffix;
  std::vector<TimedDebugStage> stages;
  double total_ms = 0;
};

void log_cpu_decompression_memory_checkpoint(const std::string &scope_label,
                                             const std::string &label);

void print_cpu_decompression_timing(
    const CpuDecompressionTimingDebugInfo &info);

void print_cpu_decompression_summary(const std::string &scope_label,
                                     const gfaz::CompressedData &data);

void print_cpu_decompression_summary(const std::string &scope_label,
                                     const gfaz::GfaGraph &graph, size_t num_rules,
                                     int delta_round);

} // namespace gfaz::decompression_debug
