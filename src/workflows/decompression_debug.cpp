#include "workflows/decompression_debug.hpp"
#include "utils/debug_log.hpp"
#include "utils/runtime_utils.hpp"

#include <iomanip>
#include <iostream>

namespace gfaz::decompression_debug {

namespace {

using gfaz::runtime_utils::format_memory_snapshot;

} // namespace

void log_cpu_decompression_memory_checkpoint(const std::string &scope_label,
                                             const std::string &label) {
  if (!gfaz_debug_enabled())
    return;

  const auto snapshot = runtime_utils::read_process_memory_snapshot();
  std::cerr << "[" << scope_label << "][Memory] " << label
            << " | " << format_memory_snapshot(snapshot) << std::endl;
}

void print_cpu_decompression_timing(
    const CpuDecompressionTimingDebugInfo &info) {
  std::cerr << "\n[" << info.scope_label << "] === TIMING BREAKDOWN";
  if (!info.header_suffix.empty())
    std::cerr << " (" << info.header_suffix << ")";
  std::cerr << " ===" << std::endl;

  int step = 1;
  for (const auto &stage : info.stages) {
    std::cerr << "    " << step++ << ". " << stage.label << ": ";
    if (stage.label.size() < 28)
      std::cerr << std::string(28 - stage.label.size(), ' ');
    std::cerr << std::fixed << std::setprecision(2) << stage.time_ms << " ms"
              << std::endl;
  }

  std::cerr << "  ─────────────────────────────────" << std::endl;
  std::cerr << "  TOTAL: " << std::fixed << std::setprecision(2)
            << info.total_ms << " ms" << std::endl;
}

void print_cpu_decompression_summary(const std::string &scope_label,
                                     const gfaz::CompressedData &data) {
  if (!gfaz_debug_enabled())
    return;

  const size_t num_segments =
      data.segment_seq_lengths_zstd.original_size / sizeof(uint32_t);

  std::cerr << "[" << scope_label << "] segments=" << num_segments
            << ", links=" << data.num_links
            << ", paths=" << data.sequence_lengths.size()
            << ", walks=" << data.walk_lengths.size()
            << ", jumps=" << data.num_jumps
            << ", containments=" << data.num_containments
            << ", rules=" << data.total_rules()
            << " (delta=" << data.delta_round << ")" << std::endl;
}

void print_cpu_decompression_summary(const std::string &scope_label,
                                     const gfaz::GfaGraph &graph, size_t num_rules,
                                     int delta_round) {
  if (!gfaz_debug_enabled())
    return;

  std::cerr << "[" << scope_label << "] segments="
            << (graph.segments.node_sequences.empty() ? 0 : graph.segments.node_sequences.size() - 1)
            << ", links=" << graph.links.from_ids.size()
            << ", paths=" << graph.paths_data.traversals.size()
            << ", walks=" << graph.walks.walks.size()
            << ", jumps=" << graph.jumps.size()
            << ", containments=" << graph.containments.size()
            << ", rules=" << num_rules << " (delta=" << delta_round << ")"
            << std::endl;
}

} // namespace gfaz::decompression_debug
