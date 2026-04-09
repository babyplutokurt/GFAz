#pragma once

#include "grammar/packed_2mer.hpp"
#include "model/compressed_data.hpp"
#include "model/gfa_graph.hpp"

#include <chrono>
#include <string>
#include <vector>

namespace gfz::compression_utils {

inline constexpr const char *kCompressionErrorPrefix = "Compression workflow error: ";

using Clock = std::chrono::high_resolution_clock;

struct ProcessMemorySnapshot {
  size_t vm_rss_kb = 0;
  size_t vm_hwm_kb = 0;
  size_t rss_anon_kb = 0;
};

double elapsed_ms(const Clock::time_point &start,
                  const Clock::time_point &end);

double gbps_from_mb(double size_mb, double time_ms);

size_t total_node_count(const std::vector<std::vector<NodeId>> &sequences);

std::string format_size(size_t bytes);

ProcessMemorySnapshot read_process_memory_snapshot();

void log_memory_checkpoint(const std::string &label);

void append_string_column(const std::vector<std::string> &values,
                          std::string &concatenated,
                          std::vector<uint32_t> &lengths);

CompressedOptionalFieldColumn
compress_optional_column(const OptionalFieldColumn &col);

void flatten_paths(const std::vector<std::vector<NodeId>> &paths,
                   const std::vector<std::string> &path_names,
                   const std::vector<std::string> &path_overlaps,
                   std::vector<int32_t> &flattened,
                   std::vector<uint32_t> &lengths, std::string &names_concat,
                   std::vector<uint32_t> &name_lengths,
                   std::string &overlaps_concat,
                   std::vector<uint32_t> &overlap_lengths);

void flatten_segments(const std::vector<std::string> &sequences,
                      std::string &concat, std::vector<uint32_t> &lengths,
                      uint32_t max_id);

void process_rules(const std::vector<Packed2mer> &rulebook,
                   uint32_t layer_start_id,
                   const std::vector<LayerRuleRange> &ranges,
                   std::vector<int32_t> &first, std::vector<int32_t> &second);

void flatten_walks(const std::vector<std::vector<NodeId>> &walks,
                   std::vector<int32_t> &flattened,
                   std::vector<uint32_t> &lengths);

void remap_rule_ids(std::vector<std::vector<NodeId>> &sequences,
                    uint32_t rules_start_id,
                    const std::vector<uint32_t> &id_map);

void print_compression_stats(const CompressedData &d, size_t num_segments,
                             bool show_stats);

} // namespace gfz::compression_utils
