#ifndef COMPRESSION_WORKFLOW_GPU_HPP
#define COMPRESSION_WORKFLOW_GPU_HPP

#include "workflows/compression_workflow.hpp"
#include "gpu/gfa_graph_gpu.hpp"
#include <cstdint>
#include <map>
#include <vector>

namespace gpu_compression {

struct GpuCompressionOptions {
  size_t rolling_chunk_bytes = 0;
  bool force_rolling_scheduler = false;
  bool force_full_device_legacy = false;
};

struct GpuGrammarRoundDebugInfo {
  int round = 0;
  size_t chunk_count = 0;
  size_t rules_found = 0;
  size_t rules_used = 0;
  double count_ms = 0;
  double apply_ms = 0;
  double remap_ms = 0;
};

struct GpuPathCompressionDebugInfo {
  std::string mode_label;
  size_t traversal_bytes = 0;
  size_t num_traversals = 0;
  size_t chunk_bytes = 0;
  size_t initial_chunk_count = 0;
  double host_to_device_ms = 0;
  double delta_ms = 0;
  double traversal_zstd_ms = 0;
  double rules_zstd_ms = 0;
  double total_ms = 0;
  size_t original_paths = 0;
  size_t encoded_paths = 0;
  size_t original_walks = 0;
  size_t encoded_walks = 0;
  std::vector<GpuGrammarRoundDebugInfo> rounds;
};

struct GpuMetadataCompressionStageDebugInfo {
  std::string label;
  std::string codec_label;
  double time_ms = 0;
  size_t original_bytes = 0;
  size_t compressed_bytes = 0;
};

struct GpuMetadataCompressionDebugInfo {
  std::vector<GpuMetadataCompressionStageDebugInfo> stages;
  double total_ms = 0;
};

/**
 * GPU path compression.
 *
 * Small traversals stay on-device for the full grammar pipeline. Large
 * traversals fall back to a rolling round scheduler that:
 * 1. Keeps the flattened traversal on host memory
 * 2. Partitions it into traversal-aligned chunks
 * 3. Discovers global 2-mer rules by merging per-chunk histograms
 * 4. Applies the merged rulebook chunk-by-chunk on GPU
 * 5. Preserves the same round/layer output structure
 *
 * @param paths Input flattened paths (host memory)
 * @param num_rounds Maximum number of compression rounds
 * @return CompressedData containing shared Zstd-compressed path/walk columns
 */
CompressedData run_path_compression_gpu(
    const FlattenedPaths &paths, uint32_t num_paths, int num_rounds,
    GpuCompressionOptions options = {},
    GpuPathCompressionDebugInfo *debug_info = nullptr);

/**
 * High-level GPU compression entry point (parse + compress).
 * Mirrors CPU compress_gfa but uses GPU path compression and shared CPU Zstd
 * for final entropy coding.
 */
CompressedData compress_gfa_gpu(const std::string &gfa_file_path,
                                int num_rounds,
                                GpuCompressionOptions options = {});

/**
 * GPU compression from GfaGraph_gpu (no parsing).
 * Use this for accurate timing of compression-only.
 *
 * @param gpu_graph Pre-converted GPU graph
 * @param num_rounds Number of compression rounds
 * @return CompressedData containing all compressed fields
 */
CompressedData compress_gpu_graph(
    const GfaGraph_gpu &gpu_graph, int num_rounds,
    GpuCompressionOptions options = {});

void set_gpu_compression_debug(bool enabled);

/**
 * Build a rulebook map from the flat rules vector and layer ranges.
 * Useful for round-trip verification with CPU reconstruction.
 *
 * @param data CompressedData containing all_rules and layer ranges
 * @return Map from rule_id -> packed_2mer
 */
std::map<uint32_t, uint64_t> build_rulebook(const CompressedData &data);

} // namespace gpu_compression

#endif // COMPRESSION_WORKFLOW_GPU_HPP
