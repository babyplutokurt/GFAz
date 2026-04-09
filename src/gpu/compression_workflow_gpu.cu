#include "io/gfa_parser.hpp"
#include "gpu/codec_gpu.cuh"
#include "gpu/compression_workflow_gpu_internal.hpp"
#include "gpu/compression_workflow_gpu.hpp"
#include "gpu/metadata_codec_gpu.hpp"
#include "gpu/path_chunk_planner.hpp"
#include "gpu/path_compression_gpu_legacy.hpp"
#include "gpu/path_compression_gpu_rolling.hpp"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>

#include "codec/codec.hpp"

namespace gpu_compression {

// Debug flag for compression stats (can be controlled via environment or
// compile flag)
static bool g_debug_compression = false;
using Clock = std::chrono::high_resolution_clock;

static double elapsed_ms(const Clock::time_point &start,
                         const Clock::time_point &end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

bool scheduler_debug_enabled() {
  const char *env = std::getenv("GFAZ_GPU_SCHED_DEBUG");
  return env && *env != '\0' && std::string(env) != "0";
}

bool compression_debug_enabled() { return g_debug_compression; }

void set_gpu_compression_debug(bool enabled) {
  g_debug_compression = enabled;
}

CompressedData run_path_compression_gpu(
    const FlattenedPaths &paths, uint32_t num_paths, int num_rounds,
    GpuCompressionOptions options) {
  const size_t traversal_bytes = paths.data.size() * sizeof(int32_t);
  const size_t chunk_bytes =
      (options.rolling_chunk_bytes > 0) ? options.rolling_chunk_bytes
                                        : default_rolling_chunk_bytes();

  if (scheduler_debug_enabled()) {
    std::cerr << "[GPU Scheduler] traversal_bytes=" << traversal_bytes
              << ", requested_chunk_bytes=" << options.rolling_chunk_bytes
              << ", resolved_chunk_bytes=" << chunk_bytes
              << ", force_rolling="
              << (options.force_rolling_scheduler ? "true" : "false")
              << ", force_legacy="
              << (options.force_full_device_legacy ? "true" : "false")
              << std::endl;
  }

  if (options.force_full_device_legacy) {
    if (scheduler_debug_enabled()) {
      std::cerr << "[GPU Scheduler] mode=full_device (forced)" << std::endl;
    }
    return run_path_compression_gpu_full_device(paths, num_paths, num_rounds);
  }

  if (options.force_rolling_scheduler) {
    if (scheduler_debug_enabled()) {
      std::cerr << "[GPU Scheduler] mode=rolling (forced)" << std::endl;
    }
    return run_path_compression_gpu_rolling(paths, num_paths, num_rounds,
                                            chunk_bytes);
  }

  if (traversal_bytes <= chunk_bytes) {
    if (scheduler_debug_enabled()) {
      std::cerr << "[GPU Scheduler] mode=full_device (fits in chunk budget)"
                << std::endl;
    }
    return run_path_compression_gpu_full_device(paths, num_paths, num_rounds);
  }
  if (scheduler_debug_enabled()) {
    std::cerr << "[GPU Scheduler] mode=rolling (exceeds chunk budget)"
              << std::endl;
  }
  return run_path_compression_gpu_rolling(paths, num_paths, num_rounds,
                                          chunk_bytes);
}


std::map<uint32_t, uint64_t> build_rulebook(const CompressedData &data) {
  std::map<uint32_t, uint64_t> rulebook;

  std::vector<int32_t> first =
      Codec::zstd_decompress_int32_vector(data.rules_first_zstd);
  std::vector<int32_t> second =
      Codec::zstd_decompress_int32_vector(data.rules_second_zstd);

  if (first.empty() || second.empty()) {
    return rulebook;
  }

  Codec::delta_decode_int32(first);
  Codec::delta_decode_int32(second);

  size_t offset = 0;
  for (const auto &range : data.layer_rule_ranges) {
    const uint32_t count = range.end_id - range.start_id;
    for (uint32_t i = 0; i < count; ++i) {
      uint32_t rule_id = range.start_id + i;
      uint64_t packed =
          (static_cast<uint64_t>(static_cast<uint32_t>(first[offset + i]))
           << 32) |
          static_cast<uint64_t>(static_cast<uint32_t>(second[offset + i]));
      rulebook[rule_id] = packed;
    }
    offset += count;
  }

  return rulebook;
}

CompressedData compress_gfa_gpu(const std::string &gfa_file_path,
                                int num_rounds,
                                GpuCompressionOptions options) {
  GfaParser parser;
  GfaGraph graph = parser.parse(gfa_file_path);

  GfaGraph_gpu gpu_graph = convert_to_gpu_layout(graph);

  return compress_gpu_graph(gpu_graph, num_rounds, options);
}

CompressedData compress_gpu_graph(const GfaGraph_gpu &gpu_graph,
                                  int num_rounds,
                                  GpuCompressionOptions options) {
  auto compress_start = Clock::now();

  CompressedData data =
      run_path_compression_gpu(gpu_graph.paths, gpu_graph.num_paths, num_rounds,
                               options);
  data.header_line = gpu_graph.header_line;
  data.delta_round = 1;
  compress_graph_metadata_gpu(gpu_graph, data);

  auto compress_end = Clock::now();
  double compress_time_ms = elapsed_ms(compress_start, compress_end);

  if (g_debug_compression) {
    std::cout << "[GPU Compression] Total compression time (GfaGraph_gpu -> "
                 "CompressedData): "
              << std::fixed << std::setprecision(2) << compress_time_ms
              << " ms (" << compress_time_ms / 1000.0 << " s)" << std::endl;
  }

  return data;
}

} // namespace gpu_compression
