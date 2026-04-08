#include "gfa_parser.hpp"
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

#include "gpu/codec_gpu_nvcomp.cuh"

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

static std::vector<int32_t>
inverse_delta_decode_host(const std::vector<int32_t> &delta_encoded) {
  if (delta_encoded.empty()) {
    return {};
  }
  std::vector<int32_t> decoded(delta_encoded.size());
  decoded[0] = delta_encoded[0];
  for (size_t i = 1; i < delta_encoded.size(); ++i) {
    decoded[i] = decoded[i - 1] + delta_encoded[i];
  }
  return decoded;
}

CompressedData_gpu run_path_compression_gpu(
    const FlattenedPaths &paths, int num_rounds,
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
    return run_path_compression_gpu_full_device(paths, num_rounds);
  }

  if (options.force_rolling_scheduler) {
    if (scheduler_debug_enabled()) {
      std::cerr << "[GPU Scheduler] mode=rolling (forced)" << std::endl;
    }
    return run_path_compression_gpu_rolling(paths, num_rounds, chunk_bytes);
  }

  if (traversal_bytes <= chunk_bytes) {
    if (scheduler_debug_enabled()) {
      std::cerr << "[GPU Scheduler] mode=full_device (fits in chunk budget)"
                << std::endl;
    }
    return run_path_compression_gpu_full_device(paths, num_rounds);
  }
  if (scheduler_debug_enabled()) {
    std::cerr << "[GPU Scheduler] mode=rolling (exceeds chunk budget)"
              << std::endl;
  }
  return run_path_compression_gpu_rolling(paths, num_rounds, chunk_bytes);
}


std::map<uint32_t, uint64_t> build_rulebook(const CompressedData_gpu &data) {
  std::map<uint32_t, uint64_t> rulebook;

  // Decompress rules_first and rules_second
  std::vector<int32_t> first_delta =
      gpu_codec::nvcomp_zstd_decompress_int32(data.rules_first_zstd_nvcomp);
  std::vector<int32_t> second_delta =
      gpu_codec::nvcomp_zstd_decompress_int32(data.rules_second_zstd_nvcomp);

  if (first_delta.empty() || second_delta.empty()) {
    return rulebook;
  }

  // Inverse delta-encode to get original rule endpoints.
  std::vector<int32_t> first = inverse_delta_decode_host(first_delta);
  std::vector<int32_t> second = inverse_delta_decode_host(second_delta);

  // Build rulebook from unpacked rules
  size_t offset = 0;
  for (const auto &range : data.layer_ranges) {
    for (uint32_t i = 0; i < range.count; ++i) {
      uint32_t rule_id = range.start_id + i;
      // Pack first and second back into uint64_t
      uint64_t packed =
          (static_cast<uint64_t>(static_cast<uint32_t>(first[offset + i]))
           << 32) |
          static_cast<uint64_t>(static_cast<uint32_t>(second[offset + i]));
      rulebook[rule_id] = packed;
    }
    offset += range.count;
  }

  return rulebook;
}

CompressedData_gpu compress_gfa_gpu(const std::string &gfa_file_path,
                                    int num_rounds,
                                    GpuCompressionOptions options) {
  GfaParser parser;
  GfaGraph graph = parser.parse(gfa_file_path);

  GfaGraph_gpu gpu_graph = convert_to_gpu_layout(graph);

  return compress_gpu_graph(gpu_graph, num_rounds, options);
}

CompressedData_gpu compress_gpu_graph(const GfaGraph_gpu &gpu_graph,
                                      int num_rounds,
                                      GpuCompressionOptions options) {
  // Start timer for compression (GfaGraph_gpu -> CompressedData_gpu)
  auto compress_start = Clock::now();

  CompressedData_gpu data =
      run_path_compression_gpu(gpu_graph.paths, num_rounds, options);
  compress_graph_metadata_gpu(gpu_graph, data);

  // End timer and print compression time
  auto compress_end = Clock::now();
  double compress_time_ms = elapsed_ms(compress_start, compress_end);

  if (g_debug_compression) {
    std::cout << "[GPU Compression] Total compression time (GfaGraph_gpu -> "
                 "CompressedData_gpu): "
              << std::fixed << std::setprecision(2) << compress_time_ms
              << " ms (" << compress_time_ms / 1000.0 << " s)" << std::endl;
  }

  return data;
}

} // namespace gpu_compression
