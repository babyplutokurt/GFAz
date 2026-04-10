#include "io/gfa_parser.hpp"
#include "gpu/compression/compression_workflow_gpu_internal.hpp"
#include "gpu/compression/compression_workflow_gpu.hpp"
#include "gpu/compression/metadata_codec_gpu.hpp"
#include "gpu/compression/traversal_compression_gpu.hpp"
#include "utils/runtime_utils.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>

#include "codec/codec.hpp"

namespace gpu_compression {

// Debug flag for compression stats (can be controlled via environment or
// compile flag)
static bool g_debug_compression = false;
using Clock = std::chrono::high_resolution_clock;
using gfz::runtime_utils::elapsed_ms;
using gfz::runtime_utils::format_memory_snapshot;
using gfz::runtime_utils::format_size;

namespace {

struct GpuMemorySnapshot {
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  bool valid = false;
};

GpuMemorySnapshot read_gpu_memory_snapshot() {
  GpuMemorySnapshot snapshot;
  cudaError_t err =
      cudaMemGetInfo(&snapshot.free_bytes, &snapshot.total_bytes);
  snapshot.valid = (err == cudaSuccess);
  return snapshot;
}

std::string format_gpu_memory_snapshot(const GpuMemorySnapshot &snapshot) {
  if (!snapshot.valid) {
    return "GPU memory unavailable";
  }

  return "GPU free=" + format_size(snapshot.free_bytes) +
         " | GPU total=" + format_size(snapshot.total_bytes);
}

void log_gpu_memory_checkpoint(const std::string &label) {
  if (!scheduler_debug_enabled()) {
    return;
  }

  const auto host_snapshot = gfz::runtime_utils::read_process_memory_snapshot();
  const auto gpu_snapshot = read_gpu_memory_snapshot();
  std::cerr << "[GPU Compress][Memory] " << label << " | "
            << format_memory_snapshot(host_snapshot) << " | "
            << format_gpu_memory_snapshot(gpu_snapshot) << std::endl;
}

std::string format_reduction_percent(size_t original, size_t encoded) {
  if (original == 0) {
    return {};
  }

  std::ostringstream oss;
  oss << " (" << std::fixed << std::setprecision(2)
      << 100.0 * static_cast<double>(encoded) / static_cast<double>(original)
      << "%)";
  return oss.str();
}

std::string format_ratio(size_t original_bytes, size_t compressed_bytes) {
  std::ostringstream oss;
  oss << format_size(original_bytes) << " -> " << format_size(compressed_bytes);
  if (compressed_bytes > 0) {
    oss << " (" << std::fixed << std::setprecision(2)
        << static_cast<double>(original_bytes) /
               static_cast<double>(compressed_bytes)
        << "x)";
  }
  return oss.str();
}

void print_gpu_path_compression_debug(const GpuPathCompressionDebugInfo &info) {
  std::cerr << "[GPU Path Compress] === " << info.path_label << " ("
            << std::fixed << std::setprecision(1)
            << (info.traversal_bytes / (1024.0 * 1024.0)) << " MB, "
            << info.num_traversals << " traversals";
  if (info.chunk_bytes > 0) {
    std::cerr << ", chunk budget " << (info.chunk_bytes / (1024.0 * 1024.0))
              << " MB";
  }
  std::cerr << ") ===" << std::endl;
  if (info.host_to_device_ms > 0.0) {
    std::cerr << "  host_to_device: " << std::fixed << std::setprecision(2)
              << info.host_to_device_ms << " ms" << std::endl;
  }
  std::cerr << "  delta_transform: " << std::fixed << std::setprecision(2)
            << info.delta_ms << " ms";
  if (info.initial_chunk_count > 0) {
    std::cerr << " | chunks=" << info.initial_chunk_count;
  }
  std::cerr << std::endl;
  if (scheduler_debug_enabled()) {
    std::cerr << "  rounds:" << std::endl;
    for (const auto &round : info.rounds) {
      std::cerr << "    " << round.round << ". chunks=" << round.chunk_count
                << " | rules=" << round.rules_used << "/"
                << round.rules_found << " | count=" << std::fixed
                << std::setprecision(2) << round.count_ms
                << " ms | apply=" << round.apply_ms << " ms | remap="
                << round.remap_ms << " ms";
      if (round.chunk_count > 0) {
        const double total_round_ms =
            round.count_ms + round.apply_ms + round.remap_ms;
        std::cerr << " | total=" << total_round_ms << " ms";
      }
      std::cerr << std::endl;
    }
  }
  std::cerr << "  traversal reduction:" << std::endl;
  std::cerr << "    paths: " << info.original_paths << " -> "
            << info.encoded_paths
            << format_reduction_percent(info.original_paths, info.encoded_paths)
            << std::endl;
  std::cerr << "    walks: " << info.original_walks << " -> "
            << info.encoded_walks
            << format_reduction_percent(info.original_walks, info.encoded_walks)
            << std::endl;
  std::cerr << "    total: " << (info.original_paths + info.original_walks)
            << " -> " << (info.encoded_paths + info.encoded_walks)
            << format_reduction_percent(
                   info.original_paths + info.original_walks,
                   info.encoded_paths + info.encoded_walks)
            << std::endl;
  std::cerr << "  compress traversals (ZSTD): " << std::fixed
            << std::setprecision(2) << info.traversal_zstd_ms << " ms"
            << std::endl;
  std::cerr << "  compress rules (ZSTD):      " << std::fixed
            << std::setprecision(2) << info.rules_zstd_ms << " ms"
            << std::endl;
  std::cerr << "  TOTAL:                      " << std::fixed
            << std::setprecision(2) << info.total_ms << " ms" << std::endl;
}

void print_gpu_metadata_compression_debug(
    const GpuMetadataCompressionDebugInfo &info) {
  std::cerr << "[GPU Metadata Compress]" << std::endl;
  for (const auto &stage : info.stages) {
    std::cerr << "  " << stage.label;
    if (!stage.codec_label.empty()) {
      std::cerr << " (" << stage.codec_label << ")";
    }
    std::cerr << ": " << std::fixed << std::setprecision(2) << stage.time_ms
              << " ms";
    if (stage.original_bytes > 0 || stage.compressed_bytes > 0) {
      std::cerr << " | "
                << format_ratio(stage.original_bytes, stage.compressed_bytes);
    }
    std::cerr << std::endl;
  }
  std::cerr << "  TOTAL: " << std::fixed << std::setprecision(2)
            << info.total_ms << " ms" << std::endl;
}

} // namespace

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
    GpuCompressionOptions options, GpuPathCompressionDebugInfo *debug_info) {
  return compress_gpu_traversals(paths, num_paths, num_rounds, options,
                                 debug_info);
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
  const auto total_start = Clock::now();
  const auto parse_start = Clock::now();
  GfaParser parser;
  GfaGraph graph = parser.parse(gfa_file_path);
  const auto parse_end = Clock::now();
  log_gpu_memory_checkpoint("after parse");

  const auto layout_start = Clock::now();
  GfaGraph_gpu gpu_graph = convert_to_gpu_layout(graph);
  const auto layout_end = Clock::now();
  log_gpu_memory_checkpoint("after GPU layout conversion");

  CompressedData data = compress_gpu_graph(gpu_graph, num_rounds, options);

  if (g_debug_compression) {
    std::cerr << "[GPU Compress] parse+layout: parse=" << std::fixed
              << std::setprecision(2) << elapsed_ms(parse_start, parse_end)
              << " ms, host_to_gpu_layout=" << elapsed_ms(layout_start, layout_end)
              << " ms, end-to-end="
              << elapsed_ms(total_start, Clock::now()) << " ms" << std::endl;
  }

  return data;
}

CompressedData compress_gpu_graph(const GfaGraph_gpu &gpu_graph,
                                  int num_rounds,
                                  GpuCompressionOptions options) {
  auto compress_start = Clock::now();
  GpuPathCompressionDebugInfo path_debug;
  GpuMetadataCompressionDebugInfo metadata_debug;

  CompressedData data =
      run_path_compression_gpu(gpu_graph.paths, gpu_graph.num_paths, num_rounds,
                               options, g_debug_compression ? &path_debug
                                                            : nullptr);
  log_gpu_memory_checkpoint("after path compression");
  data.header_line = gpu_graph.header_line;
  data.delta_round = 1;
  compress_graph_metadata_gpu(gpu_graph, data,
                              g_debug_compression ? &metadata_debug : nullptr);
  log_gpu_memory_checkpoint("after metadata compression");

  auto compress_end = Clock::now();
  double compress_time_ms = elapsed_ms(compress_start, compress_end);

  if (g_debug_compression) {
    print_gpu_path_compression_debug(path_debug);
    print_gpu_metadata_compression_debug(metadata_debug);
    std::cerr << "[GPU Compress] TOTAL (GfaGraph_gpu -> CompressedData): "
              << std::fixed << std::setprecision(2) << compress_time_ms
              << " ms" << std::endl;
  }

  return data;
}

} // namespace gpu_compression
