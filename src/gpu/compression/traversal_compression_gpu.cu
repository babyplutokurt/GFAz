#include "gpu/compression/traversal_compression_gpu.hpp"

#include "gpu/compression/compression_workflow_gpu_internal.hpp"
#include "gpu/compression/path_compression_gpu_legacy.hpp"
#include "gpu/compression/path_compression_gpu_rolling.hpp"
#include "utils/runtime_utils.hpp"

#include <iostream>

namespace gpu_compression {

using gfz::runtime_utils::format_size;

GpuTraversalCompressionPath resolve_gpu_traversal_compression_path(
    const FlattenedPaths &paths, GpuCompressionOptions options,
    size_t *resolved_chunk_bytes) {
  const size_t traversal_bytes = paths.data.size() * sizeof(int32_t);
  const size_t chunk_bytes =
      (options.rolling_input_chunk_bytes > 0)
          ? options.rolling_input_chunk_bytes
          : default_rolling_chunk_bytes();
  if (resolved_chunk_bytes != nullptr) {
    *resolved_chunk_bytes = chunk_bytes;
  }

  if (scheduler_debug_enabled()) {
    std::cerr << "[GPU Scheduler] traversal=" << format_size(traversal_bytes)
              << ", chunk_budget=" << format_size(chunk_bytes)
              << ", requested_chunk="
              << (options.rolling_input_chunk_bytes > 0
                      ? format_size(options.rolling_input_chunk_bytes)
                      : std::string("default"))
              << ", policy="
              << (options.force_rolling_scheduler
                      ? "force-rolling"
                      : (options.force_full_device_legacy ? "force-legacy"
                                                          : "auto"))
              << std::endl;
  }

  if (options.force_full_device_legacy) {
    return GpuTraversalCompressionPath::kLegacyWholeDevice;
  }
  if (options.force_rolling_scheduler) {
    return GpuTraversalCompressionPath::kRollingScheduler;
  }
  if (traversal_bytes <= chunk_bytes) {
    return GpuTraversalCompressionPath::kLegacyWholeDevice;
  }
  return GpuTraversalCompressionPath::kRollingScheduler;
}

CompressedData compress_gpu_traversals(
    const FlattenedPaths &paths, uint32_t num_paths, int num_rounds,
    GpuCompressionOptions options, GpuPathCompressionDebugInfo *debug_info) {
  size_t chunk_bytes = 0;
  const GpuTraversalCompressionPath path =
      resolve_gpu_traversal_compression_path(paths, options, &chunk_bytes);

  if (scheduler_debug_enabled()) {
    std::cerr << "[GPU Scheduler] selected path: "
              << (path == GpuTraversalCompressionPath::kLegacyWholeDevice
                      ? "legacy whole-device"
                      : "rolling scheduler");
    if (!options.force_full_device_legacy && !options.force_rolling_scheduler) {
      std::cerr << (path == GpuTraversalCompressionPath::kLegacyWholeDevice
                        ? " (fits chunk budget)"
                        : " (exceeds chunk budget)");
    }
    std::cerr << std::endl;
  }

  if (path == GpuTraversalCompressionPath::kLegacyWholeDevice) {
    return compress_gpu_traversals_legacy_whole_device(paths, num_paths,
                                                       num_rounds, debug_info);
  }
  return compress_gpu_traversals_rolling_scheduler(paths, num_paths, num_rounds,
                                                   chunk_bytes, debug_info);
}

} // namespace gpu_compression
