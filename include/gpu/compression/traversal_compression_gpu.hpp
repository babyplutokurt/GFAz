#pragma once

#include "gpu/compression/compression_workflow_gpu.hpp"

namespace gpu_compression {

enum class GpuTraversalCompressionPath {
  kLegacyWholeDevice,
  kRollingScheduler,
};

GpuTraversalCompressionPath resolve_gpu_traversal_compression_path(
    const FlattenedPaths &paths, GpuCompressionOptions options,
    size_t *resolved_chunk_bytes = nullptr);

CompressedData compress_gpu_traversals(
    const FlattenedPaths &paths, uint32_t num_paths, int num_rounds,
    GpuCompressionOptions options = {},
    GpuPathCompressionDebugInfo *debug_info = nullptr);

} // namespace gpu_compression
