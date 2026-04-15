#pragma once

#include "gpu/decompression/decompression_primitives_gpu.hpp"

namespace gpu_decompression {

enum class GpuTraversalDecompressionPath {
  kLegacyMaterialized,
  kRollingMaterialized,
  kRollingDirectWriter,
};

struct GpuTraversalDecodeStats {
  double payload_decode_ms = 0.0;
  double expand_ms = 0.0;
};

GpuTraversalDecompressionPath resolve_gpu_traversal_decompression_path(
    GpuDecompressionOptions options, bool direct_writer);

std::vector<int32_t> decompress_gpu_traversal_materialized(
    const gfaz::ZstdCompressedBlock &encoded_block,
    const std::vector<uint32_t> &final_lengths,
    const GpuTraversalRulebook &rulebook, GpuDecompressionOptions options,
    GpuTraversalDecodeStats *stats = nullptr);

void decompress_gpu_traversal_rolling_direct_writer(
    const gfaz::ZstdCompressedBlock &encoded_block,
    const std::vector<uint32_t> &final_lengths,
    const GpuTraversalRulebook &rulebook, GpuDecompressionOptions options,
    RollingPathChunkConsumer consumer,
    RollingPathStreamOptions stream_options = {},
    GpuTraversalDecodeStats *stats = nullptr);

} // namespace gpu_decompression
