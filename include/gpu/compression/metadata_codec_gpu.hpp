#ifndef METADATA_CODEC_GPU_HPP
#define METADATA_CODEC_GPU_HPP

#include "gpu/compression/compression_workflow_gpu.hpp"
#include "gpu/core/gfa_graph_gpu.hpp"

namespace gpu_compression {

void compress_graph_metadata_gpu(const GfaGraph_gpu &gpu_graph,
                                 gfaz::CompressedData &data,
                                 GpuMetadataCompressionDebugInfo
                                     *debug_info = nullptr);

} // namespace gpu_compression

namespace gpu_decompression {

void decompress_graph_metadata_gpu(const gfaz::CompressedData &data,
                                   GfaGraph_gpu &result);

} // namespace gpu_decompression

#endif // METADATA_CODEC_GPU_HPP
