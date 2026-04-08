#ifndef METADATA_CODEC_GPU_HPP
#define METADATA_CODEC_GPU_HPP

#include "gpu/compression_workflow_gpu.hpp"

namespace gpu_compression {

void compress_graph_metadata_gpu(const GfaGraph_gpu &gpu_graph,
                                 CompressedData_gpu &data);

} // namespace gpu_compression

#endif // METADATA_CODEC_GPU_HPP
