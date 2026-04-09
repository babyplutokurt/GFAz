#ifndef GPU_GFA_WRITER_GPU_HPP
#define GPU_GFA_WRITER_GPU_HPP

#ifdef ENABLE_CUDA
#include "model/compressed_data.hpp"
#include "gpu/decompression_workflow_gpu.hpp"
#include <string>


void write_gfa_from_compressed_data_gpu(
    const CompressedData &data,
    const std::string &output_path,
    gpu_decompression::GpuDecompressionOptions options = {});
#endif

#endif // GPU_GFA_WRITER_GPU_HPP
