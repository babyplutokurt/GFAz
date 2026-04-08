#ifndef GFA_WRITER_HPP
#define GFA_WRITER_HPP

#include "compression_workflow.hpp"
#include "gfa_parser.hpp"
#include <string>

void write_gfa(const GfaGraph &graph, const std::string &output_path);
void write_gfa_from_compressed_data(const CompressedData &data,
                                    const std::string &output_path,
                                    int num_threads = 0);

#ifdef ENABLE_CUDA
#include "gpu/compression_workflow_gpu.hpp"
#include "gpu/decompression_workflow_gpu.hpp"

void write_gfa_from_compressed_data_gpu(
    const gpu_compression::CompressedData_gpu &data,
    const std::string &output_path,
    gpu_decompression::GpuDecompressionOptions options = {});
#endif

#endif
