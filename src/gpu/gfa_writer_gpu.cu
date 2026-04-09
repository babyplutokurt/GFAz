#include "gpu/gfa_writer_gpu.hpp"
#include "io/gfa_writer.hpp"

#ifdef ENABLE_CUDA

void write_gfa_from_compressed_data_gpu(
    const CompressedData &data, const std::string &output_path,
    gpu_decompression::GpuDecompressionOptions options) {
  GfaGraph graph = gpu_decompression::decompress_to_host_graph(data, options);
  write_gfa(graph, output_path);
}

#endif
