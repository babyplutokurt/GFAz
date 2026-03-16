#ifndef SERIALIZATION_GPU_HPP
#define SERIALIZATION_GPU_HPP

#include "gpu/compression_workflow_gpu.hpp"
#include <string>

constexpr uint32_t GFAZ_GPU_MAGIC = 0x47505547; // "GPUG" in little-endian
constexpr uint32_t GFAZ_GPU_VERSION = 1;

void serialize_compressed_data_gpu(
    const gpu_compression::CompressedData_gpu &data,
    const std::string &output_path);

gpu_compression::CompressedData_gpu
deserialize_compressed_data_gpu(const std::string &input_path);

#endif
