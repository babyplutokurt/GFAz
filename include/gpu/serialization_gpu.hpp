#ifndef SERIALIZATION_GPU_HPP
#define SERIALIZATION_GPU_HPP

#include "codec/serialization.hpp"

inline void serialize_compressed_data_gpu(const CompressedData &data,
                                          const std::string &output_path) {
  serialize_compressed_data(data, output_path);
}

inline CompressedData deserialize_compressed_data_gpu(
    const std::string &input_path) {
  return deserialize_compressed_data(input_path);
}

#endif
