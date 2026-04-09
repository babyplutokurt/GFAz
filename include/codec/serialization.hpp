#ifndef SERIALIZATION_HPP
#define SERIALIZATION_HPP

#include "model/compressed_data.hpp"
#include <string>


// GFAZ binary file format
constexpr uint32_t GFAZ_MAGIC = 0x5A414647;  // "GFAZ" in little-endian
constexpr uint32_t GFAZ_VERSION = 5;         // v5: Added original path/walk lengths for exact allocation

void serialize_compressed_data(const CompressedData &data,
                               const std::string &output_path);

CompressedData deserialize_compressed_data(const std::string &input_path);

#endif
