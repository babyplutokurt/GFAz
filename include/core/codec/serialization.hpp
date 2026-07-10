#ifndef SERIALIZATION_HPP
#define SERIALIZATION_HPP

#include "core/model/compressed_data.hpp"
#include <istream>
#include <string>

namespace gfaz {

// GFAZ binary file format
constexpr uint32_t GFAZ_MAGIC = 0x5A414647;  // "GFAZ" in little-endian
constexpr uint32_t GFAZ_VERSION = 5;         // v5: Added original path/walk lengths for exact allocation

void serialize_compressed_data(const CompressedData &data,
                               const std::string &output_path);

CompressedData deserialize_compressed_data(const std::string &input_path);

// Deserialize from the stream's current position. Seekable streams retain
// file-size bounds checks; unseekable streams are validated by exact reads.
CompressedData deserialize_compressed_data(std::istream &input);

} // namespace gfaz

#endif
