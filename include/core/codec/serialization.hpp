#ifndef SERIALIZATION_HPP
#define SERIALIZATION_HPP

#include "core/model/compressed_data.hpp"
#include <istream>
#include <string>

namespace gfaz {

// GFAZ binary file format
constexpr uint32_t GFAZ_MAGIC = 0x5A414647;  // "GFAZ" in little-endian
// v5: original path/walk lengths for exact allocation.
// v6: v5 followed by the segment names. Written only for a graph whose
//     segments are not named 1..N; any other graph is still written as v5,
//     byte for byte.
constexpr uint32_t GFAZ_VERSION = 6;
constexpr uint32_t GFAZ_MIN_VERSION = 5;
constexpr uint32_t GFAZ_VERSION_SEGMENT_NAMES = 6;

void serialize_compressed_data(const CompressedData &data,
                               const std::string &output_path);

CompressedData deserialize_compressed_data(const std::string &input_path);

// Deserialize from the stream's current position. Seekable streams retain
// file-size bounds checks; unseekable streams are validated by exact reads.
CompressedData deserialize_compressed_data(std::istream &input);

} // namespace gfaz

#endif
