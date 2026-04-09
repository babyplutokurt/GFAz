#ifndef CODEC_HPP
#define CODEC_HPP

#include "model/compressed_data.hpp"
#include "model/gfa_graph.hpp"
#include <cstdint>
#include <string>
#include <vector>


namespace Codec {

// --- Delta Encoding for Paths ---
void delta_transform(std::vector<std::vector<NodeId>> &paths);
void inverse_delta_transform(std::vector<std::vector<NodeId>> &paths);

// Fused delta transform + max absolute value (single pass optimization)
uint32_t delta_transform_and_max_abs(std::vector<std::vector<NodeId>> &paths);

// --- Delta Encoding for int32 vectors (used for rules) ---
void delta_encode_int32(std::vector<int32_t> &data);
void delta_decode_int32(std::vector<int32_t> &data);

// --- ZSTD Compression ---
ZstdCompressedBlock
zstd_compress_int32_vector(const std::vector<int32_t> &data);
std::vector<int32_t>
zstd_decompress_int32_vector(const ZstdCompressedBlock &block);

ZstdCompressedBlock zstd_compress_string(const std::string &data);
std::string zstd_decompress_string(const ZstdCompressedBlock &block);

ZstdCompressedBlock
zstd_compress_uint32_vector(const std::vector<uint32_t> &data);
std::vector<uint32_t>
zstd_decompress_uint32_vector(const ZstdCompressedBlock &block);

ZstdCompressedBlock zstd_compress_char_vector(const std::vector<char> &data);
std::vector<char> zstd_decompress_char_vector(const ZstdCompressedBlock &block);

// Delta + Varint encoding for uint32 vectors (for link IDs)
ZstdCompressedBlock
compress_delta_varint_uint32(const std::vector<uint32_t> &data);
std::vector<uint32_t>
decompress_delta_varint_uint32(const ZstdCompressedBlock &block,
                               size_t num_elements);

// Bit-packing for orientations (+ = 0, - = 1)
ZstdCompressedBlock compress_orientations(const std::vector<char> &orients);
std::vector<char> decompress_orientations(const ZstdCompressedBlock &block,
                                          size_t num_elements);

// Varint encoding for int64 vectors (for optional field int values)
ZstdCompressedBlock compress_varint_int64(const std::vector<int64_t> &data);
std::vector<int64_t> decompress_varint_int64(const ZstdCompressedBlock &block,
                                             size_t num_elements);

// Float vector compression
ZstdCompressedBlock zstd_compress_float_vector(const std::vector<float> &data);
std::vector<float>
zstd_decompress_float_vector(const ZstdCompressedBlock &block);

} // namespace Codec

#endif // CODEC_HPP
