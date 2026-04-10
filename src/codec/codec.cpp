#include "codec/codec.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace Codec {

// --- Delta Encoding ---
void delta_transform(std::vector<std::vector<NodeId>> &paths) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (size_t p = 0; p < paths.size(); ++p) {
    auto &path = paths[p];
    if (path.size() < 2)
      continue;
    for (size_t i = path.size() - 1; i > 0; --i) {
      path[i] = path[i] - path[i - 1];
    }
  }
}

void inverse_delta_transform(std::vector<std::vector<NodeId>> &paths) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (size_t p = 0; p < paths.size(); ++p) {
    auto &path = paths[p];
    if (path.size() < 2)
      continue;
    for (size_t i = 1; i < path.size(); ++i) {
      path[i] = path[i] + path[i - 1];
    }
  }
}

// Fused delta transform + max absolute value (single pass optimization)
// Performs delta encoding and tracks max absolute value simultaneously
uint32_t delta_transform_and_max_abs(std::vector<std::vector<NodeId>> &paths) {
  uint32_t max_abs_val = 0;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) reduction(max : max_abs_val)
#endif
  for (size_t p = 0; p < paths.size(); ++p) {
    auto &path = paths[p];
    if (path.empty())
      continue;

    // Track max before delta transform (first element stays as-is)
    uint32_t local_max = static_cast<uint32_t>(std::abs(path[0]));

    if (path.size() >= 2) {
      // Delta encode backwards + track max of delta values
      for (size_t i = path.size() - 1; i > 0; --i) {
        path[i] = path[i] - path[i - 1];
        uint32_t abs_val = static_cast<uint32_t>(std::abs(path[i]));
        if (abs_val > local_max) {
          local_max = abs_val;
        }
      }
    }

    if (local_max > max_abs_val) {
      max_abs_val = local_max;
    }
  }

  return max_abs_val;
}

// --- Delta Encoding for int32 vectors (used for rules) ---
void delta_encode_int32(std::vector<int32_t> &data) {
  if (data.size() < 2)
    return;
  // Encode backwards to avoid overwriting values we still need
  for (size_t i = data.size() - 1; i > 0; --i) {
    data[i] = data[i] - data[i - 1];
  }
}

void delta_decode_int32(std::vector<int32_t> &data) {
  if (data.size() < 2)
    return;
  // Decode forwards (prefix sum)
  for (size_t i = 1; i < data.size(); ++i) {
    data[i] = data[i] + data[i - 1];
  }
}

// --- ZSTD Compression ---
#include "zstd.h"
#include <cstdlib>


namespace {
// Get ZSTD compression level from environment variable, default to 9
int get_zstd_level() {
  static int level = -1;
  if (level == -1) {
    const char *env_val = std::getenv("GFA_COMPRESSION_ZSTD_LEVEL");
    if (env_val) {
      int parsed = std::atoi(env_val);
      // Clamp to valid ZSTD range: 1-22
      if (parsed >= 1 && parsed <= 22) {
        level = parsed;
      } else {
        std::cerr << "Warning: GFA_COMPRESSION_ZSTD_LEVEL=" << env_val
                  << " is invalid (must be 1-22), using default 9."
                  << std::endl;
        level = 9;
      }
    } else {
      level = 9; // Default
    }
  }
  return level;
}

int get_zstd_workers() {
  static int env_workers = -1;
  if (env_workers == -1) {
    env_workers = 0;
    const char *env_val = std::getenv("GFA_COMPRESSION_ZSTD_WORKERS");
    if (env_val) {
      int parsed = std::atoi(env_val);
      if (parsed >= 0) {
        env_workers = parsed;
      } else {
        std::cerr << "Warning: GFA_COMPRESSION_ZSTD_WORKERS=" << env_val
                  << " is invalid (must be >= 0), using active thread count."
                  << std::endl;
      }
    }
  }

  if (env_workers > 0) {
    return env_workers;
  }

#ifdef _OPENMP
  return std::max(1, omp_get_max_threads());
#else
  return 1;
#endif
}

bool configure_zstd_context(ZSTD_CCtx *ctx) {
  const size_t level_result =
      ZSTD_CCtx_setParameter(ctx, ZSTD_c_compressionLevel, get_zstd_level());
  if (ZSTD_isError(level_result)) {
    std::cerr << "ZSTD parameter setup failed: "
              << ZSTD_getErrorName(level_result) << std::endl;
    return false;
  }

  const int workers = get_zstd_workers();
  const size_t worker_result =
      ZSTD_CCtx_setParameter(ctx, ZSTD_c_nbWorkers, workers);
  if (ZSTD_isError(worker_result)) {
    static bool warned_unsupported_workers = false;
    if (!warned_unsupported_workers) {
      std::cerr
          << "Warning: ZSTD multithreaded compression is unavailable ("
          << ZSTD_getErrorName(worker_result)
          << "); falling back to single-threaded ZSTD." << std::endl;
      warned_unsupported_workers = true;
    }
  }

  return true;
}

ZstdCompressedBlock zstd_compress_bytes(const void *data, size_t size_bytes) {
  ZstdCompressedBlock block;
  block.original_size = size_bytes;

  if (size_bytes == 0) {
    return block;
  }

  ZSTD_CCtx *ctx = ZSTD_createCCtx();
  if (ctx == nullptr) {
    std::cerr << "ZSTD compression failed: unable to create compression context"
              << std::endl;
    return block;
  }

  if (!configure_zstd_context(ctx)) {
    ZSTD_freeCCtx(ctx);
    return block;
  }

  const size_t bound = ZSTD_compressBound(size_bytes);
  block.payload.resize(bound);
  const size_t compressed_size =
      ZSTD_compress2(ctx, block.payload.data(), block.payload.size(), data,
                     size_bytes);
  ZSTD_freeCCtx(ctx);

  if (ZSTD_isError(compressed_size)) {
    std::cerr << "ZSTD compression failed: "
              << ZSTD_getErrorName(compressed_size) << std::endl;
    block.payload.clear();
    return block;
  }

  block.payload.resize(compressed_size);
  return block;
}
} // namespace

ZstdCompressedBlock
zstd_compress_int32_vector(const std::vector<int32_t> &data) {
  return zstd_compress_bytes(data.data(), data.size() * sizeof(int32_t));
}

std::vector<int32_t>
zstd_decompress_int32_vector(const ZstdCompressedBlock &block) {
  if (block.original_size == 0) {
    return {};
  }

  size_t num_elements = block.original_size / sizeof(int32_t);
  std::vector<int32_t> result(num_elements);

  size_t decompressed_size =
      ZSTD_decompress(result.data(), block.original_size, block.payload.data(),
                      block.payload.size());

  if (ZSTD_isError(decompressed_size)) {
    std::cerr << "ZSTD decompression failed: "
              << ZSTD_getErrorName(decompressed_size) << std::endl;
    return {};
  }

  return result;
}

ZstdCompressedBlock zstd_compress_string(const std::string &data) {
  return zstd_compress_bytes(data.data(), data.size());
}

std::string zstd_decompress_string(const ZstdCompressedBlock &block) {
  if (block.original_size == 0) {
    return "";
  }

  std::string result(block.original_size, '\0');
  size_t decompressed_size = ZSTD_decompress(
      &result[0], result.size(), block.payload.data(), block.payload.size());

  if (ZSTD_isError(decompressed_size)) {
    std::cerr << "ZSTD decompression failed: "
              << ZSTD_getErrorName(decompressed_size) << std::endl;
    return "";
  }

  return result;
}

ZstdCompressedBlock
zstd_compress_uint32_vector(const std::vector<uint32_t> &data) {
  return zstd_compress_bytes(data.data(), data.size() * sizeof(uint32_t));
}

std::vector<uint32_t>
zstd_decompress_uint32_vector(const ZstdCompressedBlock &block) {
  if (block.original_size == 0) {
    return {};
  }

  size_t num_elements = block.original_size / sizeof(uint32_t);
  std::vector<uint32_t> result(num_elements);

  size_t decompressed_size =
      ZSTD_decompress(result.data(), block.original_size, block.payload.data(),
                      block.payload.size());

  if (ZSTD_isError(decompressed_size)) {
    std::cerr << "ZSTD decompression failed: "
              << ZSTD_getErrorName(decompressed_size) << std::endl;
    return {};
  }

  return result;
}

ZstdCompressedBlock zstd_compress_char_vector(const std::vector<char> &data) {
  return zstd_compress_bytes(data.data(), data.size());
}

std::vector<char>
zstd_decompress_char_vector(const ZstdCompressedBlock &block) {
  if (block.original_size == 0) {
    return {};
  }

  std::vector<char> result(block.original_size);

  size_t decompressed_size =
      ZSTD_decompress(result.data(), block.original_size, block.payload.data(),
                      block.payload.size());

  if (ZSTD_isError(decompressed_size)) {
    std::cerr << "ZSTD decompression failed: "
              << ZSTD_getErrorName(decompressed_size) << std::endl;
    return {};
  }

  return result;
}

// Delta + Varint encoding for uint32 vectors
namespace {
void append_varint(uint32_t value, std::vector<uint8_t> &out) {
  while (value >= 0x80) {
    out.push_back(static_cast<uint8_t>((value & 0x7F) | 0x80));
    value >>= 7;
  }
  out.push_back(static_cast<uint8_t>(value));
}

size_t read_varint(const uint8_t *data, size_t max_len, uint32_t &out_value) {
  out_value = 0;
  size_t i = 0;
  int shift = 0;
  while (i < max_len) {
    uint8_t byte = data[i++];
    out_value |= (byte & 0x7F) << shift;
    if (!(byte & 0x80))
      break;
    shift += 7;
  }
  return i;
}
} // namespace

ZstdCompressedBlock
compress_delta_varint_uint32(const std::vector<uint32_t> &data) {
  if (data.empty()) {
    ZstdCompressedBlock block;
    block.original_size = 0;
    return block;
  }

  // Signed delta encode (handles non-sorted data correctly)
  std::vector<int32_t> deltas(data.size());
  deltas[0] = static_cast<int32_t>(data[0]);
  for (size_t i = 1; i < data.size(); ++i) {
    deltas[i] =
        static_cast<int32_t>(data[i]) - static_cast<int32_t>(data[i - 1]);
  }

  // Zigzag + Varint encode (zigzag maps signed to unsigned for efficient
  // varint)
  std::vector<uint8_t> varint_bytes;
  varint_bytes.reserve(data.size() * 2);
  for (int32_t val : deltas) {
    // Zigzag encode: (val << 1) ^ (val >> 31)
    uint32_t zigzag = static_cast<uint32_t>((val << 1) ^ (val >> 31));
    append_varint(zigzag, varint_bytes);
  }

  // ZSTD compress
  return zstd_compress_string(
      std::string(varint_bytes.begin(), varint_bytes.end()));
}

std::vector<uint32_t>
decompress_delta_varint_uint32(const ZstdCompressedBlock &block,
                               size_t num_elements) {
  if (block.original_size == 0 || num_elements == 0) {
    return {};
  }

  // ZSTD decompress
  std::string varint_str = zstd_decompress_string(block);
  const uint8_t *bytes = reinterpret_cast<const uint8_t *>(varint_str.data());
  size_t len = varint_str.size();

  // Varint decode + zigzag decode
  std::vector<int32_t> deltas;
  deltas.reserve(num_elements);
  size_t offset = 0;
  while (offset < len && deltas.size() < num_elements) {
    uint32_t zigzag;
    offset += read_varint(bytes + offset, len - offset, zigzag);
    // Zigzag decode: (zigzag >> 1) ^ -(zigzag & 1)
    int32_t val = static_cast<int32_t>((zigzag >> 1) ^ (~(zigzag & 1) + 1));
    deltas.push_back(val);
  }

  // Inverse delta (accumulate signed deltas back to unsigned values)
  std::vector<uint32_t> result(deltas.size());
  result[0] = static_cast<uint32_t>(deltas[0]);
  for (size_t i = 1; i < deltas.size(); ++i) {
    result[i] =
        static_cast<uint32_t>(static_cast<int32_t>(result[i - 1]) + deltas[i]);
  }

  return result;
}

// Bit-packing for single orientation vector (+ = 0, - = 1)
ZstdCompressedBlock compress_orientations(const std::vector<char> &orients) {
  size_t n = orients.size();
  // 1 bit per orientation, pack into bytes
  size_t num_bytes = (n + 7) / 8;
  std::vector<uint8_t> packed(num_bytes, 0);

  for (size_t i = 0; i < n; ++i) {
    size_t byte_idx = i / 8;
    size_t bit_offset = i % 8;

    uint8_t bit = (orients[i] == '-') ? 1 : 0;
    packed[byte_idx] |= (bit << bit_offset);
  }

  return zstd_compress_string(std::string(packed.begin(), packed.end()));
}

std::vector<char> decompress_orientations(const ZstdCompressedBlock &block,
                                          size_t num_elements) {
  std::vector<char> orients;

  if (block.original_size == 0 || num_elements == 0) {
    return orients;
  }

  std::string packed_str = zstd_decompress_string(block);
  const uint8_t *packed = reinterpret_cast<const uint8_t *>(packed_str.data());

  orients.reserve(num_elements);

  for (size_t i = 0; i < num_elements; ++i) {
    size_t byte_idx = i / 8;
    size_t bit_offset = i % 8;

    uint8_t bit = (packed[byte_idx] >> bit_offset) & 1;
    orients.push_back(bit ? '-' : '+');
  }

  return orients;
}

// Varint encoding for int64 with zigzag for signed values
namespace {
uint64_t zigzag_encode_64(int64_t value) {
  return (static_cast<uint64_t>(value) << 1) ^
         static_cast<uint64_t>(value >> 63);
}

int64_t zigzag_decode_64(uint64_t value) {
  return static_cast<int64_t>((value >> 1) ^ (~(value & 1) + 1));
}

void append_varint_64(uint64_t value, std::vector<uint8_t> &out) {
  while (value >= 0x80) {
    out.push_back(static_cast<uint8_t>((value & 0x7F) | 0x80));
    value >>= 7;
  }
  out.push_back(static_cast<uint8_t>(value));
}

size_t read_varint_64(const uint8_t *data, size_t max_len,
                      uint64_t &out_value) {
  out_value = 0;
  size_t i = 0;
  int shift = 0;
  while (i < max_len) {
    uint8_t byte = data[i++];
    out_value |= static_cast<uint64_t>(byte & 0x7F) << shift;
    if (!(byte & 0x80))
      break;
    shift += 7;
  }
  return i;
}
} // namespace

ZstdCompressedBlock compress_varint_int64(const std::vector<int64_t> &data) {
  if (data.empty()) {
    ZstdCompressedBlock block;
    block.original_size = 0;
    return block;
  }

  // Zigzag + Varint encode
  std::vector<uint8_t> varint_bytes;
  varint_bytes.reserve(data.size() * 4);
  for (int64_t val : data) {
    append_varint_64(zigzag_encode_64(val), varint_bytes);
  }

  return zstd_compress_string(
      std::string(varint_bytes.begin(), varint_bytes.end()));
}

std::vector<int64_t> decompress_varint_int64(const ZstdCompressedBlock &block,
                                             size_t num_elements) {
  if (block.original_size == 0 || num_elements == 0) {
    return {};
  }

  std::string varint_str = zstd_decompress_string(block);
  const uint8_t *bytes = reinterpret_cast<const uint8_t *>(varint_str.data());
  size_t len = varint_str.size();

  std::vector<int64_t> result;
  result.reserve(num_elements);
  size_t offset = 0;
  while (offset < len && result.size() < num_elements) {
    uint64_t val;
    offset += read_varint_64(bytes + offset, len - offset, val);
    result.push_back(zigzag_decode_64(val));
  }

  return result;
}

ZstdCompressedBlock zstd_compress_float_vector(const std::vector<float> &data) {
  return zstd_compress_bytes(data.data(), data.size() * sizeof(float));
}

std::vector<float>
zstd_decompress_float_vector(const ZstdCompressedBlock &block) {
  if (block.original_size == 0) {
    return {};
  }

  size_t num_elements = block.original_size / sizeof(float);
  std::vector<float> result(num_elements);

  size_t decompressed_size =
      ZSTD_decompress(result.data(), block.original_size, block.payload.data(),
                      block.payload.size());

  if (ZSTD_isError(decompressed_size)) {
    std::cerr << "ZSTD decompression failed: "
              << ZSTD_getErrorName(decompressed_size) << std::endl;
    return {};
  }

  return result;
}

} // namespace Codec
