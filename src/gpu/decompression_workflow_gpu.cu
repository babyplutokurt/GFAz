#include "gpu/codec_gpu.cuh"
#include "gpu/codec_gpu_nvcomp.cuh"
#include "gpu/decompression_workflow_gpu.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

namespace gpu_decompression {

// Debug flag for decompression stats
static bool g_debug_decompression = false;
using Clock = std::chrono::high_resolution_clock;

void set_gpu_decompression_debug(bool enabled) {
  g_debug_decompression = enabled;
}

static double elapsed_ms(const Clock::time_point &start,
                         const Clock::time_point &end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

// =============================================================================
// Decompression helper functions
// =============================================================================

// Decode a single varint from byte stream, returns number of bytes consumed
static size_t decode_varint_32(const uint8_t *data, size_t max_len,
                               uint32_t &out) {
  out = 0;
  size_t shift = 0;
  size_t i = 0;
  while (i < max_len) {
    uint8_t byte = data[i++];
    out |= static_cast<uint32_t>(byte & 0x7F) << shift;
    if ((byte & 0x80) == 0)
      break;
    shift += 7;
  }
  return i;
}

// Decode zigzag-encoded int32
static int32_t zigzag_decode_32(uint32_t n) {
  return static_cast<int32_t>((n >> 1) ^ -(static_cast<int32_t>(n & 1)));
}

// Decode a single varint64 from byte stream
static size_t decode_varint_64(const uint8_t *data, size_t max_len,
                               uint64_t &out) {
  out = 0;
  size_t shift = 0;
  size_t i = 0;
  while (i < max_len) {
    uint8_t byte = data[i++];
    out |= static_cast<uint64_t>(byte & 0x7F) << shift;
    if ((byte & 0x80) == 0)
      break;
    shift += 7;
  }
  return i;
}

// Decode zigzag-encoded int64
static int64_t zigzag_decode_64(uint64_t n) {
  return static_cast<int64_t>((n >> 1) ^ -(static_cast<int64_t>(n & 1)));
}

// Decompress delta + zigzag + varint encoded uint32 array
static std::vector<uint32_t>
decompress_delta_varint_uint32(const gpu_codec::NvcompCompressedBlock &block,
                               size_t expected_count) {

  if (block.payload.empty()) {
    return std::vector<uint32_t>();
  }

  // Decompress to get varint bytes
  std::vector<uint8_t> varint_bytes = gpu_codec::nvcomp_zstd_decompress(block);

  // Decode varints and apply inverse zigzag + delta
  std::vector<uint32_t> result;
  result.reserve(expected_count);

  size_t offset = 0;
  int32_t prev = 0;
  while (offset < varint_bytes.size() && result.size() < expected_count) {
    uint32_t zigzag_val;
    offset += decode_varint_32(varint_bytes.data() + offset,
                               varint_bytes.size() - offset, zigzag_val);
    int32_t delta = zigzag_decode_32(zigzag_val);
    int32_t current = prev + delta;
    result.push_back(static_cast<uint32_t>(current));
    prev = current;
  }

  return result;
}

// Decompress bit-packed orientations using GPU
static std::vector<char>
decompress_orientations(const gpu_codec::NvcompCompressedBlock &block,
                        size_t expected_count) {

  if (block.payload.empty()) {
    return std::vector<char>();
  }

  std::vector<uint8_t> packed = gpu_codec::nvcomp_zstd_decompress(block);

  // Use GPU kernel for bit-unpacking
  return gpu_codec::unpack_orientations_gpu(packed, expected_count);
}

// Decompress varint-encoded int64 array
static std::vector<int64_t>
decompress_varint_int64(const gpu_codec::NvcompCompressedBlock &block,
                        size_t expected_count) {

  if (block.payload.empty()) {
    return std::vector<int64_t>();
  }

  std::vector<uint8_t> varint_bytes = gpu_codec::nvcomp_zstd_decompress(block);

  std::vector<int64_t> result;
  result.reserve(expected_count);

  size_t offset = 0;
  while (offset < varint_bytes.size() && result.size() < expected_count) {
    uint64_t zigzag_val;
    offset += decode_varint_64(varint_bytes.data() + offset,
                               varint_bytes.size() - offset, zigzag_val);
    result.push_back(zigzag_decode_64(zigzag_val));
  }

  return result;
}

// Decompress raw bytes to float vector
static std::vector<float>
decompress_float(const gpu_codec::NvcompCompressedBlock &block) {

  if (block.payload.empty()) {
    return std::vector<float>();
  }

  std::vector<uint8_t> bytes = gpu_codec::nvcomp_zstd_decompress(block);
  size_t count = bytes.size() / sizeof(float);
  std::vector<float> result(count);
  std::memcpy(result.data(), bytes.data(), count * sizeof(float));
  return result;
}

// Decompress raw bytes to char vector
static std::vector<char>
decompress_char(const gpu_codec::NvcompCompressedBlock &block) {

  if (block.payload.empty()) {
    return std::vector<char>();
  }

  std::vector<uint8_t> bytes = gpu_codec::nvcomp_zstd_decompress(block);
  return std::vector<char>(bytes.begin(), bytes.end());
}

// Decompress an optional field column
static OptionalFieldColumn_gpu decompress_optional_field_column(
    const gpu_compression::CompressedOptionalFieldColumn_gpu &compressed) {

  OptionalFieldColumn_gpu result;
  result.tag = compressed.tag;
  result.type = compressed.type;
  result.num_elements = compressed.num_elements;

  switch (compressed.type) {
  case 'i':
    result.int_values = decompress_varint_int64(
        compressed.int_values_zstd_nvcomp, compressed.num_elements);
    break;

  case 'f':
    result.float_values = decompress_float(compressed.float_values_zstd_nvcomp);
    break;

  case 'A':
    result.char_values = decompress_char(compressed.char_values_zstd_nvcomp);
    break;

  case 'Z':
  case 'J':
  case 'H': {
    std::string str_data = gpu_codec::nvcomp_zstd_decompress_string(
        compressed.strings_zstd_nvcomp);
    result.strings.data = std::vector<char>(str_data.begin(), str_data.end());
    result.strings.lengths = gpu_codec::nvcomp_zstd_decompress_uint32(
        compressed.string_lengths_zstd_nvcomp);
    break;
  }

  case 'B': {
    std::vector<uint8_t> subtypes_bytes =
        gpu_codec::nvcomp_zstd_decompress(compressed.b_subtypes_zstd_nvcomp);
    result.b_subtypes =
        std::vector<char>(subtypes_bytes.begin(), subtypes_bytes.end());
    result.b_lengths = gpu_codec::nvcomp_zstd_decompress_uint32(
        compressed.b_lengths_zstd_nvcomp);
    result.b_data = gpu_codec::nvcomp_zstd_decompress(
        compressed.b_concat_bytes_zstd_nvcomp);
    break;
  }

  default:
    // Unknown type, leave empty
    break;
  }

  return result;
}

// Helper: Decompress a FlattenedStrings from data + lengths blocks
static FlattenedStrings decompress_flattened_strings(
    const gpu_codec::NvcompCompressedBlock &data_block,
    const gpu_codec::NvcompCompressedBlock &lengths_block) {
  FlattenedStrings result;
  std::string str_data = gpu_codec::nvcomp_zstd_decompress_string(data_block);
  result.data = std::vector<char>(str_data.begin(), str_data.end());
  result.lengths = gpu_codec::nvcomp_zstd_decompress_uint32(lengths_block);
  return result;
}

static void decompress_optional_columns(
    const std::vector<gpu_compression::CompressedOptionalFieldColumn_gpu>
        &compressed_columns,
    std::vector<OptionalFieldColumn_gpu> &out_columns) {
  out_columns.reserve(out_columns.size() + compressed_columns.size());
  for (const auto &compressed_col : compressed_columns) {
    out_columns.push_back(decompress_optional_field_column(compressed_col));
  }
}

// CUDA error checking macro
#define CUDA_CHECK_DECOMP(call)                                                \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      throw std::runtime_error(std::string("CUDA error at ") + __FILE__ +      \
                               ":" + std::to_string(__LINE__) + " - " +        \
                               cudaGetErrorString(err));                       \
    }                                                                          \
  } while (0)

struct ScopedCudaStreams3 {
  cudaStream_t a = nullptr;
  cudaStream_t b = nullptr;
  cudaStream_t c = nullptr;

  ScopedCudaStreams3() {
    CUDA_CHECK_DECOMP(cudaStreamCreate(&a));
    CUDA_CHECK_DECOMP(cudaStreamCreate(&b));
    CUDA_CHECK_DECOMP(cudaStreamCreate(&c));
  }

  ~ScopedCudaStreams3() {
    if (a)
      cudaStreamDestroy(a);
    if (b)
      cudaStreamDestroy(b);
    if (c)
      cudaStreamDestroy(c);
  }

  ScopedCudaStreams3(const ScopedCudaStreams3 &) = delete;
  ScopedCudaStreams3 &operator=(const ScopedCudaStreams3 &) = delete;
};

FlattenedPaths
decompress_paths_gpu(const gpu_compression::CompressedData_gpu &data) {
  auto decomp_start = Clock::now();

  FlattenedPaths result;

  // Decompress encoded path + rule arrays concurrently.
  ScopedCudaStreams3 streams;

  // =========================================================================
  // OPTIMIZATION 1: Decompress directly to device memory (avoid D->H->D)
  // OPTIMIZATION 2: Use separate CUDA streams for parallel decompression
  // =========================================================================

  if (g_debug_decompression) {
    std::cout << "[GPU Decompress] Starting parallel decompression to device..."
              << std::endl;
  }

  // Decompress all three arrays in parallel using separate streams
  int32_t *d_encoded_path_raw = nullptr;
  int32_t *d_first_delta_raw = nullptr;
  int32_t *d_second_delta_raw = nullptr;
  size_t encoded_path_count = 0;
  size_t first_count = 0;
  size_t second_count = 0;

  // Launch all three decompressions in parallel
  gpu_codec::nvcomp_zstd_decompress_int32_to_device(
      data.encoded_path_zstd_nvcomp, &d_encoded_path_raw, &encoded_path_count,
      streams.a);

  gpu_codec::nvcomp_zstd_decompress_int32_to_device(
      data.rules_first_zstd_nvcomp, &d_first_delta_raw, &first_count,
      streams.b);

  gpu_codec::nvcomp_zstd_decompress_int32_to_device(
      data.rules_second_zstd_nvcomp, &d_second_delta_raw, &second_count,
      streams.c);

  // Synchronize all streams
  CUDA_CHECK_DECOMP(cudaStreamSynchronize(streams.a));
  CUDA_CHECK_DECOMP(cudaStreamSynchronize(streams.b));
  CUDA_CHECK_DECOMP(cudaStreamSynchronize(streams.c));

  auto t_nvcomp_end = Clock::now();
  double nvcomp_time_ms = elapsed_ms(decomp_start, t_nvcomp_end);

  if (g_debug_decompression) {
    std::cout << "[GPU Decompress] nvComp decompression: " << std::fixed
              << std::setprecision(2) << nvcomp_time_ms << " ms" << std::endl;
  }

  if (d_encoded_path_raw == nullptr || encoded_path_count == 0) {
    std::cerr << "[GPU Decompress] Error: encoded_path is empty!" << std::endl;
    // Still decompress path_lengths so zero-length paths preserve structure
    result.lengths =
        gpu_codec::nvcomp_zstd_decompress_uint32(data.path_lengths_zstd_nvcomp);
    return result;
  }

  // Wrap raw pointers in thrust device_ptr for easier manipulation
  thrust::device_ptr<int32_t> d_encoded_ptr(d_encoded_path_raw);
  thrust::device_vector<int32_t> d_encoded_path(
      d_encoded_ptr, d_encoded_ptr + encoded_path_count);
  CUDA_CHECK_DECOMP(cudaFree(
      d_encoded_path_raw)); // Free original, data copied to device_vector

  if (d_first_delta_raw == nullptr || first_count == 0 ||
      d_second_delta_raw == nullptr || second_count == 0) {
    // No rules means the path is just delta-encoded raw nodes
    if (g_debug_decompression) {
      std::cout << "[GPU Decompress] No rules found, path is raw delta-encoded"
                << std::endl;
    }

    // Clean up any allocated memory
    if (d_first_delta_raw)
      CUDA_CHECK_DECOMP(cudaFree(d_first_delta_raw));
    if (d_second_delta_raw)
      CUDA_CHECK_DECOMP(cudaFree(d_second_delta_raw));

    // Decompress path_lengths (original, pre-encoding lengths)
    result.lengths =
        gpu_codec::nvcomp_zstd_decompress_uint32(data.path_lengths_zstd_nvcomp);

    // Segmented inverse delta-decode using original lengths
    uint32_t num_segs = static_cast<uint32_t>(result.lengths.size());
    thrust::device_vector<uint32_t> d_lens(result.lengths.begin(),
                                            result.lengths.end());
    thrust::device_vector<uint64_t> d_offs(num_segs);
    thrust::exclusive_scan(d_lens.begin(), d_lens.end(), d_offs.begin(),
                           uint64_t(0));

    thrust::device_vector<int32_t> d_original =
        gpu_codec::segmented_inverse_delta_decode_device_vec(
            d_encoded_path, d_offs, num_segs, d_encoded_path.size());

    result.data.resize(d_original.size());
    thrust::copy(d_original.begin(), d_original.end(), result.data.begin());
    return result;
  }

  // Wrap rules in device_vectors
  thrust::device_ptr<int32_t> d_first_ptr(d_first_delta_raw);
  thrust::device_ptr<int32_t> d_second_ptr(d_second_delta_raw);
  thrust::device_vector<int32_t> d_first_delta(d_first_ptr,
                                               d_first_ptr + first_count);
  thrust::device_vector<int32_t> d_second_delta(d_second_ptr,
                                                d_second_ptr + second_count);
  CUDA_CHECK_DECOMP(cudaFree(d_first_delta_raw));
  CUDA_CHECK_DECOMP(cudaFree(d_second_delta_raw));

  if (g_debug_decompression) {
    std::cout << "[GPU Decompress] Decompressed: encoded_path="
              << encoded_path_count << ", rules_first=" << first_count
              << ", rules_second=" << second_count << std::endl;
  }

  // =========================================================================
  // Inverse delta-decode rules
  // =========================================================================

  auto t_rules_delta_start = Clock::now();

  thrust::device_vector<int32_t> d_rules_first =
      gpu_codec::inverse_delta_decode_device_vec(d_first_delta);
  thrust::device_vector<int32_t> d_rules_second =
      gpu_codec::inverse_delta_decode_device_vec(d_second_delta);

  auto t_rules_delta_end = Clock::now();
  double rules_delta_time_ms =
      elapsed_ms(t_rules_delta_start, t_rules_delta_end);

  if (g_debug_decompression) {
    std::cout << "[GPU Decompress] Inverse delta-decode rules: " << std::fixed
              << std::setprecision(2) << rules_delta_time_ms << " ms"
              << std::endl;
  }

  // Get min_rule_id and total rules
  uint32_t min_rule_id = data.min_rule_id();
  size_t num_rules = data.total_rules();

  // Validate decoded rule array sizes match expected count from layer_ranges
  if (d_rules_first.size() != num_rules || d_rules_second.size() != num_rules) {
    std::cerr << "[GPU Decompress] ERROR: Rule count mismatch! "
              << "Expected " << num_rules << " rules, but decoded "
              << d_rules_first.size() << " first / " << d_rules_second.size()
              << " second." << std::endl;
    // Fall back: use the minimum to avoid out-of-bounds access
    num_rules =
        std::min({num_rules, d_rules_first.size(), d_rules_second.size()});
    if (num_rules == 0) {
      // No valid rules - segmented inverse-delta-decode the path
      result.lengths = gpu_codec::nvcomp_zstd_decompress_uint32(
          data.path_lengths_zstd_nvcomp);
      uint32_t ns = static_cast<uint32_t>(result.lengths.size());
      thrust::device_vector<uint32_t> dl(result.lengths.begin(),
                                          result.lengths.end());
      thrust::device_vector<uint64_t> do2(ns);
      thrust::exclusive_scan(dl.begin(), dl.end(), do2.begin(),
                             uint64_t(0));

      thrust::device_vector<int32_t> d_decoded =
          gpu_codec::segmented_inverse_delta_decode_device_vec(
              d_encoded_path, do2, ns, d_encoded_path.size());
      result.data.resize(d_decoded.size());
      thrust::copy(d_decoded.begin(), d_decoded.end(), result.data.begin());
      return result;
    }
  }

  if (g_debug_decompression) {
    std::cout << "[GPU Decompress] Expanding path with rolling chunk scheduler "
              << "(128 paths per chunk), min_rule_id=" << min_rule_id << std::endl;
  }

  // =========================================================================
  // Expand path on GPU + Inverse delta-decode the expanded path (rolling)
  // =========================================================================

  auto expand_start = Clock::now();
  
  result.lengths =
      gpu_codec::nvcomp_zstd_decompress_uint32(data.path_lengths_zstd_nvcomp);
  thrust::device_vector<uint32_t> d_lens_final(result.lengths.begin(),
                                                result.lengths.end());
                                                
  gpu_codec::rolling_expand_and_inverse_delta_decode(
      d_encoded_path, d_rules_first, d_rules_second, min_rule_id, num_rules,
      d_lens_final, result.data);

  auto expand_end = Clock::now();
  double expand_time_ms = elapsed_ms(expand_start, expand_end);

  if (g_debug_decompression) {
    std::cout << "[GPU Decompress] Rolling Rule expansion and copy to host: " << result.data.size() << " elements in " << std::fixed
              << std::setprecision(2) << expand_time_ms << " ms" << std::endl;
  }

  auto decomp_end = Clock::now();
  double decomp_time_ms = elapsed_ms(decomp_start, decomp_end);

  if (g_debug_decompression) {
    double gpu_work_ms = nvcomp_time_ms + rules_delta_time_ms + expand_time_ms;

    std::cout << "[GPU Decompress] === TIMING BREAKDOWN ===" << std::endl;
    std::cout << "  GPU Work:" << std::endl;
    std::cout << "    1. ZSTD decompress (nvComp):    " << std::fixed
              << std::setprecision(2) << nvcomp_time_ms << " ms" << std::endl;
    std::cout << "    2. Decode rules (prefix sum):   " << std::fixed
              << std::setprecision(2) << rules_delta_time_ms << " ms"
              << std::endl;
    std::cout << "    3. Rolling Expand/Decode/Copy:  " << std::fixed
              << std::setprecision(2) << expand_time_ms << " ms" << std::endl;
    std::cout << "    ─────────────────────────────" << std::endl;
    std::cout << "    GPU Total:                      " << std::fixed
              << std::setprecision(2) << gpu_work_ms << " ms" << std::endl;
    std::cout << "  TOTAL:                            " << std::fixed
              << std::setprecision(2) << decomp_time_ms << " ms" << std::endl;
    std::cout << "[GPU Decompress] Path: " << encoded_path_count
              << " compressed -> " << result.data.size() << " elements"
              << std::endl;
  }

  return result;
}

GfaGraph_gpu
decompress_to_gpu_layout(const gpu_compression::CompressedData_gpu &data) {
  auto start = Clock::now();

  GfaGraph_gpu result;

  // =========================================================================
  // 1. Decompress paths (main GPU-accelerated decompression)
  // =========================================================================
  if (g_debug_decompression) {
    std::cout << "[GPU Decompress] === Starting full decompression ==="
              << std::endl;
  }
  result.paths = decompress_paths_gpu(data);

  // Store path/walk split info
  result.num_paths = data.num_paths;
  result.num_walks = data.num_walks;

  // =========================================================================
  // 2. Decompress path names (P-lines only)
  // =========================================================================
  if (g_debug_decompression) {
    std::cout << "[GPU Decompress] Decompressing path names..." << std::endl;
  }
  result.path_names = decompress_flattened_strings(
      data.names_zstd_nvcomp, data.name_lengths_zstd_nvcomp);

  // =========================================================================
  // 3. Decompress path overlaps (P-lines only)
  // =========================================================================
  if (g_debug_decompression) {
    std::cout << "[GPU Decompress] Decompressing path overlaps..." << std::endl;
  }
  result.path_overlaps = decompress_flattened_strings(
      data.overlaps_zstd_nvcomp, data.overlap_lengths_zstd_nvcomp);

  // =========================================================================
  // 4. Decompress walk metadata (W-lines only)
  // =========================================================================
  if (data.num_walks > 0) {
    if (g_debug_decompression) {
      std::cout << "[GPU Decompress] Decompressing walk metadata ("
                << data.num_walks << " walks)..." << std::endl;
    }

    result.walk_sample_ids =
        decompress_flattened_strings(data.walk_sample_ids_zstd_nvcomp,
                                     data.walk_sample_id_lengths_zstd_nvcomp);

    result.walk_hap_indices = gpu_codec::nvcomp_zstd_decompress_uint32(
        data.walk_hap_indices_zstd_nvcomp);

    result.walk_seq_ids = decompress_flattened_strings(
        data.walk_seq_ids_zstd_nvcomp, data.walk_seq_id_lengths_zstd_nvcomp);

    result.walk_seq_starts = decompress_varint_int64(
        data.walk_seq_starts_zstd_nvcomp, data.num_walks);
    result.walk_seq_ends =
        decompress_varint_int64(data.walk_seq_ends_zstd_nvcomp, data.num_walks);
  }

  // =========================================================================
  // 5. Decompress segment sequences
  // =========================================================================
  if (g_debug_decompression) {
    std::cout << "[GPU Decompress] Decompressing segment sequences..."
              << std::endl;
  }
  result.node_sequences = decompress_flattened_strings(
      data.segment_sequences_zstd_nvcomp, data.segment_seq_lengths_zstd_nvcomp);

  // =========================================================================
  // 5b. Restore header and node names (for full-fidelity round-trip)
  // =========================================================================
  result.header_line = data.header_line;

  result.node_names = decompress_flattened_strings(
      data.node_names_zstd_nvcomp, data.node_name_lengths_zstd_nvcomp);

  // =========================================================================
  // 6. Decompress links
  // =========================================================================
  if (g_debug_decompression) {
    std::cout << "[GPU Decompress] Decompressing links (" << data.num_links
              << " links)..." << std::endl;
  }

  if (data.num_links > 0) {
    result.link_from_ids = decompress_delta_varint_uint32(
        data.link_from_ids_zstd_nvcomp, data.num_links);
    result.link_to_ids = decompress_delta_varint_uint32(
        data.link_to_ids_zstd_nvcomp, data.num_links);
    result.link_from_orients = decompress_orientations(
        data.link_from_orients_zstd_nvcomp, data.num_links);
    result.link_to_orients = decompress_orientations(
        data.link_to_orients_zstd_nvcomp, data.num_links);
    result.link_overlap_nums = gpu_codec::nvcomp_zstd_decompress_uint32(
        data.link_overlap_nums_zstd_nvcomp);
    result.link_overlap_ops =
        decompress_char(data.link_overlap_ops_zstd_nvcomp);
  }

  // =========================================================================
  // 7. Decompress segment optional fields
  // =========================================================================
  if (!data.segment_optional_fields_zstd_nvcomp.empty()) {
    if (g_debug_decompression) {
      std::cout << "[GPU Decompress] Decompressing segment optional fields ("
                << data.segment_optional_fields_zstd_nvcomp.size()
                << " columns)..." << std::endl;
    }

    decompress_optional_columns(data.segment_optional_fields_zstd_nvcomp,
                                result.segment_optional_fields);
  }

  // =========================================================================
  // 8. Decompress link optional fields
  // =========================================================================
  if (!data.link_optional_fields_zstd_nvcomp.empty()) {
    if (g_debug_decompression) {
      std::cout << "[GPU Decompress] Decompressing link optional fields ("
                << data.link_optional_fields_zstd_nvcomp.size()
                << " columns)..." << std::endl;
    }

    decompress_optional_columns(data.link_optional_fields_zstd_nvcomp,
                                result.link_optional_fields);
  }

  // =========================================================================
  // 9. Decompress J-lines (jumps) - structured columnar
  // =========================================================================
  if (data.num_jumps_stored > 0) {
    if (g_debug_decompression) {
      std::cout << "[GPU Decompress] Decompressing J-lines ("
                << data.num_jumps_stored << " jumps)..." << std::endl;
    }

    result.jump_from_ids = decompress_delta_varint_uint32(
        data.jump_from_ids_zstd_nvcomp, data.num_jumps_stored);
    result.jump_to_ids = decompress_delta_varint_uint32(
        data.jump_to_ids_zstd_nvcomp, data.num_jumps_stored);
    result.jump_from_orients = decompress_orientations(
        data.jump_from_orients_zstd_nvcomp, data.num_jumps_stored);
    result.jump_to_orients = decompress_orientations(
        data.jump_to_orients_zstd_nvcomp, data.num_jumps_stored);

    result.jump_distances =
        decompress_flattened_strings(data.jump_distances_zstd_nvcomp,
                                     data.jump_distance_lengths_zstd_nvcomp);
    result.jump_rest_fields = decompress_flattened_strings(
        data.jump_rest_fields_zstd_nvcomp, data.jump_rest_lengths_zstd_nvcomp);
  }

  // =========================================================================
  // 10. Decompress C-lines (containments) - structured columnar
  // =========================================================================
  if (data.num_containments_stored > 0) {
    if (g_debug_decompression) {
      std::cout << "[GPU Decompress] Decompressing C-lines ("
                << data.num_containments_stored << " containments)..."
                << std::endl;
    }

    result.containment_container_ids = decompress_delta_varint_uint32(
        data.containment_container_ids_zstd_nvcomp,
        data.num_containments_stored);
    result.containment_contained_ids = decompress_delta_varint_uint32(
        data.containment_contained_ids_zstd_nvcomp,
        data.num_containments_stored);
    result.containment_container_orients =
        decompress_orientations(data.containment_container_orients_zstd_nvcomp,
                                data.num_containments_stored);
    result.containment_contained_orients =
        decompress_orientations(data.containment_contained_orients_zstd_nvcomp,
                                data.num_containments_stored);
    result.containment_positions = gpu_codec::nvcomp_zstd_decompress_uint32(
        data.containment_positions_zstd_nvcomp);

    result.containment_overlaps = decompress_flattened_strings(
        data.containment_overlaps_zstd_nvcomp,
        data.containment_overlap_lengths_zstd_nvcomp);
    result.containment_rest_fields =
        decompress_flattened_strings(data.containment_rest_fields_zstd_nvcomp,
                                     data.containment_rest_lengths_zstd_nvcomp);
  }

  // =========================================================================
  // 11. Set metadata
  // =========================================================================
  result.num_segments =
      static_cast<uint32_t>(result.node_sequences.lengths.size());
  result.num_links = static_cast<uint32_t>(data.num_links);

  auto end = Clock::now();
  double time_ms = elapsed_ms(start, end);

  if (g_debug_decompression) {
    std::cout << "[GPU Decompress] === Full decompression complete ==="
              << std::endl;
    std::cout << "  Segments:      " << result.num_segments << std::endl;
    std::cout << "  Paths:         " << result.num_paths << " ("
              << result.paths.total_nodes() << " total nodes)" << std::endl;
    std::cout << "  Walks:         " << result.num_walks << std::endl;
    std::cout << "  Links:         " << result.num_links << std::endl;
    if (result.num_jumps() > 0)
      std::cout << "  Jumps:         " << result.num_jumps() << std::endl;
    if (result.num_containments() > 0)
      std::cout << "  Containments:  " << result.num_containments()
                << std::endl;
    std::cout << "  Time:          " << std::fixed << std::setprecision(2)
              << time_ms << " ms" << std::endl;
  }

  return result;
}

} // namespace gpu_decompression
