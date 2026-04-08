#include "gpu/codec_gpu.cuh"
#include "gpu/codec_gpu_nvcomp.cuh"
#include "gpu/decompression_workflow_gpu_internal.hpp"
#include "gpu/decompression_workflow_gpu.hpp"
#include "gpu/metadata_codec_gpu.hpp"
#include "gpu/path_decompression_gpu_legacy.hpp"
#include "gpu/path_decompression_gpu_rolling.hpp"
#include <algorithm>
#include <chrono>
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

bool decompression_debug_enabled() { return g_debug_decompression; }

static double elapsed_ms(const Clock::time_point &start,
                         const Clock::time_point &end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
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
decompress_paths_gpu(const gpu_compression::CompressedData_gpu &data,
                     GpuDecompressionOptions options) {
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
  const uint32_t traversals_per_chunk =
      std::max<uint32_t>(1, options.traversals_per_chunk);

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

  auto expand_start = Clock::now();

  result.lengths =
      gpu_codec::nvcomp_zstd_decompress_uint32(data.path_lengths_zstd_nvcomp);
  thrust::device_vector<uint32_t> d_lens_final(result.lengths.begin(),
                                               result.lengths.end());

  if (options.use_legacy_full_decompression) {
    decompress_paths_gpu_legacy(d_encoded_path, d_rules_first, d_rules_second,
                                min_rule_id, num_rules, d_lens_final,
                                result.data);
  } else {
    decompress_paths_gpu_rolling(
        d_encoded_path, d_rules_first, d_rules_second, min_rule_id, num_rules,
        d_lens_final, traversals_per_chunk, result.data);
  }

  auto expand_end = Clock::now();
  double expand_time_ms = elapsed_ms(expand_start, expand_end);

  if (g_debug_decompression) {
    std::cout << "[GPU Decompress] Rule expansion and copy to host: "
              << result.data.size() << " elements in " << std::fixed
              << std::setprecision(2) << expand_time_ms << " ms"
              << std::endl;
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
decompress_to_gpu_layout(const gpu_compression::CompressedData_gpu &data,
                         GpuDecompressionOptions options) {
  auto start = Clock::now();

  GfaGraph_gpu result;

  // =========================================================================
  // 1. Decompress paths (main GPU-accelerated decompression)
  // =========================================================================
  if (g_debug_decompression) {
    std::cout << "[GPU Decompress] === Starting full decompression ==="
              << std::endl;
  }

  const auto paths_start = Clock::now();
  result.paths = decompress_paths_gpu(data, options);
  const auto paths_end = Clock::now();

  const auto metadata_start = Clock::now();
  decompress_graph_metadata_gpu(data, result);
  const auto metadata_end = Clock::now();

  auto end = Clock::now();
  double time_ms = elapsed_ms(start, end);
  double paths_time_ms = elapsed_ms(paths_start, paths_end);
  double metadata_time_ms = elapsed_ms(metadata_start, metadata_end);

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
    std::cout << "  Paths phase:   " << std::fixed << std::setprecision(2)
              << paths_time_ms << " ms" << std::endl;
    std::cout << "  Metadata:      " << std::fixed << std::setprecision(2)
              << metadata_time_ms << " ms" << std::endl;
    std::cout << "  Time:          " << std::fixed << std::setprecision(2)
              << time_ms << " ms" << std::endl;
  }

  return result;
}

} // namespace gpu_decompression
