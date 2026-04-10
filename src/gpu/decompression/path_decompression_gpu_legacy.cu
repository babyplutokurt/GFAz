#include "gpu/decompression/decompression_workflow_gpu_internal.hpp"
#include "gpu/core/codec_gpu.cuh"
#include "gpu/decompression/path_decompression_gpu_legacy.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

namespace gpu_decompression {

namespace {

using Clock = std::chrono::high_resolution_clock;

double elapsed_ms(const Clock::time_point &start,
                  const Clock::time_point &end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

} // namespace

void decompress_paths_gpu_legacy(
    const thrust::device_vector<int32_t> &d_encoded_path,
    const thrust::device_vector<int32_t> &d_rules_first,
    const thrust::device_vector<int32_t> &d_rules_second,
    uint32_t min_rule_id, size_t num_rules,
    const thrust::device_vector<uint32_t> &d_lens_final,
    std::vector<int32_t> &out_data) {
  if (decompression_debug_enabled()) {
    std::cout << "[GPU Decompress] Expanding path with legacy whole-device "
                 "pipeline, min_rule_id="
              << min_rule_id << std::endl;
  }

  const auto prepare_start = Clock::now();
  uint32_t num_segs_final = static_cast<uint32_t>(d_lens_final.size());
  thrust::device_vector<uint64_t> d_offs_final(num_segs_final);
  thrust::exclusive_scan(d_lens_final.begin(), d_lens_final.end(),
                         d_offs_final.begin(), uint64_t(0));
  const auto prepare_end = Clock::now();

  const auto expand_start = Clock::now();
  thrust::device_vector<int32_t> d_expanded = gpu_codec::expand_path_device_vec(
      d_encoded_path, d_rules_first, d_rules_second, min_rule_id, num_rules);
  thrust::device_vector<int32_t> d_decoded =
      gpu_codec::segmented_inverse_delta_decode_device_vec(
          d_expanded, d_offs_final, num_segs_final, d_expanded.size());
  const auto expand_end = Clock::now();

  const auto copy_start = Clock::now();
  out_data.resize(d_decoded.size());
  thrust::copy(d_decoded.begin(), d_decoded.end(), out_data.begin());
  const auto copy_end = Clock::now();

  if (decompression_debug_enabled()) {
    std::cout << "[GPU Decompress][Legacy] prepare offsets: " << std::fixed
              << std::setprecision(2)
              << elapsed_ms(prepare_start, prepare_end) << " ms" << std::endl;
    std::cout << "[GPU Decompress][Legacy] expand + inverse delta: "
              << std::fixed << std::setprecision(2)
              << elapsed_ms(expand_start, expand_end) << " ms" << std::endl;
    std::cout << "[GPU Decompress][Legacy] device->host copy: " << std::fixed
              << std::setprecision(2) << elapsed_ms(copy_start, copy_end)
              << " ms" << std::endl;
  }
}

} // namespace gpu_decompression
