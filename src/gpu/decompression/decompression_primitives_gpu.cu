#include "gpu/decompression/decompression_primitives_gpu.hpp"

#include "codec/codec.hpp"
#include "utils/runtime_utils.hpp"

#include <chrono>
#include <thrust/device_vector.h>

namespace gpu_decompression {

namespace {

using Clock = std::chrono::high_resolution_clock;
using gfaz::runtime_utils::elapsed_ms;

} // namespace

GpuTraversalRulebook prepare_gpu_traversal_rulebook(const gfaz::CompressedData &data) {
  const auto start = Clock::now();

  std::vector<int32_t> first_host =
      gfaz::Codec::zstd_decompress_int32_vector(data.rules_first_zstd);
  std::vector<int32_t> second_host =
      gfaz::Codec::zstd_decompress_int32_vector(data.rules_second_zstd);
  gfaz::Codec::delta_decode_int32(first_host);
  gfaz::Codec::delta_decode_int32(second_host);

  GpuTraversalRulebook rulebook;
  rulebook.d_rules_first = thrust::device_vector<int32_t>(first_host.begin(),
                                                          first_host.end());
  rulebook.d_rules_second = thrust::device_vector<int32_t>(second_host.begin(),
                                                           second_host.end());
  rulebook.min_rule_id = data.min_rule_id();
  rulebook.num_rules = std::min(
      {data.total_rules(), rulebook.d_rules_first.size(),
       rulebook.d_rules_second.size()});
  rulebook.host_decode_ms = elapsed_ms(start, Clock::now());
  return rulebook;
}

GpuTraversalPayload prepare_gpu_traversal_payload(
    const gfaz::ZstdCompressedBlock &encoded_block,
    const std::vector<uint32_t> &final_lengths) {
  const auto start = Clock::now();

  std::vector<int32_t> encoded_host =
      gfaz::Codec::zstd_decompress_int32_vector(encoded_block);

  GpuTraversalPayload payload;
  payload.d_encoded =
      thrust::device_vector<int32_t>(encoded_host.begin(), encoded_host.end());
  payload.d_final_lengths =
      thrust::device_vector<uint32_t>(final_lengths.begin(), final_lengths.end());
  payload.host_decode_ms = elapsed_ms(start, Clock::now());
  return payload;
}

} // namespace gpu_decompression
