#pragma once

#include "gpu/decompression/decompression_workflow_gpu.hpp"
#include "gpu/decompression/path_decompression_gpu_rolling.hpp"

#include <cstddef>
#include <cstdint>
#include <thrust/device_vector.h>
#include <vector>

namespace gpu_decompression {

struct GpuTraversalRulebook {
  thrust::device_vector<int32_t> d_rules_first;
  thrust::device_vector<int32_t> d_rules_second;
  uint32_t min_rule_id = 0;
  size_t num_rules = 0;
  double host_decode_ms = 0.0;
};

struct GpuTraversalPayload {
  thrust::device_vector<int32_t> d_encoded;
  thrust::device_vector<uint32_t> d_final_lengths;
  double host_decode_ms = 0.0;
};

GpuTraversalRulebook prepare_gpu_traversal_rulebook(const gfaz::CompressedData &data);

GpuTraversalPayload prepare_gpu_traversal_payload(
    const gfaz::ZstdCompressedBlock &encoded_block,
    const std::vector<uint32_t> &final_lengths);

} // namespace gpu_decompression
