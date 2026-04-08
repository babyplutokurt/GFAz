#include "gpu/decompression_workflow_gpu_internal.hpp"
#include "gpu/codec_gpu.cuh"
#include "gpu/path_decompression_gpu_rolling.hpp"

#include <algorithm>
#include <iostream>

namespace gpu_decompression {

void decompress_paths_gpu_rolling(
    const thrust::device_vector<int32_t> &d_encoded_path,
    const thrust::device_vector<int32_t> &d_rules_first,
    const thrust::device_vector<int32_t> &d_rules_second,
    uint32_t min_rule_id, size_t num_rules,
    const thrust::device_vector<uint32_t> &d_lens_final,
    uint32_t traversals_per_chunk, std::vector<int32_t> &out_data) {
  const uint32_t resolved_traversals_per_chunk =
      std::max<uint32_t>(1, traversals_per_chunk);

  if (decompression_debug_enabled()) {
    std::cout << "[GPU Decompress] Expanding path with rolling chunk "
                 "scheduler ("
              << resolved_traversals_per_chunk
              << " traversals per chunk), min_rule_id=" << min_rule_id
              << std::endl;
  }

  gpu_codec::rolling_expand_and_inverse_delta_decode(
      d_encoded_path, d_rules_first, d_rules_second, min_rule_id, num_rules,
      d_lens_final, out_data, resolved_traversals_per_chunk);
}

} // namespace gpu_decompression
