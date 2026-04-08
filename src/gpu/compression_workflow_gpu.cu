#include "gfa_parser.hpp"
#include "gpu/codec_gpu.cuh"
#include "gpu/compression_workflow_gpu.hpp"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <unordered_map>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include "gpu/codec_gpu_nvcomp.cuh"

namespace gpu_compression {

// Debug flag for compression stats (can be controlled via environment or
// compile flag)
static bool g_debug_compression = false;
using Clock = std::chrono::high_resolution_clock;

static double elapsed_ms(const Clock::time_point &start,
                         const Clock::time_point &end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

static std::string flattened_to_string(const FlattenedStrings &flat) {
  return std::string(flat.data.begin(), flat.data.end());
}

static void print_compression_stats(const char *label, size_t original_size,
                                    size_t compressed_size) {
  if (!g_debug_compression || original_size == 0)
    return;

  double ratio = 100.0 * (1.0 - static_cast<double>(compressed_size) /
                                    static_cast<double>(original_size));
  std::cout << "  [nvComp] " << label << ": " << original_size << " -> "
            << compressed_size << " bytes (" << std::fixed
            << std::setprecision(1) << ratio << "% reduction)" << std::endl;
}

// =============================================================================
// GPU ZSTD compression wrappers using nvComp
// All functions follow the pattern: compress_<type>_gpu(input, label)
// =============================================================================

static gpu_codec::NvcompCompressedBlock
compress_bytes_gpu(const std::vector<uint8_t> &input,
                   const char *label = "bytes") {
  auto block = gpu_codec::nvcomp_zstd_compress(input);
  print_compression_stats(label, input.size(), block.payload.size());
  return block;
}

static gpu_codec::NvcompCompressedBlock
compress_string_gpu(const std::string &input, const char *label = "string") {
  auto block = gpu_codec::nvcomp_zstd_compress_string(input);
  print_compression_stats(label, input.size(), block.payload.size());
  return block;
}

static gpu_codec::NvcompCompressedBlock
compress_uint32_gpu(const std::vector<uint32_t> &input,
                    const char *label = "uint32_vec") {
  size_t original_bytes = input.size() * sizeof(uint32_t);
  auto block = gpu_codec::nvcomp_zstd_compress_uint32(input);
  print_compression_stats(label, original_bytes, block.payload.size());
  return block;
}

static gpu_codec::NvcompCompressedBlock
compress_int32_gpu(const std::vector<int32_t> &input,
                   const char *label = "int32_vec") {
  size_t original_bytes = input.size() * sizeof(int32_t);
  auto block = gpu_codec::nvcomp_zstd_compress_int32(input);
  print_compression_stats(label, original_bytes, block.payload.size());
  return block;
}

static gpu_codec::NvcompCompressedBlock
compress_float_gpu(const std::vector<float> &input,
                   const char *label = "float_vec") {
  const uint8_t *bytes = reinterpret_cast<const uint8_t *>(input.data());
  std::vector<uint8_t> payload(bytes, bytes + input.size() * sizeof(float));
  return compress_bytes_gpu(payload, label);
}

static gpu_codec::NvcompCompressedBlock
compress_char_gpu(const std::vector<char> &input,
                  const char *label = "char_vec") {
  const uint8_t *bytes = reinterpret_cast<const uint8_t *>(input.data());
  std::vector<uint8_t> payload(bytes, bytes + input.size() * sizeof(char));
  return compress_bytes_gpu(payload, label);
}

void set_gpu_compression_debug(bool enabled) {
  g_debug_compression = enabled;
}

// Device-resident compression (avoids D->H->D round-trip)
static gpu_codec::NvcompCompressedBlock
compress_int32_device_gpu(const thrust::device_vector<int32_t> &d_input,
                          const char *label = "int32_device") {
  auto block = gpu_codec::nvcomp_zstd_compress_int32_device(
      thrust::raw_pointer_cast(d_input.data()), d_input.size());
  size_t original_bytes = d_input.size() * sizeof(int32_t);
  print_compression_stats(label, original_bytes, block.payload.size());
  return block;
}

static uint64_t zigzag_encode_64(int64_t value) {
  return (static_cast<uint64_t>(value) << 1) ^
         static_cast<uint64_t>(value >> 63);
}

static void append_varint_64(uint64_t value, std::vector<uint8_t> &out) {
  while (value >= 0x80) {
    out.push_back(static_cast<uint8_t>((value & 0x7F) | 0x80));
    value >>= 7;
  }
  out.push_back(static_cast<uint8_t>(value));
}

static gpu_codec::NvcompCompressedBlock
compress_varint_int64_gpu(const std::vector<int64_t> &input,
                          const char *label = "varint_int64") {
  std::vector<uint8_t> varint_bytes;
  varint_bytes.reserve(input.size() * 4);
  for (int64_t val : input) {
    append_varint_64(zigzag_encode_64(val), varint_bytes);
  }
  return compress_bytes_gpu(varint_bytes, label);
}

static void append_varint_32(uint32_t value, std::vector<uint8_t> &out) {
  while (value >= 0x80) {
    out.push_back(static_cast<uint8_t>((value & 0x7F) | 0x80));
    value >>= 7;
  }
  out.push_back(static_cast<uint8_t>(value));
}

static gpu_codec::NvcompCompressedBlock
compress_delta_varint_uint32_gpu(const std::vector<uint32_t> &input,
                                 const char *label = "delta_varint_uint32") {
  if (input.empty()) {
    return gpu_codec::NvcompCompressedBlock{};
  }

  std::vector<int32_t> deltas(input.size());
  deltas[0] = static_cast<int32_t>(input[0]);
  for (size_t i = 1; i < input.size(); ++i) {
    deltas[i] =
        static_cast<int32_t>(input[i]) - static_cast<int32_t>(input[i - 1]);
  }

  std::vector<uint8_t> varint_bytes;
  varint_bytes.reserve(input.size() * 2);
  for (int32_t val : deltas) {
    // Zigzag encode: cast to unsigned first to avoid UB on negative left-shift
    uint32_t zigzag =
        (static_cast<uint32_t>(val) << 1) ^ static_cast<uint32_t>(val >> 31);
    append_varint_32(zigzag, varint_bytes);
  }

  return compress_bytes_gpu(varint_bytes, label);
}

static std::vector<int32_t>
inverse_delta_decode_host(const std::vector<int32_t> &delta_encoded) {
  if (delta_encoded.empty()) {
    return {};
  }
  std::vector<int32_t> decoded(delta_encoded.size());
  decoded[0] = delta_encoded[0];
  for (size_t i = 1; i < delta_encoded.size(); ++i) {
    decoded[i] = decoded[i - 1] + delta_encoded[i];
  }
  return decoded;
}

static gpu_codec::NvcompCompressedBlock
compress_orientations_gpu(const std::vector<char> &orients,
                          const char *label = "orientations") {

  // Use GPU kernel for bit-packing
  std::vector<uint8_t> packed = gpu_codec::pack_orientations_gpu(orients);

  return compress_bytes_gpu(packed, label);
}

// Helper: compress a FlattenedStrings into data + lengths blocks
static void compress_flattened_strings_gpu(
    const FlattenedStrings &flat, gpu_codec::NvcompCompressedBlock &data_block,
    gpu_codec::NvcompCompressedBlock &lengths_block, const char *data_label,
    const char *lengths_label) {
  data_block = compress_string_gpu(flattened_to_string(flat), data_label);
  lengths_block = compress_uint32_gpu(flat.lengths, lengths_label);
}

// Helper: compress an optional field column
static CompressedOptionalFieldColumn_gpu
compress_optional_column_gpu(const OptionalFieldColumn_gpu &col,
                             const char *prefix) {
  CompressedOptionalFieldColumn_gpu compressed_col;
  compressed_col.tag = col.tag;
  compressed_col.type = col.type;

  // Build label strings (using static buffers for simplicity)
  char label[64];

  switch (col.type) {
  case 'i':
    compressed_col.num_elements = col.int_values.size();
    snprintf(label, sizeof(label), "%s_optional_i", prefix);
    compressed_col.int_values_zstd_nvcomp =
        compress_varint_int64_gpu(col.int_values, label);
    break;
  case 'f':
    compressed_col.num_elements = col.float_values.size();
    snprintf(label, sizeof(label), "%s_optional_f", prefix);
    compressed_col.float_values_zstd_nvcomp =
        compress_float_gpu(col.float_values, label);
    break;
  case 'A':
    compressed_col.num_elements = col.char_values.size();
    snprintf(label, sizeof(label), "%s_optional_A", prefix);
    compressed_col.char_values_zstd_nvcomp =
        compress_char_gpu(col.char_values, label);
    break;
  case 'Z':
  case 'J':
  case 'H':
    compressed_col.num_elements = col.strings.lengths.size();
    snprintf(label, sizeof(label), "%s_optional_strings", prefix);
    compressed_col.strings_zstd_nvcomp = compress_string_gpu(
        std::string(col.strings.data.begin(), col.strings.data.end()), label);
    snprintf(label, sizeof(label), "%s_optional_string_lengths", prefix);
    compressed_col.string_lengths_zstd_nvcomp =
        compress_uint32_gpu(col.strings.lengths, label);
    break;
  case 'B':
    compressed_col.num_elements = col.b_subtypes.size();
    snprintf(label, sizeof(label), "%s_optional_b_subtypes", prefix);
    compressed_col.b_subtypes_zstd_nvcomp =
        compress_char_gpu(col.b_subtypes, label);
    snprintf(label, sizeof(label), "%s_optional_b_lengths", prefix);
    compressed_col.b_lengths_zstd_nvcomp =
        compress_uint32_gpu(col.b_lengths, label);
    snprintf(label, sizeof(label), "%s_optional_b_bytes", prefix);
    compressed_col.b_concat_bytes_zstd_nvcomp =
        compress_bytes_gpu(col.b_data, label);
    break;
  }
  return compressed_col;
}

static void compress_optional_columns_gpu(
    const std::vector<OptionalFieldColumn_gpu> &columns,
    std::vector<CompressedOptionalFieldColumn_gpu> &out_columns,
    const char *prefix) {
  out_columns.reserve(out_columns.size() + columns.size());
  for (const auto &col : columns) {
    out_columns.push_back(compress_optional_column_gpu(col, prefix));
  }
}

struct TraversalChunk {
  size_t segment_begin = 0;
  size_t segment_end = 0;
  size_t node_begin = 0;
  size_t node_end = 0;

  size_t num_segments() const { return segment_end - segment_begin; }
  size_t num_nodes() const { return node_end - node_begin; }
};

static size_t rolling_chunk_bytes() {
  constexpr size_t kDefaultChunkBytes = 1ull << 30; // 1 GiB
  const char *env_mb = std::getenv("GFAZ_GPU_ROLLING_CHUNK_MB");
  if (!env_mb || *env_mb == '\0') {
    return kDefaultChunkBytes;
  }

  char *end = nullptr;
  unsigned long long value_mb = std::strtoull(env_mb, &end, 10);
  if (end == env_mb || value_mb == 0) {
    return kDefaultChunkBytes;
  }
  return static_cast<size_t>(value_mb) * 1024ull * 1024ull;
}

static std::vector<TraversalChunk>
build_traversal_chunks(const std::vector<uint32_t> &lengths,
                       size_t target_chunk_nodes) {
  std::vector<TraversalChunk> chunks;
  chunks.reserve(std::max<size_t>(1, lengths.size() / 1024));

  size_t seg_begin = 0;
  size_t node_begin = 0;
  while (seg_begin < lengths.size()) {
    size_t seg_end = seg_begin;
    size_t node_end = node_begin;

    while (seg_end < lengths.size()) {
      size_t next_len = lengths[seg_end];
      size_t proposed = node_end - node_begin + next_len;
      if (seg_end > seg_begin && proposed > target_chunk_nodes) {
        break;
      }
      node_end += next_len;
      ++seg_end;
      if (node_end - node_begin >= target_chunk_nodes) {
        break;
      }
    }

    if (seg_end == seg_begin) {
      node_end += lengths[seg_end];
      ++seg_end;
    }

    chunks.push_back({seg_begin, seg_end, node_begin, node_end});
    seg_begin = seg_end;
    node_begin = node_end;
  }

  return chunks;
}

static uint32_t find_max_abs_host(const std::vector<int32_t> &data) {
  uint32_t max_abs = 0;
  for (int32_t value : data) {
    uint32_t abs_value =
        (value >= 0) ? static_cast<uint32_t>(value)
                     : (0u - static_cast<uint32_t>(value));
    if (abs_value > max_abs) {
      max_abs = abs_value;
    }
  }
  return max_abs;
}

static void compact_rules_and_remap_host(std::vector<int32_t> &data,
                                         const std::vector<uint8_t> &rules_used,
                                         std::vector<uint64_t> &rules,
                                         uint32_t start_id) {
  if (rules_used.empty()) {
    return;
  }

  std::vector<uint32_t> new_indices(rules_used.size(), 0);
  uint32_t next_index = 0;
  for (size_t i = 0; i < rules_used.size(); ++i) {
    new_indices[i] = next_index;
    if (rules_used[i]) {
      ++next_index;
    }
  }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < data.size(); ++idx) {
    int32_t value = data[idx];
    int32_t abs_value = (value >= 0) ? value : -value;
    if (static_cast<uint32_t>(abs_value) < start_id) {
      continue;
    }

    uint32_t offset = static_cast<uint32_t>(abs_value) - start_id;
    if (offset >= rules_used.size()) {
      continue;
    }

    uint32_t new_id = start_id + new_indices[offset];
    data[idx] = (value >= 0) ? static_cast<int32_t>(new_id)
                             : -static_cast<int32_t>(new_id);
  }

  std::vector<uint64_t> compacted_rules;
  compacted_rules.reserve(next_index);
  for (size_t i = 0; i < rules.size(); ++i) {
    if (rules_used[i]) {
      compacted_rules.push_back(rules[i]);
    }
  }
  rules = std::move(compacted_rules);
}

static void sort_rules_and_remap_host(std::vector<int32_t> &data,
                                      std::vector<uint64_t> &rules,
                                      uint32_t start_id) {
  if (rules.empty()) {
    return;
  }

  std::vector<uint32_t> indices(rules.size());
  std::iota(indices.begin(), indices.end(), 0u);
  std::sort(indices.begin(), indices.end(),
            [&rules](uint32_t lhs, uint32_t rhs) {
              return rules[lhs] < rules[rhs];
            });

  std::vector<uint64_t> sorted_rules(rules.size());
  std::vector<uint32_t> reorder_map(rules.size());
  for (uint32_t new_idx = 0; new_idx < indices.size(); ++new_idx) {
    uint32_t old_idx = indices[new_idx];
    sorted_rules[new_idx] = rules[old_idx];
    reorder_map[old_idx] = new_idx;
  }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < data.size(); ++idx) {
    int32_t value = data[idx];
    int32_t abs_value = (value >= 0) ? value : -value;
    if (static_cast<uint32_t>(abs_value) < start_id) {
      continue;
    }

    uint32_t offset = static_cast<uint32_t>(abs_value) - start_id;
    if (offset >= reorder_map.size()) {
      continue;
    }

    uint32_t new_id = start_id + reorder_map[offset];
    data[idx] = (value >= 0) ? static_cast<int32_t>(new_id)
                             : -static_cast<int32_t>(new_id);
  }

  rules = std::move(sorted_rules);
}

static void delta_encode_chunks_on_gpu(std::vector<int32_t> &data,
                                       const std::vector<uint32_t> &lengths,
                                       const std::vector<TraversalChunk> &chunks) {
  size_t max_nodes = 0;
  size_t max_segs = 0;
  for (const auto &chunk : chunks) {
    max_nodes = std::max(max_nodes, chunk.num_nodes());
    max_segs = std::max(max_segs, chunk.num_segments());
  }

  thrust::device_vector<int32_t> d_chunk_data;
  d_chunk_data.reserve(max_nodes);
  thrust::device_vector<uint32_t> d_chunk_lengths;
  d_chunk_lengths.reserve(max_segs);
  thrust::device_vector<uint64_t> d_chunk_offsets;
  d_chunk_offsets.reserve(max_segs);
  thrust::device_vector<uint8_t> d_is_first;
  d_is_first.reserve(max_nodes);
  thrust::device_vector<uint8_t> d_is_last;
  d_is_last.reserve(max_nodes);

  for (const auto &chunk : chunks) {
    if (chunk.num_nodes() == 0) {
      continue;
    }

    d_chunk_data.resize(chunk.num_nodes());
    thrust::copy(data.begin() + static_cast<std::ptrdiff_t>(chunk.node_begin),
                 data.begin() + static_cast<std::ptrdiff_t>(chunk.node_end),
                 d_chunk_data.begin());

    d_chunk_lengths.resize(chunk.num_segments());
    thrust::copy(lengths.begin() + static_cast<std::ptrdiff_t>(chunk.segment_begin),
                 lengths.begin() + static_cast<std::ptrdiff_t>(chunk.segment_end),
                 d_chunk_lengths.begin());

    d_chunk_offsets.resize(d_chunk_lengths.size());
    thrust::exclusive_scan(d_chunk_lengths.begin(), d_chunk_lengths.end(),
                           d_chunk_offsets.begin(), uint64_t(0));

    d_is_first.resize(d_chunk_data.size());
    d_is_last.resize(d_chunk_data.size());
    gpu_codec::compute_boundary_masks(d_chunk_offsets,
                                      static_cast<uint32_t>(chunk.num_segments()),
                                      d_chunk_data.size(), d_is_first,
                                      d_is_last);
    gpu_codec::segmented_delta_encode_device_vec(d_chunk_data, d_is_first);

    thrust::copy(d_chunk_data.begin(), d_chunk_data.end(),
                 data.begin() + static_cast<std::ptrdiff_t>(chunk.node_begin));
  }
}

static CompressedData_gpu
run_path_compression_gpu_full_device(const FlattenedPaths &paths,
                                     int num_rounds) {
  CompressedData_gpu result;

  if (paths.data.empty()) {
    // Even with empty data, store path_lengths so decompression can
    // reconstruct the correct number of (zero-length) paths/walks.
    if (!paths.lengths.empty()) {
      result.path_lengths_zstd_nvcomp =
          gpu_codec::nvcomp_zstd_compress_uint32(paths.lengths);
    }
    return result;
  }

  // 1. Copy paths data AND lengths to device
  thrust::device_vector<int32_t> d_data(paths.data.begin(), paths.data.end());
  thrust::device_vector<uint32_t> d_lengths(paths.lengths.begin(),
                                             paths.lengths.end());
  uint32_t num_segments = static_cast<uint32_t>(paths.lengths.size());

  // 2. Compute offsets (exclusive prefix sum of lengths — uint64_t to avoid overflow)
  thrust::device_vector<uint64_t> d_offsets(num_segments);
  thrust::exclusive_scan(d_lengths.begin(), d_lengths.end(), d_offsets.begin(),
                         uint64_t(0));

  // 3. Compute start_id on GPU (max abs value + 1)
  uint32_t start_id = gpu_codec::find_max_abs_device(d_data) + 1;

  // 4. Segmented delta encode (per-traversal, first element unchanged)
  {
    thrust::device_vector<uint8_t> d_is_first, d_is_last;
    gpu_codec::compute_boundary_masks(d_offsets, num_segments,
                                      d_data.size(), d_is_first, d_is_last);
    gpu_codec::segmented_delta_encode_device_vec(d_data, d_is_first);
  }

  // 5. Recompute start_id after delta encoding (delta values may be larger)
  uint32_t delta_max = gpu_codec::find_max_abs_device(d_data);
  if (delta_max >= start_id) {
    start_id = delta_max + 1;
  }

  // 6. Accumulator for all rules (device vector)
  thrust::device_vector<uint64_t> d_all_rules;
  uint32_t next_start_id = start_id;

  // 7. Compression rounds
  for (int round_idx = 0; round_idx < num_rounds; ++round_idx) {
    // Compute boundary mask for current data layout
    thrust::device_vector<uint8_t> d_is_first, d_is_last;
    gpu_codec::compute_boundary_masks(d_offsets, num_segments,
                                      d_data.size(), d_is_first, d_is_last);

    // Find repeated 2-mers (boundary-aware)
    thrust::device_vector<uint64_t> d_round_rules =
        gpu_codec::find_repeated_2mers_segmented_device_vec(d_data, d_is_last);

    if (d_round_rules.empty()) {
      break; // No more rules found
    }

    uint32_t num_rules_found = d_round_rules.size();

    // Create hash table from device memory
    void *table_ptr = gpu_codec::create_rule_table_gpu_from_device(
        d_round_rules, next_start_id);

    // Apply rules (boundary-aware) — returns new per-segment lengths
    thrust::device_vector<uint8_t> d_rules_used(num_rules_found, 0);
    d_lengths = gpu_codec::apply_2mer_rules_segmented_device_vec(
        d_data, table_ptr, d_rules_used, next_start_id,
        d_offsets, num_segments);

    // Cleanup table
    gpu_codec::free_rule_table_gpu(table_ptr);

    // Compact rules (operates on values only — boundary-agnostic)
    gpu_codec::compact_rules_and_remap_device_vec(d_data, d_rules_used,
                                                  d_round_rules, next_start_id);

    // Sort rules (operates on values only — boundary-agnostic)
    gpu_codec::sort_rules_and_remap_device_vec(d_data, d_round_rules,
                                               next_start_id);

    // Record layer range
    uint32_t num_used_rules = d_round_rules.size();
    result.layer_ranges.push_back({next_start_id, num_used_rules});

    // Append rules to accumulator (device-to-device copy)
    size_t old_size = d_all_rules.size();
    d_all_rules.resize(old_size + num_used_rules);
    thrust::copy(d_round_rules.begin(), d_round_rules.end(),
                 d_all_rules.begin() + old_size);

    // Update next start ID
    next_start_id += num_used_rules;

    // Recompute offsets from new lengths for next round
    thrust::exclusive_scan(d_lengths.begin(), d_lengths.end(),
                           d_offsets.begin(), uint64_t(0));
  }

  // 8. Compress encoded path directly from device (no D->H copy needed)
  result.encoded_path_zstd_nvcomp =
      compress_int32_device_gpu(d_data, "encoded_path");

  // 9. Compress ORIGINAL path_lengths (decompression needs these to split
  // the expanded data back into per-traversal segments)
  result.path_lengths_zstd_nvcomp =
      compress_uint32_gpu(paths.lengths, "path_lengths");

  // 10. Split rules into first/second, delta-encode on GPU, then compress
  // directly from device
  if (!d_all_rules.empty()) {
    thrust::device_vector<int32_t> d_first, d_second;
    gpu_codec::split_and_delta_encode_rules_device_vec(d_all_rules, d_first,
                                                       d_second);

    // Compress directly from device (no D->H copy needed)
    result.rules_first_zstd_nvcomp =
        compress_int32_device_gpu(d_first, "rules_first");
    result.rules_second_zstd_nvcomp =
        compress_int32_device_gpu(d_second, "rules_second");
  }

  return result;
}

static CompressedData_gpu
run_path_compression_gpu_rolling(const FlattenedPaths &paths, int num_rounds,
                                 size_t chunk_bytes) {
  CompressedData_gpu result;

  if (paths.data.empty()) {
    if (!paths.lengths.empty()) {
      result.path_lengths_zstd_nvcomp =
          gpu_codec::nvcomp_zstd_compress_uint32(paths.lengths);
    }
    return result;
  }

  std::vector<int32_t> current_data = paths.data;
  std::vector<uint32_t> current_lengths = paths.lengths;
  const uint32_t num_segments = static_cast<uint32_t>(current_lengths.size());
  const size_t target_chunk_nodes =
      std::max<size_t>(1, chunk_bytes / sizeof(int32_t));

  auto chunks = build_traversal_chunks(current_lengths, target_chunk_nodes);
  delta_encode_chunks_on_gpu(current_data, current_lengths, chunks);

  uint32_t start_id = find_max_abs_host(paths.data) + 1;
  uint32_t delta_max = find_max_abs_host(current_data);
  if (delta_max >= start_id) {
    start_id = delta_max + 1;
  }

  size_t max_nodes = 0;
  size_t max_segs = 0;
  for (const auto &chunk : chunks) {
    max_nodes = std::max(max_nodes, chunk.num_nodes());
    max_segs = std::max(max_segs, chunk.num_segments());
  }

  thrust::device_vector<int32_t> d_chunk_data;
  d_chunk_data.reserve(max_nodes);
  thrust::device_vector<uint32_t> d_chunk_lengths;
  d_chunk_lengths.reserve(max_segs);
  thrust::device_vector<uint64_t> d_chunk_offsets;
  d_chunk_offsets.reserve(max_segs);
  thrust::device_vector<uint8_t> d_is_first;
  d_is_first.reserve(max_nodes);
  thrust::device_vector<uint8_t> d_is_last;
  d_is_last.reserve(max_nodes);
  thrust::device_vector<uint64_t> d_unique_keys;
  d_unique_keys.reserve(max_nodes);
  thrust::device_vector<uint32_t> d_counts;
  d_counts.reserve(max_nodes);
  thrust::device_vector<uint8_t> d_rules_used_dev;

  std::vector<uint64_t> all_rules;
  uint32_t next_start_id = start_id;

  for (int round_idx = 0; round_idx < num_rounds; ++round_idx) {
    chunks = build_traversal_chunks(current_lengths, target_chunk_nodes);

    std::unordered_map<uint64_t, uint64_t> global_counts;
    for (const auto &chunk : chunks) {
      if (chunk.num_nodes() < 2) {
        continue;
      }

      d_chunk_data.resize(chunk.num_nodes());
      thrust::copy(current_data.begin() + static_cast<std::ptrdiff_t>(chunk.node_begin),
                   current_data.begin() + static_cast<std::ptrdiff_t>(chunk.node_end),
                   d_chunk_data.begin());

      d_chunk_lengths.resize(chunk.num_segments());
      thrust::copy(current_lengths.begin() + static_cast<std::ptrdiff_t>(chunk.segment_begin),
                   current_lengths.begin() + static_cast<std::ptrdiff_t>(chunk.segment_end),
                   d_chunk_lengths.begin());

      d_chunk_offsets.resize(d_chunk_lengths.size());
      thrust::exclusive_scan(d_chunk_lengths.begin(), d_chunk_lengths.end(),
                             d_chunk_offsets.begin(), uint64_t(0));

      d_is_first.resize(d_chunk_data.size());
      d_is_last.resize(d_chunk_data.size());
      gpu_codec::compute_boundary_masks(
          d_chunk_offsets, static_cast<uint32_t>(chunk.num_segments()),
          d_chunk_data.size(), d_is_first, d_is_last);

      gpu_codec::count_2mers_segmented_device_vec(d_chunk_data, d_is_last,
                                                  d_unique_keys, d_counts);

      if (d_unique_keys.empty()) {
        continue;
      }

      std::vector<uint64_t> h_keys(d_unique_keys.size());
      std::vector<uint32_t> h_counts(d_counts.size());
      thrust::copy(d_unique_keys.begin(), d_unique_keys.end(), h_keys.begin());
      thrust::copy(d_counts.begin(), d_counts.end(), h_counts.begin());

      for (size_t i = 0; i < h_keys.size(); ++i) {
        global_counts[h_keys[i]] += h_counts[i];
      }
    }

    std::vector<uint64_t> round_rules;
    round_rules.reserve(global_counts.size());
    for (const auto &entry : global_counts) {
      if (entry.second >= 2) {
        round_rules.push_back(entry.first);
      }
    }

    if (round_rules.empty()) {
      break;
    }

    std::sort(round_rules.begin(), round_rules.end());
    void *table_ptr = gpu_codec::create_rule_table_gpu(round_rules, next_start_id);
    std::vector<uint8_t> rules_used(round_rules.size(), 0);

    std::vector<int32_t> next_data;
    next_data.reserve(current_data.size());
    std::vector<uint32_t> next_lengths;
    next_lengths.reserve(current_lengths.size());

    for (const auto &chunk : chunks) {
      d_chunk_data.resize(chunk.num_nodes());
      thrust::copy(current_data.begin() + static_cast<std::ptrdiff_t>(chunk.node_begin),
                   current_data.begin() + static_cast<std::ptrdiff_t>(chunk.node_end),
                   d_chunk_data.begin());

      d_chunk_lengths.resize(chunk.num_segments());
      thrust::copy(current_lengths.begin() + static_cast<std::ptrdiff_t>(chunk.segment_begin),
                   current_lengths.begin() + static_cast<std::ptrdiff_t>(chunk.segment_end),
                   d_chunk_lengths.begin());

      d_chunk_offsets.resize(d_chunk_lengths.size());
      thrust::exclusive_scan(d_chunk_lengths.begin(), d_chunk_lengths.end(),
                             d_chunk_offsets.begin(), uint64_t(0));

      d_rules_used_dev.resize(round_rules.size());
      thrust::fill(d_rules_used_dev.begin(), d_rules_used_dev.end(), 0);
      thrust::device_vector<uint32_t> d_new_lengths =
          gpu_codec::apply_2mer_rules_segmented_device_vec(
              d_chunk_data, table_ptr, d_rules_used_dev, next_start_id,
              d_chunk_offsets, static_cast<uint32_t>(chunk.num_segments()));

      size_t old_size = next_data.size();
      next_data.resize(old_size + d_chunk_data.size());
      thrust::copy(d_chunk_data.begin(), d_chunk_data.end(),
                   next_data.begin() + static_cast<std::ptrdiff_t>(old_size));

      size_t old_lengths_size = next_lengths.size();
      next_lengths.resize(old_lengths_size + d_new_lengths.size());
      thrust::copy(d_new_lengths.begin(), d_new_lengths.end(),
                   next_lengths.begin() +
                       static_cast<std::ptrdiff_t>(old_lengths_size));

      std::vector<uint8_t> h_chunk_used(d_rules_used_dev.size());
      thrust::copy(d_rules_used_dev.begin(), d_rules_used_dev.end(),
                   h_chunk_used.begin());
      for (size_t i = 0; i < h_chunk_used.size(); ++i) {
        rules_used[i] = static_cast<uint8_t>(rules_used[i] | h_chunk_used[i]);
      }
    }

    gpu_codec::free_rule_table_gpu(table_ptr);

    current_data = std::move(next_data);
    current_lengths = std::move(next_lengths);

    compact_rules_and_remap_host(current_data, rules_used, round_rules,
                                 next_start_id);
    if (round_rules.empty()) {
      break;
    }

    sort_rules_and_remap_host(current_data, round_rules, next_start_id);

    result.layer_ranges.push_back(
        {next_start_id, static_cast<uint32_t>(round_rules.size())});
    all_rules.insert(all_rules.end(), round_rules.begin(), round_rules.end());
    next_start_id += static_cast<uint32_t>(round_rules.size());
  }

  result.encoded_path_zstd_nvcomp =
      compress_int32_gpu(current_data, "encoded_path");
  result.path_lengths_zstd_nvcomp =
      compress_uint32_gpu(paths.lengths, "path_lengths");

  if (!all_rules.empty()) {
    thrust::device_vector<uint64_t> d_all_rules(all_rules.begin(),
                                                all_rules.end());
    thrust::device_vector<int32_t> d_first, d_second;
    gpu_codec::split_and_delta_encode_rules_device_vec(d_all_rules, d_first,
                                                       d_second);
    result.rules_first_zstd_nvcomp =
        compress_int32_device_gpu(d_first, "rules_first");
    result.rules_second_zstd_nvcomp =
        compress_int32_device_gpu(d_second, "rules_second");
  }

  if (g_debug_compression) {
    std::cout << "[GPU Compression] Rolling scheduler used for "
              << paths.total_nodes() * sizeof(int32_t) / (1024.0 * 1024.0)
              << " MB traversal payload across " << num_segments
              << " traversals (chunk budget "
              << chunk_bytes / (1024.0 * 1024.0) << " MB)" << std::endl;
  }

  return result;
}

CompressedData_gpu run_path_compression_gpu(const FlattenedPaths &paths,
                                            int num_rounds) {
  const size_t traversal_bytes = paths.data.size() * sizeof(int32_t);
  const size_t chunk_bytes = rolling_chunk_bytes();

  if (traversal_bytes <= chunk_bytes) {
    return run_path_compression_gpu_full_device(paths, num_rounds);
  }
  return run_path_compression_gpu_rolling(paths, num_rounds, chunk_bytes);
}


std::map<uint32_t, uint64_t> build_rulebook(const CompressedData_gpu &data) {
  std::map<uint32_t, uint64_t> rulebook;

  // Decompress rules_first and rules_second
  std::vector<int32_t> first_delta =
      gpu_codec::nvcomp_zstd_decompress_int32(data.rules_first_zstd_nvcomp);
  std::vector<int32_t> second_delta =
      gpu_codec::nvcomp_zstd_decompress_int32(data.rules_second_zstd_nvcomp);

  if (first_delta.empty() || second_delta.empty()) {
    return rulebook;
  }

  // Inverse delta-encode to get original rule endpoints.
  std::vector<int32_t> first = inverse_delta_decode_host(first_delta);
  std::vector<int32_t> second = inverse_delta_decode_host(second_delta);

  // Build rulebook from unpacked rules
  size_t offset = 0;
  for (const auto &range : data.layer_ranges) {
    for (uint32_t i = 0; i < range.count; ++i) {
      uint32_t rule_id = range.start_id + i;
      // Pack first and second back into uint64_t
      uint64_t packed =
          (static_cast<uint64_t>(static_cast<uint32_t>(first[offset + i]))
           << 32) |
          static_cast<uint64_t>(static_cast<uint32_t>(second[offset + i]));
      rulebook[rule_id] = packed;
    }
    offset += range.count;
  }

  return rulebook;
}

CompressedData_gpu compress_gfa_gpu(const std::string &gfa_file_path,
                                    int num_rounds) {
  GfaParser parser;
  GfaGraph graph = parser.parse(gfa_file_path);

  GfaGraph_gpu gpu_graph = convert_to_gpu_layout(graph);

  return compress_gpu_graph(gpu_graph, num_rounds);
}

CompressedData_gpu compress_gpu_graph(const GfaGraph_gpu &gpu_graph,
                                      int num_rounds) {
  // Start timer for compression (GfaGraph_gpu -> CompressedData_gpu)
  auto compress_start = Clock::now();

  CompressedData_gpu data =
      run_path_compression_gpu(gpu_graph.paths, num_rounds);

  // Store path/walk split info
  data.num_paths = gpu_graph.num_paths;
  data.num_walks = gpu_graph.num_walks;

  // ====== PATH METADATA (P-lines only) ======
  if (g_debug_compression) {
    std::cout
        << "[GPU Compression] Using nvComp GPU ZSTD for metadata compression"
        << std::endl;
  }

  compress_flattened_strings_gpu(gpu_graph.path_names, data.names_zstd_nvcomp,
                                 data.name_lengths_zstd_nvcomp, "path_names",
                                 "name_lengths");
  compress_flattened_strings_gpu(
      gpu_graph.path_overlaps, data.overlaps_zstd_nvcomp,
      data.overlap_lengths_zstd_nvcomp, "path_overlaps", "overlap_lengths");

  // ====== WALK METADATA (W-lines only) ======
  if (gpu_graph.num_walks > 0) {
    if (g_debug_compression) {
      std::cout << "[GPU Compression] Compressing walk metadata ("
                << gpu_graph.num_walks << " walks)" << std::endl;
    }

    compress_flattened_strings_gpu(gpu_graph.walk_sample_ids,
                                   data.walk_sample_ids_zstd_nvcomp,
                                   data.walk_sample_id_lengths_zstd_nvcomp,
                                   "walk_sample_ids", "walk_sample_id_lengths");

    data.walk_hap_indices_zstd_nvcomp =
        compress_uint32_gpu(gpu_graph.walk_hap_indices, "walk_hap_indices");

    compress_flattened_strings_gpu(gpu_graph.walk_seq_ids,
                                   data.walk_seq_ids_zstd_nvcomp,
                                   data.walk_seq_id_lengths_zstd_nvcomp,
                                   "walk_seq_ids", "walk_seq_id_lengths");

    data.walk_seq_starts_zstd_nvcomp =
        compress_varint_int64_gpu(gpu_graph.walk_seq_starts, "walk_seq_starts");
    data.walk_seq_ends_zstd_nvcomp =
        compress_varint_int64_gpu(gpu_graph.walk_seq_ends, "walk_seq_ends");
  }

  // ====== SEGMENT DATA ======
  compress_flattened_strings_gpu(gpu_graph.node_sequences,
                                 data.segment_sequences_zstd_nvcomp,
                                 data.segment_seq_lengths_zstd_nvcomp,
                                 "segment_sequences", "segment_seq_lengths");

  // ====== HEADER + NODE NAMES (for full-fidelity round-trip) ======
  data.header_line = gpu_graph.header_line;

  compress_flattened_strings_gpu(
      gpu_graph.node_names, data.node_names_zstd_nvcomp,
      data.node_name_lengths_zstd_nvcomp, "node_names", "node_name_lengths");

  // ====== SEGMENT OPTIONAL FIELDS ======
  compress_optional_columns_gpu(gpu_graph.segment_optional_fields,
                                data.segment_optional_fields_zstd_nvcomp,
                                "segment");

  // ====== LINK DATA ======
  data.num_links = gpu_graph.link_from_ids.size();
  data.link_from_ids_zstd_nvcomp = compress_delta_varint_uint32_gpu(
      gpu_graph.link_from_ids, "link_from_ids");
  data.link_to_ids_zstd_nvcomp =
      compress_delta_varint_uint32_gpu(gpu_graph.link_to_ids, "link_to_ids");
  data.link_from_orients_zstd_nvcomp = compress_orientations_gpu(
      gpu_graph.link_from_orients, "link_from_orients");
  data.link_to_orients_zstd_nvcomp =
      compress_orientations_gpu(gpu_graph.link_to_orients, "link_to_orients");
  data.link_overlap_nums_zstd_nvcomp =
      compress_uint32_gpu(gpu_graph.link_overlap_nums, "link_overlap_nums");
  data.link_overlap_ops_zstd_nvcomp =
      compress_char_gpu(gpu_graph.link_overlap_ops, "link_overlap_ops");

  // ====== LINK OPTIONAL FIELDS ======
  compress_optional_columns_gpu(gpu_graph.link_optional_fields,
                                data.link_optional_fields_zstd_nvcomp, "link");

  // ====== JUMP DATA (J-lines) - structured columnar ======
  if (!gpu_graph.jump_from_ids.empty()) {
    data.num_jumps_stored = gpu_graph.jump_from_ids.size();

    if (g_debug_compression) {
      std::cout << "[GPU Compression] Compressing J-lines ("
                << data.num_jumps_stored << " jumps)" << std::endl;
    }

    data.jump_from_ids_zstd_nvcomp = compress_delta_varint_uint32_gpu(
        gpu_graph.jump_from_ids, "jump_from_ids");
    data.jump_to_ids_zstd_nvcomp =
        compress_delta_varint_uint32_gpu(gpu_graph.jump_to_ids, "jump_to_ids");
    data.jump_from_orients_zstd_nvcomp = compress_orientations_gpu(
        gpu_graph.jump_from_orients, "jump_from_orients");
    data.jump_to_orients_zstd_nvcomp =
        compress_orientations_gpu(gpu_graph.jump_to_orients, "jump_to_orients");

    compress_flattened_strings_gpu(gpu_graph.jump_distances,
                                   data.jump_distances_zstd_nvcomp,
                                   data.jump_distance_lengths_zstd_nvcomp,
                                   "jump_distances", "jump_distance_lengths");
    compress_flattened_strings_gpu(gpu_graph.jump_rest_fields,
                                   data.jump_rest_fields_zstd_nvcomp,
                                   data.jump_rest_lengths_zstd_nvcomp,
                                   "jump_rest_fields", "jump_rest_lengths");
  }

  // ====== CONTAINMENT DATA (C-lines) - structured columnar ======
  if (!gpu_graph.containment_container_ids.empty()) {
    data.num_containments_stored = gpu_graph.containment_container_ids.size();

    if (g_debug_compression) {
      std::cout << "[GPU Compression] Compressing C-lines ("
                << data.num_containments_stored << " containments)"
                << std::endl;
    }

    data.containment_container_ids_zstd_nvcomp =
        compress_delta_varint_uint32_gpu(gpu_graph.containment_container_ids,
                                         "containment_container_ids");
    data.containment_contained_ids_zstd_nvcomp =
        compress_delta_varint_uint32_gpu(gpu_graph.containment_contained_ids,
                                         "containment_contained_ids");
    data.containment_container_orients_zstd_nvcomp =
        compress_orientations_gpu(gpu_graph.containment_container_orients,
                                  "containment_container_orients");
    data.containment_contained_orients_zstd_nvcomp =
        compress_orientations_gpu(gpu_graph.containment_contained_orients,
                                  "containment_contained_orients");
    data.containment_positions_zstd_nvcomp = compress_uint32_gpu(
        gpu_graph.containment_positions, "containment_positions");

    compress_flattened_strings_gpu(
        gpu_graph.containment_overlaps, data.containment_overlaps_zstd_nvcomp,
        data.containment_overlap_lengths_zstd_nvcomp, "containment_overlaps",
        "containment_overlap_lengths");
    compress_flattened_strings_gpu(gpu_graph.containment_rest_fields,
                                   data.containment_rest_fields_zstd_nvcomp,
                                   data.containment_rest_lengths_zstd_nvcomp,
                                   "containment_rest_fields",
                                   "containment_rest_lengths");
  }

  // End timer and print compression time
  auto compress_end = Clock::now();
  double compress_time_ms = elapsed_ms(compress_start, compress_end);

  if (g_debug_compression) {
    std::cout << "[GPU Compression] Total compression time (GfaGraph_gpu -> "
                 "CompressedData_gpu): "
              << std::fixed << std::setprecision(2) << compress_time_ms
              << " ms (" << compress_time_ms / 1000.0 << " s)" << std::endl;
  }

  return data;
}

} // namespace gpu_compression
