#include "gfa_writer.hpp"

#include "codec.hpp"
#include "debug_log.hpp"
#include "gpu/codec_gpu.cuh"
#include "gpu/codec_gpu_nvcomp.cuh"
#include "gpu/decompression_workflow_gpu.hpp"
#include "gpu/gfa_graph_gpu.hpp"
#include "gpu/metadata_codec_gpu.hpp"
#include "gpu/path_decompression_gpu_rolling.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <vector>

namespace {

constexpr const char *kWriterErrorPrefix = "GFA writer error: ";

using FieldOffsets = std::vector<std::vector<size_t>>;
using SequenceOffsets = std::vector<size_t>;

std::string format_float(float val) {
  std::ostringstream oss;
  oss << val;
  return oss.str();
}

inline size_t b_elem_size(char subtype) {
  switch (subtype) {
  case 'c':
  case 'C':
    return 1;
  case 's':
  case 'S':
    return 2;
  case 'i':
  case 'I':
  case 'f':
    return 4;
  default:
    return 0;
  }
}

FieldOffsets build_field_offsets(const std::vector<OptionalFieldColumn> &cols) {
  FieldOffsets offsets(cols.size());
  for (size_t c = 0; c < cols.size(); ++c) {
    const auto &col = cols[c];
    if (col.type == 'Z' || col.type == 'J' || col.type == 'H') {
      offsets[c].resize(col.string_lengths.size() + 1, 0);
      for (size_t i = 0; i < col.string_lengths.size(); ++i)
        offsets[c][i + 1] = offsets[c][i] + col.string_lengths[i];
    } else if (col.type == 'B') {
      offsets[c].resize(col.b_lengths.size() + 1, 0);
      for (size_t i = 0; i < col.b_lengths.size(); ++i) {
        offsets[c][i + 1] =
            offsets[c][i] + col.b_lengths[i] * b_elem_size(col.b_subtypes[i]);
      }
    }
  }
  return offsets;
}

std::string format_optional_fields(const std::vector<OptionalFieldColumn> &cols,
                                   const FieldOffsets &offsets, size_t index) {
  std::string result;
  for (size_t c = 0; c < cols.size(); ++c) {
    const auto &col = cols[c];
    switch (col.type) {
    case 'i':
      if (index < col.int_values.size()) {
        const int64_t val = col.int_values[index];
        if (val != std::numeric_limits<int64_t>::min())
          result += "\t" + col.tag + ":i:" + std::to_string(val);
      }
      break;
    case 'f':
      if (index < col.float_values.size()) {
        const float val = col.float_values[index];
        if (val != std::numeric_limits<float>::lowest())
          result += "\t" + col.tag + ":f:" + format_float(val);
      }
      break;
    case 'A':
      if (index < col.char_values.size()) {
        const char val = col.char_values[index];
        if (val != '\0')
          result += "\t" + col.tag + ":A:" + std::string(1, val);
      }
      break;
    case 'Z':
    case 'J':
    case 'H':
      if (index < col.string_lengths.size()) {
        const uint32_t len = col.string_lengths[index];
        if (len > 0) {
          result += "\t" + col.tag + ":" + std::string(1, col.type) + ":";
          result.append(col.concatenated_strings, offsets[c][index], len);
        }
      }
      break;
    case 'B':
      if (index < col.b_lengths.size()) {
        const char subtype = col.b_subtypes[index];
        const uint32_t count = col.b_lengths[index];
        if (subtype != '\0' && count > 0) {
          const size_t elem_sz = b_elem_size(subtype);
          const uint8_t *ptr = col.b_concat_bytes.data() + offsets[c][index];
          result += "\t" + col.tag + ":B:" + std::string(1, subtype);
          for (uint32_t i = 0; i < count; ++i) {
            result += ',';
            if (subtype == 'c') {
              int8_t v;
              std::memcpy(&v, ptr + i * elem_sz, 1);
              result += std::to_string(v);
            } else if (subtype == 'C') {
              result += std::to_string(ptr[i]);
            } else if (subtype == 's') {
              int16_t v;
              std::memcpy(&v, ptr + i * elem_sz, 2);
              result += std::to_string(v);
            } else if (subtype == 'S') {
              uint16_t v;
              std::memcpy(&v, ptr + i * elem_sz, 2);
              result += std::to_string(v);
            } else if (subtype == 'i') {
              int32_t v;
              std::memcpy(&v, ptr + i * elem_sz, 4);
              result += std::to_string(v);
            } else if (subtype == 'I') {
              uint32_t v;
              std::memcpy(&v, ptr + i * elem_sz, 4);
              result += std::to_string(v);
            } else if (subtype == 'f') {
              float v;
              std::memcpy(&v, ptr + i * elem_sz, 4);
              result += format_float(v);
            }
          }
        }
      }
      break;
    }
  }
  return result;
}

SequenceOffsets build_offsets(const std::vector<uint32_t> &lengths) {
  SequenceOffsets offsets(lengths.size() + 1, 0);
  for (size_t i = 0; i < lengths.size(); ++i)
    offsets[i + 1] = offsets[i] + lengths[i];
  return offsets;
}

void append_numeric_node_name(std::string &out, uint32_t node_id) {
  out += std::to_string(node_id);
}

void append_flattened_string(std::string &out, const FlattenedStrings &strings,
                             const SequenceOffsets &offsets, size_t index) {
  if (index >= strings.lengths.size())
    return;
  const size_t begin = offsets[index];
  const size_t end = offsets[index + 1];
  if (end > strings.data.size()) {
    throw std::runtime_error(std::string(kWriterErrorPrefix) +
                             "flattened string column is truncated");
  }
  out.append(strings.data.data() + begin, end - begin);
}

std::string flattened_string_at(const FlattenedStrings &strings,
                                const SequenceOffsets &offsets, size_t index) {
  std::string result;
  append_flattened_string(result, strings, offsets, index);
  return result;
}

void append_flattened_node_name(std::string &out, uint32_t node_id,
                                const FlattenedStrings &node_names,
                                const SequenceOffsets &node_name_offsets) {
  if (node_id < node_names.lengths.size() && node_names.lengths[node_id] > 0) {
    append_flattened_string(out, node_names, node_name_offsets, node_id);
  } else {
    append_numeric_node_name(out, node_id);
  }
}

OptionalFieldColumn convert_optional_field_from_gpu_writer(
    const OptionalFieldColumn_gpu &gpu_col) {
  OptionalFieldColumn result;
  result.tag = gpu_col.tag;
  result.type = gpu_col.type;
  switch (gpu_col.type) {
  case 'i':
    result.int_values = gpu_col.int_values;
    break;
  case 'f':
    result.float_values = gpu_col.float_values;
    break;
  case 'A':
    result.char_values = gpu_col.char_values;
    break;
  case 'Z':
  case 'J':
  case 'H':
    result.concatenated_strings.assign(gpu_col.strings.data.begin(),
                                       gpu_col.strings.data.end());
    result.string_lengths = gpu_col.strings.lengths;
    break;
  case 'B':
    result.b_subtypes = gpu_col.b_subtypes;
    result.b_lengths = gpu_col.b_lengths;
    result.b_concat_bytes = gpu_col.b_data;
    break;
  default:
    break;
  }
  return result;
}

std::string format_path_line_flat(const std::string &path_name,
                                  const int32_t *nodes, size_t node_count,
                                  const std::string &overlap,
                                  const FlattenedStrings &node_names,
                                  const SequenceOffsets &node_name_offsets) {
  std::string line = "P\t";
  line += path_name;
  line += '\t';
  for (size_t i = 0; i < node_count; ++i) {
    if (i > 0)
      line += ',';
    const NodeId node = nodes[i];
    const bool reverse = node < 0;
    append_flattened_node_name(
        line, static_cast<uint32_t>(reverse ? -node : node), node_names,
        node_name_offsets);
    line += (reverse ? '-' : '+');
  }
  line += '\t';
  line += overlap.empty() ? "*" : overlap;
  line += '\n';
  return line;
}

std::string format_walk_line_flat(const std::string &sample_id,
                                  uint32_t hap_index,
                                  const std::string &seq_id,
                                  int64_t seq_start, int64_t seq_end,
                                  const int32_t *nodes, size_t node_count,
                                  const FlattenedStrings &node_names,
                                  const SequenceOffsets &node_name_offsets) {
  std::string line = "W\t";
  line += sample_id;
  line += '\t';
  line += std::to_string(hap_index);
  line += '\t';
  line += seq_id;
  line += '\t';
  line += (seq_start >= 0) ? std::to_string(seq_start) : "*";
  line += '\t';
  line += (seq_end >= 0) ? std::to_string(seq_end) : "*";
  line += '\t';
  for (size_t i = 0; i < node_count; ++i) {
    const NodeId node = nodes[i];
    const bool reverse = node < 0;
    line += (reverse ? '<' : '>');
    append_flattened_node_name(
        line, static_cast<uint32_t>(reverse ? -node : node), node_names,
        node_name_offsets);
  }
  line += '\n';
  return line;
}

struct ScopedCudaStreams3Writer {
  cudaStream_t a = nullptr;
  cudaStream_t b = nullptr;
  cudaStream_t c = nullptr;

  ScopedCudaStreams3Writer() {
    if (cudaStreamCreate(&a) != cudaSuccess || cudaStreamCreate(&b) != cudaSuccess ||
        cudaStreamCreate(&c) != cudaSuccess) {
      throw std::runtime_error(std::string(kWriterErrorPrefix) +
                               "failed to create CUDA streams for GPU writer");
    }
  }

  ~ScopedCudaStreams3Writer() {
    if (a)
      cudaStreamDestroy(a);
    if (b)
      cudaStreamDestroy(b);
    if (c)
      cudaStreamDestroy(c);
  }
};

} // namespace

void write_gfa_from_compressed_data_gpu(
    const gpu_compression::CompressedData_gpu &data,
    const std::string &output_path,
    gpu_decompression::GpuDecompressionOptions options) {
  if (options.use_legacy_full_decompression) {
    GfaGraph_gpu graph_gpu =
        gpu_decompression::decompress_to_gpu_layout(data, options);
    GfaGraph graph = convert_from_gpu_layout(graph_gpu);
    write_gfa(graph, output_path);
    return;
  }

  std::ofstream out(output_path);
  if (!out) {
    throw std::runtime_error(std::string(kWriterErrorPrefix) +
                             "failed to open output file: " + output_path);
  }

  GFAZ_LOG("Writing GFA directly from GPU compressed data to " << output_path
                                                                << "...");

  GfaGraph_gpu metadata;
  gpu_decompression::decompress_graph_metadata_gpu(data, metadata);

  std::vector<OptionalFieldColumn> segment_optional_fields;
  segment_optional_fields.reserve(metadata.segment_optional_fields.size());
  for (const auto &col : metadata.segment_optional_fields)
    segment_optional_fields.push_back(convert_optional_field_from_gpu_writer(col));

  std::vector<OptionalFieldColumn> link_optional_fields;
  link_optional_fields.reserve(metadata.link_optional_fields.size());
  for (const auto &col : metadata.link_optional_fields)
    link_optional_fields.push_back(convert_optional_field_from_gpu_writer(col));

  const FieldOffsets seg_offsets = build_field_offsets(segment_optional_fields);
  const FieldOffsets link_offsets = build_field_offsets(link_optional_fields);
  const SequenceOffsets node_name_offsets = build_offsets(metadata.node_names.lengths);
  const SequenceOffsets node_seq_offsets =
      build_offsets(metadata.node_sequences.lengths);
  const SequenceOffsets path_name_offsets = build_offsets(metadata.path_names.lengths);
  const SequenceOffsets path_overlap_offsets =
      build_offsets(metadata.path_overlaps.lengths);
  const SequenceOffsets walk_sample_offsets =
      build_offsets(metadata.walk_sample_ids.lengths);
  const SequenceOffsets walk_seq_id_offsets =
      build_offsets(metadata.walk_seq_ids.lengths);
  const SequenceOffsets jump_distance_offsets =
      build_offsets(metadata.jump_distances.lengths);
  const SequenceOffsets jump_rest_offsets =
      build_offsets(metadata.jump_rest_fields.lengths);
  const SequenceOffsets containment_overlap_offsets =
      build_offsets(metadata.containment_overlaps.lengths);
  const SequenceOffsets containment_rest_offsets =
      build_offsets(metadata.containment_rest_fields.lengths);

  if (!data.header_line.empty())
    out << data.header_line << "\n";

  std::string line;
  line.reserve(4096);

  for (size_t i = 1; i < metadata.node_sequences.lengths.size(); ++i) {
    line.clear();
    line += "S\t";
    append_flattened_node_name(line, static_cast<uint32_t>(i), metadata.node_names,
                               node_name_offsets);
    line += '\t';
    append_flattened_string(line, metadata.node_sequences, node_seq_offsets, i);
    line += format_optional_fields(segment_optional_fields, seg_offsets, i - 1);
    line += '\n';
    out.write(line.data(), static_cast<std::streamsize>(line.size()));
  }

  for (size_t i = 0; i < metadata.link_from_ids.size(); ++i) {
    line.clear();
    line += "L\t";
    append_flattened_node_name(line, metadata.link_from_ids[i], metadata.node_names,
                               node_name_offsets);
    line += '\t';
    line += metadata.link_from_orients[i];
    line += '\t';
    append_flattened_node_name(line, metadata.link_to_ids[i], metadata.node_names,
                               node_name_offsets);
    line += '\t';
    line += metadata.link_to_orients[i];
    line += '\t';
    if (i < metadata.link_overlap_ops.size() && metadata.link_overlap_ops[i] != '\0') {
      line += std::to_string(metadata.link_overlap_nums[i]);
      line += metadata.link_overlap_ops[i];
    } else {
      line += '*';
    }
    line += format_optional_fields(link_optional_fields, link_offsets, i);
    line += '\n';
    out.write(line.data(), static_cast<std::streamsize>(line.size()));
  }

  ScopedCudaStreams3Writer streams;
  int32_t *d_encoded_path_raw = nullptr;
  int32_t *d_first_delta_raw = nullptr;
  int32_t *d_second_delta_raw = nullptr;
  size_t encoded_path_count = 0;
  size_t first_count = 0;
  size_t second_count = 0;

  gpu_codec::nvcomp_zstd_decompress_int32_to_device(
      data.encoded_path_zstd_nvcomp, &d_encoded_path_raw, &encoded_path_count,
      streams.a);
  gpu_codec::nvcomp_zstd_decompress_int32_to_device(
      data.rules_first_zstd_nvcomp, &d_first_delta_raw, &first_count, streams.b);
  gpu_codec::nvcomp_zstd_decompress_int32_to_device(
      data.rules_second_zstd_nvcomp, &d_second_delta_raw, &second_count,
      streams.c);

  cudaStreamSynchronize(streams.a);
  cudaStreamSynchronize(streams.b);
  cudaStreamSynchronize(streams.c);

  thrust::device_vector<int32_t> d_encoded_path;
  if (d_encoded_path_raw != nullptr && encoded_path_count > 0) {
    thrust::device_ptr<int32_t> d_encoded_ptr(d_encoded_path_raw);
    d_encoded_path.assign(d_encoded_ptr, d_encoded_ptr + encoded_path_count);
  }
  if (d_encoded_path_raw != nullptr)
    cudaFree(d_encoded_path_raw);

  thrust::device_vector<int32_t> d_rules_first;
  thrust::device_vector<int32_t> d_rules_second;
  if (d_first_delta_raw != nullptr && first_count > 0) {
    thrust::device_ptr<int32_t> d_first_ptr(d_first_delta_raw);
    thrust::device_vector<int32_t> d_first_delta(d_first_ptr,
                                                 d_first_ptr + first_count);
    d_rules_first = gpu_codec::inverse_delta_decode_device_vec(d_first_delta);
  }
  if (d_first_delta_raw != nullptr)
    cudaFree(d_first_delta_raw);
  if (d_second_delta_raw != nullptr && second_count > 0) {
    thrust::device_ptr<int32_t> d_second_ptr(d_second_delta_raw);
    thrust::device_vector<int32_t> d_second_delta(d_second_ptr,
                                                  d_second_ptr + second_count);
    d_rules_second = gpu_codec::inverse_delta_decode_device_vec(d_second_delta);
  }
  if (d_second_delta_raw != nullptr)
    cudaFree(d_second_delta_raw);

  std::vector<uint32_t> path_lengths =
      gpu_codec::nvcomp_zstd_decompress_uint32(data.path_lengths_zstd_nvcomp);
  thrust::device_vector<uint32_t> d_lens_final(path_lengths.begin(),
                                               path_lengths.end());
  const size_t num_rules =
      std::min({data.total_rules(), d_rules_first.size(), d_rules_second.size()});

  gpu_decompression::RollingPathStreamOptions stream_options;
  stream_options.num_host_buffers = 3;

  gpu_decompression::stream_decompress_paths_gpu_rolling(
      d_encoded_path, d_rules_first, d_rules_second, data.min_rule_id(),
      num_rules, d_lens_final, options.traversals_per_chunk,
      [&](const gpu_decompression::RollingPathPinnedHostBuffer &chunk) {
        size_t local_offset = 0;
        for (size_t local_index = 0; local_index < chunk.lengths.size();
             ++local_index) {
          const size_t global_index = chunk.segment_begin + local_index;
          const size_t node_count = chunk.lengths[local_index];
          const int32_t *nodes = chunk.host_nodes + local_offset;

          if (global_index < metadata.num_paths) {
            const std::string name =
                flattened_string_at(metadata.path_names, path_name_offsets, global_index);
            const std::string overlap = flattened_string_at(
                metadata.path_overlaps, path_overlap_offsets, global_index);
            const std::string path_line =
                format_path_line_flat(name, nodes, node_count, overlap,
                                      metadata.node_names, node_name_offsets);
            out.write(path_line.data(),
                      static_cast<std::streamsize>(path_line.size()));
          } else {
            const size_t walk_index = global_index - metadata.num_paths;
            const std::string sample_id = flattened_string_at(
                metadata.walk_sample_ids, walk_sample_offsets, walk_index);
            const uint32_t hap_index =
                (walk_index < metadata.walk_hap_indices.size())
                    ? metadata.walk_hap_indices[walk_index]
                    : 0;
            const std::string seq_id =
                flattened_string_at(metadata.walk_seq_ids, walk_seq_id_offsets, walk_index);
            const int64_t seq_start =
                (walk_index < metadata.walk_seq_starts.size())
                    ? metadata.walk_seq_starts[walk_index]
                    : -1;
            const int64_t seq_end =
                (walk_index < metadata.walk_seq_ends.size())
                    ? metadata.walk_seq_ends[walk_index]
                    : -1;
            const std::string walk_line = format_walk_line_flat(
                sample_id, hap_index, seq_id, seq_start, seq_end, nodes,
                node_count, metadata.node_names, node_name_offsets);
            out.write(walk_line.data(),
                      static_cast<std::streamsize>(walk_line.size()));
          }
          local_offset += node_count;
        }
      },
      stream_options);

  for (size_t i = 0; i < metadata.jump_from_ids.size(); ++i) {
    line.clear();
    line += "J\t";
    append_flattened_node_name(line, metadata.jump_from_ids[i], metadata.node_names,
                               node_name_offsets);
    line += '\t';
    line += metadata.jump_from_orients[i];
    line += '\t';
    append_flattened_node_name(line, metadata.jump_to_ids[i], metadata.node_names,
                               node_name_offsets);
    line += '\t';
    line += metadata.jump_to_orients[i];
    line += '\t';
    append_flattened_string(line, metadata.jump_distances, jump_distance_offsets, i);
    if (i < metadata.jump_rest_fields.lengths.size() &&
        metadata.jump_rest_fields.lengths[i] > 0) {
      line += '\t';
      append_flattened_string(line, metadata.jump_rest_fields, jump_rest_offsets, i);
    }
    line += '\n';
    out.write(line.data(), static_cast<std::streamsize>(line.size()));
  }

  for (size_t i = 0; i < metadata.containment_container_ids.size(); ++i) {
    line.clear();
    line += "C\t";
    append_flattened_node_name(line, metadata.containment_container_ids[i],
                               metadata.node_names, node_name_offsets);
    line += '\t';
    line += metadata.containment_container_orients[i];
    line += '\t';
    append_flattened_node_name(line, metadata.containment_contained_ids[i],
                               metadata.node_names, node_name_offsets);
    line += '\t';
    line += metadata.containment_contained_orients[i];
    line += '\t';
    line += std::to_string(metadata.containment_positions[i]);
    line += '\t';
    append_flattened_string(line, metadata.containment_overlaps,
                            containment_overlap_offsets, i);
    if (i < metadata.containment_rest_fields.lengths.size() &&
        metadata.containment_rest_fields.lengths[i] > 0) {
      line += '\t';
      append_flattened_string(line, metadata.containment_rest_fields,
                              containment_rest_offsets, i);
    }
    line += '\n';
    out.write(line.data(), static_cast<std::streamsize>(line.size()));
  }

  out.close();
}
