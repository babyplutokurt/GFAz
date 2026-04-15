#pragma once

#include "gpu/decompression/path_decompression_gpu_rolling.hpp"
#include "io/gfa_write_utils.hpp"
#include "model/compressed_data.hpp"

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace gpu_decompression {

struct GpuDirectWriterStaticFields {
  std::vector<uint32_t> segment_lengths;
  std::string segment_sequences;
  std::vector<gfaz::OptionalFieldColumn> segment_optional_fields;
  std::vector<uint32_t> link_from_ids;
  std::vector<uint32_t> link_to_ids;
  std::vector<char> link_from_orients;
  std::vector<char> link_to_orients;
  std::vector<uint32_t> link_overlap_nums;
  std::vector<char> link_overlap_ops;
  std::vector<gfaz::OptionalFieldColumn> link_optional_fields;
  std::vector<uint32_t> jump_from_ids;
  std::vector<uint32_t> jump_to_ids;
  std::vector<char> jump_from_orients;
  std::vector<char> jump_to_orients;
  std::vector<std::string> jump_distances;
  std::vector<std::string> jump_rest_fields;
  std::vector<uint32_t> containment_container_ids;
  std::vector<uint32_t> containment_contained_ids;
  std::vector<char> containment_container_orients;
  std::vector<char> containment_contained_orients;
  std::vector<uint32_t> containment_positions;
  std::vector<std::string> containment_overlaps;
  std::vector<std::string> containment_rest_fields;
  gfaz::gfa_write_utils::FieldOffsets segment_offsets;
  gfaz::gfa_write_utils::FieldOffsets link_offsets;
  double decode_ms = 0.0;
};

struct GpuPathWriterMetadata {
  std::vector<std::string> names;
  std::vector<std::string> overlaps;
  double decode_ms = 0.0;
};

struct GpuWalkWriterMetadata {
  std::vector<std::string> sample_ids;
  std::vector<uint32_t> hap_indices;
  std::vector<std::string> seq_ids;
  std::vector<int64_t> seq_starts;
  std::vector<int64_t> seq_ends;
  double decode_ms = 0.0;
};

GpuDirectWriterStaticFields
decode_gpu_direct_writer_static_fields(const gfaz::CompressedData &data);

GpuPathWriterMetadata decode_gpu_path_writer_metadata(const gfaz::CompressedData &data);

GpuWalkWriterMetadata decode_gpu_walk_writer_metadata(const gfaz::CompressedData &data);

void write_gpu_direct_writer_static_fields(
    std::ofstream &out, const gfaz::CompressedData &data,
    const GpuDirectWriterStaticFields &fields);

void write_gpu_path_chunk_lines(std::ofstream &out,
                                const RollingPathPinnedHostBuffer &buffer,
                                const GpuPathWriterMetadata &metadata);

void write_gpu_walk_chunk_lines(std::ofstream &out,
                                const RollingPathPinnedHostBuffer &buffer,
                                const GpuWalkWriterMetadata &metadata);

} // namespace gpu_decompression
