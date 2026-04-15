#pragma once

#include "model/compressed_data.hpp"
#include "model/gfa_graph.hpp"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace gfaz::gfa_write_utils {

using FieldOffsets = std::vector<std::vector<size_t>>;
using SequenceOffsets = std::vector<size_t>;

FieldOffsets build_field_offsets(const std::vector<gfaz::OptionalFieldColumn> &cols);

std::string format_optional_fields(const std::vector<gfaz::OptionalFieldColumn> &cols,
                                   const FieldOffsets &offsets, size_t index);

void append_numeric_node_name(std::string &out, uint32_t node_id);

std::vector<std::string>
decompress_string_column(const gfaz::ZstdCompressedBlock &strings_zstd,
                         const gfaz::ZstdCompressedBlock &lengths_zstd);

gfaz::OptionalFieldColumn
decompress_optional_column(const gfaz::CompressedOptionalFieldColumn &c);

SequenceOffsets build_offsets(const std::vector<uint32_t> &lengths);

std::pair<std::vector<int32_t>, std::vector<int32_t>>
decode_rules(const gfaz::CompressedData &data);

std::string format_path_line_numeric(const std::string &path_name,
                                     const int32_t *path_data, size_t path_size,
                                     const std::string &overlap);

std::string format_walk_line_numeric(const std::string &sample_id,
                                     uint32_t hap_index,
                                     const std::string &seq_id,
                                     int64_t seq_start, int64_t seq_end,
                                     const int32_t *walk_data,
                                     size_t walk_size);

void write_segments_numeric(std::ofstream &out,
                            const std::string &segment_sequences,
                            const std::vector<uint32_t> &segment_lengths,
                            const std::vector<gfaz::OptionalFieldColumn>
                                &segment_optional_fields,
                            const FieldOffsets &segment_offsets);

void write_links_numeric(
    std::ofstream &out, const std::vector<uint32_t> &link_from_ids,
    const std::vector<uint32_t> &link_to_ids,
    const std::vector<char> &link_from_orients,
    const std::vector<char> &link_to_orients,
    const std::vector<uint32_t> &link_overlap_nums,
    const std::vector<char> &link_overlap_ops,
    const std::vector<gfaz::OptionalFieldColumn> &link_optional_fields,
    const FieldOffsets &link_offsets);

void write_jumps_numeric(std::ofstream &out,
                         const std::vector<uint32_t> &jump_from_ids,
                         const std::vector<uint32_t> &jump_to_ids,
                         const std::vector<char> &jump_from_orients,
                         const std::vector<char> &jump_to_orients,
                         const std::vector<std::string> &jump_distances,
                         const std::vector<std::string> &jump_rest_fields);

void write_containments_numeric(
    std::ofstream &out,
    const std::vector<uint32_t> &containment_container_ids,
    const std::vector<uint32_t> &containment_contained_ids,
    const std::vector<char> &containment_container_orients,
    const std::vector<char> &containment_contained_orients,
    const std::vector<uint32_t> &containment_positions,
    const std::vector<std::string> &containment_overlaps,
    const std::vector<std::string> &containment_rest_fields);

} // namespace gfaz::gfa_write_utils
