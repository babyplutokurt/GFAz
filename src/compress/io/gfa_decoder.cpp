#include "compress/io/gfa_decoder.hpp"

#include "compress/io/gfa_write_utils.hpp"
#include "core/codec/codec.hpp"
#include "core/codec/serialization.hpp"
#include "core/utils/threading_utils.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

constexpr const char *kDecoderErrorPrefix = "GFA decoder error: ";

using gfaz::GfaTagList;
using gfaz::NodeId;
using gfaz::gfa_write_utils::build_field_offsets;
using gfaz::gfa_write_utils::build_offsets;
using gfaz::gfa_write_utils::decode_rules;
using gfaz::gfa_write_utils::decompress_optional_column;
using gfaz::gfa_write_utils::decompress_string_column;
using gfaz::gfa_write_utils::FieldOffsets;
using gfaz::gfa_write_utils::format_optional_fields;
using gfaz::gfa_write_utils::SequenceOffsets;

void expand_rule(uint32_t rule_id, bool reverse,
                 const std::vector<int32_t> &first,
                 const std::vector<int32_t> &second, uint32_t min_id,
                 uint32_t max_id, std::vector<NodeId> &out) {
  const uint32_t idx = rule_id - min_id;
  if (idx >= first.size() || idx >= second.size())
    throw std::runtime_error(std::string(kDecoderErrorPrefix) +
                             "rule reference is out of range");

  const int32_t a = first[idx];
  const int32_t b = second[idx];

  if (!reverse) {
    const uint32_t abs_a = static_cast<uint32_t>(std::abs(a));
    if (abs_a >= min_id && abs_a < max_id)
      expand_rule(abs_a, a < 0, first, second, min_id, max_id, out);
    else
      out.push_back(a);

    const uint32_t abs_b = static_cast<uint32_t>(std::abs(b));
    if (abs_b >= min_id && abs_b < max_id)
      expand_rule(abs_b, b < 0, first, second, min_id, max_id, out);
    else
      out.push_back(b);
  } else {
    const uint32_t abs_b = static_cast<uint32_t>(std::abs(b));
    if (abs_b >= min_id && abs_b < max_id)
      expand_rule(abs_b, b >= 0, first, second, min_id, max_id, out);
    else
      out.push_back(-b);

    const uint32_t abs_a = static_cast<uint32_t>(std::abs(a));
    if (abs_a >= min_id && abs_a < max_id)
      expand_rule(abs_a, a >= 0, first, second, min_id, max_id, out);
    else
      out.push_back(-a);
  }
}

std::vector<NodeId>
decode_sequence_at_index(const std::vector<int32_t> &flat,
                         const SequenceOffsets &compressed_offsets,
                         const SequenceOffsets &original_offsets, size_t index,
                         const std::vector<int32_t> &rules_first,
                         const std::vector<int32_t> &rules_second,
                         uint32_t min_rule_id, int delta_round) {
  if (index + 1 >= compressed_offsets.size())
    throw std::out_of_range(std::string(kDecoderErrorPrefix) +
                            "sequence index out of range");

  const size_t start = compressed_offsets[index];
  const size_t end = compressed_offsets[index + 1];
  if (end > flat.size())
    throw std::runtime_error(std::string(kDecoderErrorPrefix) +
                             "flattened sequence block is truncated");

  const uint32_t max_rule_id =
      min_rule_id + static_cast<uint32_t>(rules_first.size());
  const size_t original_length =
      (index + 1 < original_offsets.size())
          ? original_offsets[index + 1] - original_offsets[index]
          : end - start;

  std::vector<NodeId> decoded;
  decoded.reserve(original_length);
  for (size_t pos = start; pos < end; ++pos) {
    const NodeId node = flat[pos];
    const uint32_t abs_id = static_cast<uint32_t>(std::abs(node));
    if (abs_id >= min_rule_id && abs_id < max_rule_id)
      expand_rule(abs_id, node < 0, rules_first, rules_second, min_rule_id,
                  max_rule_id, decoded);
    else
      decoded.push_back(node);
  }

  std::vector<std::vector<NodeId>> sequences(1);
  sequences[0] = std::move(decoded);
  for (int round = 0; round < delta_round; ++round)
    gfaz::Codec::inverse_delta_transform(sequences);
  return std::move(sequences[0]);
}

GfaTagList tags_for_row(const std::vector<gfaz::OptionalFieldColumn> &columns,
                        const FieldOffsets &offsets, size_t index) {
  const std::string formatted = format_optional_fields(columns, offsets, index);
  GfaTagList tags;
  size_t start = 0;
  while (start < formatted.size()) {
    if (formatted[start] == '\t')
      ++start;
    if (start >= formatted.size())
      break;
    const size_t end = formatted.find('\t', start);
    tags.push_back(formatted.substr(start, end - start));
    if (end == std::string::npos)
      break;
    start = end;
  }
  return tags;
}

template <typename Emitter>
void decode_sequence_batches(const std::vector<int32_t> &flat,
                             const SequenceOffsets &compressed_offsets,
                             const SequenceOffsets &original_offsets,
                             const std::vector<int32_t> &rules_first,
                             const std::vector<int32_t> &rules_second,
                             uint32_t min_rule_id, int delta_round,
                             size_t total_count, int num_threads,
                             Emitter emit) {
  if (total_count == 0)
    return;

  const int effective_threads =
      std::max(1, resolve_omp_thread_count(num_threads));
  const size_t batch_size = static_cast<size_t>(effective_threads) * 8;

  for (size_t batch_start = 0; batch_start < total_count;
       batch_start += batch_size) {
    const size_t batch_end = std::min(batch_start + batch_size, total_count);
    std::vector<std::vector<NodeId>> decoded(batch_end - batch_start);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (size_t i = batch_start; i < batch_end; ++i) {
      decoded[i - batch_start] = decode_sequence_at_index(
          flat, compressed_offsets, original_offsets, i, rules_first,
          rules_second, min_rule_id, delta_round);
    }

    for (size_t i = batch_start; i < batch_end; ++i)
      emit(i, decoded[i - batch_start]);
  }
}

} // namespace

namespace gfaz {

void decode_gfa_records(const CompressedData &data, GfaRecordVisitor &visitor,
                        int num_threads) {
  ScopedOMPThreads omp_scope(num_threads);

  std::vector<int32_t> rules_first;
  std::vector<int32_t> rules_second;
  std::vector<int32_t> paths_flat;
  std::vector<int32_t> walks_flat;
  std::vector<std::string> path_names;
  std::vector<std::string> path_overlaps;
  std::vector<std::string> walk_sample_ids;
  std::vector<uint32_t> walk_hap_indices;
  std::vector<std::string> walk_sequence_ids;
  std::vector<int64_t> walk_sequence_starts;
  std::vector<int64_t> walk_sequence_ends;
  std::string segment_sequences;
  std::vector<uint32_t> segment_lengths;
  std::vector<uint32_t> link_from_ids;
  std::vector<uint32_t> link_to_ids;
  std::vector<char> link_from_orients;
  std::vector<char> link_to_orients;
  std::vector<uint32_t> link_overlap_nums;
  std::vector<char> link_overlap_ops;

#ifdef _OPENMP
#pragma omp parallel sections
  {
#pragma omp section
    {
      auto rules = decode_rules(data);
      rules_first = std::move(rules.first);
      rules_second = std::move(rules.second);
    }
#pragma omp section
    {
      if (!data.paths_zstd.payload.empty())
        paths_flat = Codec::zstd_decompress_int32_vector(data.paths_zstd);
    }
#pragma omp section
    {
      if (!data.walks_zstd.payload.empty())
        walks_flat = Codec::zstd_decompress_int32_vector(data.walks_zstd);
    }
#pragma omp section
    {
      path_names =
          decompress_string_column(data.names_zstd, data.name_lengths_zstd);
      path_overlaps = decompress_string_column(data.overlaps_zstd,
                                               data.overlap_lengths_zstd);
    }
#pragma omp section
    {
      walk_sample_ids = decompress_string_column(
          data.walk_sample_ids_zstd, data.walk_sample_id_lengths_zstd);
      walk_hap_indices =
          Codec::zstd_decompress_uint32_vector(data.walk_hap_indices_zstd);
      walk_sequence_ids = decompress_string_column(
          data.walk_seq_ids_zstd, data.walk_seq_id_lengths_zstd);
      walk_sequence_starts = Codec::decompress_varint_int64(
          data.walk_seq_starts_zstd, data.walk_lengths.size());
      walk_sequence_ends = Codec::decompress_varint_int64(
          data.walk_seq_ends_zstd, data.walk_lengths.size());
    }
#pragma omp section
    {
      segment_sequences =
          Codec::zstd_decompress_string(data.segment_sequences_zstd);
      segment_lengths =
          Codec::zstd_decompress_uint32_vector(data.segment_seq_lengths_zstd);
    }
#pragma omp section
    {
      link_from_ids = Codec::decompress_delta_varint_uint32(
          data.link_from_ids_zstd, data.num_links);
      link_to_ids = Codec::decompress_delta_varint_uint32(data.link_to_ids_zstd,
                                                          data.num_links);
      link_from_orients = Codec::decompress_orientations(
          data.link_from_orients_zstd, data.num_links);
      link_to_orients = Codec::decompress_orientations(
          data.link_to_orients_zstd, data.num_links);
      link_overlap_nums =
          Codec::zstd_decompress_uint32_vector(data.link_overlap_nums_zstd);
      link_overlap_ops =
          Codec::zstd_decompress_char_vector(data.link_overlap_ops_zstd);
    }
  }
#else
  auto rules = decode_rules(data);
  rules_first = std::move(rules.first);
  rules_second = std::move(rules.second);
  if (!data.paths_zstd.payload.empty())
    paths_flat = Codec::zstd_decompress_int32_vector(data.paths_zstd);
  if (!data.walks_zstd.payload.empty())
    walks_flat = Codec::zstd_decompress_int32_vector(data.walks_zstd);
  path_names =
      decompress_string_column(data.names_zstd, data.name_lengths_zstd);
  path_overlaps =
      decompress_string_column(data.overlaps_zstd, data.overlap_lengths_zstd);
  walk_sample_ids = decompress_string_column(data.walk_sample_ids_zstd,
                                             data.walk_sample_id_lengths_zstd);
  walk_hap_indices =
      Codec::zstd_decompress_uint32_vector(data.walk_hap_indices_zstd);
  walk_sequence_ids = decompress_string_column(data.walk_seq_ids_zstd,
                                               data.walk_seq_id_lengths_zstd);
  walk_sequence_starts = Codec::decompress_varint_int64(
      data.walk_seq_starts_zstd, data.walk_lengths.size());
  walk_sequence_ends = Codec::decompress_varint_int64(data.walk_seq_ends_zstd,
                                                      data.walk_lengths.size());
  segment_sequences =
      Codec::zstd_decompress_string(data.segment_sequences_zstd);
  segment_lengths =
      Codec::zstd_decompress_uint32_vector(data.segment_seq_lengths_zstd);
  link_from_ids = Codec::decompress_delta_varint_uint32(data.link_from_ids_zstd,
                                                        data.num_links);
  link_to_ids = Codec::decompress_delta_varint_uint32(data.link_to_ids_zstd,
                                                      data.num_links);
  link_from_orients = Codec::decompress_orientations(
      data.link_from_orients_zstd, data.num_links);
  link_to_orients =
      Codec::decompress_orientations(data.link_to_orients_zstd, data.num_links);
  link_overlap_nums =
      Codec::zstd_decompress_uint32_vector(data.link_overlap_nums_zstd);
  link_overlap_ops =
      Codec::zstd_decompress_char_vector(data.link_overlap_ops_zstd);
#endif

  std::vector<OptionalFieldColumn> segment_optional_fields;
  segment_optional_fields.reserve(data.segment_optional_fields_zstd.size());
  for (const auto &column : data.segment_optional_fields_zstd)
    segment_optional_fields.push_back(decompress_optional_column(column));

  std::vector<OptionalFieldColumn> link_optional_fields;
  link_optional_fields.reserve(data.link_optional_fields_zstd.size());
  for (const auto &column : data.link_optional_fields_zstd)
    link_optional_fields.push_back(decompress_optional_column(column));

  const FieldOffsets segment_field_offsets =
      build_field_offsets(segment_optional_fields);
  const FieldOffsets link_field_offsets =
      build_field_offsets(link_optional_fields);
  const SequenceOffsets path_offsets = build_offsets(data.sequence_lengths);
  const SequenceOffsets walk_offsets = build_offsets(data.walk_lengths);
  const SequenceOffsets original_path_offsets =
      build_offsets(data.original_path_lengths);
  const SequenceOffsets original_walk_offsets =
      build_offsets(data.original_walk_lengths);

  if (!data.header_line.empty())
    visitor.on_header(data.header_line);

  size_t sequence_offset = 0;
  for (size_t i = 0; i < segment_lengths.size(); ++i) {
    const size_t length = segment_lengths[i];
    if (sequence_offset + length > segment_sequences.size())
      throw std::runtime_error(std::string(kDecoderErrorPrefix) +
                               "segment sequence column is truncated");
    const std::string sequence =
        segment_sequences.substr(sequence_offset, length);
    const GfaTagList tags =
        tags_for_row(segment_optional_fields, segment_field_offsets, i);
    visitor.on_segment(static_cast<uint32_t>(i + 1), sequence, tags);
    sequence_offset += length;
  }

  for (size_t i = 0; i < link_from_ids.size(); ++i) {
    std::string overlap;
    if (i < link_overlap_ops.size() && link_overlap_ops[i] != '\0') {
      overlap = std::to_string(
          i < link_overlap_nums.size() ? link_overlap_nums[i] : 0);
      overlap += link_overlap_ops[i];
    }
    const GfaTagList tags =
        tags_for_row(link_optional_fields, link_field_offsets, i);
    visitor.on_link(link_from_ids[i],
                    i < link_from_orients.size() && link_from_orients[i] == '-',
                    i < link_to_ids.size() ? link_to_ids[i] : 0,
                    i < link_to_orients.size() && link_to_orients[i] == '-',
                    overlap, tags);
  }

  if (data.num_jumps > 0) {
    const std::vector<uint32_t> from_ids =
        Codec::decompress_delta_varint_uint32(data.jump_from_ids_zstd,
                                              data.num_jumps);
    const std::vector<uint32_t> to_ids = Codec::decompress_delta_varint_uint32(
        data.jump_to_ids_zstd, data.num_jumps);
    const std::vector<char> from_orients = Codec::decompress_orientations(
        data.jump_from_orients_zstd, data.num_jumps);
    const std::vector<char> to_orients = Codec::decompress_orientations(
        data.jump_to_orients_zstd, data.num_jumps);
    const std::vector<std::string> distances = decompress_string_column(
        data.jump_distances_zstd, data.jump_distance_lengths_zstd);
    const std::vector<std::string> rest_fields = decompress_string_column(
        data.jump_rest_fields_zstd, data.jump_rest_lengths_zstd);
    for (size_t i = 0; i < from_ids.size(); ++i) {
      const std::string empty;
      visitor.on_jump(from_ids[i],
                      i < from_orients.size() && from_orients[i] == '-',
                      i < to_ids.size() ? to_ids[i] : 0,
                      i < to_orients.size() && to_orients[i] == '-',
                      i < distances.size() ? distances[i] : empty,
                      i < rest_fields.size() ? rest_fields[i] : empty);
    }
  }

  if (data.num_containments > 0) {
    const std::vector<uint32_t> container_ids =
        Codec::decompress_delta_varint_uint32(
            data.containment_container_ids_zstd, data.num_containments);
    const std::vector<uint32_t> contained_ids =
        Codec::decompress_delta_varint_uint32(
            data.containment_contained_ids_zstd, data.num_containments);
    const std::vector<char> container_orients = Codec::decompress_orientations(
        data.containment_container_orients_zstd, data.num_containments);
    const std::vector<char> contained_orients = Codec::decompress_orientations(
        data.containment_contained_orients_zstd, data.num_containments);
    const std::vector<uint32_t> positions =
        Codec::zstd_decompress_uint32_vector(data.containment_positions_zstd);
    const std::vector<std::string> overlaps = decompress_string_column(
        data.containment_overlaps_zstd, data.containment_overlap_lengths_zstd);
    const std::vector<std::string> rest_fields = decompress_string_column(
        data.containment_rest_fields_zstd, data.containment_rest_lengths_zstd);
    for (size_t i = 0; i < container_ids.size(); ++i) {
      const std::string empty;
      visitor.on_containment(
          container_ids[i],
          i < container_orients.size() && container_orients[i] == '-',
          i < contained_ids.size() ? contained_ids[i] : 0,
          i < contained_orients.size() && contained_orients[i] == '-',
          i < positions.size() ? positions[i] : 0,
          i < overlaps.size() ? overlaps[i] : empty,
          i < rest_fields.size() ? rest_fields[i] : empty);
    }
  }

  const uint32_t min_rule_id = data.min_rule_id();
  const GfaTagList no_tags;
  decode_sequence_batches(
      paths_flat, path_offsets, original_path_offsets, rules_first,
      rules_second, min_rule_id, data.delta_round, data.sequence_lengths.size(),
      num_threads, [&](size_t index, const std::vector<NodeId> &visits) {
        const std::string fallback_name = std::to_string(index);
        const std::string empty;
        visitor.on_path(
            index < path_names.size() ? path_names[index] : fallback_name,
            visits, index < path_overlaps.size() ? path_overlaps[index] : empty,
            no_tags);
      });

  decode_sequence_batches(
      walks_flat, walk_offsets, original_walk_offsets, rules_first,
      rules_second, min_rule_id, data.delta_round, data.walk_lengths.size(),
      num_threads, [&](size_t index, const std::vector<NodeId> &visits) {
        const std::string default_sample = "sample";
        const std::string default_sequence = "unknown";
        visitor.on_walk(
            index < walk_sample_ids.size() ? walk_sample_ids[index]
                                           : default_sample,
            index < walk_hap_indices.size() ? walk_hap_indices[index] : 0,
            index < walk_sequence_ids.size() ? walk_sequence_ids[index]
                                             : default_sequence,
            index < walk_sequence_starts.size() ? walk_sequence_starts[index]
                                                : -1,
            index < walk_sequence_ends.size() ? walk_sequence_ends[index] : -1,
            visits, no_tags);
      });
}

void decode_gfa_records(std::istream &input, GfaRecordVisitor &visitor,
                        int num_threads) {
  decode_gfa_records(deserialize_compressed_data(input), visitor, num_threads);
}

void decode_gfa_records(const std::string &input_path,
                        GfaRecordVisitor &visitor, int num_threads) {
  decode_gfa_records(deserialize_compressed_data(input_path), visitor,
                     num_threads);
}

} // namespace gfaz
