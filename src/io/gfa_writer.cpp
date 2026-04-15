#include "io/gfa_writer.hpp"
#include "io/gfa_write_utils.hpp"
#include "codec/codec.hpp"
#include "utils/debug_log.hpp"
#include "utils/runtime_utils.hpp"
#include "utils/threading_utils.hpp"
#include "workflows/decompression_debug.hpp"

#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

constexpr const char *kWriterErrorPrefix = "GFA writer error: ";
using Clock = std::chrono::high_resolution_clock;
using gfaz::runtime_utils::elapsed_ms;
using namespace gfaz::decompression_debug;
using namespace gfaz::gfa_write_utils;

// Get node name or numeric ID as string
inline void append_node_name(std::string &out, uint32_t node_id,
                             const std::vector<std::string> &names) {
  if (node_id < names.size() && !names[node_id].empty())
    out += names[node_id];
  else
    out += std::to_string(node_id);
}

void expand_rule(uint32_t rule_id, bool reverse,
                 const std::vector<int32_t> &first,
                 const std::vector<int32_t> &second, uint32_t min_id,
                 uint32_t max_id, std::vector<gfaz::NodeId> &out) {
  const uint32_t idx = rule_id - min_id;
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

std::vector<gfaz::NodeId>
decode_sequence_at_index(const std::vector<int32_t> &flat,
                         const SequenceOffsets &compressed_offsets,
                         const SequenceOffsets &original_offsets, size_t index,
                         const std::vector<int32_t> &rules_first,
                         const std::vector<int32_t> &rules_second,
                         uint32_t min_rule_id, int delta_round) {
  if (index + 1 >= compressed_offsets.size()) {
    throw std::out_of_range(std::string(kWriterErrorPrefix) +
                            "sequence index out of range");
  }

  const size_t start = compressed_offsets[index];
  const size_t end = compressed_offsets[index + 1];
  if (end > flat.size()) {
    throw std::runtime_error(std::string(kWriterErrorPrefix) +
                             "flattened sequence block is truncated");
  }

  const uint32_t max_rule_id =
      min_rule_id + static_cast<uint32_t>(rules_first.size());
  const size_t original_length =
      (index + 1 < original_offsets.size())
          ? (original_offsets[index + 1] - original_offsets[index])
          : (end - start);

  std::vector<gfaz::NodeId> decoded;
  decoded.reserve(original_length);

  for (size_t pos = start; pos < end; ++pos) {
    const gfaz::NodeId node = flat[pos];
    const uint32_t abs_id = static_cast<uint32_t>(std::abs(node));
    if (abs_id >= min_rule_id && abs_id < max_rule_id)
      expand_rule(abs_id, node < 0, rules_first, rules_second, min_rule_id,
                  max_rule_id, decoded);
    else
      decoded.push_back(node);
  }

  std::vector<std::vector<gfaz::NodeId>> seqs(1);
  seqs[0] = std::move(decoded);
  for (int round = 0; round < delta_round; ++round)
    gfaz::Codec::inverse_delta_transform(seqs);
  return std::move(seqs[0]);
}

template <typename Formatter>
void write_sequence_batch_stream(std::ofstream &out, size_t total_count,
                                 int num_threads, Formatter formatter) {
  if (total_count == 0)
    return;

  const int effective_threads = std::max(1, resolve_omp_thread_count(num_threads));
  const size_t batch_size = static_cast<size_t>(effective_threads) * 8;

  for (size_t batch_start = 0; batch_start < total_count;
       batch_start += batch_size) {
    const size_t batch_end = std::min(batch_start + batch_size, total_count);
    std::vector<std::string> lines(batch_end - batch_start);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (size_t i = batch_start; i < batch_end; ++i)
      lines[i - batch_start] = formatter(i);

    for (const auto &line : lines)
      out.write(line.data(), static_cast<std::streamsize>(line.size()));
  }
}

} // namespace

void write_gfa(const gfaz::GfaGraph &graph, const std::string &output_path) {
  std::ofstream out(output_path);
  if (!out)
    throw std::runtime_error(std::string(kWriterErrorPrefix) +
                             "failed to open output file: " + output_path);

  GFAZ_LOG("Writing GFA to " << output_path << "...");

  // Precompute string offsets for optional fields
  const FieldOffsets seg_offsets =
      build_field_offsets(graph.segments.optional_fields);
  const FieldOffsets link_offsets =
      build_field_offsets(graph.link_optional_fields);

  std::string line;
  line.reserve(4096);

  // H (Header)
  if (!graph.header_line.empty())
    out << graph.header_line << "\n";

  // S (Segments) - index 0 is placeholder, segments start at 1
  size_t num_segments = graph.segments.node_sequences.size();
  for (size_t i = 1; i < num_segments; ++i) {
    line.clear();
    line += "S\t";
    append_node_name(line, i, graph.segments.node_id_to_name);
    line += '\t';
    line += graph.segments.node_sequences[i];
    line += format_optional_fields(graph.segments.optional_fields, seg_offsets,
                                   i - 1);
    line += '\n';
    out.write(line.data(), line.size());

    if (i % 10000000 == 0) {
      GFAZ_LOG("  Segments: " << i << "/" << (num_segments - 1));
      out.flush();
    }
  }

  // L (Links)
  size_t num_links = graph.links.from_ids.size();
  for (size_t i = 0; i < num_links; ++i) {
    line.clear();
    line += "L\t";

    append_node_name(line, graph.links.from_ids[i], graph.segments.node_id_to_name);
    line += '\t';
    line += graph.links.from_orients[i];
    line += '\t';

    append_node_name(line, graph.links.to_ids[i], graph.segments.node_id_to_name);
    line += '\t';
    line += graph.links.to_orients[i];
    line += '\t';

    if (graph.links.overlap_ops[i] != '\0') {
      line += std::to_string(graph.links.overlap_nums[i]);
      line += graph.links.overlap_ops[i];
    } else {
      line += '*';
    }

    line += format_optional_fields(graph.link_optional_fields, link_offsets, i);
    line += '\n';
    out.write(line.data(), line.size());

    if ((i + 1) % 10000000 == 0) {
      GFAZ_LOG("  Links: " << (i + 1) << "/" << num_links);
      out.flush();
    }
  }

  // P (Paths)
  for (size_t p = 0; p < graph.paths_data.traversals.size(); ++p) {
    line.clear();
    line += "P\t";
    line += graph.paths_data.names[p];
    line += '\t';

    const auto &path = graph.paths_data.traversals[p];
    for (size_t n = 0; n < path.size(); ++n) {
      if (n > 0)
        line += ',';

      gfaz::NodeId node = path[n];
      bool is_reverse = (node < 0);
      uint32_t node_id = is_reverse ? static_cast<uint32_t>(-node)
                                    : static_cast<uint32_t>(node);
      append_node_name(line, node_id, graph.segments.node_id_to_name);
      line += (is_reverse ? '-' : '+');
    }

    line += '\t';
    if (p < graph.paths_data.overlaps.size() && !graph.paths_data.overlaps[p].empty())
      line += graph.paths_data.overlaps[p];
    else
      line += '*';
    line += '\n';

    out.write(line.data(), line.size());
  }

  // W (Walks)
  for (size_t w = 0; w < graph.walks.walks.size(); ++w) {
    line.clear();
    line += "W\t";

    // Sample ID
    if (w < graph.walks.sample_ids.size())
      line += graph.walks.sample_ids[w];
    else
      line += "sample";
    line += '\t';

    // Haplotype index
    if (w < graph.walks.hap_indices.size())
      line += std::to_string(graph.walks.hap_indices[w]);
    else
      line += "0";
    line += '\t';

    // Sequence ID
    if (w < graph.walks.seq_ids.size())
      line += graph.walks.seq_ids[w];
    else
      line += "unknown";
    line += '\t';

    // Sequence start
    if (w < graph.walks.seq_starts.size() && graph.walks.seq_starts[w] >= 0)
      line += std::to_string(graph.walks.seq_starts[w]);
    else
      line += '*';
    line += '\t';

    // Sequence end
    if (w < graph.walks.seq_ends.size() && graph.walks.seq_ends[w] >= 0)
      line += std::to_string(graph.walks.seq_ends[w]);
    else
      line += '*';
    line += '\t';

    // Walk nodes (>node or <node format)
    const auto &walk = graph.walks.walks[w];
    for (size_t n = 0; n < walk.size(); ++n) {
      gfaz::NodeId node = walk[n];
      bool is_reverse = (node < 0);
      uint32_t node_id = is_reverse ? static_cast<uint32_t>(-node)
                                    : static_cast<uint32_t>(node);
      line += (is_reverse ? '<' : '>');
      append_node_name(line, node_id, graph.segments.node_id_to_name);
    }
    line += '\n';

    out.write(line.data(), line.size());

    if ((w + 1) % 10000 == 0) {
      GFAZ_LOG("  Walks: " << (w + 1) << "/" << graph.walks.walks.size());
      out.flush();
    }
  }

  // J (Jump) lines
  for (size_t i = 0; i < graph.jumps.size(); ++i) {
    line.clear();
    line += "J\t";

    append_node_name(line, graph.jumps.from_ids[i], graph.segments.node_id_to_name);
    line += '\t';
    line += graph.jumps.from_orients[i];
    line += '\t';

    append_node_name(line, graph.jumps.to_ids[i], graph.segments.node_id_to_name);
    line += '\t';
    line += graph.jumps.to_orients[i];
    line += '\t';

    line += graph.jumps.distances[i];

    if (i < graph.jumps.rest_fields.size() &&
        !graph.jumps.rest_fields[i].empty()) {
      line += '\t';
      line += graph.jumps.rest_fields[i];
    }
    line += '\n';
    out.write(line.data(), line.size());
  }

  // C (Containment) lines
  for (size_t i = 0; i < graph.containments.size(); ++i) {
    line.clear();
    line += "C\t";

    append_node_name(line, graph.containments.container_ids[i],
                     graph.segments.node_id_to_name);
    line += '\t';
    line += graph.containments.container_orients[i];
    line += '\t';

    append_node_name(line, graph.containments.contained_ids[i],
                     graph.segments.node_id_to_name);
    line += '\t';
    line += graph.containments.contained_orients[i];
    line += '\t';

    line += std::to_string(graph.containments.positions[i]);
    line += '\t';
    line += graph.containments.overlaps[i];

    if (i < graph.containments.rest_fields.size() &&
        !graph.containments.rest_fields[i].empty()) {
      line += '\t';
      line += graph.containments.rest_fields[i];
    }
    line += '\n';
    out.write(line.data(), line.size());
  }

  out.close();

  // Report file size
  std::ifstream check(output_path, std::ios::binary | std::ios::ate);
  size_t file_size = check.tellg();
  GFAZ_LOG("Wrote GFA file: " << output_path << " (" << file_size << " bytes)");
}

void write_gfa_from_compressed_data(const gfaz::CompressedData &data,
                                    const std::string &output_path,
                                    int num_threads) {
  ScopedOMPThreads omp_scope(num_threads);
  std::vector<TimedDebugStage> debug_stages;
  const auto writer_total_start = Clock::now();

  std::ofstream out(output_path);
  if (!out) {
    throw std::runtime_error(std::string(kWriterErrorPrefix) +
                             "failed to open output file: " + output_path);
  }

  GFAZ_LOG("Writing GFA directly from compressed data to " << output_path
                                                          << "...");

  std::vector<int32_t> rules_first;
  std::vector<int32_t> rules_second;
  std::vector<int32_t> paths_flat;
  std::vector<int32_t> walks_flat;
  std::vector<std::string> path_names;
  std::vector<std::string> path_overlaps;
  std::vector<std::string> walk_sample_ids;
  std::vector<uint32_t> walk_hap_indices;
  std::vector<std::string> walk_seq_ids;
  std::vector<int64_t> walk_seq_starts;
  std::vector<int64_t> walk_seq_ends;
  std::string segment_sequences;
  std::vector<uint32_t> segment_lengths;
  std::vector<uint32_t> link_from_ids;
  std::vector<uint32_t> link_to_ids;
  std::vector<char> link_from_orients;
  std::vector<char> link_to_orients;
  std::vector<uint32_t> link_overlap_nums;
  std::vector<char> link_overlap_ops;
  auto t0 = Clock::now();

#ifdef _OPENMP
#pragma omp parallel sections
  {
#pragma omp section
    {
      auto decoded_rules = decode_rules(data);
      rules_first = std::move(decoded_rules.first);
      rules_second = std::move(decoded_rules.second);
    }
#pragma omp section
    {
      if (!data.paths_zstd.payload.empty())
        paths_flat = gfaz::Codec::zstd_decompress_int32_vector(data.paths_zstd);
    }
#pragma omp section
    {
      if (!data.walks_zstd.payload.empty())
        walks_flat = gfaz::Codec::zstd_decompress_int32_vector(data.walks_zstd);
    }
#pragma omp section
    {
      path_names = decompress_string_column(data.names_zstd, data.name_lengths_zstd);
      path_overlaps =
          decompress_string_column(data.overlaps_zstd, data.overlap_lengths_zstd);
    }
#pragma omp section
    {
      walk_sample_ids = decompress_string_column(data.walk_sample_ids_zstd,
                                                 data.walk_sample_id_lengths_zstd);
      walk_hap_indices =
          gfaz::Codec::zstd_decompress_uint32_vector(data.walk_hap_indices_zstd);
      walk_seq_ids = decompress_string_column(data.walk_seq_ids_zstd,
                                              data.walk_seq_id_lengths_zstd);
      walk_seq_starts = gfaz::Codec::decompress_varint_int64(
          data.walk_seq_starts_zstd, data.walk_lengths.size());
      walk_seq_ends =
          gfaz::Codec::decompress_varint_int64(data.walk_seq_ends_zstd,
                                        data.walk_lengths.size());
    }
#pragma omp section
    {
      segment_sequences =
          gfaz::Codec::zstd_decompress_string(data.segment_sequences_zstd);
      segment_lengths =
          gfaz::Codec::zstd_decompress_uint32_vector(data.segment_seq_lengths_zstd);
    }
#pragma omp section

    {
      link_from_ids =
          gfaz::Codec::decompress_delta_varint_uint32(data.link_from_ids_zstd,
                                                data.num_links);
      link_to_ids =
          gfaz::Codec::decompress_delta_varint_uint32(data.link_to_ids_zstd,
                                                data.num_links);
      link_from_orients = gfaz::Codec::decompress_orientations(
          data.link_from_orients_zstd, data.num_links);
      link_to_orients =
          gfaz::Codec::decompress_orientations(data.link_to_orients_zstd, data.num_links);
      link_overlap_nums =
          gfaz::Codec::zstd_decompress_uint32_vector(data.link_overlap_nums_zstd);
      link_overlap_ops =
          gfaz::Codec::zstd_decompress_char_vector(data.link_overlap_ops_zstd);
    }
  }
#else
  auto decoded_rules = decode_rules(data);
  rules_first = std::move(decoded_rules.first);
  rules_second = std::move(decoded_rules.second);
  if (!data.paths_zstd.payload.empty())
    paths_flat = gfaz::Codec::zstd_decompress_int32_vector(data.paths_zstd);
  if (!data.walks_zstd.payload.empty())
    walks_flat = gfaz::Codec::zstd_decompress_int32_vector(data.walks_zstd);
  path_names = decompress_string_column(data.names_zstd, data.name_lengths_zstd);
  path_overlaps =
      decompress_string_column(data.overlaps_zstd, data.overlap_lengths_zstd);
  walk_sample_ids = decompress_string_column(data.walk_sample_ids_zstd,
                                             data.walk_sample_id_lengths_zstd);
  walk_hap_indices =
      gfaz::Codec::zstd_decompress_uint32_vector(data.walk_hap_indices_zstd);
  walk_seq_ids =
      decompress_string_column(data.walk_seq_ids_zstd, data.walk_seq_id_lengths_zstd);
  walk_seq_starts = gfaz::Codec::decompress_varint_int64(data.walk_seq_starts_zstd,
                                                   data.walk_lengths.size());
  walk_seq_ends = gfaz::Codec::decompress_varint_int64(data.walk_seq_ends_zstd,
                                                 data.walk_lengths.size());
  segment_sequences = gfaz::Codec::zstd_decompress_string(data.segment_sequences_zstd);
  segment_lengths =
      gfaz::Codec::zstd_decompress_uint32_vector(data.segment_seq_lengths_zstd);
  link_from_ids =
      gfaz::Codec::decompress_delta_varint_uint32(data.link_from_ids_zstd,
                                            data.num_links);
  link_to_ids = gfaz::Codec::decompress_delta_varint_uint32(data.link_to_ids_zstd,
                                                      data.num_links);
  link_from_orients =
      gfaz::Codec::decompress_orientations(data.link_from_orients_zstd, data.num_links);
  link_to_orients =
      gfaz::Codec::decompress_orientations(data.link_to_orients_zstd, data.num_links);
  link_overlap_nums =
      gfaz::Codec::zstd_decompress_uint32_vector(data.link_overlap_nums_zstd);
  link_overlap_ops =
      gfaz::Codec::zstd_decompress_char_vector(data.link_overlap_ops_zstd);
#endif
  auto t1 = Clock::now();
  debug_stages.push_back({"decode core columns", elapsed_ms(t0, t1)});

  t0 = Clock::now();
  std::vector<gfaz::OptionalFieldColumn> segment_optional_fields;
  segment_optional_fields.reserve(data.segment_optional_fields_zstd.size());
  for (const auto &c : data.segment_optional_fields_zstd)
    segment_optional_fields.push_back(decompress_optional_column(c));

  std::vector<gfaz::OptionalFieldColumn> link_optional_fields;
  link_optional_fields.reserve(data.link_optional_fields_zstd.size());
  for (const auto &c : data.link_optional_fields_zstd)
    link_optional_fields.push_back(decompress_optional_column(c));
  t1 = Clock::now();
  debug_stages.push_back({"decode optional columns", elapsed_ms(t0, t1)});

  std::vector<uint32_t> jump_from_ids;
  std::vector<uint32_t> jump_to_ids;
  std::vector<char> jump_from_orients;
  std::vector<char> jump_to_orients;
  std::vector<std::string> jump_distances;
  std::vector<std::string> jump_rest_fields;
  if (data.num_jumps > 0) {
    t0 = Clock::now();
    jump_from_ids = gfaz::Codec::decompress_delta_varint_uint32(data.jump_from_ids_zstd,
                                                          data.num_jumps);
    jump_to_ids = gfaz::Codec::decompress_delta_varint_uint32(data.jump_to_ids_zstd,
                                                        data.num_jumps);
    jump_from_orients = gfaz::Codec::decompress_orientations(
        data.jump_from_orients_zstd, data.num_jumps);
    jump_to_orients =
        gfaz::Codec::decompress_orientations(data.jump_to_orients_zstd, data.num_jumps);
    jump_distances = decompress_string_column(data.jump_distances_zstd,
                                              data.jump_distance_lengths_zstd);
    jump_rest_fields = decompress_string_column(data.jump_rest_fields_zstd,
                                                data.jump_rest_lengths_zstd);
    t1 = Clock::now();
    debug_stages.push_back({"decode jump fields", elapsed_ms(t0, t1)});
  }

  std::vector<uint32_t> containment_container_ids;
  std::vector<uint32_t> containment_contained_ids;
  std::vector<char> containment_container_orients;
  std::vector<char> containment_contained_orients;
  std::vector<uint32_t> containment_positions;
  std::vector<std::string> containment_overlaps;
  std::vector<std::string> containment_rest_fields;
  if (data.num_containments > 0) {
    t0 = Clock::now();
    containment_container_ids = gfaz::Codec::decompress_delta_varint_uint32(
        data.containment_container_ids_zstd, data.num_containments);
    containment_contained_ids = gfaz::Codec::decompress_delta_varint_uint32(
        data.containment_contained_ids_zstd, data.num_containments);
    containment_container_orients = gfaz::Codec::decompress_orientations(
        data.containment_container_orients_zstd, data.num_containments);
    containment_contained_orients = gfaz::Codec::decompress_orientations(
        data.containment_contained_orients_zstd, data.num_containments);
    containment_positions =
        gfaz::Codec::zstd_decompress_uint32_vector(data.containment_positions_zstd);
    containment_overlaps = decompress_string_column(
        data.containment_overlaps_zstd, data.containment_overlap_lengths_zstd);
    containment_rest_fields = decompress_string_column(
        data.containment_rest_fields_zstd, data.containment_rest_lengths_zstd);
    t1 = Clock::now();
    debug_stages.push_back({"decode containment fields", elapsed_ms(t0, t1)});
  }
  log_cpu_decompression_memory_checkpoint("CPU Direct-Writer",
                                          "after field decode");

  t0 = Clock::now();
  const FieldOffsets seg_offsets = build_field_offsets(segment_optional_fields);
  const FieldOffsets link_offsets = build_field_offsets(link_optional_fields);
  const SequenceOffsets path_offsets = build_offsets(data.sequence_lengths);
  const SequenceOffsets walk_offsets = build_offsets(data.walk_lengths);
  const SequenceOffsets original_path_offsets =
      build_offsets(data.original_path_lengths);
  const SequenceOffsets original_walk_offsets =
      build_offsets(data.original_walk_lengths);
  t1 = Clock::now();
  debug_stages.push_back({"build traversal offsets", elapsed_ms(t0, t1)});

  if (!data.header_line.empty())
    out << data.header_line << "\n";

  std::string line;
  line.reserve(4096);

  t0 = Clock::now();
  size_t segment_seq_offset = 0;
  for (size_t i = 0; i < segment_lengths.size(); ++i) {
    line.clear();
    line += "S\t";
    append_numeric_node_name(line, static_cast<uint32_t>(i + 1));
    line += '\t';

    const uint32_t len = segment_lengths[i];
    if (segment_seq_offset + len > segment_sequences.size()) {
      throw std::runtime_error(std::string(kWriterErrorPrefix) +
                               "segment sequence column is truncated");
    }
    line.append(segment_sequences, segment_seq_offset, len);
    segment_seq_offset += len;

    line += format_optional_fields(segment_optional_fields, seg_offsets, i);
    line += '\n';
    out.write(line.data(), static_cast<std::streamsize>(line.size()));
  }
  t1 = Clock::now();
  debug_stages.push_back({"stream segments", elapsed_ms(t0, t1)});

  t0 = Clock::now();
  for (size_t i = 0; i < link_from_ids.size(); ++i) {
    line.clear();
    line += "L\t";
    append_numeric_node_name(line, link_from_ids[i]);
    line += '\t';
    line += link_from_orients[i];
    line += '\t';
    append_numeric_node_name(line, link_to_ids[i]);
    line += '\t';
    line += link_to_orients[i];
    line += '\t';
    if (i < link_overlap_ops.size() && link_overlap_ops[i] != '\0') {
      line += std::to_string(link_overlap_nums[i]);
      line += link_overlap_ops[i];
    } else {
      line += '*';
    }
    line += format_optional_fields(link_optional_fields, link_offsets, i);
    line += '\n';
    out.write(line.data(), static_cast<std::streamsize>(line.size()));
  }
  t1 = Clock::now();
  debug_stages.push_back({"stream links", elapsed_ms(t0, t1)});

  t0 = Clock::now();
  for (size_t i = 0; i < jump_from_ids.size(); ++i) {
    line.clear();
    line += "J\t";
    append_numeric_node_name(line, jump_from_ids[i]);
    line += '\t';
    line += jump_from_orients[i];
    line += '\t';
    append_numeric_node_name(line, jump_to_ids[i]);
    line += '\t';
    line += jump_to_orients[i];
    line += '\t';
    line += (i < jump_distances.size()) ? jump_distances[i] : "*";
    if (i < jump_rest_fields.size() && !jump_rest_fields[i].empty()) {
      line += '\t';
      line += jump_rest_fields[i];
    }
    line += '\n';
    out.write(line.data(), static_cast<std::streamsize>(line.size()));
  }
  t1 = Clock::now();
  if (!jump_from_ids.empty())
    debug_stages.push_back({"stream jumps", elapsed_ms(t0, t1)});

  t0 = Clock::now();
  for (size_t i = 0; i < containment_container_ids.size(); ++i) {
    line.clear();
    line += "C\t";
    append_numeric_node_name(line, containment_container_ids[i]);
    line += '\t';
    line += containment_container_orients[i];
    line += '\t';
    append_numeric_node_name(line, containment_contained_ids[i]);
    line += '\t';
    line += containment_contained_orients[i];
    line += '\t';
    line += std::to_string(containment_positions[i]);
    line += '\t';
    line += (i < containment_overlaps.size()) ? containment_overlaps[i] : "*";
    if (i < containment_rest_fields.size() &&
        !containment_rest_fields[i].empty()) {
      line += '\t';
      line += containment_rest_fields[i];
    }
    line += '\n';
    out.write(line.data(), static_cast<std::streamsize>(line.size()));
  }
  t1 = Clock::now();
  if (!containment_container_ids.empty())
    debug_stages.push_back({"stream containments", elapsed_ms(t0, t1)});

  const uint32_t min_rule_id = data.min_rule_id();

  t0 = Clock::now();
  write_sequence_batch_stream(out, data.sequence_lengths.size(), num_threads,
                              [&](size_t index) {
                                const std::vector<gfaz::NodeId> path =
                                    decode_sequence_at_index(
                                        paths_flat, path_offsets,
                                        original_path_offsets, index,
                                        rules_first, rules_second, min_rule_id,
                                        data.delta_round);
                                const std::string &name =
                                    (index < path_names.size()) ? path_names[index]
                                                                : std::to_string(index);
                                const std::string &overlap =
                                    (index < path_overlaps.size())
                                        ? path_overlaps[index]
                                        : "";
                                return format_path_line_numeric(
                                    name, path.data(), path.size(), overlap);
                              });
  t1 = Clock::now();
  debug_stages.push_back({"expand and stream paths", elapsed_ms(t0, t1)});
  log_cpu_decompression_memory_checkpoint("CPU Direct-Writer",
                                          "after path streaming");

  t0 = Clock::now();
  write_sequence_batch_stream(out, data.walk_lengths.size(), num_threads,
                              [&](size_t index) {
                                const std::vector<gfaz::NodeId> walk =
                                    decode_sequence_at_index(
                                        walks_flat, walk_offsets,
                                        original_walk_offsets, index,
                                        rules_first, rules_second, min_rule_id,
                                        data.delta_round);
                                const std::string &sample_id =
                                    (index < walk_sample_ids.size())
                                        ? walk_sample_ids[index]
                                        : "sample";
                                const uint32_t hap_index =
                                    (index < walk_hap_indices.size())
                                        ? walk_hap_indices[index]
                                        : 0;
                                const std::string &seq_id =
                                    (index < walk_seq_ids.size())
                                        ? walk_seq_ids[index]
                                        : "unknown";
                                const int64_t seq_start =
                                    (index < walk_seq_starts.size())
                                        ? walk_seq_starts[index]
                                        : -1;
                                const int64_t seq_end =
                                    (index < walk_seq_ends.size())
                                        ? walk_seq_ends[index]
                                        : -1;
                                return format_walk_line_numeric(
                                    sample_id, hap_index, seq_id, seq_start,
                                    seq_end, walk.data(), walk.size());
                              });
  t1 = Clock::now();
  debug_stages.push_back({"expand and stream walks", elapsed_ms(t0, t1)});
  log_cpu_decompression_memory_checkpoint("CPU Direct-Writer",
                                          "after walk streaming");

  out.close();
  log_cpu_decompression_memory_checkpoint("CPU Direct-Writer",
                                          "after output close");

  std::ifstream check(output_path, std::ios::binary | std::ios::ate);
  const size_t file_size = static_cast<size_t>(check.tellg());
  GFAZ_LOG("Wrote GFA file directly from compressed data: " << output_path
                                                            << " (" << file_size
                                                            << " bytes)");

  if (gfaz_debug_enabled()) {
    print_cpu_decompression_summary("CPU Direct-Writer", data);
    print_cpu_decompression_timing(
        {"CPU Direct-Writer", "streaming direct-writer path", debug_stages,
         elapsed_ms(writer_total_start, Clock::now())});
  }
}
