#include "compress/io/gfa_writer.hpp"
#include "compress/io/gfa_write_utils.hpp"
#include "core/codec/codec.hpp"
#include "core/utils/debug_log.hpp"
#include "core/utils/runtime_utils.hpp"
#include "core/utils/threading_utils.hpp"
#include "compress/decompression_debug.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string_view>
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

inline void prefetch_rule(int32_t node, const std::vector<int32_t> &first,
                          const std::vector<int32_t> &second, uint32_t min_id,
                          uint32_t max_id) {
  const uint32_t abs_id = static_cast<uint32_t>(std::abs(node));
  if (abs_id >= min_id && abs_id < max_id) {
    __builtin_prefetch(&first[abs_id - min_id]);
    __builtin_prefetch(&second[abs_id - min_id]);
  }
}

template <typename Emit>
void expand_rule_emit(uint32_t rule_id, bool reverse,
                      const std::vector<int32_t> &first,
                      const std::vector<int32_t> &second, uint32_t min_id,
                      uint32_t max_id, Emit &emit) {
  const uint32_t idx = rule_id - min_id;
  const int32_t a = first[idx];
  const int32_t b = second[idx];

  // The child expanded second is known now; fetch its rule while the other
  // child is expanded.
  prefetch_rule(reverse ? a : b, first, second, min_id, max_id);

  if (!reverse) {
    const uint32_t abs_a = static_cast<uint32_t>(std::abs(a));
    if (abs_a >= min_id && abs_a < max_id)
      expand_rule_emit(abs_a, a < 0, first, second, min_id, max_id, emit);
    else
      emit(a);

    const uint32_t abs_b = static_cast<uint32_t>(std::abs(b));
    if (abs_b >= min_id && abs_b < max_id)
      expand_rule_emit(abs_b, b < 0, first, second, min_id, max_id, emit);
    else
      emit(b);
  } else {
    const uint32_t abs_b = static_cast<uint32_t>(std::abs(b));
    if (abs_b >= min_id && abs_b < max_id)
      expand_rule_emit(abs_b, b >= 0, first, second, min_id, max_id, emit);
    else
      emit(-b);

    const uint32_t abs_a = static_cast<uint32_t>(std::abs(a));
    if (abs_a >= min_id && abs_a < max_id)
      expand_rule_emit(abs_a, a >= 0, first, second, min_id, max_id, emit);
    else
      emit(-a);
  }
}

template <typename Emit>
void decode_sequence_at_index(const std::vector<int32_t> &flat,
                              const SequenceOffsets &compressed_offsets,
                              size_t index,
                              const std::vector<int32_t> &rules_first,
                              const std::vector<int32_t> &rules_second,
                              uint32_t min_rule_id, int delta_round,
                              Emit emit) {
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
  std::vector<int64_t> delta_sums(static_cast<size_t>(delta_round), 0);
  auto emit_delta_decoded = [&](gfaz::NodeId node) {
    int64_t decoded = node;
    for (int64_t &sum : delta_sums) {
      sum += decoded;
      decoded = sum;
    }
    emit(static_cast<gfaz::NodeId>(decoded));
  };

  constexpr size_t kPrefetchAhead = 4;
  for (size_t pos = start; pos < end; ++pos) {
    if (pos + kPrefetchAhead < end)
      prefetch_rule(flat[pos + kPrefetchAhead], rules_first, rules_second,
                    min_rule_id, max_rule_id);
    const gfaz::NodeId node = flat[pos];
    const uint32_t abs_id = static_cast<uint32_t>(std::abs(node));
    if (abs_id >= min_rule_id && abs_id < max_rule_id)
      expand_rule_emit(abs_id, node < 0, rules_first, rules_second, min_rule_id,
                       max_rule_id, emit_delta_decoded);
    else
      emit_delta_decoded(node);
  }
}

// One output line built in a buffer that is not zero-filled. `ensure` makes
// room once per step; the `put` calls after it are unchecked.
class LineBuffer {
public:
  LineBuffer() = default;
  explicit LineBuffer(size_t capacity) { grow(capacity); }

  const char *data() const { return data_.get(); }
  size_t size() const { return static_cast<size_t>(cursor_ - data_.get()); }

  void ensure(size_t bytes) {
    if (static_cast<size_t>(limit_ - cursor_) < bytes)
      grow(bytes);
  }
  void put(char c) { *cursor_++ = c; }
  void put_number(uint32_t value) {
    cursor_ = std::to_chars(cursor_, limit_, value).ptr;
  }
  void append(std::string_view text) {
    ensure(text.size());
    std::memcpy(cursor_, text.data(), text.size());
    cursor_ += text.size();
  }
  void append(char c) {
    ensure(1);
    put(c);
  }

private:
  void grow(size_t bytes) {
    const size_t used = size();
    const size_t capacity = static_cast<size_t>(limit_ - data_.get());
    const size_t new_capacity = std::max(used + bytes, capacity * 2);
    std::unique_ptr<char[]> grown(new char[new_capacity]);
    if (used != 0)
      std::memcpy(grown.get(), data_.get(), used);
    data_ = std::move(grown);
    cursor_ = data_.get() + used;
    limit_ = data_.get() + new_capacity;
  }

  std::unique_ptr<char[]> data_;
  char *cursor_ = nullptr;
  char *limit_ = nullptr;
};

// Longest text of one step: orientation, ten digits and a separator.
constexpr size_t kMaxStepBytes = 12;

template <typename Formatter>
void write_sequence_batch_stream(std::ofstream &out, size_t total_count,
                                 int num_threads, Formatter formatter) {
  if (total_count == 0)
    return;

  const int effective_threads = std::max(1, resolve_omp_thread_count(num_threads));
  // Format the next batch concurrently with ordered writes from the current
  // batch. Keeping the original batch width avoids extra OpenMP team setup.
  const size_t batch_size = static_cast<size_t>(effective_threads) * 8;

  auto format_batch = [&](size_t batch_start) {
    const size_t batch_end = std::min(batch_start + batch_size, total_count);
    std::vector<LineBuffer> lines(batch_end - batch_start);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) num_threads(effective_threads)
#endif
    for (size_t i = batch_start; i < batch_end; ++i)
      lines[i - batch_start] = formatter(i);
    return lines;
  };

  size_t batch_start = 0;
  auto pending = std::async(std::launch::async, format_batch, batch_start);
  while (batch_start < total_count) {
    std::vector<LineBuffer> lines = pending.get();
    batch_start += batch_size;
    if (batch_start < total_count)
      pending = std::async(std::launch::async, format_batch, batch_start);

    for (const auto &line : lines)
      out.write(line.data(), static_cast<std::streamsize>(line.size()));
  }
}

constexpr size_t kRowsPerBlock = 16384;

// Writes text rows in order. Block b covers rows
// [block_rows[b], block_rows[b + 1]); `format_block(b, begin, end, buffer)`
// appends them to `buffer`. Blocks are formatted in parallel, a batch at a
// time, and each batch is written while the next one is formatted.
template <typename FormatBlock>
void write_row_blocks_stream(std::ofstream &out,
                             const std::vector<size_t> &block_rows,
                             int num_threads, FormatBlock format_block) {
  if (block_rows.size() < 2)
    return;

  const int effective_threads = std::max(1, resolve_omp_thread_count(num_threads));
  const size_t total_blocks = block_rows.size() - 1;
  const size_t batch_blocks = static_cast<size_t>(effective_threads) * 2;

  auto format_batch = [&](size_t first_block) {
    const size_t end_block = std::min(first_block + batch_blocks, total_blocks);
    std::vector<std::string> buffers(end_block - first_block);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) num_threads(effective_threads)
#endif
    for (size_t block = first_block; block < end_block; ++block)
      format_block(block, block_rows[block], block_rows[block + 1],
                   buffers[block - first_block]);
    return buffers;
  };

  size_t first_block = 0;
  auto pending = std::async(std::launch::async, format_batch, first_block);
  while (first_block < total_blocks) {
    std::vector<std::string> buffers = pending.get();
    first_block += batch_blocks;
    if (first_block < total_blocks)
      pending = std::async(std::launch::async, format_batch, first_block);

    for (const auto &buffer : buffers)
      out.write(buffer.data(), static_cast<std::streamsize>(buffer.size()));
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
    append_optional_fields(line, graph.segments.optional_fields, seg_offsets,
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

    append_optional_fields(line, graph.link_optional_fields, link_offsets, i);
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
  t1 = Clock::now();
  debug_stages.push_back({"build traversal offsets", elapsed_ms(t0, t1)});

  if (!data.header_line.empty())
    out << data.header_line << "\n";

  std::string line;
  line.reserve(4096);

  t0 = Clock::now();
  {
    // Blocks end after kRowsPerBlock rows or kBasesPerBlock bases, whichever
    // comes first, so a batch of blocks stays small for long sequences too.
    // block_starts holds where each block's bases start.
    constexpr size_t kBasesPerBlock = size_t{1} << 20;
    const size_t segment_count = segment_lengths.size();
    std::vector<size_t> block_rows{0};
    std::vector<size_t> block_starts{0};
    size_t total_bases = 0;
    for (size_t i = 0; i < segment_count; ++i) {
      total_bases += segment_lengths[i];
      if (i + 1 == segment_count ||
          i + 1 - block_rows.back() >= kRowsPerBlock ||
          total_bases - block_starts.back() >= kBasesPerBlock) {
        block_rows.push_back(i + 1);
        block_starts.push_back(total_bases);
      }
    }
    if (total_bases > segment_sequences.size()) {
      throw std::runtime_error(std::string(kWriterErrorPrefix) +
                               "segment sequence column is truncated");
    }
    const auto format_segments =
        [&](size_t block, size_t begin, size_t end, std::string &buffer) {
          size_t offset = block_starts[block];
          buffer.reserve((block_starts[block + 1] - offset) +
                         (end - begin) * 48);
          for (size_t i = begin; i < end; ++i) {
            buffer += "S\t";
            append_numeric_node_name(buffer, static_cast<uint32_t>(i + 1));
            buffer += '\t';
            buffer.append(segment_sequences, offset, segment_lengths[i]);
            offset += segment_lengths[i];
            append_optional_fields(buffer, segment_optional_fields,
                                   seg_offsets, i);
            buffer += '\n';
          }
        };
    // Long sequences are a plain copy that threads do not speed up; the
    // per-block buffers would only add a second copy through fresh memory.
    constexpr size_t kLongAverageBases = 512;
    if (total_bases / std::max<size_t>(segment_count, 1) >= kLongAverageBases) {
      for (size_t block = 0; block + 1 < block_rows.size(); ++block) {
        line.clear();
        format_segments(block, block_rows[block], block_rows[block + 1], line);
        out.write(line.data(), static_cast<std::streamsize>(line.size()));
      }
    } else {
      write_row_blocks_stream(out, block_rows, num_threads, format_segments);
    }
  }
  t1 = Clock::now();
  debug_stages.push_back({"stream segments", elapsed_ms(t0, t1)});

  t0 = Clock::now();
  std::vector<size_t> link_block_rows;
  for (size_t row = 0; row < link_from_ids.size(); row += kRowsPerBlock)
    link_block_rows.push_back(row);
  link_block_rows.push_back(link_from_ids.size());
  write_row_blocks_stream(
      out, link_block_rows, num_threads,
      [&](size_t, size_t begin, size_t end, std::string &buffer) {
        buffer.reserve((end - begin) * 48);
        for (size_t i = begin; i < end; ++i) {
          buffer += "L\t";
          append_numeric_node_name(buffer, link_from_ids[i]);
          buffer += '\t';
          buffer += link_from_orients[i];
          buffer += '\t';
          append_numeric_node_name(buffer, link_to_ids[i]);
          buffer += '\t';
          buffer += link_to_orients[i];
          buffer += '\t';
          if (i < link_overlap_ops.size() && link_overlap_ops[i] != '\0') {
            append_numeric_node_name(buffer, link_overlap_nums[i]);
            buffer += link_overlap_ops[i];
          } else {
            buffer += '*';
          }
          append_optional_fields(buffer, link_optional_fields, link_offsets,
                                 i);
          buffer += '\n';
        }
      });
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
                                const std::string &name =
                                    (index < path_names.size()) ? path_names[index]
                                                                : std::to_string(index);
                                const std::string &overlap =
                                    (index < path_overlaps.size())
                                        ? path_overlaps[index]
                                        : "";
                                const size_t original_length =
                                    index < data.original_path_lengths.size()
                                        ? data.original_path_lengths[index]
                                        : 0;
                                LineBuffer path_line(
                                    name.size() + overlap.size() +
                                    original_length * 10 + 8);
                                path_line.append("P\t");
                                path_line.append(name);
                                path_line.append('\t');
                                bool first_node = true;
                                decode_sequence_at_index(
                                    paths_flat, path_offsets, index, rules_first,
                                    rules_second, min_rule_id, data.delta_round,
                                    [&](gfaz::NodeId node) {
                                      path_line.ensure(kMaxStepBytes);
                                      if (!first_node)
                                        path_line.put(',');
                                      first_node = false;
                                      const bool reverse = node < 0;
                                      path_line.put_number(static_cast<uint32_t>(
                                          reverse ? -node : node));
                                      path_line.put(reverse ? '-' : '+');
                                    });
                                path_line.append('\t');
                                path_line.append(overlap.empty() ? "*" : overlap);
                                path_line.append('\n');
                                return path_line;
                              });
  t1 = Clock::now();
  debug_stages.push_back({"expand and stream paths", elapsed_ms(t0, t1)});
  log_cpu_decompression_memory_checkpoint("CPU Direct-Writer",
                                          "after path streaming");

  t0 = Clock::now();
  write_sequence_batch_stream(out, data.walk_lengths.size(), num_threads,
                              [&](size_t index) {
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
                                const size_t original_length =
                                    index < data.original_walk_lengths.size()
                                        ? data.original_walk_lengths[index]
                                        : 0;
                                LineBuffer walk_line(
                                    sample_id.size() + seq_id.size() +
                                    original_length * 10 + 80);
                                walk_line.append("W\t");
                                walk_line.append(sample_id);
                                walk_line.append('\t');
                                walk_line.append(std::to_string(hap_index));
                                walk_line.append('\t');
                                walk_line.append(seq_id);
                                walk_line.append('\t');
                                walk_line.append(seq_start >= 0
                                                     ? std::to_string(seq_start)
                                                     : "*");
                                walk_line.append('\t');
                                walk_line.append(seq_end >= 0
                                                     ? std::to_string(seq_end)
                                                     : "*");
                                walk_line.append('\t');
                                decode_sequence_at_index(
                                    walks_flat, walk_offsets, index, rules_first,
                                    rules_second, min_rule_id, data.delta_round,
                                    [&](gfaz::NodeId node) {
                                      const bool reverse = node < 0;
                                      walk_line.ensure(kMaxStepBytes);
                                      walk_line.put(reverse ? '<' : '>');
                                      walk_line.put_number(static_cast<uint32_t>(
                                          reverse ? -node : node));
                                    });
                                walk_line.append('\n');
                                return walk_line;
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
