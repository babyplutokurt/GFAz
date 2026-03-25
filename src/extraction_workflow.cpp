#include "extraction_workflow.hpp"

#include "codec.hpp"
#include "threading_utils.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {

constexpr const char *kExtractionErrorPrefix = "Extraction workflow error: ";

void reconstruct_strings(const std::string &concat,
                         const std::vector<uint32_t> &lengths,
                         std::vector<std::string> &out) {
  out.clear();
  out.reserve(lengths.size());

  size_t offset = 0;
  for (uint32_t len : lengths) {
    if (offset + len > concat.size()) {
      throw std::runtime_error(std::string(kExtractionErrorPrefix) +
                               "string column is truncated");
    }
    out.push_back(concat.substr(offset, len));
    offset += len;
  }
}

std::vector<std::string>
decompress_string_column(const ZstdCompressedBlock &strings_zstd,
                         const ZstdCompressedBlock &lengths_zstd) {
  std::vector<std::string> out;
  reconstruct_strings(Codec::zstd_decompress_string(strings_zstd),
                      Codec::zstd_decompress_uint32_vector(lengths_zstd), out);
  return out;
}

size_t compute_offset(const std::vector<uint32_t> &lengths, size_t index) {
  size_t offset = 0;
  for (size_t i = 0; i < index; ++i)
    offset += lengths[i];
  return offset;
}

std::vector<int32_t> extract_encoded_sequence(const ZstdCompressedBlock &block,
                                              const std::vector<uint32_t> &lengths,
                                              size_t index) {
  if (index >= lengths.size()) {
    throw std::out_of_range(std::string(kExtractionErrorPrefix) +
                            "sequence index out of range");
  }

  const std::vector<int32_t> flat = Codec::zstd_decompress_int32_vector(block);
  const size_t offset = compute_offset(lengths, index);
  const size_t length = lengths[index];

  if (offset + length > flat.size()) {
    throw std::runtime_error(std::string(kExtractionErrorPrefix) +
                             "flattened sequence block is truncated");
  }

  return std::vector<int32_t>(flat.begin() + static_cast<std::ptrdiff_t>(offset),
                              flat.begin() + static_cast<std::ptrdiff_t>(offset + length));
}

void expand_rule(uint32_t rule_id, bool reverse,
                 const std::vector<int32_t> &first,
                 const std::vector<int32_t> &second, uint32_t min_id,
                 uint32_t max_id, std::vector<NodeId> &out) {
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

std::vector<NodeId> decode_sequence(const std::vector<int32_t> &encoded,
                                    const std::vector<int32_t> &rules_first,
                                    const std::vector<int32_t> &rules_second,
                                    uint32_t min_rule_id,
                                    uint32_t original_length,
                                    int delta_round) {
  std::vector<NodeId> decoded;
  decoded.reserve(original_length);

  const uint32_t max_rule_id =
      min_rule_id + static_cast<uint32_t>(rules_first.size());

  for (NodeId node : encoded) {
    const uint32_t abs_id = static_cast<uint32_t>(std::abs(node));
    if (abs_id >= min_rule_id && abs_id < max_rule_id) {
      expand_rule(abs_id, node < 0, rules_first, rules_second, min_rule_id,
                  max_rule_id, decoded);
    } else {
      decoded.push_back(node);
    }
  }

  std::vector<std::vector<NodeId>> seqs(1);
  seqs[0] = std::move(decoded);
  for (int i = 0; i < delta_round; ++i)
    Codec::inverse_delta_transform(seqs);
  return std::move(seqs[0]);
}

void append_node_name(std::string &line, uint32_t node_id) {
  line += std::to_string(node_id);
}

std::pair<std::vector<int32_t>, std::vector<int32_t>>
decode_rules(const CompressedData &data) {
  std::vector<int32_t> first = Codec::zstd_decompress_int32_vector(data.rules_first_zstd);
  std::vector<int32_t> second =
      Codec::zstd_decompress_int32_vector(data.rules_second_zstd);
  Codec::delta_decode_int32(first);
  Codec::delta_decode_int32(second);
  return {std::move(first), std::move(second)};
}

std::string format_path_line(const std::string &path_name,
                             const std::vector<NodeId> &path,
                             const std::string &overlap) {
  std::string line = "P\t";
  line += path_name;
  line += '\t';

  for (size_t i = 0; i < path.size(); ++i) {
    if (i > 0)
      line += ',';

    const NodeId node = path[i];
    const bool is_reverse = node < 0;
    append_node_name(line, static_cast<uint32_t>(is_reverse ? -node : node));
    line += (is_reverse ? '-' : '+');
  }

  line += '\t';
  line += overlap.empty() ? "*" : overlap;
  line += '\n';
  return line;
}

std::string format_walk_line(const std::string &sample_id, uint32_t hap_index,
                             const std::string &seq_id, int64_t seq_start,
                             int64_t seq_end,
                             const std::vector<NodeId> &walk) {
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

  for (NodeId node : walk) {
    const bool is_reverse = node < 0;
    line += (is_reverse ? '<' : '>');
    append_node_name(line, static_cast<uint32_t>(is_reverse ? -node : node));
  }

  line += '\n';
  return line;
}

} // namespace

std::string extract_path_line_by_name(const CompressedData &data,
                                      const std::string &path_name,
                                      int num_threads) {
  ScopedOMPThreads omp_scope(num_threads);

  const std::vector<std::string> path_names =
      decompress_string_column(data.names_zstd, data.name_lengths_zstd);
  size_t path_index = path_names.size();
  for (size_t i = 0; i < path_names.size(); ++i) {
    if (path_names[i] == path_name) {
      path_index = i;
      break;
    }
  }

  if (path_index == path_names.size()) {
    throw std::runtime_error(std::string(kExtractionErrorPrefix) +
                             "path not found: " + path_name);
  }

  const std::vector<std::string> overlaps =
      decompress_string_column(data.overlaps_zstd, data.overlap_lengths_zstd);
  const auto [rules_first, rules_second] = decode_rules(data);
  const std::vector<int32_t> encoded =
      extract_encoded_sequence(data.paths_zstd, data.sequence_lengths, path_index);
  const uint32_t original_length =
      (path_index < data.original_path_lengths.size())
          ? data.original_path_lengths[path_index]
          : static_cast<uint32_t>(encoded.size());
  const std::vector<NodeId> decoded =
      decode_sequence(encoded, rules_first, rules_second, data.min_rule_id(),
                      original_length, data.delta_round);

  const std::string overlap =
      (path_index < overlaps.size()) ? overlaps[path_index] : "";
  return format_path_line(path_name, decoded, overlap);
}

std::string extract_walk_line(const CompressedData &data,
                              const std::string &sample_id,
                              uint32_t hap_index,
                              const std::string &seq_id,
                              int64_t seq_start,
                              int64_t seq_end,
                              int num_threads) {
  ScopedOMPThreads omp_scope(num_threads);

  const std::vector<std::string> sample_ids = decompress_string_column(
      data.walk_sample_ids_zstd, data.walk_sample_id_lengths_zstd);
  const std::vector<uint32_t> hap_indices =
      Codec::zstd_decompress_uint32_vector(data.walk_hap_indices_zstd);
  const std::vector<std::string> seq_ids =
      decompress_string_column(data.walk_seq_ids_zstd,
                               data.walk_seq_id_lengths_zstd);
  const std::vector<int64_t> seq_starts =
      Codec::decompress_varint_int64(data.walk_seq_starts_zstd,
                                     data.walk_lengths.size());
  const std::vector<int64_t> seq_ends =
      Codec::decompress_varint_int64(data.walk_seq_ends_zstd,
                                     data.walk_lengths.size());

  size_t walk_index = data.walk_lengths.size();
  for (size_t i = 0; i < data.walk_lengths.size(); ++i) {
    if (i < sample_ids.size() && i < hap_indices.size() && i < seq_ids.size() &&
        i < seq_starts.size() && i < seq_ends.size() &&
        sample_ids[i] == sample_id && hap_indices[i] == hap_index &&
        seq_ids[i] == seq_id && seq_starts[i] == seq_start &&
        seq_ends[i] == seq_end) {
      walk_index = i;
      break;
    }
  }

  if (walk_index == data.walk_lengths.size()) {
    throw std::runtime_error(std::string(kExtractionErrorPrefix) +
                             "walk not found for the provided identifier");
  }

  const auto [rules_first, rules_second] = decode_rules(data);
  const std::vector<int32_t> encoded =
      extract_encoded_sequence(data.walks_zstd, data.walk_lengths, walk_index);
  const uint32_t original_length =
      (walk_index < data.original_walk_lengths.size())
          ? data.original_walk_lengths[walk_index]
          : static_cast<uint32_t>(encoded.size());
  const std::vector<NodeId> decoded =
      decode_sequence(encoded, rules_first, rules_second, data.min_rule_id(),
                      original_length, data.delta_round);

  return format_walk_line(sample_id, hap_index, seq_id, seq_start, seq_end,
                          decoded);
}

std::string extract_walk_line_by_name(const CompressedData &data,
                                      const std::string &walk_name,
                                      int num_threads) {
  ScopedOMPThreads omp_scope(num_threads);

  const std::vector<std::string> sample_ids = decompress_string_column(
      data.walk_sample_ids_zstd, data.walk_sample_id_lengths_zstd);

  size_t walk_index = data.walk_lengths.size();
  bool ambiguous = false;
  for (size_t i = 0; i < sample_ids.size(); ++i) {
    if (sample_ids[i] == walk_name) {
      if (walk_index != data.walk_lengths.size()) {
        ambiguous = true;
        break;
      }
      walk_index = i;
    }
  }

  if (ambiguous) {
    throw std::runtime_error(std::string(kExtractionErrorPrefix) +
                             "walk name is ambiguous: " + walk_name);
  }

  if (walk_index == data.walk_lengths.size()) {
    throw std::runtime_error(std::string(kExtractionErrorPrefix) +
                             "walk not found: " + walk_name);
  }

  const std::vector<uint32_t> hap_indices =
      Codec::zstd_decompress_uint32_vector(data.walk_hap_indices_zstd);
  const std::vector<std::string> seq_ids =
      decompress_string_column(data.walk_seq_ids_zstd,
                               data.walk_seq_id_lengths_zstd);
  const std::vector<int64_t> seq_starts =
      Codec::decompress_varint_int64(data.walk_seq_starts_zstd,
                                     data.walk_lengths.size());
  const std::vector<int64_t> seq_ends =
      Codec::decompress_varint_int64(data.walk_seq_ends_zstd,
                                     data.walk_lengths.size());

  const auto [rules_first, rules_second] = decode_rules(data);
  const std::vector<int32_t> encoded =
      extract_encoded_sequence(data.walks_zstd, data.walk_lengths, walk_index);
  const uint32_t original_length =
      (walk_index < data.original_walk_lengths.size())
          ? data.original_walk_lengths[walk_index]
          : static_cast<uint32_t>(encoded.size());
  const std::vector<NodeId> decoded =
      decode_sequence(encoded, rules_first, rules_second, data.min_rule_id(),
                      original_length, data.delta_round);

  const uint32_t hap_index =
      (walk_index < hap_indices.size()) ? hap_indices[walk_index] : 0;
  const std::string seq_id =
      (walk_index < seq_ids.size()) ? seq_ids[walk_index] : "unknown";
  const int64_t seq_start =
      (walk_index < seq_starts.size()) ? seq_starts[walk_index] : -1;
  const int64_t seq_end =
      (walk_index < seq_ends.size()) ? seq_ends[walk_index] : -1;

  return format_walk_line(walk_name, hap_index, seq_id, seq_start, seq_end,
                          decoded);
}
