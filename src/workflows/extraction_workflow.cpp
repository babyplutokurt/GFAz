#include "workflows/extraction_workflow.hpp"

#include "codec/codec.hpp"
#include "utils/threading_utils.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
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

std::vector<NodeId>
extract_encoded_sequence_from_flat(const std::vector<int32_t> &flat,
                                   const std::vector<uint32_t> &lengths,
                                   size_t index) {
  if (index >= lengths.size()) {
    throw std::out_of_range(std::string(kExtractionErrorPrefix) +
                            "sequence index out of range");
  }

  const size_t offset = compute_offset(lengths, index);
  const size_t length = lengths[index];
  if (offset + length > flat.size()) {
    throw std::runtime_error(std::string(kExtractionErrorPrefix) +
                             "flattened sequence block is truncated");
  }

  return std::vector<NodeId>(flat.begin() + static_cast<std::ptrdiff_t>(offset),
                             flat.begin() +
                                 static_cast<std::ptrdiff_t>(offset + length));
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
  return extract_path_lines_by_name(data, {path_name}, num_threads).front();
}

std::vector<std::string>
extract_path_lines_by_name(const CompressedData &data,
                           const std::vector<std::string> &requested_path_names,
                           int num_threads) {
  ScopedOMPThreads omp_scope(num_threads);
  if (requested_path_names.empty())
    return {};

  const std::vector<std::string> stored_path_names =
      decompress_string_column(data.names_zstd, data.name_lengths_zstd);
  std::unordered_map<std::string, size_t> path_name_to_index;
  for (size_t i = 0; i < stored_path_names.size(); ++i) {
    const auto [it, inserted] =
        path_name_to_index.emplace(stored_path_names[i], i);
    if (!inserted) {
      throw std::runtime_error(std::string(kExtractionErrorPrefix) +
                               "path name is ambiguous: " +
                               stored_path_names[i]);
    }
  }

  std::vector<size_t> indices;
  indices.reserve(requested_path_names.size());
  for (const auto &path_name : requested_path_names) {
    const auto it = path_name_to_index.find(path_name);
    if (it == path_name_to_index.end()) {
      throw std::runtime_error(std::string(kExtractionErrorPrefix) +
                               "path not found: " + path_name);
    }
    indices.push_back(it->second);
  }

  const std::vector<std::string> overlaps =
      decompress_string_column(data.overlaps_zstd, data.overlap_lengths_zstd);
  const auto [rules_first, rules_second] = decode_rules(data);
  const std::vector<int32_t> flat =
      Codec::zstd_decompress_int32_vector(data.paths_zstd);

  std::vector<std::string> lines;
  lines.reserve(requested_path_names.size());
  for (size_t request_idx = 0; request_idx < requested_path_names.size();
       ++request_idx) {
    const size_t path_index = indices[request_idx];
    const std::vector<NodeId> encoded = extract_encoded_sequence_from_flat(
        flat, data.sequence_lengths, path_index);
    const uint32_t original_length =
        (path_index < data.original_path_lengths.size())
            ? data.original_path_lengths[path_index]
            : static_cast<uint32_t>(encoded.size());
    const std::vector<NodeId> decoded =
        decode_sequence(encoded, rules_first, rules_second, data.min_rule_id(),
                        original_length, data.delta_round);
    const std::string overlap =
        (path_index < overlaps.size()) ? overlaps[path_index] : "";
    lines.push_back(
        format_path_line(requested_path_names[request_idx], decoded, overlap));
  }
  return lines;
}

std::string extract_walk_line(const CompressedData &data,
                              const std::string &sample_id,
                              uint32_t hap_index,
                              const std::string &seq_id,
                              int64_t seq_start,
                              int64_t seq_end,
                              int num_threads) {
  WalkLookupKey key;
  key.sample_id = sample_id;
  key.hap_index = hap_index;
  key.seq_id = seq_id;
  key.seq_start = seq_start;
  key.seq_end = seq_end;
  return extract_walk_lines(data, {key}, num_threads).front();
}

std::vector<std::string>
extract_walk_lines(const CompressedData &data,
                   const std::vector<WalkLookupKey> &walk_keys,
                   int num_threads) {
  ScopedOMPThreads omp_scope(num_threads);
  if (walk_keys.empty())
    return {};

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

  std::vector<size_t> indices;
  indices.reserve(walk_keys.size());
  for (const auto &walk_key : walk_keys) {
    size_t walk_index = data.walk_lengths.size();
    for (size_t i = 0; i < data.walk_lengths.size(); ++i) {
      if (i < sample_ids.size() && i < hap_indices.size() && i < seq_ids.size() &&
          i < seq_starts.size() && i < seq_ends.size() &&
          sample_ids[i] == walk_key.sample_id &&
          hap_indices[i] == walk_key.hap_index &&
          seq_ids[i] == walk_key.seq_id &&
          seq_starts[i] == walk_key.seq_start &&
          seq_ends[i] == walk_key.seq_end) {
        walk_index = i;
        break;
      }
    }

    if (walk_index == data.walk_lengths.size()) {
      throw std::runtime_error(std::string(kExtractionErrorPrefix) +
                               "walk not found for the provided identifier: " +
                               walk_key.sample_id + "\t" +
                               std::to_string(walk_key.hap_index) + "\t" +
                               walk_key.seq_id + "\t" +
                               std::to_string(walk_key.seq_start) + "\t" +
                               std::to_string(walk_key.seq_end));
    }
    indices.push_back(walk_index);
  }

  const auto [rules_first, rules_second] = decode_rules(data);
  const std::vector<int32_t> flat =
      Codec::zstd_decompress_int32_vector(data.walks_zstd);

  std::vector<std::string> lines;
  lines.reserve(walk_keys.size());
  for (size_t request_idx = 0; request_idx < walk_keys.size(); ++request_idx) {
    const size_t walk_index = indices[request_idx];
    const std::vector<NodeId> encoded =
        extract_encoded_sequence_from_flat(flat, data.walk_lengths, walk_index);
    const uint32_t original_length =
        (walk_index < data.original_walk_lengths.size())
            ? data.original_walk_lengths[walk_index]
            : static_cast<uint32_t>(encoded.size());
    const std::vector<NodeId> decoded =
        decode_sequence(encoded, rules_first, rules_second, data.min_rule_id(),
                        original_length, data.delta_round);

    const auto &walk_key = walk_keys[request_idx];
    lines.push_back(format_walk_line(walk_key.sample_id, walk_key.hap_index,
                                     walk_key.seq_id, walk_key.seq_start,
                                     walk_key.seq_end, decoded));
  }
  return lines;
}

std::string extract_walk_line_by_name(const CompressedData &data,
                                      const std::string &walk_name,
                                      int num_threads) {
  return extract_walk_lines_by_name(data, {walk_name}, num_threads).front();
}

std::vector<std::string>
extract_walk_lines_by_name(const CompressedData &data,
                           const std::vector<std::string> &walk_names,
                           int num_threads) {
  ScopedOMPThreads omp_scope(num_threads);
  if (walk_names.empty())
    return {};

  const std::vector<std::string> sample_ids = decompress_string_column(
      data.walk_sample_ids_zstd, data.walk_sample_id_lengths_zstd);
  std::unordered_map<std::string, size_t> walk_name_to_index;
  for (size_t i = 0; i < sample_ids.size(); ++i) {
    const auto [it, inserted] = walk_name_to_index.emplace(sample_ids[i], i);
    if (!inserted) {
      throw std::runtime_error(std::string(kExtractionErrorPrefix) +
                               "walk name is ambiguous: " + sample_ids[i]);
    }
  }

  std::vector<size_t> indices;
  indices.reserve(walk_names.size());
  for (const auto &walk_name : walk_names) {
    const auto it = walk_name_to_index.find(walk_name);
    if (it == walk_name_to_index.end()) {
      throw std::runtime_error(std::string(kExtractionErrorPrefix) +
                               "walk not found: " + walk_name);
    }
    indices.push_back(it->second);
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

  std::vector<WalkLookupKey> walk_keys;
  walk_keys.reserve(walk_names.size());
  for (size_t request_idx = 0; request_idx < walk_names.size(); ++request_idx) {
    const size_t walk_index = indices[request_idx];
    WalkLookupKey walk_key;
    walk_key.sample_id = walk_names[request_idx];
    walk_key.hap_index =
        (walk_index < hap_indices.size()) ? hap_indices[walk_index] : 0;
    walk_key.seq_id =
        (walk_index < seq_ids.size()) ? seq_ids[walk_index] : "unknown";
    walk_key.seq_start =
        (walk_index < seq_starts.size()) ? seq_starts[walk_index] : -1;
    walk_key.seq_end =
        (walk_index < seq_ends.size()) ? seq_ends[walk_index] : -1;
    walk_keys.push_back(std::move(walk_key));
  }
  return extract_walk_lines(data, walk_keys, num_threads);
}

