#include "add_haplotypes_workflow.hpp"

#include "codec.hpp"
#include "packed_2mer.hpp"
#include "path_encoder.hpp"
#include "threading_utils.hpp"

#include <cctype>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

    namespace {

  constexpr const char *kAddHaplotypesErrorPrefix =
      "Add-haplotypes workflow error: ";

  struct AppendInput {
    std::vector<std::vector<NodeId>> paths;
    std::vector<std::string> path_names;
    std::vector<std::string> path_overlaps;
    WalkData walks;
    bool has_paths = false;
    bool has_walks = false;
  };

  inline std::string_view next_field(std::string_view line, size_t &pos) {
    while (pos < line.size() && (line[pos] == ' ' || line[pos] == '\t'))
      ++pos;
    const size_t start = pos;
    while (pos < line.size() && line[pos] != ' ' && line[pos] != '\t')
      ++pos;
    return line.substr(start, pos - start);
  }

  bool is_numeric(std::string_view value) {
    if (value.empty())
      return false;
    for (char c : value) {
      if (!std::isdigit(static_cast<unsigned char>(c)))
        return false;
    }
    return true;
  }

  uint32_t parse_uint32(std::string_view value, const char *field_name) {
    if (!is_numeric(value)) {
      throw std::runtime_error(std::string(kAddHaplotypesErrorPrefix) +
                               field_name + " must be numeric");
    }

    uint64_t parsed = 0;
    for (char c : value) {
      parsed = parsed * 10 + static_cast<uint64_t>(c - '0');
      if (parsed > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error(std::string(kAddHaplotypesErrorPrefix) +
                                 field_name + " overflows uint32");
      }
    }
    return static_cast<uint32_t>(parsed);
  }

  int64_t parse_int64_or_star(std::string_view value, const char *field_name) {
    if (value == "*")
      return -1;
    if (value.empty()) {
      throw std::runtime_error(std::string(kAddHaplotypesErrorPrefix) +
                               field_name + " is empty");
    }

    size_t pos = 0;
    bool negative = false;
    if (value[pos] == '-') {
      negative = true;
      ++pos;
    }
    if (pos >= value.size()) {
      throw std::runtime_error(std::string(kAddHaplotypesErrorPrefix) +
                               field_name + " is invalid");
    }

    int64_t parsed = 0;
    for (; pos < value.size(); ++pos) {
      char c = value[pos];
      if (!std::isdigit(static_cast<unsigned char>(c))) {
        throw std::runtime_error(std::string(kAddHaplotypesErrorPrefix) +
                                 field_name + " must be an integer or '*'");
      }
      const int digit = c - '0';
      if (!negative) {
        if (parsed > (std::numeric_limits<int64_t>::max() - digit) / 10) {
          throw std::runtime_error(std::string(kAddHaplotypesErrorPrefix) +
                                   field_name + " overflows int64");
        }
        parsed = parsed * 10 + digit;
      } else {
        if (parsed < (std::numeric_limits<int64_t>::min() + digit) / 10) {
          throw std::runtime_error(std::string(kAddHaplotypesErrorPrefix) +
                                   field_name + " overflows int64");
        }
        parsed = parsed * 10 - digit;
      }
    }
    return parsed;
  }

  NodeId parse_numeric_node(std::string_view value, char orientation,
                            const char *context) {
    const uint32_t node_id = parse_uint32(value, context);
    if (node_id == 0) {
      throw std::runtime_error(std::string(kAddHaplotypesErrorPrefix) +
                               context + " must be 1-based and non-zero");
    }
    return (orientation == '-' || orientation == '<')
               ? -static_cast<NodeId>(node_id)
               : static_cast<NodeId>(node_id);
  }

  void parse_path_line(std::string_view line, AppendInput & out) {
    size_t pos = 1;
    const std::string_view path_name = next_field(line, pos);
    const std::string_view nodes_str = next_field(line, pos);
    while (pos < line.size() && (line[pos] == ' ' || line[pos] == '\t'))
      ++pos;
    const std::string overlap(line.substr(pos));

    if (path_name.empty()) {
      throw std::runtime_error(std::string(kAddHaplotypesErrorPrefix) +
                               "P-line path name is empty");
    }

    std::vector<NodeId> path;
    size_t node_start = 0;
    for (size_t i = 0; i <= nodes_str.size(); ++i) {
      if (i == nodes_str.size() || nodes_str[i] == ',') {
        if (i > node_start) {
          const std::string_view node_token =
              nodes_str.substr(node_start, i - node_start);
          if (node_token.size() < 2) {
            throw std::runtime_error(std::string(kAddHaplotypesErrorPrefix) +
                                     "invalid P-line node token");
          }
          const char orientation = node_token.back();
          if (orientation != '+' && orientation != '-') {
            throw std::runtime_error(
                std::string(kAddHaplotypesErrorPrefix) +
                "P-line node orientation must be '+' or '-'");
          }
          const std::string_view node_name =
              node_token.substr(0, node_token.size() - 1);
          if (!is_numeric(node_name)) {
            throw std::runtime_error(std::string(kAddHaplotypesErrorPrefix) +
                                     "CPU .gfaz does not store original "
                                     "segment names; appended P-lines "
                                     "must use numeric segment IDs");
          }
          path.push_back(
              parse_numeric_node(node_name, orientation, "P-line segment ID"));
        }
        node_start = i + 1;
      }
    }

    out.path_names.emplace_back(path_name);
    out.path_overlaps.push_back(overlap);
    out.paths.push_back(std::move(path));
    out.has_paths = true;
  }

  void parse_walk_line(std::string_view line, AppendInput & out) {
    size_t pos = 1;
    const std::string_view sample_id = next_field(line, pos);
    const std::string_view hap_index = next_field(line, pos);
    const std::string_view seq_id = next_field(line, pos);
    const std::string_view seq_start = next_field(line, pos);
    const std::string_view seq_end = next_field(line, pos);
    const std::string_view walk_str = next_field(line, pos);

    if (sample_id.empty()) {
      throw std::runtime_error(std::string(kAddHaplotypesErrorPrefix) +
                               "W-line sample_id is empty");
    }

    std::vector<NodeId> walk;
    size_t walk_pos = 0;
    while (walk_pos < walk_str.size()) {
      const char orientation = walk_str[walk_pos];
      if (orientation != '>' && orientation != '<') {
        throw std::runtime_error(std::string(kAddHaplotypesErrorPrefix) +
                                 "W-line walk string is malformed");
      }

      const size_t name_start = walk_pos + 1;
      size_t name_end = name_start;
      while (name_end < walk_str.size() && walk_str[name_end] != '>' &&
             walk_str[name_end] != '<') {
        ++name_end;
      }

      const std::string_view node_name =
          walk_str.substr(name_start, name_end - name_start);
      if (!is_numeric(node_name)) {
        throw std::runtime_error(
            std::string(kAddHaplotypesErrorPrefix) +
            "CPU .gfaz does not store original segment names; appended W-lines "
            "must use numeric segment IDs");
      }
      walk.push_back(
          parse_numeric_node(node_name, orientation, "W-line segment ID"));
      walk_pos = name_end;
    }

    out.walks.sample_ids.emplace_back(sample_id);
    out.walks.hap_indices.push_back(
        parse_uint32(hap_index, "W-line hap_index"));
    out.walks.seq_ids.emplace_back(seq_id);
    out.walks.seq_starts.push_back(
        parse_int64_or_star(seq_start, "W-line seq_start"));
    out.walks.seq_ends.push_back(
        parse_int64_or_star(seq_end, "W-line seq_end"));
    out.walks.walks.push_back(std::move(walk));
    out.has_walks = true;
  }

  AppendInput parse_append_file(const std::string &haplotypes_path) {
    std::ifstream input(haplotypes_path);
    if (!input.is_open()) {
      throw std::runtime_error(std::string(kAddHaplotypesErrorPrefix) +
                               "failed to open input file '" + haplotypes_path +
                               "'");
    }

    AppendInput parsed;
    std::string line;
    size_t line_number = 0;
    while (std::getline(input, line)) {
      ++line_number;
      if (line.empty())
        continue;

      const char type = line[0];
      if (type == 'H' || type == '#')
        continue;

      if (type == 'P') {
        if (parsed.has_walks) {
          throw std::runtime_error(std::string(kAddHaplotypesErrorPrefix) +
                                   "input file mixes P-lines and W-lines; use "
                                   "a pure path or pure walk file");
        }
        parse_path_line(line, parsed);
        continue;
      }

      if (type == 'W') {
        if (parsed.has_paths) {
          throw std::runtime_error(std::string(kAddHaplotypesErrorPrefix) +
                                   "input file mixes P-lines and W-lines; use "
                                   "a pure path or pure walk file");
        }
        parse_walk_line(line, parsed);
        continue;
      }

      throw std::runtime_error(
          std::string(kAddHaplotypesErrorPrefix) + "unsupported line type '" +
          std::string(1, type) + "' at line " + std::to_string(line_number) +
          "; only H/P/W lines are accepted");
    }

    if (!parsed.has_paths && !parsed.has_walks) {
      throw std::runtime_error(
          std::string(kAddHaplotypesErrorPrefix) +
          "input file does not contain any P-lines or W-lines");
    }

    return parsed;
  }

  void reconstruct_strings(const std::string &concat,
                           const std::vector<uint32_t> &lengths,
                           std::vector<std::string> &out) {
    out.clear();
    out.reserve(lengths.size());

    size_t offset = 0;
    for (uint32_t len : lengths) {
      if (offset + len > concat.size()) {
        throw std::runtime_error(std::string(kAddHaplotypesErrorPrefix) +
                                 "string column is truncated");
      }
      out.push_back(concat.substr(offset, len));
      offset += len;
    }
  }

  std::vector<std::string> decompress_string_column(
      const ZstdCompressedBlock &strings_zstd,
      const ZstdCompressedBlock &lengths_zstd) {
    std::vector<std::string> out;
    reconstruct_strings(Codec::zstd_decompress_string(strings_zstd),
                        Codec::zstd_decompress_uint32_vector(lengths_zstd),
                        out);
    return out;
  }

  void append_string_column(const std::vector<std::string> &values,
                            std::string &concatenated,
                            std::vector<uint32_t> &lengths) {
    concatenated.clear();
    lengths.clear();
    for (const auto &value : values) {
      concatenated += value;
      lengths.push_back(static_cast<uint32_t>(value.size()));
    }
  }

  std::pair<std::vector<int32_t>, std::vector<int32_t>> decode_rules(
      const CompressedData &data) {
    std::vector<int32_t> first =
        Codec::zstd_decompress_int32_vector(data.rules_first_zstd);
    std::vector<int32_t> second =
        Codec::zstd_decompress_int32_vector(data.rules_second_zstd);
    Codec::delta_decode_int32(first);
    Codec::delta_decode_int32(second);
    return {std::move(first), std::move(second)};
  }

  std::vector<CompressionRules2Mer> build_layered_rules(
      const CompressedData &data, const std::vector<int32_t> &rules_first,
      const std::vector<int32_t> &rules_second) {
    std::vector<CompressionRules2Mer> layers;
    layers.reserve(data.layer_rule_ranges.size());

    for (const auto &range : data.layer_rule_ranges) {
      CompressionRules2Mer layer;
      layer.rules_start_id = range.start_id;
      layer.next_available_id = range.end_id;

      const uint32_t rule_count = range.end_id - range.start_id;
      const size_t rule_offset =
          range.k == 0 ? 0 : (range.flattened_offset / range.k);
      layer.rule_id_to_kmer.reserve(rule_count);
      layer.kmer_to_rule_id.reserve(rule_count);

      for (uint32_t i = 0; i < rule_count; ++i) {
        const size_t idx = rule_offset + i;
        if (idx >= rules_first.size() || idx >= rules_second.size()) {
          throw std::runtime_error(std::string(kAddHaplotypesErrorPrefix) +
                                   "rule column is truncated");
        }

        const Packed2mer kmer = pack_2mer(rules_first[idx], rules_second[idx]);
        layer.rule_id_to_kmer.push_back(kmer);
        layer.kmer_to_rule_id.emplace(kmer, range.start_id + i);
      }

      layers.push_back(std::move(layer));
    }

    return layers;
  }

  void encode_with_existing_rules(
      std::vector<std::vector<NodeId>> & sequences,
      const std::vector<CompressionRules2Mer> &layers) {
    PathEncoder encoder;
    for (const auto &layer : layers) {
      std::vector<uint8_t> rules_used;
      encoder.encode_paths_2mer(sequences, layer, rules_used);
    }
  }

  void validate_unique_names(const std::vector<std::string> &existing,
                             const std::vector<std::string> &incoming,
                             const char *kind) {
    std::unordered_set<std::string> seen;
    seen.reserve(existing.size() + incoming.size());
    for (const auto &name : existing)
      seen.insert(name);

    for (const auto &name : incoming) {
      if (!seen.insert(name).second) {
        throw std::runtime_error(
            std::string(kAddHaplotypesErrorPrefix) + kind +
            " name already exists or is duplicated: " + name);
      }
    }
  }

  std::string make_walk_identity_key(
      std::string_view sample_id, uint32_t hap_index, std::string_view seq_id,
      int64_t seq_start, int64_t seq_end) {
    std::string key;
    key.reserve(sample_id.size() + seq_id.size() + 64);
    key += sample_id;
    key.push_back('\x1f');
    key += std::to_string(hap_index);
    key.push_back('\x1f');
    key += seq_id;
    key.push_back('\x1f');
    key += std::to_string(seq_start);
    key.push_back('\x1f');
    key += std::to_string(seq_end);
    return key;
  }

  void validate_unique_walk_keys(
      const std::vector<std::string> &existing_sample_ids,
      const std::vector<uint32_t> &existing_hap_indices,
      const std::vector<std::string> &existing_seq_ids,
      const std::vector<int64_t> &existing_seq_starts,
      const std::vector<int64_t> &existing_seq_ends, const WalkData &incoming) {
    std::unordered_set<std::string> seen;
    seen.reserve(existing_sample_ids.size() + incoming.sample_ids.size());

    for (size_t i = 0; i < existing_sample_ids.size(); ++i) {
      seen.insert(make_walk_identity_key(
          existing_sample_ids[i], existing_hap_indices[i], existing_seq_ids[i],
          existing_seq_starts[i], existing_seq_ends[i]));
    }

    for (size_t i = 0; i < incoming.sample_ids.size(); ++i) {
      const std::string key = make_walk_identity_key(
          incoming.sample_ids[i], incoming.hap_indices[i], incoming.seq_ids[i],
          incoming.seq_starts[i], incoming.seq_ends[i]);
      if (!seen.insert(key).second) {
        throw std::runtime_error(
            std::string(kAddHaplotypesErrorPrefix) +
            "walk identifier already exists or is duplicated: " +
            incoming.sample_ids[i] + "\t" +
            std::to_string(incoming.hap_indices[i]) + "\t" +
            incoming.seq_ids[i] + "\t" +
            std::to_string(incoming.seq_starts[i]) + "\t" +
            std::to_string(incoming.seq_ends[i]));
      }
    }
  }

  void apply_delta_and_validate(std::vector<std::vector<NodeId>> & sequences,
                                int delta_round, uint32_t min_rule_id,
                                bool has_rule_region) {
    uint32_t max_abs = 0;
    for (int i = 0; i < delta_round; ++i) {
      const uint32_t round_max = Codec::delta_transform_and_max_abs(sequences);
      if (round_max > max_abs)
        max_abs = round_max;
    }

    if (has_rule_region && max_abs >= min_rule_id) {
      throw std::runtime_error(
          std::string(kAddHaplotypesErrorPrefix) +
          "delta-encoded appended sequence reaches ID " +
          std::to_string(max_abs) +
          ", which collides with the stored rule region starting at " +
          std::to_string(min_rule_id));
    }
  }

  void append_paths(CompressedData & data, const AppendInput &input,
                    const std::vector<CompressionRules2Mer> &layers) {
    std::vector<std::string> existing_names =
        decompress_string_column(data.names_zstd, data.name_lengths_zstd);
    validate_unique_names(existing_names, input.path_names, "path");

    std::vector<std::string> existing_overlaps =
        decompress_string_column(data.overlaps_zstd, data.overlap_lengths_zstd);
    std::vector<int32_t> flat =
        Codec::zstd_decompress_int32_vector(data.paths_zstd);

    std::vector<std::vector<NodeId>> appended_paths = input.paths;
    data.original_path_lengths.reserve(data.original_path_lengths.size() +
                                       appended_paths.size());
    for (const auto &path : appended_paths)
      data.original_path_lengths.push_back(static_cast<uint32_t>(path.size()));

    apply_delta_and_validate(appended_paths, data.delta_round,
                             data.min_rule_id(),
                             !data.layer_rule_ranges.empty());
    encode_with_existing_rules(appended_paths, layers);

    for (const auto &path : appended_paths) {
      data.sequence_lengths.push_back(static_cast<uint32_t>(path.size()));
      flat.insert(flat.end(), path.begin(), path.end());
    }

    existing_names.insert(existing_names.end(), input.path_names.begin(),
                          input.path_names.end());
    existing_overlaps.insert(existing_overlaps.end(),
                             input.path_overlaps.begin(),
                             input.path_overlaps.end());

    std::string names_concat, overlaps_concat;
    std::vector<uint32_t> name_lengths, overlap_lengths;
    append_string_column(existing_names, names_concat, name_lengths);
    append_string_column(existing_overlaps, overlaps_concat, overlap_lengths);

    data.paths_zstd = Codec::zstd_compress_int32_vector(flat);
    data.names_zstd = Codec::zstd_compress_string(names_concat);
    data.name_lengths_zstd = Codec::zstd_compress_uint32_vector(name_lengths);
    data.overlaps_zstd = Codec::zstd_compress_string(overlaps_concat);
    data.overlap_lengths_zstd =
        Codec::zstd_compress_uint32_vector(overlap_lengths);
  }

  void append_walks(CompressedData & data, const AppendInput &input,
                    const std::vector<CompressionRules2Mer> &layers) {
    std::vector<std::string> existing_sample_ids = decompress_string_column(
        data.walk_sample_ids_zstd, data.walk_sample_id_lengths_zstd);
    std::vector<uint32_t> hap_indices =
        Codec::zstd_decompress_uint32_vector(data.walk_hap_indices_zstd);
    std::vector<std::string> seq_ids = decompress_string_column(
        data.walk_seq_ids_zstd, data.walk_seq_id_lengths_zstd);
    std::vector<int64_t> seq_starts = Codec::decompress_varint_int64(
        data.walk_seq_starts_zstd, data.walk_lengths.size());
    std::vector<int64_t> seq_ends = Codec::decompress_varint_int64(
        data.walk_seq_ends_zstd, data.walk_lengths.size());
    validate_unique_walk_keys(existing_sample_ids, hap_indices, seq_ids,
                              seq_starts, seq_ends, input.walks);
    std::vector<int32_t> flat =
        Codec::zstd_decompress_int32_vector(data.walks_zstd);

    std::vector<std::vector<NodeId>> appended_walks = input.walks.walks;
    data.original_walk_lengths.reserve(data.original_walk_lengths.size() +
                                       appended_walks.size());
    for (const auto &walk : appended_walks)
      data.original_walk_lengths.push_back(static_cast<uint32_t>(walk.size()));

    apply_delta_and_validate(appended_walks, data.delta_round,
                             data.min_rule_id(),
                             !data.layer_rule_ranges.empty());
    encode_with_existing_rules(appended_walks, layers);

    for (const auto &walk : appended_walks) {
      data.walk_lengths.push_back(static_cast<uint32_t>(walk.size()));
      flat.insert(flat.end(), walk.begin(), walk.end());
    }

    existing_sample_ids.insert(existing_sample_ids.end(),
                               input.walks.sample_ids.begin(),
                               input.walks.sample_ids.end());
    hap_indices.insert(hap_indices.end(), input.walks.hap_indices.begin(),
                       input.walks.hap_indices.end());
    seq_ids.insert(seq_ids.end(), input.walks.seq_ids.begin(),
                   input.walks.seq_ids.end());
    seq_starts.insert(seq_starts.end(), input.walks.seq_starts.begin(),
                      input.walks.seq_starts.end());
    seq_ends.insert(seq_ends.end(), input.walks.seq_ends.begin(),
                    input.walks.seq_ends.end());

    std::string sample_ids_concat, seq_ids_concat;
    std::vector<uint32_t> sample_id_lengths, seq_id_lengths;
    append_string_column(existing_sample_ids, sample_ids_concat,
                         sample_id_lengths);
    append_string_column(seq_ids, seq_ids_concat, seq_id_lengths);

    data.walks_zstd = Codec::zstd_compress_int32_vector(flat);
    data.walk_sample_ids_zstd = Codec::zstd_compress_string(sample_ids_concat);
    data.walk_sample_id_lengths_zstd =
        Codec::zstd_compress_uint32_vector(sample_id_lengths);
    data.walk_hap_indices_zstd =
        Codec::zstd_compress_uint32_vector(hap_indices);
    data.walk_seq_ids_zstd = Codec::zstd_compress_string(seq_ids_concat);
    data.walk_seq_id_lengths_zstd =
        Codec::zstd_compress_uint32_vector(seq_id_lengths);
    data.walk_seq_starts_zstd = Codec::compress_varint_int64(seq_starts);
    data.walk_seq_ends_zstd = Codec::compress_varint_int64(seq_ends);
  }

} // namespace

void add_haplotypes(CompressedData &data, const std::string &haplotypes_path,
                    int num_threads) {
  ScopedOMPThreads omp_scope(num_threads);

  const AppendInput input = parse_append_file(haplotypes_path);
  const auto [rules_first, rules_second] = decode_rules(data);
  const std::vector<CompressionRules2Mer> layers =
      build_layered_rules(data, rules_first, rules_second);

  if (input.has_paths)
    append_paths(data, input, layers);
  else
    append_walks(data, input, layers);
}
