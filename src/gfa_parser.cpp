#include "gfa_parser.hpp"
#include "debug_log.hpp"
#include "threading_utils.hpp"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

constexpr const char *kParserErrorPrefix = "GFA parser error: ";
constexpr const char *kParserWarningPrefix = "GFA parser warning: ";

struct ProcessMemorySnapshot {
  size_t vm_rss_kb = 0;
  size_t vm_hwm_kb = 0;
};

std::string format_memory_size(size_t bytes) {
  std::ostringstream oss;
  if (bytes >= 1024 * 1024)
    oss << std::fixed << std::setprecision(2) << (bytes / (1024.0 * 1024.0))
        << " MB";
  else if (bytes >= 1024)
    oss << std::fixed << std::setprecision(1) << (bytes / 1024.0) << " KB";
  else
    oss << bytes << " Bytes";
  return oss.str();
}

ProcessMemorySnapshot read_process_memory_snapshot() {
  ProcessMemorySnapshot snapshot;

  std::ifstream status("/proc/self/status");
  if (!status)
    return snapshot;

  std::string line;
  while (std::getline(status, line)) {
    auto parse_kb_field = [&](const char *prefix, size_t &out_value) {
      const std::string_view view(line);
      const std::string_view key(prefix);
      if (view.size() < key.size() || view.substr(0, key.size()) != key)
        return false;

      std::istringstream iss(std::string(view.substr(key.size())));
      size_t value = 0;
      std::string unit;
      if (iss >> value >> unit) {
        out_value = value;
        return true;
      }
      return false;
    };

    if (parse_kb_field("VmRSS:", snapshot.vm_rss_kb))
      continue;
    parse_kb_field("VmHWM:", snapshot.vm_hwm_kb);
  }

  return snapshot;
}

void log_parser_memory_checkpoint(const std::string &label) {
  if (!gfaz_debug_enabled())
    return;

  const ProcessMemorySnapshot snapshot = read_process_memory_snapshot();
  std::cerr << "[GfaParser][Memory] " << label
            << " | VmRSS=" << format_memory_size(snapshot.vm_rss_kb * 1024)
            << " | VmHWM=" << format_memory_size(snapshot.vm_hwm_kb * 1024)
            << std::endl;
}

inline int64_t parse_int64(std::string_view s) {
  if (s.empty())
    throw std::invalid_argument("parse_int64: empty");

  size_t i = 0;
  bool neg = (s[0] == '-');
  if (neg)
    i = 1;
  if (i >= s.size())
    throw std::invalid_argument("parse_int64: no digits");

  int64_t val = 0;
  int64_t limit = neg ? std::numeric_limits<int64_t>::min()
                      : std::numeric_limits<int64_t>::max();
  int64_t cutoff = limit / 10;
  int64_t cutlim = neg ? -(limit % 10) : limit % 10;

  for (; i < s.size(); ++i) {
    char c = s[i];
    if (c < '0' || c > '9')
      break;
    int digit = c - '0';
    if (!neg) {
      if (val > cutoff || (val == cutoff && digit > cutlim))
        throw std::out_of_range("parse_int64: overflow");
      val = val * 10 + digit;
    } else {
      if (val < cutoff || (val == cutoff && digit > cutlim))
        throw std::out_of_range("parse_int64: overflow");
      val = val * 10 - digit;
    }
  }
  return val;
}

inline uint32_t parse_uint32(std::string_view s) {
  if (s.empty())
    throw std::invalid_argument("parse_uint32: empty");

  uint32_t val = 0;
  uint32_t cutoff = std::numeric_limits<uint32_t>::max() / 10;
  uint32_t cutlim = std::numeric_limits<uint32_t>::max() % 10;
  bool any = false;

  for (char c : s) {
    if (c < '0' || c > '9')
      break;
    any = true;
    uint32_t digit = static_cast<uint32_t>(c - '0');
    if (val > cutoff || (val == cutoff && digit > cutlim))
      throw std::out_of_range("parse_uint32: overflow");
    val = val * 10 + digit;
  }

  if (!any)
    throw std::invalid_argument("parse_uint32: no digits");
  return val;
}

inline float parse_float(std::string_view s) {
  if (s.empty())
    throw std::invalid_argument("parse_float: empty");

  char buf[128];
  const char *cstr;
  std::string tmp;

  if (s.size() < sizeof(buf)) {
    std::memcpy(buf, s.data(), s.size());
    buf[s.size()] = '\0';
    cstr = buf;
  } else {
    tmp = std::string(s);
    cstr = tmp.c_str();
  }

  errno = 0;
  char *endptr = nullptr;
  float val = std::strtof(cstr, &endptr);

  if (endptr == cstr)
    throw std::invalid_argument("parse_float: no digits");
  if (errno == ERANGE)
    throw std::out_of_range("parse_float: overflow");
  return val;
}

inline std::string_view next_field(std::string_view line, size_t &pos) {
  while (pos < line.size() && (line[pos] == ' ' || line[pos] == '\t'))
    ++pos;
  size_t start = pos;
  while (pos < line.size() && line[pos] != ' ' && line[pos] != '\t')
    ++pos;
  return line.substr(start, pos - start);
}

inline uint32_t parse_overlap_num(std::string_view overlap_view, char &op) {
  uint32_t num = 0;
  size_t i = 0;
  while (i < overlap_view.size() &&
         std::isdigit(static_cast<unsigned char>(overlap_view[i]))) {
    num = num * 10 + static_cast<uint32_t>(overlap_view[i] - '0');
    ++i;
  }
  op = (i < overlap_view.size()) ? overlap_view[i] : '\0';
  return num;
}

} // namespace

GfaParser::GfaParser() = default;

bool GfaParser::is_numeric(std::string_view s) {
  if (s.empty())
    return false;
  for (char c : s) {
    if (!std::isdigit(static_cast<unsigned char>(c)))
      return false;
  }
  return true;
}

GfaGraph GfaParser::parse(const std::string &gfa_file_path, int num_threads) {
  ScopedOMPThreads omp_scope(num_threads);
  GfaGraph graph;
  segment_field_meta_.clear();
  link_field_meta_.clear();
  node_name_lookup_.clear();
  num_segments_hint_ = 0;
  num_links_hint_ = 0;
  all_segment_names_numeric_ = true;

  // Index 0 is a placeholder to support 1-based node IDs.
  // This allows NodeId sign to encode orientation without ambiguity.
  graph.node_id_to_name.push_back("");
  graph.node_sequences.push_back("");

  int fd = open(gfa_file_path.c_str(), O_RDONLY);
  if (fd == -1) {
    throw std::runtime_error(std::string(kParserErrorPrefix) +
                             "failed to open input file '" + gfa_file_path +
                             "': " + std::strerror(errno));
  }

  struct stat sb;
  if (fstat(fd, &sb) == -1) {
    close(fd);
    throw std::runtime_error(std::string(kParserErrorPrefix) +
                             "failed to stat input file '" + gfa_file_path +
                             "': " + std::strerror(errno));
  }
  size_t file_size = sb.st_size;

  const char *mmap_data = static_cast<const char *>(
      mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0));
  if (mmap_data == MAP_FAILED) {
    close(fd);
    throw std::runtime_error(std::string(kParserErrorPrefix) +
                             "failed to mmap input file '" + gfa_file_path +
                             "': " + std::strerror(errno));
  }

  madvise(const_cast<char *>(mmap_data), file_size, MADV_SEQUENTIAL);

  // Single-pass line classification
  std::vector<LineOffset> s_offsets, l_offsets, p_offsets, w_offsets;
  std::vector<LineOffset> j_offsets, c_offsets;
  size_t line_start = 0;

  for (size_t i = 0; i <= file_size; ++i) {
    if (i == file_size || mmap_data[i] == '\n') {
      size_t line_len = i - line_start;
      if (line_len > 0) {
        char line_type = mmap_data[line_start];
        switch (line_type) {
        case 'S':
          s_offsets.push_back({line_start, line_len});
          break;
        case 'H':
          graph.header_line = std::string(mmap_data + line_start, line_len);
          break;
        case 'L':
          l_offsets.push_back({line_start, line_len});
          break;
        case 'P':
          p_offsets.push_back({line_start, line_len});
          break;
        case 'W':
          w_offsets.push_back({line_start, line_len});
          break;
        case 'J':
          j_offsets.push_back({line_start, line_len});
          break;
        case 'C':
          c_offsets.push_back({line_start, line_len});
          break;
        }
      }
      line_start = i + 1;
    }
  }
  log_parser_memory_checkpoint("after line classification");

  num_segments_hint_ = s_offsets.size();
  num_links_hint_ = l_offsets.size();

  graph.node_id_to_name.reserve(num_segments_hint_ + 1);
  graph.node_sequences.reserve(num_segments_hint_ + 1);
  graph.node_name_to_id.reserve(num_segments_hint_);
  node_name_lookup_.reserve(num_segments_hint_);

  graph.links.from_ids.reserve(num_links_hint_);
  graph.links.to_ids.reserve(num_links_hint_);
  graph.links.from_orients.reserve(num_links_hint_);
  graph.links.to_orients.reserve(num_links_hint_);
  graph.links.overlap_nums.reserve(num_links_hint_);
  graph.links.overlap_ops.reserve(num_links_hint_);

  graph.jumps.from_ids.reserve(j_offsets.size());
  graph.jumps.from_orients.reserve(j_offsets.size());
  graph.jumps.to_ids.reserve(j_offsets.size());
  graph.jumps.to_orients.reserve(j_offsets.size());
  graph.jumps.distances.reserve(j_offsets.size());
  graph.jumps.rest_fields.reserve(j_offsets.size());

  graph.containments.container_ids.reserve(c_offsets.size());
  graph.containments.container_orients.reserve(c_offsets.size());
  graph.containments.contained_ids.reserve(c_offsets.size());
  graph.containments.contained_orients.reserve(c_offsets.size());
  graph.containments.positions.reserve(c_offsets.size());
  graph.containments.overlaps.reserve(c_offsets.size());
  graph.containments.rest_fields.reserve(c_offsets.size());
  log_parser_memory_checkpoint("after reserve");

  // Phase 1: Parse S-lines (sequential - must populate node_name_to_id first)
  for (const auto &off : s_offsets) {
    std::string_view line(mmap_data + off.offset, off.length);
    parse_s_line(line, graph);
  }

  // Pad optional field columns to segment count
  size_t num_segments = graph.node_id_to_name.size() - 1;
  for (auto &col : graph.segment_optional_fields) {
    switch (col.type) {
    case 'i':
      while (col.int_values.size() < num_segments)
        col.int_values.push_back(std::numeric_limits<int64_t>::min());
      break;
    case 'f':
      while (col.float_values.size() < num_segments)
        col.float_values.push_back(std::numeric_limits<float>::lowest());
      break;
    case 'A':
      while (col.char_values.size() < num_segments)
        col.char_values.push_back('\0');
      break;
    case 'Z':
    case 'J':
    case 'H':
      while (col.string_lengths.size() < num_segments)
        col.string_lengths.push_back(0);
      break;
    case 'B':
      while (col.b_subtypes.size() < num_segments) {
        col.b_subtypes.push_back('\0');
        col.b_lengths.push_back(0);
      }
      break;
    }
  }

  if (gfaz_debug_enabled()) {
    std::cerr << "Parsed " << num_segments << " segments";
    if (!graph.segment_optional_fields.empty())
      std::cerr << " (" << graph.segment_optional_fields.size()
                << " optional columns)";
    std::cerr << std::endl;
  }
  log_parser_memory_checkpoint("after S-lines");
  s_offsets.clear();
  s_offsets.shrink_to_fit();
  log_parser_memory_checkpoint("after clearing S offsets");

  // Phase 2: Parse L-lines (sequential)
  for (const auto &off : l_offsets) {
    std::string_view line(mmap_data + off.offset, off.length);
    parse_l_line(line, graph);
  }

  // Pad link optional field columns
  size_t num_links = graph.links.from_ids.size();
  for (auto &col : graph.link_optional_fields) {
    switch (col.type) {
    case 'i':
      while (col.int_values.size() < num_links)
        col.int_values.push_back(std::numeric_limits<int64_t>::min());
      break;
    case 'f':
      while (col.float_values.size() < num_links)
        col.float_values.push_back(std::numeric_limits<float>::lowest());
      break;
    case 'A':
      while (col.char_values.size() < num_links)
        col.char_values.push_back('\0');
      break;
    case 'Z':
    case 'J':
    case 'H':
      while (col.string_lengths.size() < num_links)
        col.string_lengths.push_back(0);
      break;
    case 'B':
      while (col.b_subtypes.size() < num_links) {
        col.b_subtypes.push_back('\0');
        col.b_lengths.push_back(0);
      }
      break;
    }
  }
  log_parser_memory_checkpoint("after L-lines");
  l_offsets.clear();
  l_offsets.shrink_to_fit();
  log_parser_memory_checkpoint("after clearing L offsets");

  // Phase 3: Parse P/W-lines (parallel - each writes to pre-allocated index)
  graph.paths.resize(p_offsets.size());
  graph.path_names.resize(p_offsets.size());
  graph.path_overlaps.resize(p_offsets.size());

  graph.walks.walks.resize(w_offsets.size());
  graph.walks.sample_ids.resize(w_offsets.size());
  graph.walks.hap_indices.resize(w_offsets.size());
  graph.walks.seq_ids.resize(w_offsets.size());
  graph.walks.seq_starts.resize(w_offsets.size());
  graph.walks.seq_ends.resize(w_offsets.size());

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (size_t i = 0; i < p_offsets.size(); ++i) {
    std::string_view line(mmap_data + p_offsets[i].offset, p_offsets[i].length);
    parse_p_line(line, graph, i);
  }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (size_t i = 0; i < w_offsets.size(); ++i) {
    std::string_view line(mmap_data + w_offsets[i].offset, w_offsets[i].length);
    parse_w_line(line, graph, i);
  }
  log_parser_memory_checkpoint("after P/W lines");
  p_offsets.clear();
  p_offsets.shrink_to_fit();
  w_offsets.clear();
  w_offsets.shrink_to_fit();
  log_parser_memory_checkpoint("after clearing P/W offsets");

  // Phase 4: Parse J/C lines (after S-lines, so node_name_to_id is populated)
  for (const auto &off : j_offsets) {
    std::string_view line(mmap_data + off.offset, off.length);
    parse_j_line(line, graph);
  }

  for (const auto &off : c_offsets) {
    std::string_view line(mmap_data + off.offset, off.length);
    parse_c_line(line, graph);
  }
  log_parser_memory_checkpoint("after J/C lines");
  j_offsets.clear();
  j_offsets.shrink_to_fit();
  c_offsets.clear();
  c_offsets.shrink_to_fit();
  log_parser_memory_checkpoint("after clearing J/C offsets");

  munmap(const_cast<char *>(mmap_data), file_size);
  close(fd);
  log_parser_memory_checkpoint("after munmap");

  if (gfaz_debug_enabled()) {
    std::cerr << "Parsed " << num_links << " links, " << graph.paths.size()
              << " paths, " << graph.walks.size() << " walks";
    if (graph.jumps.size() > 0 || graph.containments.size() > 0)
      std::cerr << ", " << graph.jumps.size() << " jumps, "
                << graph.containments.size() << " containments";
    std::cerr << std::endl;
  }

  return graph;
}

void GfaParser::parse_s_line(std::string_view line, GfaGraph &graph) {
  size_t pos = 1;
  std::string_view node_name_view = next_field(line, pos);
  std::string_view sequence_view = next_field(line, pos);

  size_t segment_index = 0;
  auto lookup_it = node_name_lookup_.find(node_name_view);
  if (lookup_it == node_name_lookup_.end()) {
    if (all_segment_names_numeric_ && !is_numeric(node_name_view))
      all_segment_names_numeric_ = false;

    uint32_t new_id = graph.node_id_to_name.size();

    // Validate fast-path: numeric segment names must match their assigned IDs
    if (all_segment_names_numeric_) {
      uint32_t numeric_name = parse_uint32(node_name_view);
      if (numeric_name != new_id)
        all_segment_names_numeric_ = false;
    }

    std::string node_name(node_name_view);
    graph.node_name_to_id[node_name] = new_id;
    graph.node_id_to_name.push_back(node_name);
    graph.node_sequences.emplace_back(sequence_view);
    node_name_lookup_.emplace(graph.node_id_to_name.back(), new_id);
    segment_index = new_id - 1;
  } else {
    segment_index = lookup_it->second - 1;
  }

  while (pos < line.size()) {
    std::string_view field = next_field(line, pos);
    if (field.empty())
      break;
    parse_segment_field(field, segment_index, graph);
  }
}

void GfaParser::parse_l_line(std::string_view line, GfaGraph &graph) {
  size_t pos = 1;
  std::string_view from_name_view = next_field(line, pos);
  std::string_view from_orient_view = next_field(line, pos);
  std::string_view to_name_view = next_field(line, pos);
  std::string_view to_orient_view = next_field(line, pos);
  std::string_view overlap_view = next_field(line, pos);

  uint32_t from_id = resolve_node_id(from_name_view);
  uint32_t to_id = resolve_node_id(to_name_view);
  if (from_id == 0 || to_id == 0)
    return;

  graph.links.from_ids.push_back(from_id);
  graph.links.to_ids.push_back(to_id);
  graph.links.from_orients.push_back(
      from_orient_view.empty() ? '+' : from_orient_view[0]);
  graph.links.to_orients.push_back(to_orient_view.empty() ? '+'
                                                          : to_orient_view[0]);

  if (overlap_view.empty() || overlap_view == "*") {
    graph.links.overlap_nums.push_back(0);
    graph.links.overlap_ops.push_back('\0');
  } else {
    char op = '\0';
    graph.links.overlap_nums.push_back(parse_overlap_num(overlap_view, op));
    graph.links.overlap_ops.push_back(op);
  }

  size_t link_index = graph.links.from_ids.size() - 1;
  while (pos < line.size()) {
    std::string_view field = next_field(line, pos);
    if (field.empty())
      break;
    parse_link_field(field, link_index, graph);
  }
}

void GfaParser::parse_p_line(std::string_view line, GfaGraph &graph,
                             size_t index) {
  size_t pos = 1;

  size_t name_start = pos;
  while (pos < line.size() && (line[pos] == ' ' || line[pos] == '\t'))
    ++pos;
  name_start = pos;
  while (pos < line.size() && line[pos] != ' ' && line[pos] != '\t')
    ++pos;
  std::string path_name(line.substr(name_start, pos - name_start));

  while (pos < line.size() && (line[pos] == ' ' || line[pos] == '\t'))
    ++pos;
  size_t nodes_start = pos;
  while (pos < line.size() && line[pos] != ' ' && line[pos] != '\t')
    ++pos;
  std::string_view nodes_str = line.substr(nodes_start, pos - nodes_start);

  while (pos < line.size() && (line[pos] == ' ' || line[pos] == '\t'))
    ++pos;
  std::string overlaps(line.substr(pos));

  std::vector<NodeId> path;
  path.reserve(1 + std::count(nodes_str.begin(), nodes_str.end(), ','));
  size_t node_start = 0;

  for (size_t i = 0; i <= nodes_str.size(); ++i) {
    if (i == nodes_str.size() || nodes_str[i] == ',') {
      if (i > node_start) {
        std::string_view node_with_orient =
            nodes_str.substr(node_start, i - node_start);
        if (!node_with_orient.empty()) {
          char orientation = node_with_orient.back();
          std::string_view node_name_view =
              node_with_orient.substr(0, node_with_orient.size() - 1);

          uint32_t node_id = resolve_node_id(node_name_view);
          if (node_id != 0) {
            NodeId oriented_node_id = node_id;
            if (orientation == '-')
              oriented_node_id = -node_id;
            path.push_back(oriented_node_id);
          }
        }
      }
      node_start = i + 1;
    }
  }

  graph.paths[index] = std::move(path);
  graph.path_names[index] = std::move(path_name);
  graph.path_overlaps[index] = std::move(overlaps);
}

void GfaParser::parse_w_line(std::string_view line, GfaGraph &graph,
                             size_t index) {
  size_t pos = 1;
  std::string_view sample_id_view = next_field(line, pos);
  std::string_view hap_index_view = next_field(line, pos);
  std::string_view seq_id_view = next_field(line, pos);
  std::string_view seq_start_view = next_field(line, pos);
  std::string_view seq_end_view = next_field(line, pos);
  std::string_view walk_str = next_field(line, pos);

  std::vector<NodeId> walk;
  size_t walk_steps = 0;
  for (char c : walk_str) {
    if (c == '>' || c == '<')
      ++walk_steps;
  }
  walk.reserve(walk_steps);
  size_t walk_pos = 0;

  while (walk_pos < walk_str.size()) {
    char orient_char = walk_str[walk_pos];
    if (orient_char != '>' && orient_char != '<') {
      ++walk_pos;
      continue;
    }

    size_t name_start = walk_pos + 1;
    size_t name_end = name_start;
    while (name_end < walk_str.size() && walk_str[name_end] != '>' &&
           walk_str[name_end] != '<')
      ++name_end;

    if (name_end > name_start) {
      std::string_view node_name_view =
          walk_str.substr(name_start, name_end - name_start);

      uint32_t node_id = resolve_node_id(node_name_view);
      if (node_id != 0) {
        NodeId oriented_node_id = node_id;
        if (orient_char == '<')
          oriented_node_id = -node_id;
        walk.push_back(oriented_node_id);
      }
    }

    walk_pos = name_end;
  }

  graph.walks.walks[index] = std::move(walk);
  graph.walks.sample_ids[index] = std::string(sample_id_view);
  graph.walks.hap_indices[index] = 0;
  for (char c : hap_index_view)
    graph.walks.hap_indices[index] =
        graph.walks.hap_indices[index] * 10 + (c - '0');
  graph.walks.seq_ids[index] = std::string(seq_id_view);
  graph.walks.seq_starts[index] =
      (seq_start_view == "*") ? -1 : parse_int64(seq_start_view);
  graph.walks.seq_ends[index] =
      (seq_end_view == "*") ? -1 : parse_int64(seq_end_view);
}

void GfaParser::parse_segment_field(std::string_view field,
                                    size_t segment_index, GfaGraph &graph) {
  if (field.size() < 5 || field[2] != ':' || field[4] != ':')
    return;

  uint16_t tag_key = field_tag_key(field);
  char type = field[3];
  std::string_view value_view = field.substr(5);

  auto it = segment_field_meta_.find(tag_key);
  if (it == segment_field_meta_.end()) {
    size_t col_index = graph.segment_optional_fields.size();
    segment_field_meta_[tag_key] = {type, col_index};

    OptionalFieldColumn col;
    col.tag = std::string(field.substr(0, 2));
    col.type = type;
    if (type == 'i')
      col.int_values.reserve(num_segments_hint_);
    else if (type == 'f')
      col.float_values.reserve(num_segments_hint_);
    else if (type == 'A')
      col.char_values.reserve(num_segments_hint_);
    else if (type == 'Z' || type == 'J' || type == 'H')
      col.string_lengths.reserve(num_segments_hint_);
    else if (type == 'B') {
      col.b_subtypes.reserve(num_segments_hint_);
      col.b_lengths.reserve(num_segments_hint_);
    }
    graph.segment_optional_fields.push_back(col);

    it = segment_field_meta_.find(tag_key);
  }

  char expected_type = it->second.first;
  size_t col_index = it->second.second;

  if (type != expected_type) {
    throw std::runtime_error("Type mismatch for tag '" + graph.segment_optional_fields[col_index].tag +
                             "': expected '" +
                             std::string(1, expected_type) + "', got '" +
                             std::string(1, type) + "'");
  }

  OptionalFieldColumn &col = graph.segment_optional_fields[col_index];

  switch (type) {
  case 'i':
    while (col.int_values.size() < segment_index)
      col.int_values.push_back(std::numeric_limits<int64_t>::min());
    col.int_values.push_back(parse_int64(value_view));
    break;

  case 'f':
    while (col.float_values.size() < segment_index)
      col.float_values.push_back(std::numeric_limits<float>::lowest());
    col.float_values.push_back(parse_float(value_view));
    break;

  case 'A':
    while (col.char_values.size() < segment_index)
      col.char_values.push_back('\0');
    col.char_values.push_back(value_view.empty() ? '\0' : value_view[0]);
    break;

  case 'Z':
  case 'J':
  case 'H':
    while (col.string_lengths.size() < segment_index)
      col.string_lengths.push_back(0);
    col.concatenated_strings.append(value_view.data(), value_view.size());
    col.string_lengths.push_back(static_cast<uint32_t>(value_view.size()));
    break;

  case 'B': {
    while (col.b_subtypes.size() < segment_index) {
      col.b_subtypes.push_back('\0');
      col.b_lengths.push_back(0);
    }

    if (value_view.size() < 2 || value_view[1] != ',') {
      col.b_subtypes.push_back('\0');
      col.b_lengths.push_back(0);
      break;
    }

    char subtype = value_view[0];
    col.b_subtypes.push_back(subtype);

    size_t elem_size = 0;
    switch (subtype) {
    case 'c':
    case 'C':
      elem_size = 1;
      break;
    case 's':
    case 'S':
      elem_size = 2;
      break;
    case 'i':
    case 'I':
    case 'f':
      elem_size = 4;
      break;
    }

    if (elem_size == 0) {
      col.b_lengths.push_back(0);
      break;
    }

    std::vector<uint8_t> bytes;
    std::string values_str(value_view.substr(2));
    std::istringstream vss(values_str);
    std::string token;
    uint32_t count = 0;

    while (std::getline(vss, token, ',')) {
      if (token.empty())
        continue;
      count++;

      if (subtype == 'f') {
        float fval = parse_float(token);
        auto *ptr = reinterpret_cast<uint8_t *>(&fval);
        for (size_t b = 0; b < 4; ++b)
          bytes.push_back(ptr[b]);
      } else if (subtype == 'c') {
        auto ival = static_cast<int8_t>(parse_int64(token));
        bytes.push_back(static_cast<uint8_t>(ival));
      } else if (subtype == 'C') {
        auto ival = static_cast<uint8_t>(parse_uint32(token));
        bytes.push_back(ival);
      } else if (subtype == 's') {
        auto ival = static_cast<int16_t>(parse_int64(token));
        auto *ptr = reinterpret_cast<uint8_t *>(&ival);
        for (size_t b = 0; b < 2; ++b)
          bytes.push_back(ptr[b]);
      } else if (subtype == 'S') {
        auto ival = static_cast<uint16_t>(parse_uint32(token));
        auto *ptr = reinterpret_cast<uint8_t *>(&ival);
        for (size_t b = 0; b < 2; ++b)
          bytes.push_back(ptr[b]);
      } else if (subtype == 'i') {
        auto ival = static_cast<int32_t>(parse_int64(token));
        auto *ptr = reinterpret_cast<uint8_t *>(&ival);
        for (size_t b = 0; b < 4; ++b)
          bytes.push_back(ptr[b]);
      } else if (subtype == 'I') {
        uint32_t ival = parse_uint32(token);
        auto *ptr = reinterpret_cast<uint8_t *>(&ival);
        for (size_t b = 0; b < 4; ++b)
          bytes.push_back(ptr[b]);
      }
    }

    col.b_lengths.push_back(count);
    col.b_concat_bytes.insert(col.b_concat_bytes.end(), bytes.begin(),
                              bytes.end());
    break;
  }

  default:
    std::cerr << kParserWarningPrefix << "unsupported optional field type '"
              << type << "' for tag '" << col.tag << "'" << std::endl;
    break;
  }
}

void GfaParser::parse_link_field(std::string_view field, size_t link_index,
                                 GfaGraph &graph) {
  if (field.size() < 5 || field[2] != ':' || field[4] != ':')
    return;

  uint16_t tag_key = field_tag_key(field);
  char type = field[3];
  std::string_view value_view = field.substr(5);

  auto it = link_field_meta_.find(tag_key);
  if (it == link_field_meta_.end()) {
    size_t col_index = graph.link_optional_fields.size();
    link_field_meta_[tag_key] = {type, col_index};

    OptionalFieldColumn col;
    col.tag = std::string(field.substr(0, 2));
    col.type = type;
    if (type == 'i')
      col.int_values.reserve(num_links_hint_);
    else if (type == 'f')
      col.float_values.reserve(num_links_hint_);
    else if (type == 'A')
      col.char_values.reserve(num_links_hint_);
    else if (type == 'Z' || type == 'J' || type == 'H')
      col.string_lengths.reserve(num_links_hint_);
    else if (type == 'B') {
      col.b_subtypes.reserve(num_links_hint_);
      col.b_lengths.reserve(num_links_hint_);
    }
    graph.link_optional_fields.push_back(col);

    GFAZ_LOG("Discovered link optional field: "
             << col.tag << " (type: " << type << ") at link index " << link_index);

    it = link_field_meta_.find(tag_key);
  }

  char expected_type = it->second.first;
  size_t col_index = it->second.second;

  if (type != expected_type) {
    throw std::runtime_error("Type mismatch for link tag '" +
                             graph.link_optional_fields[col_index].tag +
                             "': expected '" + std::string(1, expected_type) +
                             "', got '" + std::string(1, type) + "'");
  }

  OptionalFieldColumn &col = graph.link_optional_fields[col_index];

  switch (type) {
  case 'i':
    while (col.int_values.size() < link_index)
      col.int_values.push_back(std::numeric_limits<int64_t>::min());
    col.int_values.push_back(parse_int64(value_view));
    break;

  case 'f':
    while (col.float_values.size() < link_index)
      col.float_values.push_back(std::numeric_limits<float>::lowest());
    col.float_values.push_back(parse_float(value_view));
    break;

  case 'A':
    while (col.char_values.size() < link_index)
      col.char_values.push_back('\0');
    col.char_values.push_back(value_view.empty() ? '\0' : value_view[0]);
    break;

  case 'Z':
  case 'J':
  case 'H':
    while (col.string_lengths.size() < link_index)
      col.string_lengths.push_back(0);
    col.concatenated_strings.append(value_view.data(), value_view.size());
    col.string_lengths.push_back(static_cast<uint32_t>(value_view.size()));
    break;

  case 'B': {
    while (col.b_subtypes.size() < link_index) {
      col.b_subtypes.push_back('\0');
      col.b_lengths.push_back(0);
    }

    if (value_view.size() < 2 || value_view[1] != ',') {
      col.b_subtypes.push_back('\0');
      col.b_lengths.push_back(0);
      break;
    }

    char subtype = value_view[0];
    col.b_subtypes.push_back(subtype);

    size_t elem_size = 0;
    switch (subtype) {
    case 'c':
    case 'C':
      elem_size = 1;
      break;
    case 's':
    case 'S':
      elem_size = 2;
      break;
    case 'i':
    case 'I':
    case 'f':
      elem_size = 4;
      break;
    }

    if (elem_size == 0) {
      col.b_lengths.push_back(0);
      break;
    }

    std::vector<uint8_t> bytes;
    std::string values_str(value_view.substr(2));
    std::istringstream vss(values_str);
    std::string token;
    uint32_t count = 0;

    while (std::getline(vss, token, ',')) {
      if (token.empty())
        continue;
      count++;

      if (subtype == 'f') {
        float fval = parse_float(token);
        auto *ptr = reinterpret_cast<uint8_t *>(&fval);
        for (size_t b = 0; b < 4; ++b)
          bytes.push_back(ptr[b]);
      } else if (subtype == 'c') {
        auto ival = static_cast<int8_t>(parse_int64(token));
        bytes.push_back(static_cast<uint8_t>(ival));
      } else if (subtype == 'C') {
        auto ival = static_cast<uint8_t>(parse_uint32(token));
        bytes.push_back(ival);
      } else if (subtype == 's') {
        auto ival = static_cast<int16_t>(parse_int64(token));
        auto *ptr = reinterpret_cast<uint8_t *>(&ival);
        for (size_t b = 0; b < 2; ++b)
          bytes.push_back(ptr[b]);
      } else if (subtype == 'S') {
        auto ival = static_cast<uint16_t>(parse_uint32(token));
        auto *ptr = reinterpret_cast<uint8_t *>(&ival);
        for (size_t b = 0; b < 2; ++b)
          bytes.push_back(ptr[b]);
      } else if (subtype == 'i') {
        auto ival = static_cast<int32_t>(parse_int64(token));
        auto *ptr = reinterpret_cast<uint8_t *>(&ival);
        for (size_t b = 0; b < 4; ++b)
          bytes.push_back(ptr[b]);
      } else if (subtype == 'I') {
        uint32_t ival = parse_uint32(token);
        auto *ptr = reinterpret_cast<uint8_t *>(&ival);
        for (size_t b = 0; b < 4; ++b)
          bytes.push_back(ptr[b]);
      }
    }

    col.b_lengths.push_back(count);
    col.b_concat_bytes.insert(col.b_concat_bytes.end(), bytes.begin(),
                              bytes.end());
    break;
  }

  default:
    std::cerr << kParserWarningPrefix
              << "unsupported link optional field type '" << type
              << "' for tag '" << graph.link_optional_fields[col_index].tag
              << "'" << std::endl;
    break;
  }
}

void GfaParser::parse_j_line(std::string_view line, GfaGraph &graph) {
  // J-line format: J <from_name> <from_orient> <to_name> <to_orient> <distance>
  // [optional fields...]
  size_t pos = 1;
  std::string_view from_name_view = next_field(line, pos);
  std::string_view from_orient_view = next_field(line, pos);
  std::string_view to_name_view = next_field(line, pos);
  std::string_view to_orient_view = next_field(line, pos);
  std::string_view distance_view = next_field(line, pos);

  uint32_t from_id = resolve_node_id(from_name_view);
  uint32_t to_id = resolve_node_id(to_name_view);
  if (from_id == 0 || to_id == 0)
    return;

  graph.jumps.from_ids.push_back(from_id);
  graph.jumps.from_orients.push_back(
      from_orient_view.empty() ? '+' : from_orient_view[0]);
  graph.jumps.to_ids.push_back(to_id);
  graph.jumps.to_orients.push_back(to_orient_view.empty() ? '+'
                                                          : to_orient_view[0]);
  graph.jumps.distances.emplace_back(distance_view);

  // Capture remaining optional fields
  std::string rest;
  while (pos < line.size()) {
    std::string_view field = next_field(line, pos);
    if (field.empty())
      break;
    if (!rest.empty())
      rest += '\t';
    rest.append(field.data(), field.size());
  }
  graph.jumps.rest_fields.push_back(std::move(rest));
}

void GfaParser::parse_c_line(std::string_view line, GfaGraph &graph) {
  // C-line format: C <container> <orient> <contained> <orient> <pos> <overlap>
  // [optional fields...]
  size_t pos = 1;
  std::string_view container_view = next_field(line, pos);
  std::string_view container_orient_view = next_field(line, pos);
  std::string_view contained_view = next_field(line, pos);
  std::string_view contained_orient_view = next_field(line, pos);
  std::string_view position_view = next_field(line, pos);
  std::string_view overlap_view = next_field(line, pos);

  uint32_t container_id = resolve_node_id(container_view);
  uint32_t contained_id = resolve_node_id(contained_view);
  if (container_id == 0 || contained_id == 0)
    return;

  graph.containments.container_ids.push_back(container_id);
  graph.containments.container_orients.push_back(
      container_orient_view.empty() ? '+' : container_orient_view[0]);
  graph.containments.contained_ids.push_back(contained_id);
  graph.containments.contained_orients.push_back(
      contained_orient_view.empty() ? '+' : contained_orient_view[0]);
  graph.containments.positions.push_back(parse_uint32(position_view));
  graph.containments.overlaps.emplace_back(overlap_view);

  // Capture remaining optional fields
  std::string rest;
  while (pos < line.size()) {
    std::string_view field = next_field(line, pos);
    if (field.empty())
      break;
    if (!rest.empty())
      rest += '\t';
    rest.append(field.data(), field.size());
  }
  graph.containments.rest_fields.push_back(std::move(rest));
}

uint32_t GfaParser::resolve_node_id(std::string_view node_name_view) const {
  if (node_name_view.empty())
    return 0;

  if (all_segment_names_numeric_)
    return parse_uint32(node_name_view);

  auto it = node_name_lookup_.find(node_name_view);
  return (it == node_name_lookup_.end()) ? 0 : it->second;
}

uint16_t GfaParser::field_tag_key(std::string_view field) {
  return (static_cast<uint16_t>(static_cast<unsigned char>(field[0])) << 8) |
         static_cast<uint16_t>(static_cast<unsigned char>(field[1]));
}
