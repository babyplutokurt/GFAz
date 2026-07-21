#include "compress/io/gfa_parser.hpp"
#include "core/utils/debug_log.hpp"
#include "core/utils/runtime_utils.hpp"
#include "core/utils/threading_utils.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#if (defined(__x86_64__) || defined(__i386__)) &&                         \
    (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define GFAZ_X86_SIMD 1
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

constexpr const char *kParserErrorPrefix = "GFA parser error: ";
constexpr const char *kParserWarningPrefix = "GFA parser warning: ";
using Clock = std::chrono::steady_clock;

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

#ifdef GFAZ_X86_SIMD
inline bool parse_numeric_path_token(const char *begin, const char *end,
                                     std::vector<gfaz::NodeId> &path) {
  if (begin == end)
    return true;
  if (end - begin < 2)
    return false;

  const char orientation = end[-1];
  if (orientation != '+' && orientation != '-')
    return false;

  const size_t digit_count = static_cast<size_t>(end - begin - 1);
  if (digit_count == 0 || digit_count > 10)
    return false;

  uint64_t value = 0;
  for (size_t i = 0; i < digit_count; ++i) {
    const unsigned digit = static_cast<unsigned>(begin[i] - '0');
    if (digit > 9)
      return false;
    value = value * 10 + digit;
  }
  if (value > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()))
    return false;
  if (value == 0)
    return true;

  const auto node = static_cast<gfaz::NodeId>(value);
  path.push_back(orientation == '-' ? -node : node);
  return true;
}

__attribute__((target("avx2")))
bool parse_numeric_path_avx2(std::string_view nodes,
                             std::vector<gfaz::NodeId> &path) {
  const char *data = nodes.data();
  const size_t size = nodes.size();
  const __m256i commas = _mm256_set1_epi8(',');

  size_t comma_count = 0;
  size_t offset = 0;
  for (; offset + 32 <= size; offset += 32) {
    const __m256i chars =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(data + offset));
    const uint32_t mask = static_cast<uint32_t>(
        _mm256_movemask_epi8(_mm256_cmpeq_epi8(chars, commas)));
    comma_count += static_cast<size_t>(__builtin_popcount(mask));
  }
  for (size_t i = offset; i < size; ++i)
    comma_count += data[i] == ',';

  path.clear();
  path.reserve(comma_count + 1);
  const char *token_begin = data;
  offset = 0;
  for (; offset + 32 <= size; offset += 32) {
    const __m256i chars =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(data + offset));
    uint32_t mask = static_cast<uint32_t>(
        _mm256_movemask_epi8(_mm256_cmpeq_epi8(chars, commas)));
    while (mask != 0) {
      const size_t delimiter =
          offset + static_cast<size_t>(__builtin_ctz(mask));
      if (!parse_numeric_path_token(token_begin, data + delimiter, path))
        return false;
      token_begin = data + delimiter + 1;
      mask &= mask - 1;
    }
  }
  for (size_t i = offset; i < size; ++i) {
    if (data[i] != ',')
      continue;
    if (!parse_numeric_path_token(token_begin, data + i, path))
      return false;
    token_begin = data + i + 1;
  }
  return parse_numeric_path_token(token_begin, data + size, path);
}

__attribute__((target("avx512f,avx512bw")))
bool parse_numeric_path_avx512(std::string_view nodes,
                               std::vector<gfaz::NodeId> &path) {
  const char *data = nodes.data();
  const size_t size = nodes.size();
  const __m512i commas = _mm512_set1_epi8(',');

  size_t comma_count = 0;
  size_t offset = 0;
  for (; offset + 64 <= size; offset += 64) {
    const __m512i chars = _mm512_loadu_si512(data + offset);
    const uint64_t mask = static_cast<uint64_t>(
        _mm512_cmpeq_epi8_mask(chars, commas));
    comma_count += static_cast<size_t>(__builtin_popcountll(mask));
  }
  for (size_t i = offset; i < size; ++i)
    comma_count += data[i] == ',';

  path.clear();
  path.reserve(comma_count + 1);
  const char *token_begin = data;
  offset = 0;
  for (; offset + 64 <= size; offset += 64) {
    const __m512i chars = _mm512_loadu_si512(data + offset);
    uint64_t mask = static_cast<uint64_t>(
        _mm512_cmpeq_epi8_mask(chars, commas));
    while (mask != 0) {
      const size_t delimiter =
          offset + static_cast<size_t>(__builtin_ctzll(mask));
      if (!parse_numeric_path_token(token_begin, data + delimiter, path))
        return false;
      token_begin = data + delimiter + 1;
      mask &= mask - 1;
    }
  }
  for (size_t i = offset; i < size; ++i) {
    if (data[i] != ',')
      continue;
    if (!parse_numeric_path_token(token_begin, data + i, path))
      return false;
    token_begin = data + i + 1;
  }
  return parse_numeric_path_token(token_begin, data + size, path);
}

bool try_parse_numeric_path_simd(std::string_view nodes,
                                 std::vector<gfaz::NodeId> &path) {
  static const bool has_avx512bw = __builtin_cpu_supports("avx512bw");
  static const bool has_avx2 = __builtin_cpu_supports("avx2");
  if (has_avx512bw)
    return parse_numeric_path_avx512(nodes, path);
  return has_avx2 && parse_numeric_path_avx2(nodes, path);
}
#else
bool try_parse_numeric_path_simd(std::string_view,
                                 std::vector<gfaz::NodeId> &) {
  return false;
}
#endif

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

// Parse a SAM/GFA 'B' optional field value ("<subtype>,<v1>,<v2>,...") into the
// column's byte-array storage. Appends exactly one entry to b_subtypes/b_lengths
// (and the packed bytes); callers handle any leading padding to the row index.
// Shared by segment- and link-field parsing.
void parse_b_array(std::string_view value_view, gfaz::OptionalFieldColumn &col) {
  if (value_view.size() < 2 || value_view[1] != ',') {
    col.b_subtypes.push_back('\0');
    col.b_lengths.push_back(0);
    return;
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
    return;
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
}

// Reserve storage for the active type variant of a freshly-discovered optional
// column, sized to the expected row count. Shared by segment/link parsing.
void reserve_optional_column(gfaz::OptionalFieldColumn &col, char type,
                             size_t hint) {
  if (type == 'i')
    col.int_values.reserve(hint);
  else if (type == 'f')
    col.float_values.reserve(hint);
  else if (type == 'A')
    col.char_values.reserve(hint);
  else if (type == 'Z' || type == 'J' || type == 'H')
    col.string_lengths.reserve(hint);
  else if (type == 'B') {
    col.b_subtypes.reserve(hint);
    col.b_lengths.reserve(hint);
  }
}

// Append one optional-field value to `col` for row `row_index`, back-filling any
// skipped rows with the per-type sentinel. Shared by segment- and link-field
// parsing so both materialize columns identically.
void append_optional_value(gfaz::OptionalFieldColumn &col, char type,
                           size_t row_index, std::string_view value_view) {
  switch (type) {
  case 'i':
    while (col.int_values.size() < row_index)
      col.int_values.push_back(std::numeric_limits<int64_t>::min());
    col.int_values.push_back(parse_int64(value_view));
    break;

  case 'f':
    while (col.float_values.size() < row_index)
      col.float_values.push_back(std::numeric_limits<float>::lowest());
    col.float_values.push_back(parse_float(value_view));
    break;

  case 'A':
    while (col.char_values.size() < row_index)
      col.char_values.push_back('\0');
    col.char_values.push_back(value_view.empty() ? '\0' : value_view[0]);
    break;

  case 'Z':
  case 'J':
  case 'H':
    while (col.string_lengths.size() < row_index)
      col.string_lengths.push_back(0);
    col.concatenated_strings.append(value_view.data(), value_view.size());
    col.string_lengths.push_back(static_cast<uint32_t>(value_view.size()));
    break;

  case 'B': {
    while (col.b_subtypes.size() < row_index) {
      col.b_subtypes.push_back('\0');
      col.b_lengths.push_back(0);
    }
    parse_b_array(value_view, col);
    break;
  }

  default:
    std::cerr << kParserWarningPrefix << "unsupported optional field type '"
              << type << "' for tag '" << col.tag << "'" << std::endl;
    break;
  }
}

bool numeric_name_matches_id(std::string_view name, uint32_t expected) {
  if (name.empty())
    return false;
  uint64_t value = 0;
  for (char c : name) {
    if (c < '0' || c > '9')
      return false;
    value = value * 10 + static_cast<uint32_t>(c - '0');
    if (value > std::numeric_limits<uint32_t>::max())
      return false;
  }
  return value == expected;
}

using FixedFieldMeta =
    std::unordered_map<uint16_t, std::pair<char, size_t>>;

bool initialize_fixed_optional_columns(
    std::string_view line, size_t pos, size_t row_count,
    std::vector<gfaz::OptionalFieldColumn> &columns, FixedFieldMeta &meta) {
  columns.clear();
  meta.clear();
  while (pos < line.size()) {
    const std::string_view field = next_field(line, pos);
    if (field.empty())
      break;
    if (field.size() < 5 || field[2] != ':' || field[4] != ':')
      continue;
    const char type = field[3];
    if (type != 'i' && type != 'f' && type != 'A')
      return false;
    const uint16_t key =
        (static_cast<uint16_t>(static_cast<unsigned char>(field[0])) << 8) |
        static_cast<uint16_t>(static_cast<unsigned char>(field[1]));
    if (meta.find(key) != meta.end())
      return false;

    const size_t index = columns.size();
    if (index >= 64)
      return false;
    meta.emplace(key, std::make_pair(type, index));
    gfaz::OptionalFieldColumn col;
    col.tag = std::string(field.substr(0, 2));
    col.type = type;
    if (type == 'i')
      col.int_values.assign(row_count, std::numeric_limits<int64_t>::min());
    else if (type == 'f')
      col.float_values.assign(row_count, std::numeric_limits<float>::lowest());
    else
      col.char_values.assign(row_count, '\0');
    columns.push_back(std::move(col));
  }
  return true;
}

bool assign_fixed_optional_fields(
    std::string_view line, size_t pos, size_t row,
    std::vector<gfaz::OptionalFieldColumn> &columns,
    const FixedFieldMeta &meta) {
  uint64_t assigned = 0;
  while (pos < line.size()) {
    const std::string_view field = next_field(line, pos);
    if (field.empty())
      break;
    if (field.size() < 5 || field[2] != ':' || field[4] != ':')
      continue;
    const uint16_t key =
        (static_cast<uint16_t>(static_cast<unsigned char>(field[0])) << 8) |
        static_cast<uint16_t>(static_cast<unsigned char>(field[1]));
    const auto it = meta.find(key);
    if (it == meta.end() || it->second.first != field[3])
      return false;
    const uint64_t bit = uint64_t{1} << it->second.second;
    if ((assigned & bit) != 0)
      return false;
    assigned |= bit;

    auto &col = columns[it->second.second];
    const std::string_view value = field.substr(5);
    if (col.type == 'i')
      col.int_values[row] = parse_int64(value);
    else if (col.type == 'f')
      col.float_values[row] = parse_float(value);
    else
      col.char_values[row] = value.empty() ? '\0' : value[0];
  }
  return true;
}

} // namespace

using gfaz::runtime_utils::format_memory_snapshot;
using gfaz::runtime_utils::format_size;
using gfaz::runtime_utils::read_process_memory_snapshot;

namespace {

template <typename T> size_t vector_buffer_bytes(const std::vector<T> &values) {
  return values.capacity() * sizeof(T);
}

size_t string_owned_bytes(const std::string &value) {
  return sizeof(std::string) + value.capacity();
}

size_t string_vector_owned_bytes(const std::vector<std::string> &values) {
  size_t bytes = values.capacity() * sizeof(std::string);
  for (const auto &value : values)
    bytes += value.capacity();
  return bytes;
}

size_t
nested_node_vector_bytes(const std::vector<std::vector<gfaz::NodeId>> &sequences) {
  size_t bytes = sequences.capacity() * sizeof(std::vector<gfaz::NodeId>);
  for (const auto &seq : sequences)
    bytes += seq.capacity() * sizeof(gfaz::NodeId);
  return bytes;
}

size_t optional_field_column_bytes(const gfaz::OptionalFieldColumn &col) {
  return string_owned_bytes(col.tag) + vector_buffer_bytes(col.int_values) +
         vector_buffer_bytes(col.float_values) +
         vector_buffer_bytes(col.char_values) +
         string_owned_bytes(col.concatenated_strings) +
         vector_buffer_bytes(col.string_lengths) +
         vector_buffer_bytes(col.b_subtypes) +
         vector_buffer_bytes(col.b_lengths) +
         vector_buffer_bytes(col.b_concat_bytes);
}

size_t
optional_field_columns_bytes(const std::vector<gfaz::OptionalFieldColumn> &cols) {
  size_t bytes = cols.capacity() * sizeof(gfaz::OptionalFieldColumn);
  for (const auto &col : cols)
    bytes += optional_field_column_bytes(col);
  return bytes;
}

size_t segment_bytes(const gfaz::GfaGraph &graph) {
  return string_vector_owned_bytes(graph.segments.node_id_to_name) +
         string_vector_owned_bytes(graph.segments.node_sequences) +
         string_owned_bytes(graph.segments.node_sequences_concat) +
         vector_buffer_bytes(graph.segments.node_sequence_lengths);
}

size_t path_bytes(const gfaz::GfaGraph &graph) {
  return nested_node_vector_bytes(graph.paths_data.traversals) +
         string_vector_owned_bytes(graph.paths_data.names) +
         string_vector_owned_bytes(graph.paths_data.overlaps);
}

size_t walk_bytes(const gfaz::GfaGraph &graph) {
  return nested_node_vector_bytes(graph.walks.walks) +
         string_vector_owned_bytes(graph.walks.sample_ids) +
         vector_buffer_bytes(graph.walks.hap_indices) +
         string_vector_owned_bytes(graph.walks.seq_ids) +
         vector_buffer_bytes(graph.walks.seq_starts) +
         vector_buffer_bytes(graph.walks.seq_ends);
}

size_t link_bytes(const gfaz::GfaGraph &graph) {
  return vector_buffer_bytes(graph.links.from_ids) +
         vector_buffer_bytes(graph.links.to_ids) +
         vector_buffer_bytes(graph.links.from_orients) +
         vector_buffer_bytes(graph.links.to_orients) +
         vector_buffer_bytes(graph.links.overlap_nums) +
         vector_buffer_bytes(graph.links.overlap_ops);
}

size_t jump_bytes(const gfaz::GfaGraph &graph) {
  return vector_buffer_bytes(graph.jumps.from_ids) +
         vector_buffer_bytes(graph.jumps.from_orients) +
         vector_buffer_bytes(graph.jumps.to_ids) +
         vector_buffer_bytes(graph.jumps.to_orients) +
         string_vector_owned_bytes(graph.jumps.distances) +
         string_vector_owned_bytes(graph.jumps.rest_fields);
}

size_t containment_bytes(const gfaz::GfaGraph &graph) {
  return vector_buffer_bytes(graph.containments.container_ids) +
         vector_buffer_bytes(graph.containments.container_orients) +
         vector_buffer_bytes(graph.containments.contained_ids) +
         vector_buffer_bytes(graph.containments.contained_orients) +
         vector_buffer_bytes(graph.containments.positions) +
         string_vector_owned_bytes(graph.containments.overlaps) +
         string_vector_owned_bytes(graph.containments.rest_fields);
}

void print_graph_memory_breakdown(const gfaz::GfaGraph &graph) {
  const size_t segment_data = segment_bytes(graph);
  const size_t path_data = path_bytes(graph);
  const size_t walk_data = walk_bytes(graph);
  const size_t link_data = link_bytes(graph);
  const size_t jump_data = jump_bytes(graph);
  const size_t containment_data = containment_bytes(graph);
  const size_t segment_optional =
      optional_field_columns_bytes(graph.segments.optional_fields);
  const size_t link_optional =
      optional_field_columns_bytes(graph.link_optional_fields);
  // node_name_to_id is intentionally empty on the CPU parse path (see
  // parse_s_line), so it is not tracked here.
  const size_t total = segment_data + path_data + walk_data + link_data +
                       jump_data + containment_data + segment_optional +
                       link_optional;

  std::cerr << "[GfaParser] approximate graph memory:" << std::endl;
  std::cerr << "  segments:                 " << format_size(segment_data)
            << std::endl;
  std::cerr << "  paths:                    " << format_size(path_data)
            << std::endl;
  std::cerr << "  walks:                    " << format_size(walk_data)
            << std::endl;
  std::cerr << "  links:                    " << format_size(link_data)
            << std::endl;
  std::cerr << "  segment optional fields:  " << format_size(segment_optional)
            << std::endl;
  std::cerr << "  link optional fields:     " << format_size(link_optional)
            << std::endl;
  if (graph.jumps.size() > 0) {
    std::cerr << "  jumps:                    " << format_size(jump_data)
              << std::endl;
  }
  if (graph.containments.size() > 0) {
    std::cerr << "  containments:             " << format_size(containment_data)
              << std::endl;
  }
  std::cerr << "  total tracked:            " << format_size(total)
            << std::endl;
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

gfaz::GfaGraph GfaParser::parse(const std::string &gfa_file_path, int num_threads,
                                bool direct_segment_columns) {
  ScopedOMPThreads omp_scope(num_threads);
  const auto parse_start = Clock::now();
  auto phase_start = parse_start;
  gfaz::GfaGraph graph;
  segment_field_meta_.clear();
  link_field_meta_.clear();
  node_name_lookup_.clear();
  num_segments_hint_ = 0;
  num_links_hint_ = 0;
  all_segment_names_numeric_ = true;

  auto log_phase = [&](const std::string &label) {
    if (!gfaz_debug_enabled())
      return;
    const auto now = Clock::now();
    const double phase_ms =
        std::chrono::duration<double, std::milli>(now - phase_start).count();
    phase_start = now;

    const auto snapshot = read_process_memory_snapshot();
    std::cerr << "[GfaParser] " << label << ": " << std::fixed
              << std::setprecision(2) << phase_ms << " ms"
              << " | " << format_memory_snapshot(snapshot) << std::endl;
  };

  // Index 0 is a placeholder to support 1-based node IDs.
  // This allows gfaz::NodeId sign to encode orientation without ambiguity.
  graph.segments.node_id_to_name.push_back("");
  graph.segments.node_sequences.push_back("");

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

  // mmap() rejects a zero-length mapping; a 0-byte GFA is simply an empty graph
  // (the index-0 placeholder segment is already in place).
  if (file_size == 0) {
    close(fd);
    return graph;
  }

  const char *mmap_data = static_cast<const char *>(
      mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0));
  if (mmap_data == MAP_FAILED) {
    close(fd);
    throw std::runtime_error(std::string(kParserErrorPrefix) +
                             "failed to mmap input file '" + gfa_file_path +
                             "': " + std::strerror(errno));
  }

  madvise(const_cast<char *>(mmap_data), file_size, MADV_SEQUENTIAL);
  log_phase("mmap+madvise");

  // Dense numeric GFAs can be parsed without retaining an offset for every
  // record. The first pass counts each record family per ordered byte range;
  // the second uses family-specific prefix sums to write directly into exact
  // output slots. Cross-family records may be arbitrarily interleaved.
  bool try_direct_parser = true;
  size_t direct_probe_start = 0;
  while (direct_probe_start < file_size) {
    const void *newline =
        std::memchr(mmap_data + direct_probe_start, '\n',
                    file_size - direct_probe_start);
    const size_t line_end =
        newline == nullptr
            ? file_size
            : static_cast<size_t>(
                  static_cast<const char *>(newline) - mmap_data);
    size_t line_len = line_end - direct_probe_start;
    if (line_len > 0 &&
        mmap_data[direct_probe_start + line_len - 1] == '\r')
      --line_len;

    if (line_len > 0 && mmap_data[direct_probe_start] == 'S') {
      const std::string_view line(mmap_data + direct_probe_start, line_len);
      size_t pos = 1;
      const std::string_view name = next_field(line, pos);
      (void)next_field(line, pos);
      try_direct_parser = numeric_name_matches_id(name, 1);
      if (try_direct_parser) {
        std::vector<gfaz::OptionalFieldColumn> probe_columns;
        FixedFieldMeta probe_meta;
        try_direct_parser = initialize_fixed_optional_columns(
            line, pos, 0, probe_columns, probe_meta);
      }
      break;
    }

    direct_probe_start = line_end + (newline == nullptr ? 0 : 1);
  }

  if (try_direct_parser) {
    struct DirectChunkStats {
      size_t segments = 0;
      size_t links = 0;
      size_t paths = 0;
      size_t walks = 0;
      size_t jumps = 0;
      size_t containments = 0;
      size_t sequence_bytes = 0;
      uint32_t first_segment_id = 0;
      uint32_t last_segment_id = 0;
      bool numeric_segments = true;
      bool locally_sequential_segments = true;
      gfaz::LineOffset first_segment{};
      gfaz::LineOffset first_link{};
      gfaz::LineOffset last_header{};
      bool has_first_segment = false;
      bool has_first_link = false;
      bool has_header = false;
    };

    const int direct_threads =
        std::max(1, resolve_omp_thread_count(num_threads));
    std::vector<size_t> direct_boundaries(
        static_cast<size_t>(direct_threads) + 1);
    direct_boundaries.front() = 0;
    direct_boundaries.back() = file_size;
    for (int t = 1; t < direct_threads; ++t) {
      size_t boundary =
          (file_size / static_cast<size_t>(direct_threads)) *
          static_cast<size_t>(t);
      while (boundary < file_size && mmap_data[boundary - 1] != '\n')
        ++boundary;
      direct_boundaries[static_cast<size_t>(t)] = boundary;
    }

    const size_t direct_page_size =
        static_cast<size_t>(sysconf(_SC_PAGESIZE));
    constexpr size_t kDirectReleaseWindow = 16ull * 1024ull * 1024ull;
    std::vector<DirectChunkStats> direct_stats(
        static_cast<size_t>(direct_threads));
    std::atomic<bool> direct_scan_failed{false};

    auto parse_numeric_id_noexcept = [](std::string_view value,
                                        uint32_t &parsed) {
      if (value.empty())
        return false;
      uint64_t result = 0;
      for (char c : value) {
        if (c < '0' || c > '9')
          return false;
        result = result * 10 + static_cast<uint32_t>(c - '0');
        if (result > std::numeric_limits<uint32_t>::max())
          return false;
      }
      parsed = static_cast<uint32_t>(result);
      return true;
    };

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int t = 0; t < direct_threads; ++t) {
      auto &local = direct_stats[static_cast<size_t>(t)];
      const size_t range_start =
          direct_boundaries[static_cast<size_t>(t)];
      const size_t range_end =
          direct_boundaries[static_cast<size_t>(t) + 1];
      size_t line_start = range_start;
      size_t release_cursor =
          ((range_start + direct_page_size - 1) / direct_page_size) *
          direct_page_size;

      while (line_start < range_end) {
        if (direct_scan_failed.load(std::memory_order_relaxed))
          break;

        const void *newline =
            std::memchr(mmap_data + line_start, '\n', range_end - line_start);
        const size_t line_end =
            newline == nullptr
                ? range_end
                : static_cast<size_t>(static_cast<const char *>(newline) -
                                      mmap_data);
        size_t line_len = line_end - line_start;
        if (line_len > 0 && mmap_data[line_start + line_len - 1] == '\r')
          --line_len;

        if (line_len > 0) {
          const gfaz::LineOffset offset{line_start, line_len};
          switch (mmap_data[line_start]) {
          case 'S': {
            if (!local.has_first_segment) {
              local.first_segment = offset;
              local.has_first_segment = true;
            }
            const std::string_view line(mmap_data + line_start, line_len);
            size_t pos = 1;
            const std::string_view name = next_field(line, pos);
            const std::string_view sequence = next_field(line, pos);
            if (sequence.size() > std::numeric_limits<uint32_t>::max())
              direct_scan_failed.store(true, std::memory_order_relaxed);

            uint32_t id = 0;
            if (!parse_numeric_id_noexcept(name, id) || id == 0) {
              local.numeric_segments = false;
              direct_scan_failed.store(true, std::memory_order_relaxed);
              break;
            } else {
              if (local.segments == 0) {
                local.first_segment_id = id;
              } else if (id != local.last_segment_id + 1) {
                local.locally_sequential_segments = false;
                direct_scan_failed.store(true, std::memory_order_relaxed);
                break;
              }
              local.last_segment_id = id;
            }
            if (sequence.size() >
                std::numeric_limits<size_t>::max() - local.sequence_bytes) {
              direct_scan_failed.store(true, std::memory_order_relaxed);
            } else {
              local.sequence_bytes += sequence.size();
            }
            ++local.segments;
            break;
          }
          case 'L':
            if (!local.has_first_link) {
              local.first_link = offset;
              local.has_first_link = true;
            }
            ++local.links;
            break;
          case 'P':
            ++local.paths;
            break;
          case 'W':
            ++local.walks;
            break;
          case 'J':
            ++local.jumps;
            break;
          case 'C':
            ++local.containments;
            break;
          case 'H':
            local.last_header = offset;
            local.has_header = true;
            break;
          }
        }

        line_start = line_end + (newline == nullptr ? 0 : 1);
        if (line_start >= release_cursor + kDirectReleaseWindow) {
          const size_t release_end =
              (line_start / direct_page_size) * direct_page_size;
          if (release_end > release_cursor) {
            madvise(const_cast<char *>(mmap_data) + release_cursor,
                    release_end - release_cursor, MADV_DONTNEED);
            release_cursor = release_end;
          }
        }
      }

      const size_t release_end =
          (range_end / direct_page_size) * direct_page_size;
      if (release_end > release_cursor) {
        madvise(const_cast<char *>(mmap_data) + release_cursor,
                release_end - release_cursor, MADV_DONTNEED);
      }
    }

    std::vector<size_t> segment_bases(
        static_cast<size_t>(direct_threads) + 1, 0);
    std::vector<size_t> link_bases(
        static_cast<size_t>(direct_threads) + 1, 0);
    std::vector<size_t> path_bases(
        static_cast<size_t>(direct_threads) + 1, 0);
    std::vector<size_t> walk_bases(
        static_cast<size_t>(direct_threads) + 1, 0);
    std::vector<size_t> jump_bases(
        static_cast<size_t>(direct_threads) + 1, 0);
    std::vector<size_t> containment_bases(
        static_cast<size_t>(direct_threads) + 1, 0);
    std::vector<size_t> sequence_byte_bases(
        static_cast<size_t>(direct_threads) + 1, 0);

    bool direct_eligible =
        !direct_scan_failed.load(std::memory_order_relaxed);
    gfaz::LineOffset first_segment{};
    gfaz::LineOffset first_link{};
    bool have_first_segment = false;
    bool have_first_link = false;
    gfaz::LineOffset last_header{};
    bool have_header = false;

    for (int t = 0; t < direct_threads; ++t) {
      const size_t i = static_cast<size_t>(t);
      const auto &local = direct_stats[i];
      segment_bases[i + 1] = segment_bases[i] + local.segments;
      link_bases[i + 1] = link_bases[i] + local.links;
      path_bases[i + 1] = path_bases[i] + local.paths;
      walk_bases[i + 1] = walk_bases[i] + local.walks;
      jump_bases[i + 1] = jump_bases[i] + local.jumps;
      containment_bases[i + 1] =
          containment_bases[i] + local.containments;
      if (local.sequence_bytes >
          std::numeric_limits<size_t>::max() - sequence_byte_bases[i]) {
        direct_eligible = false;
      } else {
        sequence_byte_bases[i + 1] =
            sequence_byte_bases[i] + local.sequence_bytes;
      }

      if (local.segments > 0) {
        const uint64_t expected_first =
            static_cast<uint64_t>(segment_bases[i]) + 1;
        if (!local.numeric_segments ||
            !local.locally_sequential_segments ||
            expected_first > std::numeric_limits<uint32_t>::max() ||
            local.first_segment_id != expected_first) {
          direct_eligible = false;
        }
        if (!have_first_segment) {
          first_segment = local.first_segment;
          have_first_segment = true;
        }
      }
      if (local.has_first_link && !have_first_link) {
        first_link = local.first_link;
        have_first_link = true;
      }
      if (local.has_header) {
        last_header = local.last_header;
        have_header = true;
      }
    }

    FixedFieldMeta direct_segment_meta;
    FixedFieldMeta direct_link_meta;
    std::vector<gfaz::OptionalFieldColumn> direct_segment_optional_columns;
    std::vector<gfaz::OptionalFieldColumn> direct_link_optional_columns;
    if (direct_eligible && have_first_segment) {
      const std::string_view line(mmap_data + first_segment.offset,
                                  first_segment.length);
      size_t pos = 1;
      (void)next_field(line, pos);
      (void)next_field(line, pos);
      direct_eligible = initialize_fixed_optional_columns(
          line, pos, segment_bases.back(), direct_segment_optional_columns,
          direct_segment_meta);
    }
    if (direct_eligible && have_first_link) {
      const std::string_view line(mmap_data + first_link.offset,
                                  first_link.length);
      size_t pos = 1;
      for (int field = 0; field < 5; ++field)
        (void)next_field(line, pos);
      direct_eligible = initialize_fixed_optional_columns(
          line, pos, link_bases.back(), direct_link_optional_columns,
          direct_link_meta);
    }
    log_phase(direct_eligible ? "direct pass 1" : "direct pass 1 fallback");

    if (direct_eligible) {
      gfaz::GfaGraph direct_graph;
      direct_graph.segments.node_id_to_name.push_back("");
      direct_graph.segments.node_sequences.push_back("");
      if (have_header) {
        direct_graph.header_line =
            std::string(mmap_data + last_header.offset, last_header.length);
      }

      if (direct_segment_columns) {
        direct_graph.segments.node_sequence_lengths.resize(
            segment_bases.back());
        direct_graph.segments.node_sequences_concat.resize(
            sequence_byte_bases.back());
      } else {
        direct_graph.segments.node_id_to_name.resize(
            segment_bases.back() + 1);
        direct_graph.segments.node_sequences.resize(
            segment_bases.back() + 1);
      }
      direct_graph.segments.optional_fields =
          std::move(direct_segment_optional_columns);

      direct_graph.links.from_ids.resize(link_bases.back());
      direct_graph.links.to_ids.resize(link_bases.back());
      direct_graph.links.from_orients.resize(link_bases.back());
      direct_graph.links.to_orients.resize(link_bases.back());
      direct_graph.links.overlap_nums.resize(link_bases.back());
      direct_graph.links.overlap_ops.resize(link_bases.back());
      direct_graph.link_optional_fields =
          std::move(direct_link_optional_columns);

      direct_graph.paths_data.traversals.resize(path_bases.back());
      direct_graph.paths_data.names.resize(path_bases.back());
      direct_graph.paths_data.overlaps.resize(path_bases.back());
      direct_graph.walks.walks.resize(walk_bases.back());
      direct_graph.walks.sample_ids.resize(walk_bases.back());
      direct_graph.walks.hap_indices.resize(walk_bases.back());
      direct_graph.walks.seq_ids.resize(walk_bases.back());
      direct_graph.walks.seq_starts.resize(walk_bases.back());
      direct_graph.walks.seq_ends.resize(walk_bases.back());

      direct_graph.jumps.from_ids.resize(jump_bases.back());
      direct_graph.jumps.from_orients.resize(jump_bases.back());
      direct_graph.jumps.to_ids.resize(jump_bases.back());
      direct_graph.jumps.to_orients.resize(jump_bases.back());
      direct_graph.jumps.distances.resize(jump_bases.back());
      direct_graph.jumps.rest_fields.resize(jump_bases.back());

      direct_graph.containments.container_ids.resize(
          containment_bases.back());
      direct_graph.containments.container_orients.resize(
          containment_bases.back());
      direct_graph.containments.contained_ids.resize(
          containment_bases.back());
      direct_graph.containments.contained_orients.resize(
          containment_bases.back());
      direct_graph.containments.positions.resize(
          containment_bases.back());
      direct_graph.containments.overlaps.resize(
          containment_bases.back());
      direct_graph.containments.rest_fields.resize(
          containment_bases.back());

      num_segments_hint_ = segment_bases.back();
      num_links_hint_ = link_bases.back();
      all_segment_names_numeric_ = true;
      std::atomic<bool> direct_fill_failed{false};

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (int t = 0; t < direct_threads; ++t) {
        const size_t chunk = static_cast<size_t>(t);
        const size_t range_start = direct_boundaries[chunk];
        const size_t range_end = direct_boundaries[chunk + 1];
        size_t segment_index = segment_bases[chunk];
        size_t link_index = link_bases[chunk];
        size_t path_index = path_bases[chunk];
        size_t walk_index = walk_bases[chunk];
        size_t jump_index = jump_bases[chunk];
        size_t containment_index = containment_bases[chunk];
        size_t sequence_byte_index = sequence_byte_bases[chunk];
        size_t line_start = range_start;
        size_t release_cursor =
            ((range_start + direct_page_size - 1) / direct_page_size) *
            direct_page_size;

        while (line_start < range_end) {
          const void *newline =
              std::memchr(mmap_data + line_start, '\n',
                          range_end - line_start);
          const size_t line_end =
              newline == nullptr
                  ? range_end
                  : static_cast<size_t>(
                        static_cast<const char *>(newline) - mmap_data);
          size_t line_len = line_end - line_start;
          if (line_len > 0 && mmap_data[line_start + line_len - 1] == '\r')
            --line_len;

          if (line_len > 0 &&
              !direct_fill_failed.load(std::memory_order_relaxed)) {
            try {
              const std::string_view line(mmap_data + line_start, line_len);
              switch (mmap_data[line_start]) {
              case 'S': {
                size_t pos = 1;
                const std::string_view name = next_field(line, pos);
                const std::string_view sequence = next_field(line, pos);
                if (!numeric_name_matches_id(
                        name, static_cast<uint32_t>(segment_index + 1)) ||
                    !assign_fixed_optional_fields(
                        line, pos, segment_index,
                        direct_graph.segments.optional_fields,
                        direct_segment_meta)) {
                  direct_fill_failed.store(true, std::memory_order_relaxed);
                  break;
                }
                if (direct_segment_columns) {
                  direct_graph.segments.node_sequence_lengths[segment_index] =
                      static_cast<uint32_t>(sequence.size());
                  std::memcpy(
                      direct_graph.segments.node_sequences_concat.data() +
                          sequence_byte_index,
                      sequence.data(), sequence.size());
                  sequence_byte_index += sequence.size();
                } else {
                  direct_graph.segments.node_id_to_name[segment_index + 1] =
                      std::string(name);
                  direct_graph.segments.node_sequences[segment_index + 1] =
                      std::string(sequence);
                }
                ++segment_index;
                break;
              }
              case 'L': {
                size_t pos = 1;
                const uint32_t from_id = parse_uint32(next_field(line, pos));
                const std::string_view from_orient = next_field(line, pos);
                const uint32_t to_id = parse_uint32(next_field(line, pos));
                const std::string_view to_orient = next_field(line, pos);
                const std::string_view overlap = next_field(line, pos);
                if (from_id == 0 || to_id == 0 ||
                    !assign_fixed_optional_fields(
                        line, pos, link_index,
                        direct_graph.link_optional_fields,
                        direct_link_meta)) {
                  direct_fill_failed.store(true, std::memory_order_relaxed);
                  break;
                }
                direct_graph.links.from_ids[link_index] = from_id;
                direct_graph.links.to_ids[link_index] = to_id;
                direct_graph.links.from_orients[link_index] =
                    from_orient.empty() ? '+' : from_orient[0];
                direct_graph.links.to_orients[link_index] =
                    to_orient.empty() ? '+' : to_orient[0];
                if (overlap.empty() || overlap == "*") {
                  direct_graph.links.overlap_nums[link_index] = 0;
                  direct_graph.links.overlap_ops[link_index] = '\0';
                } else {
                  char op = '\0';
                  direct_graph.links.overlap_nums[link_index] =
                      parse_overlap_num(overlap, op);
                  direct_graph.links.overlap_ops[link_index] = op;
                }
                ++link_index;
                break;
              }
              case 'P':
                parse_p_line(line, direct_graph, path_index++);
                break;
              case 'W':
                parse_w_line(line, direct_graph, walk_index++);
                break;
              case 'J': {
                size_t pos = 1;
                const uint32_t from_id =
                    parse_uint32(next_field(line, pos));
                const std::string_view from_orient = next_field(line, pos);
                const uint32_t to_id = parse_uint32(next_field(line, pos));
                const std::string_view to_orient = next_field(line, pos);
                const std::string_view distance = next_field(line, pos);
                if (from_id == 0 || to_id == 0) {
                  direct_fill_failed.store(true, std::memory_order_relaxed);
                  break;
                }
                direct_graph.jumps.from_ids[jump_index] = from_id;
                direct_graph.jumps.from_orients[jump_index] =
                    from_orient.empty() ? '+' : from_orient[0];
                direct_graph.jumps.to_ids[jump_index] = to_id;
                direct_graph.jumps.to_orients[jump_index] =
                    to_orient.empty() ? '+' : to_orient[0];
                direct_graph.jumps.distances[jump_index] =
                    std::string(distance);
                std::string rest;
                while (pos < line.size()) {
                  const std::string_view field = next_field(line, pos);
                  if (field.empty())
                    break;
                  if (!rest.empty())
                    rest += '\t';
                  rest.append(field.data(), field.size());
                }
                direct_graph.jumps.rest_fields[jump_index] = std::move(rest);
                ++jump_index;
                break;
              }
              case 'C': {
                size_t pos = 1;
                const uint32_t container_id =
                    parse_uint32(next_field(line, pos));
                const std::string_view container_orient =
                    next_field(line, pos);
                const uint32_t contained_id =
                    parse_uint32(next_field(line, pos));
                const std::string_view contained_orient =
                    next_field(line, pos);
                const std::string_view position = next_field(line, pos);
                const std::string_view overlap = next_field(line, pos);
                if (container_id == 0 || contained_id == 0) {
                  direct_fill_failed.store(true, std::memory_order_relaxed);
                  break;
                }
                direct_graph.containments.container_ids[containment_index] =
                    container_id;
                direct_graph.containments.container_orients[
                    containment_index] =
                    container_orient.empty() ? '+' : container_orient[0];
                direct_graph.containments.contained_ids[containment_index] =
                    contained_id;
                direct_graph.containments.contained_orients[
                    containment_index] =
                    contained_orient.empty() ? '+' : contained_orient[0];
                direct_graph.containments.positions[containment_index] =
                    parse_uint32(position);
                direct_graph.containments.overlaps[containment_index] =
                    std::string(overlap);
                std::string rest;
                while (pos < line.size()) {
                  const std::string_view field = next_field(line, pos);
                  if (field.empty())
                    break;
                  if (!rest.empty())
                    rest += '\t';
                  rest.append(field.data(), field.size());
                }
                direct_graph.containments.rest_fields[containment_index] =
                    std::move(rest);
                ++containment_index;
                break;
              }
              }
            } catch (...) {
              direct_fill_failed.store(true, std::memory_order_relaxed);
            }
          }

          line_start = line_end + (newline == nullptr ? 0 : 1);
          if (line_start >= release_cursor + kDirectReleaseWindow) {
            const size_t release_end =
                (line_start / direct_page_size) * direct_page_size;
            if (release_end > release_cursor) {
              madvise(const_cast<char *>(mmap_data) + release_cursor,
                      release_end - release_cursor, MADV_DONTNEED);
              release_cursor = release_end;
            }
          }
        }

        const size_t release_end =
            (range_end / direct_page_size) * direct_page_size;
        if (release_end > release_cursor) {
          madvise(const_cast<char *>(mmap_data) + release_cursor,
                  release_end - release_cursor, MADV_DONTNEED);
        }
      }

      if (!direct_fill_failed.load(std::memory_order_relaxed)) {
        graph = std::move(direct_graph);
        log_phase("direct pass 2");
        madvise(const_cast<char *>(mmap_data), file_size, MADV_DONTNEED);
        munmap(const_cast<char *>(mmap_data), file_size);
        close(fd);
        log_phase("munmap+close");

        if (gfaz_debug_enabled()) {
          const auto parse_end = Clock::now();
          const double parse_ms =
              std::chrono::duration<double, std::milli>(parse_end - parse_start)
                  .count();
          const auto snapshot = read_process_memory_snapshot();
          std::cerr << "[GfaParser] segments=" << graph.segments.size()
                    << ", links=" << graph.links.from_ids.size()
                    << ", paths=" << graph.paths_data.traversals.size()
                    << ", walks=" << graph.walks.size()
                    << ", jumps=" << graph.jumps.size()
                    << ", containments=" << graph.containments.size()
                    << ", mode=direct-two-pass"
                    << ", time=" << std::fixed << std::setprecision(2)
                    << parse_ms << " ms"
                    << " | " << format_memory_snapshot(snapshot) << std::endl;
          print_graph_memory_breakdown(graph);
        }
        return graph;
      }

      log_phase("direct pass 2 fallback");
      segment_field_meta_.clear();
      link_field_meta_.clear();
      node_name_lookup_.clear();
      num_segments_hint_ = 0;
      num_links_hint_ = 0;
      all_segment_names_numeric_ = true;
    }

    madvise(const_cast<char *>(mmap_data), file_size, MADV_DONTNEED);
  } else {
    madvise(const_cast<char *>(mmap_data), file_size, MADV_DONTNEED);
    log_phase("direct preflight fallback");
  }

  // Parallel line classification. Split the mapping at newline boundaries so
  // each worker owns complete lines, then concatenate the per-range offsets in
  // range order to preserve the input order within each record type.
  std::vector<gfaz::LineOffset> s_offsets, l_offsets, p_offsets, w_offsets;
  std::vector<gfaz::LineOffset> j_offsets, c_offsets;

  struct ClassifiedLines {
    std::vector<gfaz::LineOffset> segments;
    std::vector<gfaz::LineOffset> links;
    std::vector<gfaz::LineOffset> paths;
    std::vector<gfaz::LineOffset> walks;
    std::vector<gfaz::LineOffset> jumps;
    std::vector<gfaz::LineOffset> containments;
    gfaz::LineOffset last_header{};
    bool has_header = false;
  };

  const int classifier_threads =
      std::max(1, resolve_omp_thread_count(num_threads));
  std::vector<size_t> boundaries(static_cast<size_t>(classifier_threads) + 1);
  boundaries.front() = 0;
  boundaries.back() = file_size;
  for (int t = 1; t < classifier_threads; ++t) {
    size_t boundary =
        (file_size / static_cast<size_t>(classifier_threads)) *
        static_cast<size_t>(t);
    while (boundary < file_size && mmap_data[boundary - 1] != '\n')
      ++boundary;
    boundaries[static_cast<size_t>(t)] = boundary;
  }

  std::vector<ClassifiedLines> classified(
      static_cast<size_t>(classifier_threads));
  const size_t mapping_page_size =
      static_cast<size_t>(sysconf(_SC_PAGESIZE));
  // Classification is a single forward pass. Release completed windows so
  // later parsing phases can fault them in with their own access pattern.
  constexpr size_t kClassificationReleaseWindow = 64ull * 1024ull * 1024ull;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int t = 0; t < classifier_threads; ++t) {
    auto &local = classified[static_cast<size_t>(t)];
    const size_t range_start = boundaries[static_cast<size_t>(t)];
    size_t line_start = range_start;
    const size_t range_end = boundaries[static_cast<size_t>(t) + 1];
    size_t release_cursor =
        ((range_start + mapping_page_size - 1) / mapping_page_size) *
        mapping_page_size;

    while (line_start < range_end) {
      const void *newline =
          std::memchr(mmap_data + line_start, '\n', range_end - line_start);
      const size_t line_end =
          newline == nullptr
              ? range_end
              : static_cast<size_t>(static_cast<const char *>(newline) -
                                    mmap_data);
      size_t line_len = line_end - line_start;
      if (line_len > 0 && mmap_data[line_start + line_len - 1] == '\r')
        --line_len;

      if (line_len > 0) {
        const gfaz::LineOffset offset{line_start, line_len};
        switch (mmap_data[line_start]) {
        case 'S':
          local.segments.push_back(offset);
          break;
        case 'H':
          local.last_header = offset;
          local.has_header = true;
          break;
        case 'L':
          local.links.push_back(offset);
          break;
        case 'P':
          local.paths.push_back(offset);
          break;
        case 'W':
          local.walks.push_back(offset);
          break;
        case 'J':
          local.jumps.push_back(offset);
          break;
        case 'C':
          local.containments.push_back(offset);
          break;
        }
      }

      line_start = line_end + (newline == nullptr ? 0 : 1);

      if (line_start >= release_cursor + kClassificationReleaseWindow) {
        const size_t release_end =
            (line_start / mapping_page_size) * mapping_page_size;
        if (release_end > release_cursor) {
          madvise(const_cast<char *>(mmap_data) + release_cursor,
                  release_end - release_cursor, MADV_DONTNEED);
          release_cursor = release_end;
        }
      }
    }

    const size_t release_end =
        (range_end / mapping_page_size) * mapping_page_size;
    if (release_end > release_cursor) {
      madvise(const_cast<char *>(mmap_data) + release_cursor,
              release_end - release_cursor, MADV_DONTNEED);
    }
  }

  auto merge_offsets = [&](auto ClassifiedLines::*member,
                           std::vector<gfaz::LineOffset> &output) {
    size_t total = 0;
    for (const auto &local : classified)
      total += (local.*member).size();
    output.reserve(total);
    for (auto &local : classified) {
      auto &offsets = local.*member;
      output.insert(output.end(), offsets.begin(), offsets.end());
    }
  };

  merge_offsets(&ClassifiedLines::segments, s_offsets);
  merge_offsets(&ClassifiedLines::links, l_offsets);
  merge_offsets(&ClassifiedLines::paths, p_offsets);
  merge_offsets(&ClassifiedLines::walks, w_offsets);
  merge_offsets(&ClassifiedLines::jumps, j_offsets);
  merge_offsets(&ClassifiedLines::containments, c_offsets);
  for (const auto &local : classified) {
    if (local.has_header) {
      graph.header_line = std::string(mmap_data + local.last_header.offset,
                                      local.last_header.length);
    }
  }
  classified.clear();
  classified.shrink_to_fit();

  // Classification touches the entire mapping. Record the last parse phase
  // that needs each page, discard the classification residency, and release
  // pages again as their owning phase completes. Pages shared by interleaved
  // record types remain resident until the latest consumer is finished.
  const size_t page_size = mapping_page_size;
  const size_t page_count = (file_size + page_size - 1) / page_size;
  std::vector<uint8_t> page_last_phase(page_count, 0);
  auto mark_last_phase = [&](const std::vector<gfaz::LineOffset> &offsets,
                             uint8_t phase) {
    for (const auto &off : offsets) {
      const size_t first_page = off.offset / page_size;
      const size_t last_page = (off.offset + off.length - 1) / page_size;
      for (size_t page = first_page; page <= last_page && page < page_count;
           ++page) {
        page_last_phase[page] = std::max(page_last_phase[page], phase);
      }
    }
  };
  mark_last_phase(s_offsets, 1);
  mark_last_phase(l_offsets, 2);
  mark_last_phase(p_offsets, 3);
  mark_last_phase(w_offsets, 3);
  mark_last_phase(j_offsets, 4);
  mark_last_phase(c_offsets, 4);

  auto release_phase_pages = [&](uint8_t phase) {
    size_t run_start = page_count;
    for (size_t page = 0; page <= page_count; ++page) {
      const bool in_phase =
          page < page_count && page_last_phase[page] == phase;
      if (in_phase && run_start == page_count) {
        run_start = page;
      } else if (!in_phase && run_start != page_count) {
        const size_t byte_start = run_start * page_size;
        const size_t byte_end = std::min(page * page_size, file_size);
        madvise(const_cast<char *>(mmap_data) + byte_start,
                byte_end - byte_start, MADV_DONTNEED);
        run_start = page_count;
      }
    }
  };

  madvise(const_cast<char *>(mmap_data), file_size, MADV_DONTNEED);
  num_segments_hint_ = s_offsets.size();
  num_links_hint_ = l_offsets.size();
  log_phase("line classification");

  if (!direct_segment_columns) {
    graph.segments.node_id_to_name.reserve(num_segments_hint_ + 1);
    graph.segments.node_sequences.reserve(num_segments_hint_ + 1);
  }

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
  log_phase("reserve");
  // Phase 1: Numeric, sequential segment IDs can be written directly by row.
  // The fast path supports fixed-width optional columns discovered on the
  // first row; any irregularity falls back to the general sequential parser.
  bool parsed_segments_in_parallel = false;
  if (!s_offsets.empty()) {
    const auto &first_offset = s_offsets.front();
    const std::string_view first_line(mmap_data + first_offset.offset,
                                      first_offset.length);
    size_t first_pos = 1;
    (void)next_field(first_line, first_pos);
    (void)next_field(first_line, first_pos);
    FixedFieldMeta fixed_meta;
    std::vector<gfaz::OptionalFieldColumn> fixed_columns;
    if (initialize_fixed_optional_columns(first_line, first_pos,
                                          s_offsets.size(), fixed_columns,
                                          fixed_meta)) {
      if (direct_segment_columns) {
        graph.segments.node_sequence_lengths.resize(s_offsets.size());
      } else {
        graph.segments.node_id_to_name.resize(s_offsets.size() + 1);
        graph.segments.node_sequences.resize(s_offsets.size() + 1);
      }
      graph.segments.optional_fields = std::move(fixed_columns);
      segment_field_meta_ = fixed_meta;
      std::atomic<bool> failed{false};

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (size_t i = 0; i < s_offsets.size(); ++i) {
        if (failed.load(std::memory_order_relaxed))
          continue;
        try {
          const auto &off = s_offsets[i];
          const std::string_view line(mmap_data + off.offset, off.length);
          size_t pos = 1;
          const std::string_view name = next_field(line, pos);
          const std::string_view sequence = next_field(line, pos);
          if (!numeric_name_matches_id(name, static_cast<uint32_t>(i + 1)) ||
              !assign_fixed_optional_fields(
                  line, pos, i, graph.segments.optional_fields, fixed_meta)) {
            failed.store(true, std::memory_order_relaxed);
            continue;
          }
          if (direct_segment_columns) {
            if (sequence.size() > std::numeric_limits<uint32_t>::max()) {
              failed.store(true, std::memory_order_relaxed);
              continue;
            }
            graph.segments.node_sequence_lengths[i] =
                static_cast<uint32_t>(sequence.size());
          } else {
            graph.segments.node_id_to_name[i + 1] = std::string(name);
            graph.segments.node_sequences[i + 1] = std::string(sequence);
          }
        } catch (...) {
          failed.store(true, std::memory_order_relaxed);
        }
      }
      parsed_segments_in_parallel = !failed.load(std::memory_order_relaxed);
      if (parsed_segments_in_parallel && direct_segment_columns) {
        std::vector<size_t> sequence_offsets(s_offsets.size() + 1, 0);
        for (size_t i = 0; i < s_offsets.size(); ++i) {
          sequence_offsets[i + 1] =
              sequence_offsets[i] + graph.segments.node_sequence_lengths[i];
        }
        graph.segments.node_sequences_concat.resize(sequence_offsets.back());
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (size_t i = 0; i < s_offsets.size(); ++i) {
          const auto &off = s_offsets[i];
          const std::string_view line(mmap_data + off.offset, off.length);
          size_t pos = 1;
          (void)next_field(line, pos);
          const std::string_view sequence = next_field(line, pos);
          std::memcpy(graph.segments.node_sequences_concat.data() +
                          sequence_offsets[i],
                      sequence.data(), sequence.size());
        }
      }
      if (!parsed_segments_in_parallel) {
        graph.segments.node_id_to_name.clear();
        graph.segments.node_sequences.clear();
        graph.segments.node_sequences_concat.clear();
        graph.segments.node_sequence_lengths.clear();
        graph.segments.node_id_to_name.push_back("");
        graph.segments.node_sequences.push_back("");
        graph.segments.optional_fields.clear();
        segment_field_meta_.clear();
      }
    }
  }

  if (!parsed_segments_in_parallel) {
    if (direct_segment_columns) {
      graph.segments.node_id_to_name.reserve(num_segments_hint_ + 1);
      graph.segments.node_sequences.reserve(num_segments_hint_ + 1);
    }
    for (const auto &off : s_offsets) {
      std::string_view line(mmap_data + off.offset, off.length);
      parse_s_line(line, graph);
    }
  }

  // Pad optional field columns to segment count
  size_t num_segments = graph.segments.size();
  for (auto &col : graph.segments.optional_fields) {
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

  s_offsets.clear();
  s_offsets.shrink_to_fit();
  release_phase_pages(1);
  log_phase("parse S-lines");

  // Phase 2: With numeric node IDs, links are independent indexed rows and can
  // use the same fixed-width optional-column fast path.
  bool parsed_links_in_parallel = false;
  if (all_segment_names_numeric_ && !l_offsets.empty()) {
    const auto &first_offset = l_offsets.front();
    const std::string_view first_line(mmap_data + first_offset.offset,
                                      first_offset.length);
    size_t first_pos = 1;
    for (int field = 0; field < 5; ++field)
      (void)next_field(first_line, first_pos);
    FixedFieldMeta fixed_meta;
    std::vector<gfaz::OptionalFieldColumn> fixed_columns;
    if (initialize_fixed_optional_columns(first_line, first_pos,
                                          l_offsets.size(), fixed_columns,
                                          fixed_meta)) {
      graph.links.from_ids.resize(l_offsets.size());
      graph.links.to_ids.resize(l_offsets.size());
      graph.links.from_orients.resize(l_offsets.size());
      graph.links.to_orients.resize(l_offsets.size());
      graph.links.overlap_nums.resize(l_offsets.size());
      graph.links.overlap_ops.resize(l_offsets.size());
      graph.link_optional_fields = std::move(fixed_columns);
      link_field_meta_ = fixed_meta;
      std::atomic<bool> failed{false};

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (size_t i = 0; i < l_offsets.size(); ++i) {
        if (failed.load(std::memory_order_relaxed))
          continue;
        try {
          const auto &off = l_offsets[i];
          const std::string_view line(mmap_data + off.offset, off.length);
          size_t pos = 1;
          const uint32_t from_id = parse_uint32(next_field(line, pos));
          const std::string_view from_orient = next_field(line, pos);
          const uint32_t to_id = parse_uint32(next_field(line, pos));
          const std::string_view to_orient = next_field(line, pos);
          const std::string_view overlap = next_field(line, pos);
          if (from_id == 0 || to_id == 0 ||
              !assign_fixed_optional_fields(
                  line, pos, i, graph.link_optional_fields, fixed_meta)) {
            failed.store(true, std::memory_order_relaxed);
            continue;
          }
          graph.links.from_ids[i] = from_id;
          graph.links.to_ids[i] = to_id;
          graph.links.from_orients[i] =
              from_orient.empty() ? '+' : from_orient[0];
          graph.links.to_orients[i] = to_orient.empty() ? '+' : to_orient[0];
          if (overlap.empty() || overlap == "*") {
            graph.links.overlap_nums[i] = 0;
            graph.links.overlap_ops[i] = '\0';
          } else {
            char op = '\0';
            graph.links.overlap_nums[i] = parse_overlap_num(overlap, op);
            graph.links.overlap_ops[i] = op;
          }
        } catch (...) {
          failed.store(true, std::memory_order_relaxed);
        }
      }
      parsed_links_in_parallel = !failed.load(std::memory_order_relaxed);
      if (!parsed_links_in_parallel) {
        graph.links = {};
        graph.link_optional_fields.clear();
        link_field_meta_.clear();
      }
    }
  }

  if (!parsed_links_in_parallel) {
    for (const auto &off : l_offsets) {
      std::string_view line(mmap_data + off.offset, off.length);
      parse_l_line(line, graph);
    }
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
  l_offsets.clear();
  l_offsets.shrink_to_fit();
  release_phase_pages(2);
  log_phase("parse L-lines");

  // Phase 3: Parse P/W-lines (parallel - each writes to pre-allocated index)
  graph.paths_data.traversals.resize(p_offsets.size());
  graph.paths_data.names.resize(p_offsets.size());
  graph.paths_data.overlaps.resize(p_offsets.size());

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
  p_offsets.clear();
  p_offsets.shrink_to_fit();
  w_offsets.clear();
  w_offsets.shrink_to_fit();
  release_phase_pages(3);
  log_phase("parse P/W-lines");

  // Phase 4: Parse J/C lines (after S-lines, so the parser-local name lookup is
  // populated)
  for (const auto &off : j_offsets) {
    std::string_view line(mmap_data + off.offset, off.length);
    parse_j_line(line, graph);
  }

  for (const auto &off : c_offsets) {
    std::string_view line(mmap_data + off.offset, off.length);
    parse_c_line(line, graph);
  }
  j_offsets.clear();
  j_offsets.shrink_to_fit();
  c_offsets.clear();
  c_offsets.shrink_to_fit();
  release_phase_pages(4);
  log_phase("parse J/C-lines");

  munmap(const_cast<char *>(mmap_data), file_size);
  close(fd);
  log_phase("munmap+close");

  if (gfaz_debug_enabled()) {
    const auto parse_end = Clock::now();
    const double parse_ms =
        std::chrono::duration<double, std::milli>(parse_end - parse_start)
            .count();
    const auto snapshot = read_process_memory_snapshot();
    std::cerr << "[GfaParser] segments=" << num_segments
              << ", links=" << num_links << ", paths=" << graph.paths_data.traversals.size()
              << ", walks=" << graph.walks.size()
              << ", jumps=" << graph.jumps.size()
              << ", containments=" << graph.containments.size();
    if (!graph.segments.optional_fields.empty()) {
      std::cerr << ", segment_optional_columns="
                << graph.segments.optional_fields.size();
    }
    if (!graph.link_optional_fields.empty()) {
      std::cerr << ", link_optional_columns="
                << graph.link_optional_fields.size();
    }
    std::cerr << ", time=" << std::fixed << std::setprecision(2) << parse_ms
              << " ms"
              << " | " << format_memory_snapshot(snapshot) << std::endl;
    print_graph_memory_breakdown(graph);
  }

  return graph;
}

void GfaParser::parse_s_line(std::string_view line, gfaz::GfaGraph &graph) {
  size_t pos = 1;
  std::string_view node_name_view = next_field(line, pos);
  std::string_view sequence_view = next_field(line, pos);

  size_t segment_index = 0;
  const uint32_t next_id = graph.segments.node_id_to_name.size();

  if (all_segment_names_numeric_ && is_numeric(node_name_view) &&
      parse_uint32(node_name_view) == next_id) {
    graph.segments.node_id_to_name.emplace_back(node_name_view);
    graph.segments.node_sequences.emplace_back(sequence_view);
    segment_index = next_id - 1;
  } else {
    if (all_segment_names_numeric_) {
      all_segment_names_numeric_ = false;
      node_name_lookup_.reserve(num_segments_hint_);
      for (uint32_t id = 1; id < graph.segments.node_id_to_name.size(); ++id) {
        node_name_lookup_.emplace(graph.segments.node_id_to_name[id], id);
      }
    }

    auto lookup_it = node_name_lookup_.find(node_name_view);
    if (lookup_it == node_name_lookup_.end()) {
      const uint32_t new_id = graph.segments.node_id_to_name.size();
      // Invariant: the CPU compressor never reads graph.node_name_to_id (name
      // resolution during parsing goes through the parser-local
      // node_name_lookup_), so it is intentionally left empty here.
      graph.segments.node_id_to_name.emplace_back(node_name_view);
      graph.segments.node_sequences.emplace_back(sequence_view);
      node_name_lookup_.emplace(graph.segments.node_id_to_name.back(), new_id);
      segment_index = new_id - 1;
    } else {
      segment_index = lookup_it->second - 1;
    }
  }

  while (pos < line.size()) {
    std::string_view field = next_field(line, pos);
    if (field.empty())
      break;
    parse_segment_field(field, segment_index, graph);
  }
}

void GfaParser::parse_l_line(std::string_view line, gfaz::GfaGraph &graph) {
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

void GfaParser::parse_p_line(std::string_view line, gfaz::GfaGraph &graph,
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

  std::vector<gfaz::NodeId> path;
  if (!(all_segment_names_numeric_ &&
        try_parse_numeric_path_simd(nodes_str, path))) {
    path.clear();
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
              gfaz::NodeId oriented_node_id = node_id;
              if (orientation == '-')
                oriented_node_id = -node_id;
              path.push_back(oriented_node_id);
            }
          }
        }
        node_start = i + 1;
      }
    }
  }

  graph.paths_data.traversals[index] = std::move(path);
  graph.paths_data.names[index] = std::move(path_name);
  graph.paths_data.overlaps[index] = std::move(overlaps);
}

void GfaParser::parse_w_line(std::string_view line, gfaz::GfaGraph &graph,
                             size_t index) {
  size_t pos = 1;
  std::string_view sample_id_view = next_field(line, pos);
  std::string_view hap_index_view = next_field(line, pos);
  std::string_view seq_id_view = next_field(line, pos);
  std::string_view seq_start_view = next_field(line, pos);
  std::string_view seq_end_view = next_field(line, pos);
  std::string_view walk_str = next_field(line, pos);

  std::vector<gfaz::NodeId> walk;
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
        gfaz::NodeId oriented_node_id = node_id;
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
  for (char c : hap_index_view) {
    if (c < '0' || c > '9')
      break;
    graph.walks.hap_indices[index] =
        graph.walks.hap_indices[index] * 10 + static_cast<uint32_t>(c - '0');
  }
  graph.walks.seq_ids[index] = std::string(seq_id_view);
  graph.walks.seq_starts[index] =
      (seq_start_view == "*") ? -1 : parse_int64(seq_start_view);
  graph.walks.seq_ends[index] =
      (seq_end_view == "*") ? -1 : parse_int64(seq_end_view);
}

void GfaParser::parse_segment_field(std::string_view field,
                                    size_t segment_index, gfaz::GfaGraph &graph) {
  if (field.size() < 5 || field[2] != ':' || field[4] != ':')
    return;

  uint16_t tag_key = field_tag_key(field);
  char type = field[3];
  std::string_view value_view = field.substr(5);

  auto it = segment_field_meta_.find(tag_key);
  if (it == segment_field_meta_.end()) {
    size_t col_index = graph.segments.optional_fields.size();
    segment_field_meta_[tag_key] = {type, col_index};

    gfaz::OptionalFieldColumn col;
    col.tag = std::string(field.substr(0, 2));
    col.type = type;
    reserve_optional_column(col, type, num_segments_hint_);
    graph.segments.optional_fields.push_back(col);

    it = segment_field_meta_.find(tag_key);
  }

  char expected_type = it->second.first;
  size_t col_index = it->second.second;

  if (type != expected_type) {
    throw std::runtime_error("Type mismatch for tag '" +
                             graph.segments.optional_fields[col_index].tag +
                             "': expected '" + std::string(1, expected_type) +
                             "', got '" + std::string(1, type) + "'");
  }

  gfaz::OptionalFieldColumn &col = graph.segments.optional_fields[col_index];
  append_optional_value(col, type, segment_index, value_view);
}

void GfaParser::parse_link_field(std::string_view field, size_t link_index,
                                 gfaz::GfaGraph &graph) {
  if (field.size() < 5 || field[2] != ':' || field[4] != ':')
    return;

  uint16_t tag_key = field_tag_key(field);
  char type = field[3];
  std::string_view value_view = field.substr(5);

  auto it = link_field_meta_.find(tag_key);
  if (it == link_field_meta_.end()) {
    size_t col_index = graph.link_optional_fields.size();
    link_field_meta_[tag_key] = {type, col_index};

    gfaz::OptionalFieldColumn col;
    col.tag = std::string(field.substr(0, 2));
    col.type = type;
    reserve_optional_column(col, type, num_links_hint_);
    graph.link_optional_fields.push_back(col);

    GFAZ_LOG("Discovered link optional field: " << col.tag << " (type: " << type
                                                << ") at link index "
                                                << link_index);

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

  gfaz::OptionalFieldColumn &col = graph.link_optional_fields[col_index];
  append_optional_value(col, type, link_index, value_view);
}

void GfaParser::parse_j_line(std::string_view line, gfaz::GfaGraph &graph) {
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

void GfaParser::parse_c_line(std::string_view line, gfaz::GfaGraph &graph) {
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
