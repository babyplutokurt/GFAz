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

#if defined(__GNUC__) || defined(__clang__)
#define GFAZ_NOINLINE __attribute__((noinline))
#else
#define GFAZ_NOINLINE
#endif

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
                                     uint64_t max_id,
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
  // An ID outside 1..max_id names no segment: hand the path back so the
  // general parser reports it.
  if (value == 0 || value > max_id ||
      value > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()))
    return false;

  const auto node = static_cast<gfaz::NodeId>(value);
  path.push_back(orientation == '-' ? -node : node);
  return true;
}

__attribute__((target("avx2")))
size_t count_walk_steps_avx2(std::string_view walk) {
  const char *data = walk.data();
  const size_t size = walk.size();
  const __m256i forward = _mm256_set1_epi8('>');
  const __m256i reverse = _mm256_set1_epi8('<');
  size_t steps = 0;
  size_t offset = 0;
  for (; offset + 32 <= size; offset += 32) {
    const __m256i chars =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(data + offset));
    const __m256i hits = _mm256_or_si256(_mm256_cmpeq_epi8(chars, forward),
                                         _mm256_cmpeq_epi8(chars, reverse));
    steps += static_cast<size_t>(__builtin_popcount(
        static_cast<uint32_t>(_mm256_movemask_epi8(hits))));
  }
  for (; offset < size; ++offset)
    steps += data[offset] == '>' || data[offset] == '<';
  return steps;
}

__attribute__((target("avx2")))
bool parse_numeric_path_avx2(std::string_view nodes, uint64_t max_id,
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
      if (!parse_numeric_path_token(token_begin, data + delimiter, max_id, path))
        return false;
      token_begin = data + delimiter + 1;
      mask &= mask - 1;
    }
  }
  for (size_t i = offset; i < size; ++i) {
    if (data[i] != ',')
      continue;
    if (!parse_numeric_path_token(token_begin, data + i, max_id, path))
      return false;
    token_begin = data + i + 1;
  }
  return parse_numeric_path_token(token_begin, data + size, max_id, path);
}

__attribute__((target("avx512f,avx512bw")))
bool parse_numeric_path_avx512(std::string_view nodes, uint64_t max_id,
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
      if (!parse_numeric_path_token(token_begin, data + delimiter, max_id, path))
        return false;
      token_begin = data + delimiter + 1;
      mask &= mask - 1;
    }
  }
  for (size_t i = offset; i < size; ++i) {
    if (data[i] != ',')
      continue;
    if (!parse_numeric_path_token(token_begin, data + i, max_id, path))
      return false;
    token_begin = data + i + 1;
  }
  return parse_numeric_path_token(token_begin, data + size, max_id, path);
}

bool try_parse_numeric_path_simd(std::string_view nodes, uint64_t max_id,
                                 std::vector<gfaz::NodeId> &path) {
  static const bool has_avx512bw = __builtin_cpu_supports("avx512bw");
  static const bool has_avx2 = __builtin_cpu_supports("avx2");
  if (has_avx512bw)
    return parse_numeric_path_avx512(nodes, max_id, path);
  return has_avx2 && parse_numeric_path_avx2(nodes, max_id, path);
}
#else
bool try_parse_numeric_path_simd(std::string_view, uint64_t,
                                 std::vector<gfaz::NodeId> &) {
  return false;
}
#endif

// Number of '>' and '<' characters, i.e. an upper bound on the walk's steps.
size_t count_walk_steps(std::string_view walk) {
#ifdef GFAZ_X86_SIMD
  static const bool has_avx2 = __builtin_cpu_supports("avx2");
  if (has_avx2)
    return count_walk_steps_avx2(walk);
#endif
  size_t steps = 0;
  for (char c : walk) {
    if (c == '>' || c == '<')
      ++steps;
  }
  return steps;
}

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
// Value of the first `digits` (1..8) ASCII digits of an 8-byte little-endian
// load; the other bytes are ignored.
inline uint32_t parse_up_to_8_digits(uint64_t word, unsigned digits) {
  uint64_t t = (word ^ 0x3030303030303030ull) << (64 - 8 * digits);
  t = ((t * 2561) >> 8) & 0x00FF00FF00FF00FFull;
  t = ((t * 6553601) >> 16) & 0x0000FFFF0000FFFFull;
  return static_cast<uint32_t>((t * 42949672960001ull) >> 32);
}

// Parses the digits at `name`, where at least 10 bytes are readable. Returns
// the digit count (1..9) with the value in `value`, or 0 when the name is not
// 1 to 9 digits followed by '>' or '<'.
inline unsigned parse_simple_walk_name(const char *name, uint32_t &value) {
  uint64_t word;
  std::memcpy(&word, name, sizeof(word));
  const uint64_t t = word ^ 0x3030303030303030ull;
  const uint64_t non_digit =
      ((t + 0x7676767676767676ull) | t) & 0x8080808080808080ull;
  const unsigned leading =
      non_digit != 0 ? static_cast<unsigned>(__builtin_ctzll(non_digit)) >> 3
                     : 8;
  if (leading == 0)
    return 0;
  uint32_t parsed = parse_up_to_8_digits(word, leading);
  const unsigned ninth = static_cast<unsigned char>(name[8] - '0');
  const bool has_ninth = (leading == 8) & (ninth <= 9);
  parsed = has_ninth ? parsed * 10 + ninth : parsed;
  const unsigned digits = leading + (has_ninth ? 1 : 0);
  const char next = name[digits];
  if (next != '>' && next != '<')
    return 0;
  value = parsed;
  return digits;
}
#define GFAZ_HAVE_SWAR_WALK_NAME 1
#else
#define GFAZ_HAVE_SWAR_WALK_NAME 0
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
// Faults in the pages of a freshly reserved buffer from all threads. A large
// buffer that one thread then fills (a resize or an append) would otherwise
// take every page fault on that thread. Only a hint: failure is ignored.
void populate_pages(const void *begin, size_t bytes) {
#if defined(__linux__)
#ifndef MADV_POPULATE_WRITE
#define MADV_POPULATE_WRITE 23
#endif
  constexpr size_t kMinBytes = 64ull * 1024ull * 1024ull;
  if (begin == nullptr || bytes < kMinBytes)
    return;
  const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  const size_t first =
      (reinterpret_cast<uintptr_t>(begin) + page - 1) / page * page;
  const size_t last = (reinterpret_cast<uintptr_t>(begin) + bytes) / page * page;
  if (last <= first)
    return;
  const size_t pages = (last - first) / page;
  constexpr size_t kPagesPerPiece = 4096;
  const size_t pieces = (pages + kPagesPerPiece - 1) / kPagesPerPiece;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (size_t piece = 0; piece < pieces; ++piece) {
    const size_t piece_first = first + piece * kPagesPerPiece * page;
    const size_t piece_last =
        std::min(last, piece_first + kPagesPerPiece * page);
    madvise(reinterpret_cast<void *>(piece_first), piece_last - piece_first,
            MADV_POPULATE_WRITE);
  }
#else
  (void)begin;
  (void)bytes;
#endif
}

template <typename T> void reserve_populated(std::vector<T> &values, size_t n) {
  values.reserve(n);
  populate_pages(values.data(), n * sizeof(T));
}

inline void reserve_populated(std::string &value, size_t n) {
  value.reserve(n);
  populate_pages(value.data(), n);
}

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

// True if `name` is the decimal number `expected` as it would be printed, so
// that writing the ID back reproduces the name ("007" does not match 7).
bool numeric_name_matches_id(std::string_view name, uint32_t expected) {
  if (name.empty() || name[0] == '0')
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

// Strict decimal parse: digits only, no leading zero, fits in uint32. Two
// names that pass are the same string exactly when they are the same number.
inline bool parse_canonical_number(std::string_view s, uint32_t &value) {
  if (s.empty() || s.size() > 10 || s[0] == '0')
    return false;
  uint64_t result = 0;
  for (char c : s) {
    const unsigned digit = static_cast<unsigned>(c - '0');
    if (digit > 9)
      return false;
    result = result * 10 + digit;
  }
  if (result > std::numeric_limits<uint32_t>::max())
    return false;
  value = static_cast<uint32_t>(result);
  return true;
}

size_t optional_column_rows(const gfaz::OptionalFieldColumn &col) {
  switch (col.type) {
  case 'i':
    return col.int_values.size();
  case 'f':
    return col.float_values.size();
  case 'A':
    return col.char_values.size();
  case 'Z':
  case 'J':
  case 'H':
    return col.string_lengths.size();
  case 'B':
    return col.b_subtypes.size();
  }
  return 0;
}

void pad_optional_column(gfaz::OptionalFieldColumn &col, size_t rows) {
  switch (col.type) {
  case 'i':
    if (col.int_values.size() < rows)
      col.int_values.resize(rows, std::numeric_limits<int64_t>::min());
    break;
  case 'f':
    if (col.float_values.size() < rows)
      col.float_values.resize(rows, std::numeric_limits<float>::lowest());
    break;
  case 'A':
    if (col.char_values.size() < rows)
      col.char_values.resize(rows, '\0');
    break;
  case 'Z':
  case 'J':
  case 'H':
    if (col.string_lengths.size() < rows)
      col.string_lengths.resize(rows, 0);
    break;
  case 'B':
    if (col.b_subtypes.size() < rows) {
      col.b_subtypes.resize(rows, '\0');
      col.b_lengths.resize(rows, 0);
    }
    break;
  }
}

inline uint16_t optional_tag_key(std::string_view tag) {
  return (static_cast<uint16_t>(static_cast<unsigned char>(tag[0])) << 8) |
         static_cast<uint16_t>(static_cast<unsigned char>(tag[1]));
}

// Optional-field columns for one contiguous block of rows, filled by a single
// worker with row numbers relative to the block. Blocks are concatenated in
// row order afterwards, which reproduces the columns (and their order of first
// appearance) that the serial parser builds.
struct OptionalColumnChunk {
  std::vector<gfaz::OptionalFieldColumn> columns;
  FixedFieldMeta meta;
};

// Returns false for rows the serial parser must handle: a tag whose type
// changes, a tag repeated within one row, or an unsupported type.
bool append_chunk_optional_fields(std::string_view line, size_t pos,
                                  size_t local_row,
                                  OptionalColumnChunk &chunk) {
  while (pos < line.size()) {
    const std::string_view field = next_field(line, pos);
    if (field.empty())
      break;
    if (field.size() < 5 || field[2] != ':' || field[4] != ':')
      continue;
    const char type = field[3];
    if (type != 'i' && type != 'f' && type != 'A' && type != 'Z' &&
        type != 'J' && type != 'H' && type != 'B')
      return false;
    const uint16_t key = optional_tag_key(field);
    auto it = chunk.meta.find(key);
    if (it == chunk.meta.end()) {
      it = chunk.meta.emplace(key, std::make_pair(type, chunk.columns.size()))
               .first;
      gfaz::OptionalFieldColumn col;
      col.tag = std::string(field.substr(0, 2));
      col.type = type;
      chunk.columns.push_back(std::move(col));
    } else if (it->second.first != type) {
      return false;
    }
    gfaz::OptionalFieldColumn &col = chunk.columns[it->second.second];
    if (optional_column_rows(col) > local_row)
      return false;
    append_optional_value(col, type, local_row, field.substr(5));
  }
  return true;
}

// Concatenates per-block columns in row order. `row_bases[k]` is the first row
// of block k and `row_bases.back()` the total row count. Blocks are freed as
// they are consumed. Returns false on a type conflict between blocks.
bool merge_optional_chunks(std::vector<OptionalColumnChunk> &chunks,
                           const std::vector<size_t> &row_bases,
                           std::vector<gfaz::OptionalFieldColumn> &out,
                           FixedFieldMeta &out_meta) {
  out.clear();
  out_meta.clear();
  std::vector<size_t> string_bytes;
  std::vector<size_t> b_bytes;
  for (const auto &chunk : chunks) {
    for (const auto &col : chunk.columns) {
      const uint16_t key = optional_tag_key(col.tag);
      auto it = out_meta.find(key);
      if (it == out_meta.end()) {
        it = out_meta.emplace(key, std::make_pair(col.type, out.size())).first;
        gfaz::OptionalFieldColumn merged;
        merged.tag = col.tag;
        merged.type = col.type;
        out.push_back(std::move(merged));
        string_bytes.push_back(0);
        b_bytes.push_back(0);
      } else if (it->second.first != col.type) {
        out.clear();
        out_meta.clear();
        return false;
      }
      string_bytes[it->second.second] += col.concatenated_strings.size();
      b_bytes[it->second.second] += col.b_concat_bytes.size();
    }
  }

  const size_t total_rows = row_bases.back();
  for (size_t i = 0; i < out.size(); ++i) {
    reserve_optional_column(out[i], out[i].type, total_rows);
    // Every column ends up with one value per row, so the whole reservation
    // is used; fault it in from all threads before the serial appends below.
    populate_pages(out[i].int_values.data(),
                   out[i].int_values.capacity() * sizeof(int64_t));
    populate_pages(out[i].float_values.data(),
                   out[i].float_values.capacity() * sizeof(float));
    populate_pages(out[i].char_values.data(), out[i].char_values.capacity());
    populate_pages(out[i].string_lengths.data(),
                   out[i].string_lengths.capacity() * sizeof(uint32_t));
    reserve_populated(out[i].concatenated_strings, string_bytes[i]);
    reserve_populated(out[i].b_concat_bytes, b_bytes[i]);
  }

  for (size_t k = 0; k < chunks.size(); ++k) {
    for (auto &col : chunks[k].columns) {
      gfaz::OptionalFieldColumn &dst =
          out[out_meta.find(optional_tag_key(col.tag))->second.second];
      pad_optional_column(dst, row_bases[k]);
      dst.int_values.insert(dst.int_values.end(), col.int_values.begin(),
                            col.int_values.end());
      dst.float_values.insert(dst.float_values.end(), col.float_values.begin(),
                              col.float_values.end());
      dst.char_values.insert(dst.char_values.end(), col.char_values.begin(),
                             col.char_values.end());
      dst.concatenated_strings.append(col.concatenated_strings);
      dst.string_lengths.insert(dst.string_lengths.end(),
                                col.string_lengths.begin(),
                                col.string_lengths.end());
      dst.b_subtypes.insert(dst.b_subtypes.end(), col.b_subtypes.begin(),
                            col.b_subtypes.end());
      dst.b_lengths.insert(dst.b_lengths.end(), col.b_lengths.begin(),
                           col.b_lengths.end());
      dst.b_concat_bytes.insert(dst.b_concat_bytes.end(),
                                col.b_concat_bytes.begin(),
                                col.b_concat_bytes.end());
    }
    chunks[k] = OptionalColumnChunk{};
  }
  return true;
}

// Splits `count` rows into contiguous blocks for the chunked parsers; more
// blocks than threads so dynamic scheduling can even out uneven line lengths.
std::vector<size_t> chunk_row_bases(size_t count) {
  int threads = 1;
#ifdef _OPENMP
  threads = std::max(1, omp_get_max_threads());
#endif
  const size_t chunks =
      std::max<size_t>(1, std::min(count, static_cast<size_t>(threads) * 8));
  std::vector<size_t> bases(chunks + 1);
  for (size_t k = 0; k <= chunks; ++k)
    bases[k] = count / chunks * k + std::min(k, count % chunks);
  return bases;
}

// Drops input pages a block's forward scan has finished with, one window at a
// time, so the text does not stay resident for the whole phase.
inline void release_chunk_pages(const char *data, size_t page_size,
                                size_t offset, size_t &cursor) {
  constexpr size_t kWindow = 16ull * 1024ull * 1024ull;
  if (offset < cursor + kWindow)
    return;
  const size_t end = (offset / page_size) * page_size;
  madvise(const_cast<char *>(data) + cursor, end - cursor, MADV_DONTNEED);
  cursor = end;
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
                                bool direct_segment_columns,
                                gfaz::TraversalSink *traversal_sink) {
  traversal_sink_ = traversal_sink;
  ScopedOMPThreads omp_scope(num_threads);
  const auto parse_start = Clock::now();
  missing_segment_refs_.store(0);
  first_missing_segment_.clear();
  auto phase_start = parse_start;
  gfaz::GfaGraph graph;
  segment_field_meta_.clear();
  link_field_meta_.clear();
  node_name_lookup_.clear();
  numeric_name_table_.clear();
  segment_name_arena_.clear();
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
  // Keep this large branch out of the legacy parser's compiler function. When
  // inlined here it changes register allocation and slows large fallback GFAs.
  // Clang only accepts a lambda attribute after the parameter list; GCC accepts
  // it in either position, so keep it here to build with both.
  auto try_parse_direct =
      [&]() GFAZ_NOINLINE -> bool {
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
      if (traversal_sink_)
        traversal_sink_->prepare(path_bases.back(), walk_bases.back());

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
      const size_t direct_dense_limit = dense_id_limit();

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
                    from_id > direct_dense_limit ||
                    to_id > direct_dense_limit ||
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
                if (from_id == 0 || to_id == 0 ||
                    from_id > direct_dense_limit ||
                    to_id > direct_dense_limit) {
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
                if (container_id == 0 || contained_id == 0 ||
                    container_id > direct_dense_limit ||
                    contained_id > direct_dense_limit) {
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
        return true;
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

  return false;
  };

  if (try_parse_direct()) {
    report_missing_segments();
    return graph;
  }
  // The general parser starts over, so forget what the direct pass noted.
  missing_segment_refs_.store(0);
  first_missing_segment_.clear();

  // Parallel line classification. Split the mapping at newline boundaries so
  // each worker owns complete lines, then concatenate the per-range offsets in
  // range order to preserve the input order within each record type.
  LineOffsets s_offsets, l_offsets, p_offsets, w_offsets;
  LineOffsets j_offsets, c_offsets;

  struct ClassifiedLines {
    std::vector<gfaz::LineOffset> segments;
    std::vector<gfaz::LineOffset> links;
    std::vector<gfaz::LineOffset> paths;
    std::vector<gfaz::LineOffset> walks;
    std::vector<gfaz::LineOffset> jumps;
    std::vector<gfaz::LineOffset> containments;
    gfaz::LineOffset last_header{};
    bool has_header = false;
    // Last parse phase needing the first and the last page of this range.
    // Neighbouring ranges share those pages, so they are merged afterwards.
    uint8_t first_page_phase = 0;
    uint8_t last_page_phase = 0;
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

  // Classification touches the entire mapping. Record the last parse phase
  // that needs each page (1: S, 2: L, 3: P/W, 4: J/C), discard the
  // classification residency, and release pages again as their owning phase
  // completes. Pages shared by interleaved record types remain resident until
  // the latest consumer is finished.
  const size_t page_size = mapping_page_size;
  const size_t page_count = (file_size + page_size - 1) / page_size;
  std::vector<uint8_t> page_last_phase(page_count, 0);

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
    const size_t range_first_page = range_start / page_size;
    const size_t range_last_page =
        range_end > range_start ? (range_end - 1) / page_size : range_first_page;
    auto mark_pages = [&](const gfaz::LineOffset &off, uint8_t phase) {
      const size_t first_page = off.offset / page_size;
      const size_t last_page = (off.offset + off.length - 1) / page_size;
      for (size_t page = first_page; page <= last_page; ++page) {
        uint8_t &slot = page == range_first_page  ? local.first_page_phase
                        : page == range_last_page ? local.last_page_phase
                                                  : page_last_phase[page];
        if (slot < phase)
          slot = phase;
      }
    };

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
          mark_pages(offset, 1);
          break;
        case 'H':
          local.last_header = offset;
          local.has_header = true;
          break;
        case 'L':
          local.links.push_back(offset);
          mark_pages(offset, 2);
          break;
        case 'P':
          local.paths.push_back(offset);
          mark_pages(offset, 3);
          break;
        case 'W':
          local.walks.push_back(offset);
          mark_pages(offset, 3);
          break;
        case 'J':
          local.jumps.push_back(offset);
          mark_pages(offset, 4);
          break;
        case 'C':
          local.containments.push_back(offset);
          mark_pages(offset, 4);
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

  for (int t = 0; t < classifier_threads; ++t) {
    const auto &local = classified[static_cast<size_t>(t)];
    const size_t range_start = boundaries[static_cast<size_t>(t)];
    const size_t range_end = boundaries[static_cast<size_t>(t) + 1];
    if (range_end <= range_start)
      continue;
    uint8_t &first = page_last_phase[range_start / page_size];
    first = std::max(first, local.first_page_phase);
    uint8_t &last = page_last_phase[(range_end - 1) / page_size];
    last = std::max(last, local.last_page_phase);
  }

  // Concatenate the per-range offsets in range order, copying in parallel.
  auto merge_offsets = [&](auto ClassifiedLines::*member, LineOffsets &output) {
    std::vector<size_t> bases(classified.size() + 1, 0);
    for (size_t t = 0; t < classified.size(); ++t)
      bases[t + 1] = bases[t] + (classified[t].*member).size();
    output.resize(bases.back());
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1)
#endif
    for (size_t t = 0; t < classified.size(); ++t) {
      auto &offsets = classified[t].*member;
      if (!offsets.empty()) {
        std::memcpy(static_cast<void *>(output.data() + bases[t]),
                    offsets.data(), offsets.size() * sizeof(gfaz::LineOffset));
      }
      std::vector<gfaz::LineOffset>().swap(offsets);
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

  // For the serial record loops, which walk the file in order: drop the
  // mapping behind the cursor every window so the input text does not pile up
  // in resident memory for the whole phase. Pages a later phase needs are
  // simply read back from the page cache.
  constexpr size_t kSequentialReleaseWindow = 64ull * 1024ull * 1024ull;
  auto release_behind = [&](size_t offset, size_t &cursor) {
    if (offset < cursor + kSequentialReleaseWindow)
      return;
    const size_t end = (offset / page_size) * page_size;
    madvise(const_cast<char *>(mmap_data) + cursor, end - cursor,
            MADV_DONTNEED);
    cursor = end;
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
        reserve_populated(graph.segments.node_sequences_concat,
                          sequence_offsets.back());
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

  // GFAZ_SERIAL_PARSE forces the line-at-a-time parsers below; tests use it
  // as the reference the chunked parsers must reproduce.
  const bool allow_chunked_parse = std::getenv("GFAZ_SERIAL_PARSE") == nullptr;
  if (allow_chunked_parse && !parsed_segments_in_parallel &&
      !s_offsets.empty()) {
    parsed_segments_in_parallel =
        parse_segments_in_chunks(mmap_data, s_offsets, page_size,
                                 direct_segment_columns, graph);
  }

  if (!parsed_segments_in_parallel) {
    if (direct_segment_columns) {
      graph.segments.node_id_to_name.reserve(num_segments_hint_ + 1);
      graph.segments.node_sequences.reserve(num_segments_hint_ + 1);
    }
    size_t release_cursor = 0;
    for (const auto &off : s_offsets) {
      std::string_view line(mmap_data + off.offset, off.length);
      parse_s_line(line, graph);
      release_behind(off.offset, release_cursor);
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
  log_phase(std::string("parse S-lines") +
            (!parsed_segments_in_parallel
                 ? " (serial)"
                 : !numeric_name_table_.empty() ? " (parallel, number table)"
                 : !segment_name_arena_.empty() ? " (parallel, name lookup)"
                                                : " (parallel)"));

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
      reserve_populated(graph.links.from_ids, l_offsets.size());
      reserve_populated(graph.links.to_ids, l_offsets.size());
      reserve_populated(graph.links.from_orients, l_offsets.size());
      reserve_populated(graph.links.to_orients, l_offsets.size());
      reserve_populated(graph.links.overlap_nums, l_offsets.size());
      reserve_populated(graph.links.overlap_ops, l_offsets.size());
      graph.links.from_ids.resize(l_offsets.size());
      graph.links.to_ids.resize(l_offsets.size());
      graph.links.from_orients.resize(l_offsets.size());
      graph.links.to_orients.resize(l_offsets.size());
      graph.links.overlap_nums.resize(l_offsets.size());
      graph.links.overlap_ops.resize(l_offsets.size());
      graph.link_optional_fields = std::move(fixed_columns);
      link_field_meta_ = fixed_meta;
      std::atomic<bool> failed{false};
      const size_t dense_limit = dense_id_limit();

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
          if (from_id == 0 || to_id == 0 || from_id > dense_limit ||
              to_id > dense_limit ||
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

  if (allow_chunked_parse && !parsed_links_in_parallel &&
      !l_offsets.empty()) {
    parsed_links_in_parallel =
        parse_links_in_chunks(mmap_data, l_offsets, page_size, graph);
  }

  if (!parsed_links_in_parallel) {
    graph.links.from_ids.reserve(num_links_hint_);
    graph.links.to_ids.reserve(num_links_hint_);
    graph.links.from_orients.reserve(num_links_hint_);
    graph.links.to_orients.reserve(num_links_hint_);
    graph.links.overlap_nums.reserve(num_links_hint_);
    graph.links.overlap_ops.reserve(num_links_hint_);
    size_t release_cursor = 0;
    for (const auto &off : l_offsets) {
      std::string_view line(mmap_data + off.offset, off.length);
      parse_l_line(line, graph);
      release_behind(off.offset, release_cursor);
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
  log_phase(std::string("parse L-lines") +
            (parsed_links_in_parallel ? " (parallel)" : " (serial)"));

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
  if (traversal_sink_)
    traversal_sink_->prepare(p_offsets.size(), w_offsets.size());

  // Traversal lines are long, so most of their pages hold nothing else. Drop
  // those pages as soon as the line is parsed; otherwise the whole P/W text
  // stays resident until the end of this phase, on top of the parsed data.
  auto release_line_pages = [&](const gfaz::LineOffset &off) {
    const size_t first = ((off.offset + page_size - 1) / page_size) * page_size;
    const size_t last = ((off.offset + off.length) / page_size) * page_size;
    if (last > first) {
      madvise(const_cast<char *>(mmap_data) + first, last - first,
              MADV_DONTNEED);
    }
  };

  // An exception must not leave a parallel region: keep the first one and
  // report it once the loops are done.
  std::atomic<bool> traversal_failed{false};
  std::string traversal_error;
  auto record_traversal_error = [&](char record, size_t index,
                                    const char *what) {
    if (traversal_failed.exchange(true))
      return;
    traversal_error = std::string("malformed ") + record + "-line (record " +
                      std::to_string(index + 1) + "): " + what;
  };

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (size_t i = 0; i < p_offsets.size(); ++i) {
    if (traversal_failed.load(std::memory_order_relaxed))
      continue;
    try {
      std::string_view line(mmap_data + p_offsets[i].offset,
                            p_offsets[i].length);
      parse_p_line(line, graph, i);
    } catch (const std::exception &e) {
      record_traversal_error('P', i, e.what());
    }
    release_line_pages(p_offsets[i]);
  }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (size_t i = 0; i < w_offsets.size(); ++i) {
    if (traversal_failed.load(std::memory_order_relaxed))
      continue;
    try {
      std::string_view line(mmap_data + w_offsets[i].offset,
                            w_offsets[i].length);
      parse_w_line(line, graph, i);
    } catch (const std::exception &e) {
      record_traversal_error('W', i, e.what());
    }
    release_line_pages(w_offsets[i]);
  }
  if (traversal_failed.load()) {
    munmap(const_cast<char *>(mmap_data), file_size);
    close(fd);
    throw std::runtime_error(std::string(kParserErrorPrefix) + traversal_error);
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
  numeric_name_table_ = {};
  segment_name_arena_ = {};
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

  report_missing_segments();
  return graph;
}

// Copy the name field of every S-line, in file order, into the flattened
// name columns of `segments`.
static void collect_segment_names(const char *data, const LineOffsets &offsets,
                                  const std::vector<size_t> &row_bases,
                                  gfaz::SegmentData &segments) {
  const size_t count = offsets.size();
  const size_t chunk_count = row_bases.size() - 1;
  auto &lengths = segments.node_name_lengths;
  lengths.resize(count);
  std::vector<size_t> name_bases(chunk_count + 1, 0);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (size_t chunk = 0; chunk < chunk_count; ++chunk) {
    size_t bytes = 0;
    for (size_t i = row_bases[chunk]; i < row_bases[chunk + 1]; ++i) {
      const std::string_view line(data + offsets[i].offset, offsets[i].length);
      size_t pos = 1;
      const std::string_view name = next_field(line, pos);
      lengths[i] = static_cast<uint32_t>(name.size());
      bytes += name.size();
    }
    name_bases[chunk + 1] = bytes;
  }
  for (size_t chunk = 0; chunk < chunk_count; ++chunk)
    name_bases[chunk + 1] += name_bases[chunk];
  segments.node_names_concat.resize(name_bases.back());
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (size_t chunk = 0; chunk < chunk_count; ++chunk) {
    char *out = segments.node_names_concat.data() + name_bases[chunk];
    for (size_t i = row_bases[chunk]; i < row_bases[chunk + 1]; ++i) {
      const std::string_view line(data + offsets[i].offset, offsets[i].length);
      size_t pos = 1;
      const std::string_view name = next_field(line, pos);
      std::memcpy(out, name.data(), name.size());
      out += name.size();
    }
  }
}

bool GfaParser::parse_segments_in_chunks(
    const char *data, const LineOffsets &offsets,
    size_t page_size, bool direct_segment_columns, gfaz::GfaGraph &graph) {
  const size_t count = offsets.size();
  if (count >= std::numeric_limits<uint32_t>::max())
    return false;

  const std::vector<size_t> row_bases = chunk_row_bases(count);
  const size_t chunk_count = row_bases.size() - 1;
  std::vector<OptionalColumnChunk> optional(chunk_count);
  std::vector<uint32_t> name_numbers(count);
  auto &names = graph.segments.node_id_to_name;
  auto &sequences = graph.segments.node_sequences;
  // With direct columns the sequences go straight into the flattened layout
  // the entropy stage consumes (lengths now, bytes in a second pass once the
  // offsets are known) and no per-segment strings are created.
  auto &lengths = graph.segments.node_sequence_lengths;
  auto &concat = graph.segments.node_sequences_concat;
  std::vector<size_t> sequence_bases(chunk_count + 1, 0);
  // Set when some name is not a canonical number: names are then resolved
  // through node_name_lookup_, as in the serial parser.
  std::atomic<bool> named{false};
  if (direct_segment_columns) {
    reserve_populated(lengths, count);
    lengths.resize(count);
  } else {
    names.resize(count + 1);
    sequences.resize(count + 1);
  }
  std::atomic<bool> failed{false};

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (size_t chunk = 0; chunk < chunk_count; ++chunk) {
    const size_t base = row_bases[chunk];
    size_t release_cursor =
        ((offsets[base].offset + page_size - 1) / page_size) * page_size;
    size_t sequence_bytes = 0;
    try {
      for (size_t i = base; i < row_bases[chunk + 1]; ++i) {
        if (failed.load(std::memory_order_relaxed))
          break;
        const auto &off = offsets[i];
        const std::string_view line(data + off.offset, off.length);
        size_t pos = 1;
        const std::string_view name = next_field(line, pos);
        const std::string_view sequence = next_field(line, pos);
        bool name_ok = true;
        if (!parse_canonical_number(name, name_numbers[i])) {
          name_numbers[i] = 0;
          // A number written with leading zeros (or too large) is matched by
          // value in the serial parser while names are numeric: hand back.
          // The serial parser also never resolves an empty name.
          name_ok = !name.empty() && !is_numeric(name);
          if (name_ok && !named.load(std::memory_order_relaxed))
            named.store(true, std::memory_order_relaxed);
        }
        if (!name_ok ||
            sequence.size() > std::numeric_limits<uint32_t>::max() ||
            !append_chunk_optional_fields(line, pos, i - base,
                                          optional[chunk])) {
          failed.store(true, std::memory_order_relaxed);
          break;
        }
        if (direct_segment_columns) {
          lengths[i] = static_cast<uint32_t>(sequence.size());
          sequence_bytes += sequence.size();
        } else {
          names[i + 1].assign(name.data(), name.size());
          sequences[i + 1].assign(sequence.data(), sequence.size());
        }
        release_chunk_pages(data, page_size, off.offset, release_cursor);
      }
    } catch (...) {
      failed.store(true, std::memory_order_relaxed);
    }
    sequence_bases[chunk + 1] = sequence_bytes;
  }

  std::vector<uint32_t> table;
  bool ok = !failed.load(std::memory_order_relaxed);
  const bool use_name_lookup = ok && named.load(std::memory_order_relaxed);
  if (use_name_lookup) {
    // Copy the names into one arena and index them; IDs are file order.
    std::vector<uint32_t>().swap(name_numbers);
    std::vector<size_t> name_bases(chunk_count + 1, 0);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (size_t chunk = 0; chunk < chunk_count; ++chunk) {
      size_t bytes = 0;
      for (size_t i = row_bases[chunk]; i < row_bases[chunk + 1]; ++i) {
        const std::string_view line(data + offsets[i].offset,
                                    offsets[i].length);
        size_t pos = 1;
        bytes += next_field(line, pos).size();
      }
      name_bases[chunk + 1] = bytes;
    }
    for (size_t chunk = 0; chunk < chunk_count; ++chunk)
      name_bases[chunk + 1] += name_bases[chunk];
    segment_name_arena_.resize(name_bases.back());
    std::vector<std::string_view> name_views(count);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (size_t chunk = 0; chunk < chunk_count; ++chunk) {
      char *out = segment_name_arena_.data() + name_bases[chunk];
      for (size_t i = row_bases[chunk]; i < row_bases[chunk + 1]; ++i) {
        const std::string_view line(data + offsets[i].offset,
                                    offsets[i].length);
        size_t pos = 1;
        const std::string_view name = next_field(line, pos);
        std::memcpy(out, name.data(), name.size());
        name_views[i] = std::string_view(out, name.size());
        out += name.size();
      }
    }
    node_name_lookup_.clear();
    node_name_lookup_.reserve(count);
    for (size_t i = 0; i < count && ok; ++i) {
      // A repeated name merges rows in the serial parser.
      ok = node_name_lookup_
               .emplace(name_views[i], static_cast<uint32_t>(i + 1))
               .second;
    }
  }
  bool dense = ok && !use_name_lookup;
  if (dense) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(&& : dense)
#endif
    for (size_t i = 0; i < count; ++i)
      dense = dense && name_numbers[i] == i + 1;
  }
  if (ok && !dense && !use_name_lookup) {
    // Names are numbers but not 1..N in file order. Segment IDs are still
    // assigned in file order; the table maps each number to its ID. Bound the
    // table so sparse numbering does not cost more than the hash lookup.
    uint32_t max_number = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(max : max_number)
#endif
    for (size_t i = 0; i < count; ++i)
      max_number = std::max(max_number, name_numbers[i]);
    const uint64_t limit =
        std::max<uint64_t>(uint64_t{16} * count, uint64_t{1} << 20);
    ok = max_number <= limit;
    if (ok) {
      // Walk parsing looks names up here at random: ask for huge pages
      // before the first touch.
      table.reserve(static_cast<size_t>(max_number) + 1);
      gfaz::runtime_utils::advise_huge_pages(
          table.data(), (static_cast<size_t>(max_number) + 1) * sizeof(uint32_t));
      table.assign(static_cast<size_t>(max_number) + 1, 0);
      for (size_t i = 0; i < count && ok; ++i) {
        uint32_t &slot = table[name_numbers[i]];
        ok = slot == 0; // a repeated name merges rows in the serial parser
        slot = static_cast<uint32_t>(i + 1);
      }
    }
  }
  if (ok) {
    ok = merge_optional_chunks(optional, row_bases,
                               graph.segments.optional_fields,
                               segment_field_meta_);
  }
  if (!ok) {
    names.resize(1);
    sequences.resize(1);
    lengths = {};
    graph.segments.optional_fields.clear();
    segment_field_meta_.clear();
    node_name_lookup_.clear();
    segment_name_arena_ = {};
    return false;
  }

  if (direct_segment_columns) {
    for (size_t chunk = 0; chunk < chunk_count; ++chunk)
      sequence_bases[chunk + 1] += sequence_bases[chunk];
    reserve_populated(concat, sequence_bases.back());
    concat.resize(sequence_bases.back());
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (size_t chunk = 0; chunk < chunk_count; ++chunk) {
      const size_t base = row_bases[chunk];
      size_t release_cursor =
          ((offsets[base].offset + page_size - 1) / page_size) * page_size;
      char *out = concat.data() + sequence_bases[chunk];
      for (size_t i = base; i < row_bases[chunk + 1]; ++i) {
        const auto &off = offsets[i];
        const std::string_view line(data + off.offset, off.length);
        size_t pos = 1;
        (void)next_field(line, pos);
        const std::string_view sequence = next_field(line, pos);
        std::memcpy(out, sequence.data(), sequence.size());
        out += sequence.size();
        release_chunk_pages(data, page_size, off.offset, release_cursor);
      }
    }
  }

  if (!dense) {
    all_segment_names_numeric_ = false;
    numeric_name_table_ = std::move(table);
    // The names are not the IDs, so the container has to carry them. Without
    // direct columns they are already in node_id_to_name.
    if (direct_segment_columns)
      collect_segment_names(data, offsets, row_bases, graph.segments);
  }
  return true;
}

bool GfaParser::parse_links_in_chunks(
    const char *data, const LineOffsets &offsets,
    size_t page_size, gfaz::GfaGraph &graph) {
  const size_t count = offsets.size();
  const std::vector<size_t> row_bases = chunk_row_bases(count);
  const size_t chunk_count = row_bases.size() - 1;
  std::vector<OptionalColumnChunk> optional(chunk_count);
  auto &links = graph.links;
  reserve_populated(links.from_ids, count);
  reserve_populated(links.to_ids, count);
  reserve_populated(links.from_orients, count);
  reserve_populated(links.to_orients, count);
  reserve_populated(links.overlap_nums, count);
  reserve_populated(links.overlap_ops, count);
  links.from_ids.resize(count);
  links.to_ids.resize(count);
  links.from_orients.resize(count);
  links.to_orients.resize(count);
  links.overlap_nums.resize(count);
  links.overlap_ops.resize(count);
  std::atomic<bool> failed{false};

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (size_t chunk = 0; chunk < chunk_count; ++chunk) {
    const size_t base = row_bases[chunk];
    size_t release_cursor =
        ((offsets[base].offset + page_size - 1) / page_size) * page_size;
    try {
      for (size_t i = base; i < row_bases[chunk + 1]; ++i) {
        if (failed.load(std::memory_order_relaxed))
          break;
        const auto &off = offsets[i];
        const std::string_view line(data + off.offset, off.length);
        size_t pos = 1;
        const uint32_t from_id = resolve_node_id(next_field(line, pos));
        const std::string_view from_orient = next_field(line, pos);
        const uint32_t to_id = resolve_node_id(next_field(line, pos));
        const std::string_view to_orient = next_field(line, pos);
        const std::string_view overlap = next_field(line, pos);
        // The serial parser drops a link with an unknown endpoint, which
        // shifts every later row.
        if (from_id == 0 || to_id == 0 ||
            !append_chunk_optional_fields(line, pos, i - base,
                                          optional[chunk])) {
          failed.store(true, std::memory_order_relaxed);
          break;
        }
        links.from_ids[i] = from_id;
        links.to_ids[i] = to_id;
        links.from_orients[i] = from_orient.empty() ? '+' : from_orient[0];
        links.to_orients[i] = to_orient.empty() ? '+' : to_orient[0];
        if (overlap.empty() || overlap == "*") {
          links.overlap_nums[i] = 0;
          links.overlap_ops[i] = '\0';
        } else {
          char op = '\0';
          links.overlap_nums[i] = parse_overlap_num(overlap, op);
          links.overlap_ops[i] = op;
        }
        release_chunk_pages(data, page_size, off.offset, release_cursor);
      }
    } catch (...) {
      failed.store(true, std::memory_order_relaxed);
    }
  }

  if (failed.load(std::memory_order_relaxed) ||
      !merge_optional_chunks(optional, row_bases, graph.link_optional_fields,
                             link_field_meta_)) {
    graph.links = {};
    graph.link_optional_fields.clear();
    link_field_meta_.clear();
    return false;
  }
  return true;
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
  if (from_id == 0 || to_id == 0) {
    note_missing_segment('L', from_id == 0 ? from_name_view : to_name_view);
    return;
  }

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

  // With a sink, decode into the sink's reused per-thread buffer: a spilled
  // traversal never needs its own allocation.
  std::vector<gfaz::NodeId> local_path;
  std::vector<gfaz::NodeId> &path =
      traversal_sink_ ? traversal_sink_->scratch() : local_path;
  if (!(all_segment_names_numeric_ &&
        try_parse_numeric_path_simd(nodes_str, dense_id_limit(), path))) {
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
            if (node_id == 0)
              note_missing_segment('P', node_name_view);
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

  if (traversal_sink_) {
    traversal_sink_->consume(false, index, path,
                             graph.paths_data.traversals[index]);
  } else {
    graph.paths_data.traversals[index] = std::move(path);
  }
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

  std::vector<gfaz::NodeId> local_walk;
  std::vector<gfaz::NodeId> &walk =
      traversal_sink_ ? traversal_sink_->scratch() : local_walk;
  walk.clear();
  walk.reserve(count_walk_steps(walk_str));
  size_t walk_pos = 0;

  // Fast path for numeric segment names: one pass over the digits, no call
  // per step. A name of up to nine digits cannot overflow; anything else goes
  // through resolve_node_id() as below, so the result is the same.
  const bool table_names = !all_segment_names_numeric_ &&
                           !numeric_name_table_.empty();
  if (all_segment_names_numeric_ || table_names) {
    const uint32_t *table = numeric_name_table_.data();
    const size_t table_size = numeric_name_table_.size();
    const size_t dense_limit = dense_id_limit();
    const char *p = walk_str.data();
    const char *const end = p + walk_str.size();
    while (p < end) {
      const char orient_char = *p++;
      if (orient_char != '>' && orient_char != '<')
        continue;
      const char *const name = p;
      uint32_t node_id = 0;
      bool simple = false;
#if GFAZ_HAVE_SWAR_WALK_NAME
      // Common case, eight bytes at a time; anything it rejects is parsed
      // again by the loop below.
      if (end - name >= 10) {
        const unsigned digits = parse_simple_walk_name(name, node_id);
        if (digits != 0 && !(table_names && *name == '0')) {
          simple = true;
          p = name + digits;
        }
      }
#endif
      if (!simple) {
        node_id = 0;
        while (p < end && static_cast<unsigned char>(*p - '0') <= 9)
          node_id = node_id * 10 + static_cast<uint32_t>(*p++ - '0');
        const size_t digits = static_cast<size_t>(p - name);
        simple = digits != 0 && digits <= 9 &&
                 (p == end || *p == '>' || *p == '<') &&
                 !(table_names && *name == '0');
      }
      // With segments named 1..N a larger number names none; resolve_node_id()
      // below turns it into an unknown segment.
      if (simple && !table_names && node_id > dense_limit)
        simple = false;
      if (simple) {
        if (table_names)
          node_id = node_id < table_size ? table[node_id] : 0;
      } else {
        while (p < end && *p != '>' && *p != '<')
          ++p;
        if (p == name)
          continue;
        node_id = resolve_node_id(
            std::string_view(name, static_cast<size_t>(p - name)));
      }
      if (node_id != 0) {
        gfaz::NodeId oriented_node_id = node_id;
        if (orient_char == '<')
          oriented_node_id = -node_id;
        walk.push_back(oriented_node_id);
      } else {
        note_missing_segment(
            'W', std::string_view(name, static_cast<size_t>(p - name)));
      }
    }
    walk_pos = walk_str.size();
  }

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
      if (node_id == 0)
        note_missing_segment('W', node_name_view);
      if (node_id != 0) {
        gfaz::NodeId oriented_node_id = node_id;
        if (orient_char == '<')
          oriented_node_id = -node_id;
        walk.push_back(oriented_node_id);
      }
    }

    walk_pos = name_end;
  }

  if (traversal_sink_) {
    traversal_sink_->consume(true, index, walk, graph.walks.walks[index]);
  } else {
    graph.walks.walks[index] = std::move(walk);
  }
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
  if (from_id == 0 || to_id == 0) {
    note_missing_segment('J', from_id == 0 ? from_name_view : to_name_view);
    return;
  }

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
  if (container_id == 0 || contained_id == 0) {
    note_missing_segment('C',
                         container_id == 0 ? container_view : contained_view);
    return;
  }

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

void GfaParser::note_missing_segment(char record,
                                     std::string_view name) const {
  if (missing_segment_refs_.fetch_add(1, std::memory_order_relaxed) != 0)
    return;
  constexpr size_t kMaxShown = 64;
  first_missing_segment_ = std::string(1, record) + "-line references '" +
                           std::string(name.substr(0, kMaxShown)) +
                           (name.size() > kMaxShown ? "..." : "") + "'";
}

void GfaParser::report_missing_segments() const {
  const uint64_t count = missing_segment_refs_.load();
  if (count == 0)
    return;
  const std::string summary =
      std::to_string(count) +
      (count == 1 ? " reference" : " references") +
      " to segments that no S-line defines (first: " + first_missing_segment_ +
      ")";
  if (!allow_missing_segments_) {
    throw std::runtime_error(
        std::string(kParserErrorPrefix) + summary +
        "; pass --allow-missing-segments (allow_missing_segments in the API) "
        "to drop them");
  }
  std::cerr << "Warning: dropped " << summary << std::endl;
}

uint32_t GfaParser::resolve_node_id(std::string_view node_name_view) const {
  if (node_name_view.empty())
    return 0;

  if (all_segment_names_numeric_) {
    // Same reading as parse_uint32(), except that a name with no leading
    // digits or one that overflows is an unknown segment (0), as in the other
    // naming modes, instead of an exception.
    uint32_t value = 0;
    bool any = false;
    for (char c : node_name_view) {
      if (c < '0' || c > '9')
        break;
      const uint32_t digit = static_cast<uint32_t>(c - '0');
      if (value > (std::numeric_limits<uint32_t>::max() - digit) / 10)
        return 0;
      value = value * 10 + digit;
      any = true;
    }
    // Segments are named 1..N here, so a larger number names none.
    return any && value <= dense_id_limit() ? value : 0;
  }

  if (!numeric_name_table_.empty()) {
    uint32_t number = 0;
    if (!parse_canonical_number(node_name_view, number) ||
        number >= numeric_name_table_.size())
      return 0;
    return numeric_name_table_[number];
  }

  auto it = node_name_lookup_.find(node_name_view);
  return (it == node_name_lookup_.end()) ? 0 : it->second;
}

uint16_t GfaParser::field_tag_key(std::string_view field) {
  return (static_cast<uint16_t>(static_cast<unsigned char>(field[0])) << 8) |
         static_cast<uint16_t>(static_cast<unsigned char>(field[1]));
}
