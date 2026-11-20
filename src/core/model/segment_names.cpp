#include "core/model/segment_names.hpp"

#include "core/codec/codec.hpp"

#include <algorithm>
#include <charconv>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace gfaz {

namespace {

constexpr const char *kErrorPrefix = "Segment names error: ";
// Most digits a zero-padded number may have.
constexpr size_t kMaxWidth = 18;
// Numbers are kept as runs when there are this many segments per run or more.
constexpr size_t kRunsWorthIt = 4;

// The digits of `text` as a number; false if it is not all digits or does not
// fit 32 bits.
bool parse_digits(std::string_view text, uint32_t &value) {
  if (text.empty() || text.size() > kMaxWidth)
    return false;
  uint64_t result = 0;
  for (const char c : text) {
    const unsigned digit = static_cast<unsigned>(c - '0');
    if (digit > 9)
      return false;
    result = result * 10 + digit;
    if (result > std::numeric_limits<uint32_t>::max())
      return false;
  }
  value = static_cast<uint32_t>(result);
  return true;
}

// A decimal number as GFA tools write it: no sign, no leading zero.
bool parse_canonical_number(std::string_view text, uint32_t &value) {
  return !text.empty() && (text[0] != '0' || text.size() == 1) &&
         parse_digits(text, value);
}

// A number zero-padded to `width` digits, as printf("%0*u") writes it: exactly
// `width` digits, or more with no leading zero.
bool parse_padded_number(std::string_view text, size_t width, uint32_t &value) {
  return (text.size() == width || (text.size() > width && text[0] != '0')) &&
         parse_digits(text, value);
}

bool is_digit(char c) { return c >= '0' && c <= '9'; }

// Split a name into prefix, digits and suffix around its last run of digits.
// False if it has no digit.
bool split_name(std::string_view name, std::string_view &prefix,
                std::string_view &digits, std::string_view &suffix) {
  size_t end = name.size();
  while (end > 0 && !is_digit(name[end - 1]))
    --end;
  size_t begin = end;
  while (begin > 0 && is_digit(name[begin - 1]))
    --begin;
  prefix = name.substr(0, begin);
  digits = name.substr(begin, end - begin);
  suffix = name.substr(end);
  return !digits.empty();
}

// The digits of `name` if it is `prefix` + digits + `suffix`, else empty.
std::string_view number_field(std::string_view name, std::string_view prefix,
                              std::string_view suffix) {
  if (name.size() <= prefix.size() + suffix.size() ||
      name.compare(0, prefix.size(), prefix) != 0 ||
      name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0)
    return {};
  return name.substr(prefix.size(),
                     name.size() - prefix.size() - suffix.size());
}

size_t segment_count(const CompressedData &data) {
  return data.segment_seq_lengths_zstd.original_size / sizeof(uint32_t);
}

} // namespace

SegmentNames SegmentNames::from_strings(std::string concat,
                                        const std::vector<uint32_t> &lengths) {
  SegmentNames names;
  if (lengths.empty())
    return names;

  // Try the numbered form: prefix, number, suffix, taken from the first name.
  // The numbers are plain unless the first one starts with a zero, in which
  // case every number must be zero-padded to the first one's width.
  const size_t count = lengths.size();
  const std::string_view text(concat);
  std::string_view prefix, first_digits, suffix;
  bool numbered = !text.empty() &&
                  split_name(text.substr(0, lengths[0]), prefix, first_digits,
                             suffix) &&
                  first_digits.size() <= kMaxWidth;
  const bool padded =
      numbered && first_digits.size() > 1 && first_digits[0] == '0';
  std::vector<uint32_t> numbers(numbered ? count : 0);
  bool dense = numbered && !padded && prefix.empty() && suffix.empty();
  size_t offset = 0;
  size_t max_length = 0;
  for (size_t i = 0; i < count; ++i) {
    const std::string_view name = text.substr(offset, lengths[i]);
    if (name.size() != lengths[i])
      throw std::runtime_error(std::string(kErrorPrefix) +
                               "name lengths exceed the name text");
    offset += lengths[i];
    max_length = std::max(max_length, name.size());
    if (!numbered)
      continue;
    const std::string_view digits = number_field(name, prefix, suffix);
    numbered = padded ? parse_padded_number(digits, first_digits.size(),
                                            numbers[i])
                      : parse_canonical_number(digits, numbers[i]);
    dense = dense && numbered && numbers[i] == i + 1;
  }
  if (dense)
    return names;

  names.max_length_ = std::max<size_t>(max_length, 10);
  names.count_ = count;
  if (numbered) {
    names.encoding_ = SegmentNameEncoding::kNumbered;
    names.prefix_ = std::string(prefix);
    names.suffix_ = std::string(suffix);
    names.width_ = padded ? static_cast<uint8_t>(first_digits.size()) : 0;
    names.set_numbers(std::move(numbers));
    names.note_plain();
    return names;
  }
  names.encoding_ = SegmentNameEncoding::kStrings;
  names.offsets_.resize(count + 1);
  names.offsets_[0] = 0;
  for (size_t i = 0; i < count; ++i)
    names.offsets_[i + 1] = names.offsets_[i] + lengths[i];
  concat.resize(names.offsets_.back());
  names.concat_ = std::move(concat);
  return names;
}

SegmentNames SegmentNames::from_numbers(std::vector<uint32_t> numbers) {
  SegmentNames names;
  bool dense = true;
  const long long count = static_cast<long long>(numbers.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(&& : dense)
#endif
  for (long long i = 0; i < count; ++i)
    dense = dense && numbers[i] == static_cast<uint32_t>(i + 1);
  if (dense)
    return names;
  names.encoding_ = SegmentNameEncoding::kNumbered;
  names.count_ = numbers.size();
  names.set_numbers(std::move(numbers));
  names.note_plain();
  return names;
}

void SegmentNames::set_numbers(std::vector<uint32_t> numbers) {
  const size_t count = numbers.size();
  size_t runs = count == 0 ? 0 : 1;
  for (size_t i = 1; i < count; ++i)
    runs += numbers[i] != numbers[i - 1] + 1;
  // Runs take two values each and a search to read; use them when they are
  // clearly the smaller layout.
  if (runs * kRunsWorthIt > count) {
    numbers_ = std::move(numbers);
    return;
  }
  run_begin_.reserve(runs);
  run_number_.reserve(runs);
  for (size_t i = 0; i < count; ++i) {
    if (i == 0 || numbers[i] != numbers[i - 1] + 1) {
      run_begin_.push_back(static_cast<uint32_t>(i + 1));
      run_number_.push_back(numbers[i]);
    }
  }
}

void SegmentNames::find_run(uint32_t id, RunHint &hint) const {
  const size_t run = static_cast<size_t>(
      std::upper_bound(run_begin_.begin(), run_begin_.end(), id) -
      run_begin_.begin() - 1);
  hint.begin = run_begin_[run];
  hint.end = run + 1 < run_begin_.size() ? run_begin_[run + 1]
                                         : static_cast<uint32_t>(count_ + 1);
  hint.first_number = run_number_[run];
}

SegmentNames
SegmentNames::from_id_table(const std::vector<std::string> &table) {
  if (table.size() <= 1)
    return {};
  std::string concat;
  std::vector<uint32_t> lengths;
  lengths.reserve(table.size() - 1);
  size_t bytes = 0;
  for (size_t id = 1; id < table.size(); ++id)
    bytes += table[id].size();
  concat.reserve(bytes);
  for (size_t id = 1; id < table.size(); ++id) {
    concat += table[id];
    lengths.push_back(static_cast<uint32_t>(table[id].size()));
  }
  return from_strings(std::move(concat), lengths);
}

SegmentNames SegmentNames::decode(const CompressedData &data) {
  SegmentNames names;
  const size_t count = segment_count(data);
  switch (static_cast<SegmentNameEncoding>(data.segment_name_encoding)) {
  case SegmentNameEncoding::kDense:
    return names;
  case SegmentNameEncoding::kNumbered: {
    names.encoding_ = SegmentNameEncoding::kNumbered;
    names.prefix_ = data.segment_name_prefix;
    names.suffix_ = data.segment_name_suffix;
    names.width_ = data.segment_name_width;
    if (names.width_ > kMaxWidth)
      throw std::runtime_error(std::string(kErrorPrefix) +
                               "segment number width is out of range");
    names.count_ = count;
    const size_t runs = data.segment_name_run_count;
    if (runs == 0) {
      names.numbers_ = Codec::decompress_delta_varint_uint32(
          data.segment_name_numbers_zstd, count);
      if (names.numbers_.size() != count)
        throw std::runtime_error(std::string(kErrorPrefix) +
                                 "segment number column is truncated");
    } else {
      names.run_number_ = Codec::decompress_delta_varint_uint32(
          data.segment_name_numbers_zstd, runs);
      names.run_begin_ = Codec::decompress_delta_varint_uint32(
          data.segment_name_run_starts_zstd, runs);
      if (names.run_number_.size() != runs || names.run_begin_.size() != runs)
        throw std::runtime_error(std::string(kErrorPrefix) +
                                 "segment number runs are truncated");
      // Runs start at segment 1 and at strictly increasing segments after it.
      bool ordered = names.run_begin_[0] == 1 && names.run_begin_[runs - 1] <= count;
      for (size_t run = 1; run < runs && ordered; ++run)
        ordered = names.run_begin_[run] > names.run_begin_[run - 1];
      if (!ordered)
        throw std::runtime_error(std::string(kErrorPrefix) +
                                 "segment number runs do not cover the segments");
    }
    names.max_length_ = names.prefix_.size() + names.suffix_.size() +
                        std::max<size_t>(10, names.width_);
    names.note_plain();
    return names;
  }
  case SegmentNameEncoding::kStrings: {
    names.encoding_ = SegmentNameEncoding::kStrings;
    names.concat_ = Codec::zstd_decompress_string(data.segment_names_zstd);
    const std::vector<uint32_t> lengths =
        Codec::zstd_decompress_uint32_vector(data.segment_name_lengths_zstd);
    if (lengths.size() != count)
      throw std::runtime_error(std::string(kErrorPrefix) +
                               "segment name column is truncated");
    names.count_ = count;
    names.offsets_.resize(count + 1);
    names.offsets_[0] = 0;
    size_t max_length = 10;
    for (size_t i = 0; i < count; ++i) {
      names.offsets_[i + 1] = names.offsets_[i] + lengths[i];
      max_length = std::max<size_t>(max_length, lengths[i]);
    }
    if (names.offsets_.back() != names.concat_.size())
      throw std::runtime_error(std::string(kErrorPrefix) +
                               "segment name lengths do not match the names");
    names.max_length_ = max_length;
    return names;
  }
  }
  throw std::runtime_error(std::string(kErrorPrefix) +
                           "unknown segment name encoding " +
                           std::to_string(data.segment_name_encoding));
}

void SegmentNames::encode(CompressedData &data) const {
  data.segment_name_encoding = static_cast<uint8_t>(encoding_);
  data.segment_name_prefix.clear();
  data.segment_name_suffix.clear();
  data.segment_name_width = 0;
  data.segment_name_run_count = 0;
  data.segment_name_numbers_zstd = {};
  data.segment_name_run_starts_zstd = {};
  data.segment_names_zstd = {};
  data.segment_name_lengths_zstd = {};
  switch (encoding_) {
  case SegmentNameEncoding::kDense:
    break;
  case SegmentNameEncoding::kNumbered:
    data.segment_name_prefix = prefix_;
    data.segment_name_suffix = suffix_;
    data.segment_name_width = width_;
    if (run_begin_.empty()) {
      data.segment_name_numbers_zstd =
          Codec::compress_delta_varint_uint32(numbers_);
    } else {
      data.segment_name_run_count = run_begin_.size();
      data.segment_name_numbers_zstd =
          Codec::compress_delta_varint_uint32(run_number_);
      data.segment_name_run_starts_zstd =
          Codec::compress_delta_varint_uint32(run_begin_);
    }
    break;
  case SegmentNameEncoding::kStrings: {
    std::vector<uint32_t> lengths(offsets_.size() - 1);
    for (size_t i = 0; i + 1 < offsets_.size(); ++i)
      lengths[i] = static_cast<uint32_t>(offsets_[i + 1] - offsets_[i]);
    data.segment_names_zstd = Codec::zstd_compress_string(concat_);
    data.segment_name_lengths_zstd = Codec::zstd_compress_uint32_vector(lengths);
    break;
  }
  }
}

size_t SegmentNames::write_general(uint32_t id, char *out,
                                   RunHint &hint) const {
  if (stored(id)) {
    if (encoding_ == SegmentNameEncoding::kStrings) {
      const size_t length = offsets_[id] - offsets_[id - 1];
      std::memcpy(out, concat_.data() + offsets_[id - 1], length);
      return length;
    }
    std::memcpy(out, prefix_.data(), prefix_.size());
    char *cursor = out + prefix_.size();
    char digits[10];
    const size_t count = static_cast<size_t>(
        std::to_chars(digits, digits + 10, number(id, hint)).ptr - digits);
    for (size_t pad = count; pad < width_; ++pad)
      *cursor++ = '0';
    std::memcpy(cursor, digits, count);
    cursor += count;
    std::memcpy(cursor, suffix_.data(), suffix_.size());
    return static_cast<size_t>(cursor - out) + suffix_.size();
  }
  return static_cast<size_t>(std::to_chars(out, out + 10, id).ptr - out);
}

void SegmentNames::append(std::string &out, uint32_t id,
                          RunHint &hint) const {
  char buffer[64];
  if (max_length_ <= sizeof(buffer)) {
    out.append(buffer, write(id, buffer, hint));
    return;
  }
  const size_t used = out.size();
  out.resize(used + max_length_);
  out.resize(used + write(id, out.data() + used, hint));
}

std::string SegmentNames::name(uint32_t id) const {
  std::string out;
  append(out, id);
  return out;
}

SegmentNameLookup::SegmentNameLookup(const SegmentNames &names,
                                     size_t num_segments)
    : names_(names), num_segments_(num_segments) {
  switch (names.encoding_) {
  case SegmentNameEncoding::kDense:
    break;
  case SegmentNameEncoding::kNumbered:
    by_number_.reserve(names.size());
    {
      SegmentNames::RunHint hint;
      for (uint32_t id = 1; id <= names.size(); ++id)
        by_number_.emplace_back(names.number(id, hint), id);
    }
    std::sort(by_number_.begin(), by_number_.end());
    break;
  case SegmentNameEncoding::kStrings:
    by_string_.reserve(names.size());
    for (size_t i = 0; i + 1 < names.offsets_.size(); ++i) {
      by_string_.emplace(
          std::string_view(names.concat_.data() + names.offsets_[i],
                           names.offsets_[i + 1] - names.offsets_[i]),
          static_cast<uint32_t>(i + 1));
    }
    break;
  }
}

uint32_t SegmentNameLookup::find(std::string_view name) const {
  switch (names_.encoding_) {
  case SegmentNameEncoding::kDense: {
    uint32_t id = 0;
    return parse_canonical_number(name, id) && id <= num_segments_ ? id : 0;
  }
  case SegmentNameEncoding::kNumbered: {
    const std::string_view digits =
        number_field(name, names_.prefix_, names_.suffix_);
    uint32_t number = 0;
    if (names_.width_ == 0
            ? !parse_canonical_number(digits, number)
            : !parse_padded_number(digits, names_.width_, number))
      return 0;
    const auto it = std::lower_bound(
        by_number_.begin(), by_number_.end(),
        std::make_pair(number, uint32_t{0}));
    return it != by_number_.end() && it->first == number ? it->second : 0;
  }
  case SegmentNameEncoding::kStrings: {
    const auto it = by_string_.find(name);
    return it == by_string_.end() ? 0 : it->second;
  }
  }
  return 0;
}

} // namespace gfaz
