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

// A decimal number as GFA tools write it: no sign, no leading zero.
bool parse_canonical_number(std::string_view text, uint32_t &value) {
  if (text.empty() || text.size() > 10 || text[0] == '0')
    return false;
  uint64_t result = 0;
  for (const char c : text) {
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

// Where the trailing run of digits of `name` starts.
size_t number_start(std::string_view name) {
  size_t start = name.size();
  while (start > 0 && name[start - 1] >= '0' && name[start - 1] <= '9')
    --start;
  return start;
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

  // Try the numbered form: one prefix, then a canonical number.
  const size_t count = lengths.size();
  const std::string_view text(concat);
  const std::string_view first = text.substr(0, lengths[0]);
  const std::string_view prefix = first.substr(0, number_start(first));
  std::vector<uint32_t> numbers(count);
  bool numbered = true;
  bool dense = prefix.empty();
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
    numbered = name.size() > prefix.size() &&
               name.compare(0, prefix.size(), prefix) == 0 &&
               parse_canonical_number(name.substr(prefix.size()), numbers[i]);
    dense = dense && numbered && numbers[i] == i + 1;
  }
  if (dense)
    return names;

  names.max_length_ = std::max<size_t>(max_length, 10);
  if (numbered) {
    names.encoding_ = SegmentNameEncoding::kNumbered;
    names.prefix_ = std::string(prefix);
    names.numbers_ = std::move(numbers);
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
    names.numbers_ = Codec::decompress_delta_varint_uint32(
        data.segment_name_numbers_zstd, count);
    if (names.numbers_.size() != count)
      throw std::runtime_error(std::string(kErrorPrefix) +
                               "segment number column is truncated");
    names.max_length_ = names.prefix_.size() + 10;
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
  data.segment_name_numbers_zstd = {};
  data.segment_names_zstd = {};
  data.segment_name_lengths_zstd = {};
  switch (encoding_) {
  case SegmentNameEncoding::kDense:
    break;
  case SegmentNameEncoding::kNumbered:
    data.segment_name_prefix = prefix_;
    data.segment_name_numbers_zstd =
        Codec::compress_delta_varint_uint32(numbers_);
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

size_t SegmentNames::size() const {
  switch (encoding_) {
  case SegmentNameEncoding::kNumbered:
    return numbers_.size();
  case SegmentNameEncoding::kStrings:
    return offsets_.size() - 1;
  case SegmentNameEncoding::kDense:
    break;
  }
  return 0;
}

size_t SegmentNames::write(uint32_t id, char *out) const {
  if (stored(id)) {
    if (encoding_ == SegmentNameEncoding::kStrings) {
      const size_t length = offsets_[id] - offsets_[id - 1];
      std::memcpy(out, concat_.data() + offsets_[id - 1], length);
      return length;
    }
    std::memcpy(out, prefix_.data(), prefix_.size());
    char *number = out + prefix_.size();
    return static_cast<size_t>(
        std::to_chars(number, number + 10, numbers_[id - 1]).ptr - out);
  }
  return static_cast<size_t>(std::to_chars(out, out + 10, id).ptr - out);
}

void SegmentNames::append(std::string &out, uint32_t id) const {
  const size_t used = out.size();
  out.resize(used + max_length_);
  out.resize(used + write(id, out.data() + used));
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
    by_number_.reserve(names.numbers_.size());
    for (size_t i = 0; i < names.numbers_.size(); ++i)
      by_number_.emplace_back(names.numbers_[i], static_cast<uint32_t>(i + 1));
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
    const std::string &prefix = names_.prefix_;
    uint32_t number = 0;
    if (name.size() <= prefix.size() ||
        name.compare(0, prefix.size(), prefix) != 0 ||
        !parse_canonical_number(name.substr(prefix.size()), number))
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
