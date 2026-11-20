#include "handlegraph_adapter/packed_sequences.hpp"

#include <algorithm>
#include <array>

namespace gfaz {

namespace {

constexpr char kBases[4] = {'A', 'C', 'G', 'T'};
constexpr uint8_t kOther = 4;

// A side list longer than one entry per this many bases costs more than the
// packing saves (an entry is nine bytes; packing saves three quarters of one).
constexpr uint64_t kBasesPerOther = 32;
constexpr uint64_t kMinOthersBeforeGivingUp = 1 << 16;

// Bases one thread packs at a time; a multiple of 32, the bases per word.
constexpr size_t kSliceBases = size_t{1} << 20;

std::array<uint8_t, 256> make_codes() {
  std::array<uint8_t, 256> codes;
  codes.fill(kOther);
  for (uint8_t code = 0; code < 4; ++code)
    codes[static_cast<unsigned char>(kBases[code])] = code;
  return codes;
}

const std::array<uint8_t, 256> kCodes = make_codes();

} // namespace

void PackedSequences::append(const char *text, size_t count) {
  if (plain_) {
    text_.append(text, count);
    size_ += count;
    return;
  }
  words_.resize((size_ + count + 31) / 32, 0);

  // Pack text[begin, end) and list its other characters.
  struct Others {
    std::vector<uint64_t> positions;
    std::string chars;
  };
  auto pack = [&](size_t begin, size_t end, Others &others) {
    for (size_t i = begin; i < end; ++i) {
      uint8_t code = kCodes[static_cast<unsigned char>(text[i])];
      const uint64_t position = size_ + i;
      if (code == kOther) {
        others.positions.push_back(position);
        others.chars.push_back(text[i]);
        code = 0;
      }
      words_[position / 32] |= static_cast<uint64_t>(code)
                               << (2 * (position % 32));
    }
  };

  // Cut the piece where a word ends, so no two parts write the same word.
  const size_t to_word_end = static_cast<size_t>((32 - size_ % 32) % 32);
  const size_t head = std::min(count, to_word_end);
  std::vector<Others> parts(1);
  pack(0, head, parts[0]);
  const size_t num_slices = (count - head + kSliceBases - 1) / kSliceBases;
  parts.resize(1 + num_slices);
#pragma omp parallel for schedule(dynamic, 1) if (num_slices > 1)
  for (long long slice = 0; slice < static_cast<long long>(num_slices);
       ++slice) {
    const size_t begin = head + static_cast<size_t>(slice) * kSliceBases;
    pack(begin, std::min(count, begin + kSliceBases), parts[1 + slice]);
  }
  for (const Others &others : parts) {
    other_positions_.insert(other_positions_.end(), others.positions.begin(),
                            others.positions.end());
    other_chars_ += others.chars;
  }

  size_ += count;
  if (other_positions_.size() > kMinOthersBeforeGivingUp &&
      other_positions_.size() > size_ / kBasesPerOther)
    unpack_all();
}

void PackedSequences::unpack_all() {
  text_ = substr(0, size_);
  plain_ = true;
  std::vector<uint64_t>().swap(words_);
  std::vector<uint64_t>().swap(other_positions_);
  std::string().swap(other_chars_);
}

char PackedSequences::at(uint64_t position) const {
  if (plain_)
    return text_[position];
  if (!other_positions_.empty()) {
    const auto it = std::lower_bound(other_positions_.begin(),
                                     other_positions_.end(), position);
    if (it != other_positions_.end() && *it == position)
      return other_chars_[it - other_positions_.begin()];
  }
  return kBases[(words_[position / 32] >> (2 * (position % 32))) & 3];
}

std::string PackedSequences::substr(uint64_t position, size_t count) const {
  if (plain_)
    return text_.substr(position, count);
  std::string out(count, '\0');
  for (size_t i = 0; i < count; ++i) {
    const uint64_t p = position + i;
    out[i] = kBases[(words_[p / 32] >> (2 * (p % 32))) & 3];
  }
  if (!other_positions_.empty()) {
    auto it = std::lower_bound(other_positions_.begin(),
                               other_positions_.end(), position);
    for (; it != other_positions_.end() && *it < position + count; ++it)
      out[*it - position] = other_chars_[it - other_positions_.begin()];
  }
  return out;
}

} // namespace gfaz
