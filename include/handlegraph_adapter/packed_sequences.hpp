#ifndef GFAZ_PACKED_SEQUENCES_HPP
#define GFAZ_PACKED_SEQUENCES_HPP

// The segment sequences of a graph back to back, two bits per base.
//
// A, C, G and T are packed; any other character (N, lower case, IUPAC codes)
// is kept in a side list by position. Text with many such characters would
// make that list larger than the packing saves, so it is then kept as plain
// bytes instead. Either way at() and substr() return the original characters.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gfaz {

class PackedSequences {
public:
  // Add text at the end. Call in order; the pieces may be cut anywhere.
  void append(const char *text, size_t count);

  uint64_t size() const { return size_; }
  char at(uint64_t position) const;
  std::string substr(uint64_t position, size_t count) const;

private:
  void unpack_all();

  uint64_t size_ = 0;
  // Packed form: base i is bits 2 * (i % 32) of words_[i / 32], as an index
  // into "ACGT". Other characters are packed as A and listed in
  // other_positions_ (increasing) with their character in other_chars_.
  std::vector<uint64_t> words_;
  std::vector<uint64_t> other_positions_;
  std::string other_chars_;
  // Plain form, used instead once the side list grows too long.
  bool plain_ = false;
  std::string text_;
};

} // namespace gfaz

#endif // GFAZ_PACKED_SEQUENCES_HPP
