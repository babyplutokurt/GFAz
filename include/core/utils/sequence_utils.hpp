#ifndef UTILS_SEQUENCE_UTILS_HPP
#define UTILS_SEQUENCE_UTILS_HPP

#include <string>

namespace gfaz {

// Complement a single nucleotide, preserving case. Unknown characters
// (including 'N'/'n' and IUPAC ambiguity codes not handled here) are returned
// unchanged so the function is total over arbitrary segment sequences.
inline char complement_base(char c) {
  switch (c) {
  case 'A': return 'T';
  case 'C': return 'G';
  case 'G': return 'C';
  case 'T': return 'A';
  case 'a': return 't';
  case 'c': return 'g';
  case 'g': return 'c';
  case 't': return 'a';
  default: return c;
  }
}

// Reverse-complement a DNA string. Used when a traversal visits a segment in
// reverse orientation (negative node id), since the segment sequence is stored
// in forward orientation.
inline std::string reverse_complement(const std::string &seq) {
  std::string out;
  out.resize(seq.size());
  for (size_t i = 0; i < seq.size(); ++i)
    out[seq.size() - 1 - i] = complement_base(seq[i]);
  return out;
}

inline void reverse_complement_inplace(std::string &seq) {
  const size_t n = seq.size();
  for (size_t i = 0; i < n / 2; ++i) {
    const char a = complement_base(seq[i]);
    const char b = complement_base(seq[n - 1 - i]);
    seq[i] = b;
    seq[n - 1 - i] = a;
  }
  if (n & 1)
    seq[n / 2] = complement_base(seq[n / 2]);
}

} // namespace gfaz

#endif // UTILS_SEQUENCE_UTILS_HPP
