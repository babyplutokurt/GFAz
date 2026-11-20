#ifndef MODEL_SEGMENT_NAMES_HPP
#define MODEL_SEGMENT_NAMES_HPP

// The names of a graph's segments, by 1-based segment ID.
//
// A container numbers segments 1..N in the order of their S-lines and refers
// to them by that ID everywhere. When the GFA already names its segments
// "1".."N" in that order (dense names) nothing more is stored. Otherwise the
// container carries the names, in one of two forms:
//   - numbered: every name is one shared prefix, a decimal number and one
//     shared suffix; the number is either written plainly or zero-padded to
//     one width ("s1", "s2", ...; plain numbers that are not 1..N in file
//     order; "utg000001l", "utg000002l", ...). Numbers that mostly count up
//     by one are kept as runs (first number, length), so a graph numbered
//     from 3463 with a gap now and then costs a few bytes per gap;
//   - strings:  anything else, kept as text.
//
// Code that prints a segment goes through SegmentNames::append / write, so it
// emits the original name whichever form is stored.

#include "core/model/compressed_data.hpp"

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace gfaz {

// Values of CompressedData::segment_name_encoding.
enum class SegmentNameEncoding : uint8_t {
  kDense = 0,
  kNumbered = 1,
  kStrings = 2,
};

class SegmentNames {
public:
  // Dense names: segment `id` is called by its decimal ID.
  SegmentNames() = default;

  // From the names in ID order, concatenated. Picks the dense form when the
  // names are "1".."N", else numbered when they fit it, else strings.
  static SegmentNames from_strings(std::string concat,
                                   const std::vector<uint32_t> &lengths);
  // From names that are all plain decimal numbers, in ID order.
  static SegmentNames from_numbers(std::vector<uint32_t> numbers);
  // From a vector indexed by ID (index 0 unused), as GfaGraph holds them.
  static SegmentNames from_id_table(const std::vector<std::string> &names);

  static SegmentNames decode(const CompressedData &data);
  // Sets the segment-name fields of `data` (clears them for dense names).
  void encode(CompressedData &data) const;

  SegmentNameEncoding encoding() const { return encoding_; }
  bool dense() const { return encoding_ == SegmentNameEncoding::kDense; }
  // Number of stored names; 0 for dense names.
  size_t size() const { return count_; }
  // True when every stored name is a plain decimal number: number(id).
  bool plain_numbers() const { return plain_; }

  // Remembers the run of consecutive numbers last looked up. Code that prints
  // many names in a row (the steps of a path) keeps one and passes it in, so
  // that a name in the same run needs no search.
  struct RunHint {
    uint32_t begin = 0; // IDs [begin, end) are numbered first_number, +1, ...
    uint32_t end = 0;
    uint32_t first_number = 0;
  };

  // The number in the name of stored segment `id` (numbered names only).
  uint32_t number(uint32_t id, RunHint &hint) const {
    if (run_begin_.empty())
      return numbers_[id - 1];
    if (id - hint.begin >= hint.end - hint.begin)
      find_run(id, hint);
    return hint.first_number + (id - hint.begin);
  }
  uint32_t number(uint32_t id) const {
    RunHint hint;
    return number(id, hint);
  }

  // The longest text write() can produce.
  size_t max_length() const { return max_length_; }
  // Write the name of segment `id` at `out`, which must have max_length()
  // bytes of room; returns the number of bytes written. An ID outside the
  // stored names is written as its decimal number.
  // Inline for names that are plain numbers, which is what a path step of a
  // large graph usually prints.
  size_t write(uint32_t id, char *out, RunHint &hint) const {
    if (plain_ && stored(id))
      return static_cast<size_t>(
          std::to_chars(out, out + 10, number(id, hint)).ptr - out);
    return write_general(id, out, hint);
  }
  size_t write(uint32_t id, char *out) const {
    RunHint hint;
    return write(id, out, hint);
  }
  void append(std::string &out, uint32_t id, RunHint &hint) const;
  void append(std::string &out, uint32_t id) const {
    RunHint hint;
    append(out, id, hint);
  }
  std::string name(uint32_t id) const;

private:
  friend class SegmentNameLookup;

  bool stored(uint32_t id) const { return id != 0 && id <= count_; }
  size_t write_general(uint32_t id, char *out, RunHint &hint) const;
  void find_run(uint32_t id, RunHint &hint) const;
  // Take the numbers of a numbered form, as runs when that is smaller.
  void set_numbers(std::vector<uint32_t> numbers);

  // Call after the numbered fields are set.
  void note_plain() {
    plain_ = encoding_ == SegmentNameEncoding::kNumbered && prefix_.empty() &&
             suffix_.empty() && width_ == 0;
  }

  SegmentNameEncoding encoding_ = SegmentNameEncoding::kDense;
  bool plain_ = false; // numbered, with no prefix, suffix or padding
  size_t count_ = 0;
  size_t max_length_ = 10; // digits of a uint32_t
  // Numbered: prefix_, the number, suffix_. A non-zero width_ pads the number
  // with zeros to at least that many digits.
  std::string prefix_;
  std::string suffix_;
  uint8_t width_ = 0;
  // The numbers, in one of two layouts: numbers_[id - 1], or runs of
  // consecutive numbers, run r covering the IDs from run_begin_[r] (up to the
  // next run) and starting at run_number_[r].
  std::vector<uint32_t> numbers_;
  std::vector<uint32_t> run_begin_;
  std::vector<uint32_t> run_number_;
  // Strings: name of `id` is concat_[offsets_[id - 1], offsets_[id]).
  std::string concat_;
  std::vector<uint64_t> offsets_;
};

// Resolves a segment name to its ID. Built on demand: only code that takes
// segment names as input needs it.
class SegmentNameLookup {
public:
  // `names` must outlive the lookup. `num_segments` bounds dense names.
  SegmentNameLookup(const SegmentNames &names, size_t num_segments);

  // The ID of the segment called `name`, or 0 if there is none.
  uint32_t find(std::string_view name) const;

private:
  const SegmentNames &names_;
  size_t num_segments_;
  std::vector<std::pair<uint32_t, uint32_t>> by_number_; // (number, id), sorted
  std::unordered_map<std::string_view, uint32_t> by_string_;
};

} // namespace gfaz

#endif // MODEL_SEGMENT_NAMES_HPP
