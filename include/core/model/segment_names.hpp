#ifndef MODEL_SEGMENT_NAMES_HPP
#define MODEL_SEGMENT_NAMES_HPP

// The names of a graph's segments, by 1-based segment ID.
//
// A container numbers segments 1..N in the order of their S-lines and refers
// to them by that ID everywhere. When the GFA already names its segments
// "1".."N" in that order (dense names) nothing more is stored. Otherwise the
// container carries the names, in one of two forms:
//   - numbered: every name is one shared prefix followed by a decimal number
//     ("s1", "s2", ... or plain numbers that are not 1..N in file order);
//   - strings:  anything else.
//
// Code that prints a segment goes through SegmentNames::append / write, so it
// emits the original name whichever form is stored.

#include "core/model/compressed_data.hpp"

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
  // From a vector indexed by ID (index 0 unused), as GfaGraph holds them.
  static SegmentNames from_id_table(const std::vector<std::string> &names);

  static SegmentNames decode(const CompressedData &data);
  // Sets the segment-name fields of `data` (clears them for dense names).
  void encode(CompressedData &data) const;

  SegmentNameEncoding encoding() const { return encoding_; }
  bool dense() const { return encoding_ == SegmentNameEncoding::kDense; }
  // Number of stored names; 0 for dense names.
  size_t size() const;
  // True when every stored name is a plain decimal number; numbers()[id - 1]
  // is then the number segment `id` is called by.
  bool plain_numbers() const {
    return encoding_ == SegmentNameEncoding::kNumbered && prefix_.empty();
  }
  const std::vector<uint32_t> &numbers() const { return numbers_; }

  // The longest text write() can produce.
  size_t max_length() const { return max_length_; }
  // Write the name of segment `id` at `out`, which must have max_length()
  // bytes of room; returns the number of bytes written. An ID outside the
  // stored names is written as its decimal number.
  size_t write(uint32_t id, char *out) const;
  void append(std::string &out, uint32_t id) const;
  std::string name(uint32_t id) const;

private:
  friend class SegmentNameLookup;

  bool stored(uint32_t id) const { return id != 0 && id <= size(); }

  SegmentNameEncoding encoding_ = SegmentNameEncoding::kDense;
  size_t max_length_ = 10; // digits of a uint32_t
  // Numbered.
  std::string prefix_;
  std::vector<uint32_t> numbers_; // numbers_[id - 1]
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
