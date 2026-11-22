#ifndef COMPUTE_REPEAT_INDEX_HPP
#define COMPUTE_REPEAT_INDEX_HPP

// Every visit of the nodes in crowded zones.
//
// The PathIndex finds the steps on a node by decoding each block listed for
// the node's zone of ids. That is cheap where a zone is crossed about once per
// haplotype. Inside a repeat (a satellite array, say) one zone is listed by
// tens of thousands of blocks, and a single lookup has to decode them all.
// For the zones listed by more than `min_zone_blocks` blocks this index keeps
// every visit of every node as (haplotype, step), 8 bytes a visit, so a lookup
// there decodes nothing.

#include "compute/path_access.hpp"
#include "core/defaults.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace gfaz {

struct RepeatIndexOptions {
  // Zones listed by more blocks than this are covered.
  size_t min_zone_blocks = 256;
  int num_threads = kDefaultNumThreads;
};

class RepeatIndex {
public:
  // Decodes the blocks of the crowded zones twice, in parallel across blocks.
  static RepeatIndex build(const PathAccess &paths,
                           const RepeatIndexOptions &options = {});

  // Whether the visits of a node are listed here.
  bool covers(uint32_t node) const {
    const size_t zone = node >> PathIndex::kZoneShift;
    return zone < zone_slot_.size() && zone_slot_[zone] != kNoSlot;
  }

  // Call visit(haplotype, step) for every visit of a covered node, in
  // increasing (haplotype, step) order, until one returns false. Returns
  // whether all of them returned true.
  template <typename Visitor>
  bool for_each_visit(uint32_t node, Visitor &&visit) const {
    const size_t row = row_of(node);
    for (uint64_t i = starts_[row]; i < starts_[row + 1]; ++i)
      if (!visit(static_cast<size_t>(visits_[i] >> 32),
                 static_cast<uint32_t>(visits_[i])))
        return false;
    return true;
  }

  size_t num_covered_zones() const { return (starts_.size() - 1) / kZoneNodes; }
  uint64_t num_visits() const { return visits_.size(); }

private:
  static constexpr uint32_t kNoSlot = std::numeric_limits<uint32_t>::max();
  static constexpr size_t kZoneNodes = PathIndex::kZoneNodes;

  size_t row_of(uint32_t node) const {
    return static_cast<size_t>(zone_slot_[node >> PathIndex::kZoneShift]) *
               kZoneNodes +
           (node & (kZoneNodes - 1));
  }

  // The rank of each zone among the covered ones, or kNoSlot.
  std::vector<uint32_t> zone_slot_;
  // Visits in CSR form, a row per node id of a covered zone (see row_of); a
  // visit is haplotype << 32 | step.
  std::vector<uint64_t> starts_{0};
  std::vector<uint64_t> visits_;
};

} // namespace gfaz

#endif // COMPUTE_REPEAT_INDEX_HPP
