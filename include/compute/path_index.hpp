#ifndef COMPUTE_PATH_INDEX_HPP
#define COMPUTE_PATH_INDEX_HPP

// Checkpoint index over the encoded traversals.
//
// Every haplotype is cut into blocks of about `block_steps` decoded steps, each
// starting at a top-level encoded element. A block records where it starts (in
// encoded elements, steps and bases), the delta state needed to decode from
// there, and the smallest and largest node id it visits. That is enough for:
//   - random access: decode from the block that holds a step or a base position
//     instead of from the start of the haplotype;
//   - node lookup: node ids are grouped into zones of kZoneNodes consecutive
//     ids, and every zone lists the blocks that visit one of its nodes. Only
//     those blocks can visit a node of the zone. On a pangenome graph that is
//     about one block per haplotype crossing the locus.
// The index is a few hundred bytes per block and is independent of the
// container format: it is built from a TraversalSet and can be saved next to
// the .gfaz (`gfaz index`), never inside it.

#include "compute/traversal_set.hpp"
#include "core/defaults.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gfaz {

struct PathIndexOptions {
  uint32_t block_steps = 4096;
  int num_threads = kDefaultNumThreads;
};

struct PathBlock {
  uint64_t bp_begin = 0;   // bases before the block
  uint32_t enc_begin = 0;  // first top-level encoded element
  uint32_t step_begin = 0; // decoded steps before the block
  uint32_t min_node = 0;   // unsigned node id range visited by the block;
  uint32_t max_node = 0;   // min_node > max_node for an empty block
  DeltaState state;        // delta state at enc_begin
};

class PathIndex {
public:
  // One pass over every haplotype, parallel across haplotypes.
  // segment_lengths[n - 1] is the length of node n; a node outside that table
  // counts as zero bases.
  static PathIndex build(const TraversalSet &traversals,
                         const std::vector<uint32_t> &segment_lengths,
                         const PathIndexOptions &options = {});

  // A value that changes when the traversals of a container change; stored in
  // a saved index so a stale file is rejected.
  static uint64_t fingerprint(const CompressedData &data);

  // Write the index to a file, and read it back. load() throws if the file is
  // not an index, or was built from a container with another fingerprint.
  void save(const std::string &path, uint64_t fingerprint) const;
  static PathIndex load(const std::string &path, uint64_t fingerprint);

  uint32_t block_steps() const { return block_steps_; }

  size_t num_haplotypes() const { return first_block_.size() - 1; }
  size_t num_blocks(size_t hap) const {
    // The last stored block of a haplotype is its end sentinel.
    return first_block_[hap + 1] - first_block_[hap] - 1;
  }
  size_t total_blocks() const { return blocks_.size() - num_haplotypes(); }

  // Block `block` of a haplotype; block == num_blocks(hap) is the end sentinel,
  // whose *_begin fields are the totals of the haplotype. A block therefore
  // ends where block(hap, block + 1) begins.
  const PathBlock &block(size_t hap, size_t block) const {
    return blocks_[first_block_[hap] + block];
  }

  uint32_t step_count(size_t hap) const {
    return block(hap, num_blocks(hap)).step_begin;
  }
  uint64_t bp_length(size_t hap) const {
    return block(hap, num_blocks(hap)).bp_begin;
  }

  // The block holding a step or a base offset. The position must be inside the
  // haplotype (step < step_count, bp < bp_length).
  size_t block_of_step(size_t hap, uint32_t step) const;
  size_t block_of_bp(size_t hap, uint64_t bp) const;

  // Call visit(hap, block) for every block that may visit a node in
  // [min_node, max_node]: one that visits a node in a zone the range touches
  // and whose own node range intersects it. Each block is reported once, in
  // increasing (hap, block) order.
  template <typename Visitor>
  void for_each_block_overlapping(uint32_t min_node, uint32_t max_node,
                                  Visitor &&visit) const {
    if (min_node > max_node || zone_starts_.size() < 2)
      return;
    auto report = [&](uint32_t stored) {
      const PathBlock &b = blocks_[stored];
      if (b.min_node <= max_node && b.max_node >= min_node) {
        const size_t hap = haplotype_of(stored);
        visit(hap, static_cast<size_t>(stored - first_block_[hap]));
      }
    };
    const size_t num_zones = zone_starts_.size() - 1;
    const size_t first_zone = min_node >> kZoneShift;
    const size_t last_zone =
        std::min<size_t>(max_node >> kZoneShift, num_zones - 1);
    if (first_zone > last_zone)
      return;
    if (first_zone == last_zone) {
      for (uint64_t i = zone_starts_[first_zone];
           i < zone_starts_[first_zone + 1]; ++i)
        report(zone_blocks_[i]);
      return;
    }
    // A block is listed in every zone it visits: merge the lists.
    std::vector<uint32_t> listed(zone_blocks_.begin() + zone_starts_[first_zone],
                                 zone_blocks_.begin() +
                                     zone_starts_[last_zone + 1]);
    std::sort(listed.begin(), listed.end());
    listed.erase(std::unique(listed.begin(), listed.end()), listed.end());
    for (uint32_t stored : listed)
      report(stored);
  }

  // Node ids per zone of the node lookup.
  static constexpr uint32_t kZoneShift = 8;
  static constexpr uint32_t kZoneNodes = uint32_t{1} << kZoneShift;

private:
  size_t haplotype_of(uint32_t stored_block) const;

  // Blocks of all haplotypes back to back, each haplotype followed by its end
  // sentinel. first_block_[hap] indexes blocks_; it has one extra entry.
  std::vector<PathBlock> blocks_;
  std::vector<uint64_t> first_block_{0};

  uint32_t block_steps_ = 0;

  // Zone map in CSR form: zone_blocks_[zone_starts_[z] .. zone_starts_[z + 1])
  // are the stored blocks that visit a node id in zone z, in increasing order.
  std::vector<uint64_t> zone_starts_{0};
  std::vector<uint32_t> zone_blocks_;
};

} // namespace gfaz

#endif // COMPUTE_PATH_INDEX_HPP
