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
//   - node lookup: only blocks whose [min_node, max_node] range covers a node
//     can visit it, and pangenome graphs keep node ids close to genome order, so
//     those are few.
// The index is a few tens of bytes per block and is independent of the
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

  // Call visit(hap, block) for every block whose node range intersects
  // [min_node, max_node]. Each block is reported once, in no particular order.
  template <typename Visitor>
  void for_each_block_overlapping(uint32_t min_node, uint32_t max_node,
                                  Visitor &&visit) const {
    if (min_node > max_node || zone_starts_.empty())
      return;
    auto report = [&](uint32_t stored) {
      const PathBlock &b = blocks_[stored];
      if (b.min_node <= max_node && b.max_node >= min_node) {
        const size_t hap = haplotype_of(stored);
        visit(hap, static_cast<size_t>(stored - first_block_[hap]));
      }
    };
    const size_t num_zones = zone_starts_.size() - 1;
    const size_t first_zone = min_node >> zone_shift_;
    const size_t last_zone =
        std::min<size_t>(max_node >> zone_shift_, num_zones - 1);
    for (size_t zone = first_zone; zone <= last_zone; ++zone) {
      for (uint64_t i = zone_starts_[zone]; i < zone_starts_[zone + 1]; ++i) {
        // A block is listed in every zone it spans; report it from the first
        // one this query looks at.
        const uint32_t stored = zone_blocks_[i];
        const size_t home = blocks_[stored].min_node >> zone_shift_;
        if (zone == std::max(home, first_zone))
          report(stored);
      }
    }
    for (uint32_t stored : wide_blocks_)
      report(stored);
  }

private:
  size_t haplotype_of(uint32_t stored_block) const;
  void build_zone_map();

  // Blocks of all haplotypes back to back, each haplotype followed by its end
  // sentinel. first_block_[hap] indexes blocks_; it has one extra entry.
  std::vector<PathBlock> blocks_;
  std::vector<uint64_t> first_block_{0};

  uint32_t block_steps_ = 0;

  // Zone map: node ids are bucketed into zones of 2^zone_shift_ ids, and each
  // zone lists the blocks that span it. Blocks spanning many zones go to
  // wide_blocks_ and are checked on every query.
  uint32_t zone_shift_ = 0;
  std::vector<uint64_t> zone_starts_;
  std::vector<uint32_t> zone_blocks_;
  std::vector<uint32_t> wide_blocks_;
};

} // namespace gfaz

#endif // COMPUTE_PATH_INDEX_HPP
