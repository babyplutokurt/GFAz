#ifndef COMPUTE_PATH_ACCESS_HPP
#define COMPUTE_PATH_ACCESS_HPP

// Positional iteration over haplotypes: a TraversalSet for decoding, a
// PathIndex for where to start, and the segment lengths for base coordinates.
// A non-owning view; the three objects must outlive it.

#include "compute/path_index.hpp"
#include "compute/traversal_set.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace gfaz {

struct PathStep {
  uint32_t step = 0;     // index of the step in its haplotype
  uint64_t bp_begin = 0; // bases of the haplotype before this step
  uint32_t length = 0;   // bases of the node
  NodeId node = 0;       // signed node id; negative is reverse orientation

  uint64_t bp_end() const { return bp_begin + length; }
};

class PathAccess {
public:
  PathAccess(const TraversalSet &traversals, const PathIndex &index,
             const std::vector<uint32_t> &segment_lengths)
      : traversals_(traversals), index_(index), lengths_(segment_lengths) {}

  const TraversalSet &traversals() const { return traversals_; }
  const PathIndex &index() const { return index_; }

  uint32_t node_length(uint32_t node) const {
    return (node >= 1 && node <= lengths_.size()) ? lengths_[node - 1] : 0;
  }

  // Call visit(const PathStep &) for every step of blocks
  // [block_begin, block_end) of a haplotype, in path order.
  template <typename Visitor>
  void for_each_step_in_blocks(size_t hap, size_t block_begin, size_t block_end,
                               Visitor &&visit) const {
    if (block_begin >= block_end)
      return;
    const PathBlock &first = index_.block(hap, block_begin);
    const PathBlock &last = index_.block(hap, block_end);
    DeltaState state = first.state;
    PathStep step;
    step.step = first.step_begin;
    step.bp_begin = first.bp_begin;
    auto node_visit = [&](NodeId node) {
      step.node = node;
      step.length = node_length(tquery::abs_node_id(node));
      visit(static_cast<const PathStep &>(step));
      step.bp_begin += step.length;
      ++step.step;
    };
    traversals_.decode(hap, first.enc_begin, last.enc_begin, state, node_visit);
  }

  // Steps [step_begin, step_end) of a haplotype.
  template <typename Visitor>
  void for_each_step(size_t hap, uint32_t step_begin, uint32_t step_end,
                     Visitor &&visit) const {
    if (step_begin >= step_end)
      return;
    for_each_step_in_blocks(hap, index_.block_of_step(hap, step_begin),
                            index_.block_of_step(hap, step_end - 1) + 1,
                            [&](const PathStep &step) {
                              if (step.step >= step_begin &&
                                  step.step < step_end)
                                visit(step);
                            });
  }

  // Steps whose bases intersect [bp_begin, bp_end) of a haplotype.
  template <typename Visitor>
  void for_each_step_in_bases(size_t hap, uint64_t bp_begin, uint64_t bp_end,
                              Visitor &&visit) const {
    bp_end = std::min(bp_end, index_.bp_length(hap));
    if (bp_begin >= bp_end)
      return;
    for_each_step_in_blocks(hap, index_.block_of_bp(hap, bp_begin),
                            index_.block_of_bp(hap, bp_end - 1) + 1,
                            [&](const PathStep &step) {
                              if (step.bp_begin < bp_end &&
                                  step.bp_end() > bp_begin)
                                visit(step);
                            });
  }

  // Call visit(hap, const PathStep &) for every step of every haplotype on a
  // node in [min_node, max_node], ordered by haplotype and then by step. Only
  // the blocks whose node range reaches the interval are decoded.
  template <typename Visitor>
  void for_each_step_on_nodes(uint32_t min_node, uint32_t max_node,
                              Visitor &&visit) const {
    std::vector<std::pair<uint32_t, uint32_t>> blocks;
    index_.for_each_block_overlapping(
        min_node, max_node, [&](size_t hap, size_t block) {
          blocks.emplace_back(static_cast<uint32_t>(hap),
                              static_cast<uint32_t>(block));
        });
    std::sort(blocks.begin(), blocks.end());
    for (const auto &[hap, block] : blocks) {
      const size_t h = hap;
      for_each_step_in_blocks(h, block, block + 1, [&](const PathStep &step) {
        const uint32_t node = tquery::abs_node_id(step.node);
        if (node >= min_node && node <= max_node)
          visit(h, step);
      });
    }
  }

private:
  const TraversalSet &traversals_;
  const PathIndex &index_;
  const std::vector<uint32_t> &lengths_;
};

} // namespace gfaz

#endif // COMPUTE_PATH_ACCESS_HPP
