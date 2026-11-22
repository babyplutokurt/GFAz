#include "compute/repeat_index.hpp"

#include "compute/traversal_query.hpp"
#include "core/utils/threading_utils.hpp"

#include <algorithm>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace gfaz {

RepeatIndex RepeatIndex::build(const PathAccess &paths,
                               const RepeatIndexOptions &options) {
  ScopedOMPThreads omp_scope(options.num_threads);
  const PathIndex &index = paths.index();
  RepeatIndex out;

  // The crowded zones, and the blocks that visit them.
  const size_t num_zones = index.num_zones();
  out.zone_slot_.assign(num_zones, kNoSlot);
  std::vector<std::pair<uint32_t, uint32_t>> blocks; // (haplotype, block)
  uint32_t num_slots = 0;
  for (size_t zone = 0; zone < num_zones; ++zone) {
    const uint32_t first = static_cast<uint32_t>(zone << PathIndex::kZoneShift);
    const uint32_t last = first + (PathIndex::kZoneNodes - 1);
    const size_t before = blocks.size();
    index.for_each_block_overlapping(first, last, [&](size_t hap, size_t b) {
      blocks.emplace_back(static_cast<uint32_t>(hap), static_cast<uint32_t>(b));
    });
    if (blocks.size() - before > options.min_zone_blocks)
      out.zone_slot_[zone] = num_slots++;
    else
      blocks.resize(before);
  }
  std::sort(blocks.begin(), blocks.end());
  blocks.erase(std::unique(blocks.begin(), blocks.end()), blocks.end());

  // Call visit(row, haplotype, step) for every step of the blocks that lands
  // in a crowded zone.
  auto for_each_covered_step = [&](auto &&visit) {
#pragma omp parallel for schedule(dynamic, 64)
    for (size_t i = 0; i < blocks.size(); ++i) {
      const size_t hap = blocks[i].first;
      const size_t block = blocks[i].second;
      paths.for_each_step_in_blocks(hap, block, block + 1,
                                    [&](const PathStep &step) {
        const uint32_t node = tquery::abs_node_id(step.node);
        if (out.covers(node))
          visit(out.row_of(node), hap, step.step);
      });
    }
  };

  // Count the visits of each row, then fill the rows.
  std::vector<uint64_t> &starts = out.starts_;
  starts.assign(static_cast<size_t>(num_slots) * kZoneNodes + 1, 0);
  for_each_covered_step([&](size_t row, size_t, uint32_t) {
#pragma omp atomic
    ++starts[row + 1];
  });
  for (size_t row = 0; row + 1 < starts.size(); ++row)
    starts[row + 1] += starts[row];

  out.visits_.resize(starts.back());
  std::vector<uint64_t> next(starts.begin(), starts.end() - 1);
  for_each_covered_step([&](size_t row, size_t hap, uint32_t step) {
    uint64_t slot;
#pragma omp atomic capture
    slot = next[row]++;
    out.visits_[slot] = (static_cast<uint64_t>(hap) << 32) | step;
  });

  // Threads filled a row in no particular order.
#pragma omp parallel for schedule(dynamic, 256)
  for (size_t row = 0; row < starts.size() - 1; ++row)
    std::sort(out.visits_.begin() + starts[row],
              out.visits_.begin() + starts[row + 1]);
  return out;
}

} // namespace gfaz
