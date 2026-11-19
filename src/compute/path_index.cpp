#include "compute/path_index.hpp"

#include "core/utils/threading_utils.hpp"

#include <limits>
#include <stdexcept>

namespace gfaz {

namespace {

// A block that spans more zones than this is kept in the wide list instead of
// being listed zone by zone.
constexpr size_t kMaxZonesPerBlock = 32;

std::vector<PathBlock> index_haplotype(const TraversalSet &traversals,
                                       const std::vector<uint32_t> &lengths,
                                       size_t hap, uint32_t block_steps) {
  std::vector<PathBlock> blocks;
  const uint32_t enc_len = traversals.encoded_length(hap);
  const uint32_t num_nodes = static_cast<uint32_t>(lengths.size());

  DeltaState state;
  uint64_t bp = 0;
  uint32_t steps = 0;
  PathBlock current;
  current.min_node = std::numeric_limits<uint32_t>::max();
  bool open = false;

  auto visit = [&](NodeId signed_node) {
    const uint32_t node = tquery::abs_node_id(signed_node);
    current.min_node = std::min(current.min_node, node);
    current.max_node = std::max(current.max_node, node);
    if (node >= 1 && node <= num_nodes)
      bp += lengths[node - 1];
    ++steps;
  };

  for (uint32_t i = 0; i < enc_len; ++i) {
    if (open && steps - current.step_begin >= block_steps) {
      blocks.push_back(current);
      open = false;
    }
    if (!open) {
      current = PathBlock{};
      current.bp_begin = bp;
      current.enc_begin = i;
      current.step_begin = steps;
      current.min_node = std::numeric_limits<uint32_t>::max();
      current.state = state;
      open = true;
    }
    traversals.decode(hap, i, i + 1, state, visit);
  }
  if (open)
    blocks.push_back(current);

  PathBlock sentinel;
  sentinel.bp_begin = bp;
  sentinel.enc_begin = enc_len;
  sentinel.step_begin = steps;
  sentinel.min_node = 1; // empty range
  sentinel.max_node = 0;
  sentinel.state = state;
  blocks.push_back(sentinel);
  return blocks;
}

} // namespace

PathIndex PathIndex::build(const TraversalSet &traversals,
                           const std::vector<uint32_t> &segment_lengths,
                           const PathIndexOptions &options) {
  if (options.block_steps == 0)
    throw std::runtime_error("path index: block_steps must be positive");

  const size_t num_haps = traversals.size();
  std::vector<std::vector<PathBlock>> per_hap(num_haps);
  {
    ScopedOMPThreads omp_scope(options.num_threads);
#pragma omp parallel for schedule(dynamic, 1)
    for (long long h = 0; h < static_cast<long long>(num_haps); ++h)
      per_hap[h] = index_haplotype(traversals, segment_lengths,
                                   static_cast<size_t>(h), options.block_steps);
  }

  PathIndex index;
  index.first_block_.assign(num_haps + 1, 0);
  for (size_t h = 0; h < num_haps; ++h)
    index.first_block_[h + 1] = index.first_block_[h] + per_hap[h].size();
  if (index.first_block_[num_haps] > std::numeric_limits<uint32_t>::max())
    throw std::runtime_error("path index: too many blocks; raise block_steps");
  index.blocks_.reserve(index.first_block_[num_haps]);
  for (std::vector<PathBlock> &blocks : per_hap) {
    index.blocks_.insert(index.blocks_.end(), blocks.begin(), blocks.end());
    std::vector<PathBlock>().swap(blocks);
  }

  // Zones about as wide as a block is long: on a graph sorted along the genome
  // a block then spans only a few zones.
  index.zone_shift_ = 0;
  while ((uint64_t{1} << index.zone_shift_) < options.block_steps)
    ++index.zone_shift_;
  index.build_zone_map();
  return index;
}

void PathIndex::build_zone_map() {
  uint32_t max_node = 0;
  for (const PathBlock &b : blocks_)
    if (b.min_node <= b.max_node)
      max_node = std::max(max_node, b.max_node);
  const size_t num_zones = (static_cast<size_t>(max_node) >> zone_shift_) + 1;

  auto zone_span = [&](const PathBlock &b, size_t &first, size_t &last) {
    first = b.min_node >> zone_shift_;
    last = b.max_node >> zone_shift_;
    return last - first + 1 <= kMaxZonesPerBlock;
  };

  // Counting pass, then fill: a CSR layout with one list of blocks per zone.
  zone_starts_.assign(num_zones + 1, 0);
  wide_blocks_.clear();
  for (size_t i = 0; i < blocks_.size(); ++i) {
    const PathBlock &b = blocks_[i];
    if (b.min_node > b.max_node)
      continue;
    size_t first, last;
    if (!zone_span(b, first, last)) {
      wide_blocks_.push_back(static_cast<uint32_t>(i));
      continue;
    }
    for (size_t zone = first; zone <= last; ++zone)
      ++zone_starts_[zone + 1];
  }
  for (size_t zone = 0; zone < num_zones; ++zone)
    zone_starts_[zone + 1] += zone_starts_[zone];

  zone_blocks_.assign(zone_starts_[num_zones], 0);
  std::vector<uint64_t> cursor(zone_starts_.begin(), zone_starts_.end() - 1);
  for (size_t i = 0; i < blocks_.size(); ++i) {
    const PathBlock &b = blocks_[i];
    size_t first, last;
    if (b.min_node > b.max_node || !zone_span(b, first, last))
      continue;
    for (size_t zone = first; zone <= last; ++zone)
      zone_blocks_[cursor[zone]++] = static_cast<uint32_t>(i);
  }
}

size_t PathIndex::haplotype_of(uint32_t stored_block) const {
  const auto it = std::upper_bound(first_block_.begin(), first_block_.end(),
                                   static_cast<uint64_t>(stored_block));
  return static_cast<size_t>(it - first_block_.begin()) - 1;
}

size_t PathIndex::block_of_step(size_t hap, uint32_t step) const {
  const PathBlock *begin = blocks_.data() + first_block_[hap];
  const PathBlock *end = begin + num_blocks(hap);
  const PathBlock *it = std::upper_bound(
      begin, end, step,
      [](uint32_t value, const PathBlock &b) { return value < b.step_begin; });
  return static_cast<size_t>(it - begin) - 1;
}

size_t PathIndex::block_of_bp(size_t hap, uint64_t bp) const {
  const PathBlock *begin = blocks_.data() + first_block_[hap];
  const PathBlock *end = begin + num_blocks(hap);
  const PathBlock *it = std::upper_bound(
      begin, end, bp,
      [](uint64_t value, const PathBlock &b) { return value < b.bp_begin; });
  return static_cast<size_t>(it - begin) - 1;
}

} // namespace gfaz
