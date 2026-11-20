#include "compute/path_index.hpp"

#include "core/utils/threading_utils.hpp"

#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <type_traits>

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
  index.block_steps_ = options.block_steps;
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

  index.build_zone_map();
  return index;
}

void PathIndex::build_zone_map() {
  // Zones about as wide as a block is long: on a graph sorted along the genome
  // a block then spans only a few zones.
  zone_shift_ = 0;
  while ((uint64_t{1} << zone_shift_) < block_steps_)
    ++zone_shift_;

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

namespace {

// Saved index layout, little-endian: the header below, first_block_
// (num_haplotypes + 1 values), then the blocks. The zone map is rebuilt on load.
constexpr char kIndexMagic[8] = {'G', 'F', 'Z', 'P', 'I', 'D', 'X', '1'};

struct IndexFileHeader {
  char magic[8];
  uint64_t fingerprint;
  uint64_t num_haplotypes;
  uint64_t num_stored_blocks;
  uint32_t block_steps;
  uint32_t block_bytes;
};

static_assert(std::is_trivially_copyable<PathBlock>::value &&
                  sizeof(PathBlock) == 40,
              "PathBlock is written to index files as raw bytes");

void fnv1a(uint64_t &hash, const void *bytes, size_t count) {
  const unsigned char *p = static_cast<const unsigned char *>(bytes);
  for (size_t i = 0; i < count; ++i) {
    hash ^= p[i];
    hash *= 1099511628211ull;
  }
}

template <typename T> void fnv1a_vector(uint64_t &hash, const std::vector<T> &v) {
  const uint64_t count = v.size();
  fnv1a(hash, &count, sizeof(count));
  if (!v.empty())
    fnv1a(hash, v.data(), v.size() * sizeof(T));
}

} // namespace

uint64_t PathIndex::fingerprint(const CompressedData &data) {
  // The per-haplotype lengths and the sizes of the three encoded blocks pin
  // down the traversals without reading them.
  uint64_t hash = 14695981039346656037ull;
  fnv1a_vector(hash, data.sequence_lengths);
  fnv1a_vector(hash, data.original_path_lengths);
  fnv1a_vector(hash, data.walk_lengths);
  fnv1a_vector(hash, data.original_walk_lengths);
  const uint64_t sizes[] = {
      data.rules_first_zstd.payload.size(), data.rules_second_zstd.payload.size(),
      data.paths_zstd.payload.size(),       data.walks_zstd.payload.size(),
      data.segment_seq_lengths_zstd.original_size,
      static_cast<uint64_t>(data.delta_round)};
  fnv1a(hash, sizes, sizeof(sizes));
  return hash;
}

void PathIndex::save(const std::string &path, uint64_t fingerprint) const {
  IndexFileHeader header{};
  std::memcpy(header.magic, kIndexMagic, sizeof(kIndexMagic));
  header.fingerprint = fingerprint;
  header.num_haplotypes = num_haplotypes();
  header.num_stored_blocks = blocks_.size();
  header.block_steps = block_steps_;
  header.block_bytes = sizeof(PathBlock);

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(reinterpret_cast<const char *>(&header), sizeof(header));
  out.write(reinterpret_cast<const char *>(first_block_.data()),
            static_cast<std::streamsize>(first_block_.size() * sizeof(uint64_t)));
  out.write(reinterpret_cast<const char *>(blocks_.data()),
            static_cast<std::streamsize>(blocks_.size() * sizeof(PathBlock)));
  out.close();
  if (!out)
    throw std::runtime_error("path index: cannot write " + path);
}

PathIndex PathIndex::load(const std::string &path, uint64_t fingerprint) {
  std::ifstream in(path, std::ios::binary);
  IndexFileHeader header{};
  in.read(reinterpret_cast<char *>(&header), sizeof(header));
  if (!in || std::memcmp(header.magic, kIndexMagic, sizeof(kIndexMagic)) != 0 ||
      header.block_bytes != sizeof(PathBlock) || header.block_steps == 0)
    throw std::runtime_error("path index: " + path +
                             " is not a readable path index");
  if (header.fingerprint != fingerprint)
    throw std::runtime_error("path index: " + path +
                             " was built from a different container; rebuild "
                             "it with `gfaz index`");

  PathIndex index;
  index.block_steps_ = header.block_steps;
  index.first_block_.resize(header.num_haplotypes + 1);
  index.blocks_.resize(header.num_stored_blocks);
  in.read(reinterpret_cast<char *>(index.first_block_.data()),
          static_cast<std::streamsize>(index.first_block_.size() *
                                       sizeof(uint64_t)));
  in.read(reinterpret_cast<char *>(index.blocks_.data()),
          static_cast<std::streamsize>(index.blocks_.size() *
                                       sizeof(PathBlock)));
  if (!in || index.first_block_.front() != 0 ||
      index.first_block_.back() != index.blocks_.size())
    throw std::runtime_error("path index: " + path + " is truncated");
  index.build_zone_map();
  return index;
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
