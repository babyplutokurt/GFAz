#include "compute/path_index.hpp"

#include "core/utils/threading_utils.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace gfaz {

namespace {

// The blocks of one haplotype, and the zones each block visits: block b
// visits zones[zone_begin[b] .. zone_begin[b + 1]).
struct HaplotypeBlocks {
  std::vector<PathBlock> blocks; // ends with the sentinel
  std::vector<uint32_t> zones;
  std::vector<uint64_t> zone_begin{0};
};

HaplotypeBlocks index_haplotype(const TraversalSet &traversals,
                                       const std::vector<uint32_t> &lengths,
                                       size_t hap, uint32_t block_steps) {
  HaplotypeBlocks out;
  std::vector<PathBlock> &blocks = out.blocks;
  const uint32_t enc_len = traversals.encoded_length(hap);
  const uint32_t num_nodes = static_cast<uint32_t>(lengths.size());

  DeltaState state;
  uint64_t bp = 0;
  uint32_t steps = 0;
  PathBlock current;
  current.min_node = std::numeric_limits<uint32_t>::max();
  bool open = false;

  // Zones of the open block. Steps mostly stay in the zone of the step before,
  // so skipping repeats of the last zone keeps this short.
  std::vector<uint32_t> zones;
  uint32_t last_zone = std::numeric_limits<uint32_t>::max();
  auto close_block = [&]() {
    std::sort(zones.begin(), zones.end());
    zones.erase(std::unique(zones.begin(), zones.end()), zones.end());
    out.zones.insert(out.zones.end(), zones.begin(), zones.end());
    out.zone_begin.push_back(out.zones.size());
    zones.clear();
    last_zone = std::numeric_limits<uint32_t>::max();
    blocks.push_back(current);
    open = false;
  };

  auto visit = [&](NodeId signed_node) {
    const uint32_t node = tquery::abs_node_id(signed_node);
    current.min_node = std::min(current.min_node, node);
    current.max_node = std::max(current.max_node, node);
    const uint32_t zone = node >> PathIndex::kZoneShift;
    if (zone != last_zone) {
      zones.push_back(zone);
      last_zone = zone;
    }
    if (node >= 1 && node <= num_nodes)
      bp += lengths[node - 1];
    ++steps;
  };

  for (uint32_t i = 0; i < enc_len; ++i) {
    if (open && steps - current.step_begin >= block_steps)
      close_block();
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
    close_block();

  PathBlock sentinel;
  sentinel.bp_begin = bp;
  sentinel.enc_begin = enc_len;
  sentinel.step_begin = steps;
  sentinel.min_node = 1; // empty range
  sentinel.max_node = 0;
  sentinel.state = state;
  blocks.push_back(sentinel);
  return out;
}

} // namespace

PathIndex PathIndex::build(const TraversalSet &traversals,
                           const std::vector<uint32_t> &segment_lengths,
                           const PathIndexOptions &options) {
  if (options.block_steps == 0)
    throw std::runtime_error("path index: block_steps must be positive");

  const size_t num_haps = traversals.size();
  std::vector<HaplotypeBlocks> per_hap(num_haps);
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
    index.first_block_[h + 1] =
        index.first_block_[h] + per_hap[h].blocks.size();
  if (index.first_block_[num_haps] > std::numeric_limits<uint32_t>::max())
    throw std::runtime_error("path index: too many blocks; raise block_steps");
  index.blocks_.reserve(index.first_block_[num_haps]);
  for (HaplotypeBlocks &hap : per_hap) {
    index.blocks_.insert(index.blocks_.end(), hap.blocks.begin(),
                         hap.blocks.end());
    std::vector<PathBlock>().swap(hap.blocks);
  }

  // Zone map: count the blocks of every zone, then fill. Going through the
  // haplotypes in order leaves each zone's list sorted.
  uint32_t max_zone = 0;
  bool any_zone = false;
  for (const HaplotypeBlocks &hap : per_hap)
    for (uint32_t zone : hap.zones) {
      max_zone = std::max(max_zone, zone);
      any_zone = true;
    }
  const size_t num_zones = any_zone ? static_cast<size_t>(max_zone) + 1 : 0;
  index.zone_starts_.assign(num_zones + 1, 0);
  for (const HaplotypeBlocks &hap : per_hap)
    for (uint32_t zone : hap.zones)
      ++index.zone_starts_[zone + 1];
  for (size_t zone = 0; zone < num_zones; ++zone)
    index.zone_starts_[zone + 1] += index.zone_starts_[zone];
  index.zone_blocks_.assign(index.zone_starts_[num_zones], 0);
  std::vector<uint64_t> cursor(index.zone_starts_.begin(),
                               index.zone_starts_.end() - 1);
  for (size_t h = 0; h < num_haps; ++h) {
    HaplotypeBlocks &hap = per_hap[h];
    for (size_t b = 0; b + 1 < hap.zone_begin.size(); ++b) {
      const uint32_t stored =
          static_cast<uint32_t>(index.first_block_[h] + b);
      for (uint64_t i = hap.zone_begin[b]; i < hap.zone_begin[b + 1]; ++i)
        index.zone_blocks_[cursor[hap.zones[i]]++] = stored;
    }
    std::vector<uint32_t>().swap(hap.zones);
  }
  return index;
}

namespace {

// Saved index layout, little-endian: the header below, first_block_
// (num_haplotypes + 1 values), the blocks, then the zone map (num_zones + 1
// starts, num_zone_blocks block numbers).
constexpr char kIndexMagic[8] = {'G', 'F', 'Z', 'P', 'I', 'D', 'X', '2'};
// Files of the first layout had no zone map.
constexpr char kOldIndexMagic[8] = {'G', 'F', 'Z', 'P', 'I', 'D', 'X', '1'};

struct IndexFileHeader {
  char magic[8];
  uint64_t fingerprint;
  uint64_t num_haplotypes;
  uint64_t num_stored_blocks;
  uint32_t block_steps;
  uint32_t block_bytes;
  uint64_t num_zones;
  uint64_t num_zone_blocks;
  uint32_t zone_shift;
  uint32_t reserved;
};

template <typename T>
void write_vector(std::ofstream &out, const std::vector<T> &values) {
  out.write(reinterpret_cast<const char *>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(T)));
}

template <typename T>
void read_vector(std::ifstream &in, std::vector<T> &values, uint64_t count) {
  values.resize(count);
  in.read(reinterpret_cast<char *>(values.data()),
          static_cast<std::streamsize>(count * sizeof(T)));
}

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
  header.num_zones = zone_starts_.size() - 1;
  header.num_zone_blocks = zone_blocks_.size();
  header.zone_shift = kZoneShift;

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(reinterpret_cast<const char *>(&header), sizeof(header));
  write_vector(out, first_block_);
  write_vector(out, blocks_);
  write_vector(out, zone_starts_);
  write_vector(out, zone_blocks_);
  out.close();
  if (!out)
    throw std::runtime_error("path index: cannot write " + path);
}

uint32_t PathIndex::saved_block_steps(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  IndexFileHeader header{};
  in.read(reinterpret_cast<char *>(&header), sizeof(header));
  if (!in || std::memcmp(header.magic, kIndexMagic, sizeof(kIndexMagic)) != 0 ||
      header.block_bytes != sizeof(PathBlock) ||
      header.zone_shift != kZoneShift)
    return 0;
  return header.block_steps;
}

PathIndex PathIndex::load(const std::string &path, uint64_t fingerprint) {
  std::ifstream in(path, std::ios::binary);
  IndexFileHeader header{};
  in.read(reinterpret_cast<char *>(&header), sizeof(header));
  if (in.gcount() >= static_cast<std::streamsize>(sizeof(kOldIndexMagic)) &&
      std::memcmp(header.magic, kOldIndexMagic, sizeof(kOldIndexMagic)) == 0)
    throw std::runtime_error("path index: " + path +
                             " was written by an older gfaz; rebuild it with "
                             "`gfaz index`");
  if (!in || std::memcmp(header.magic, kIndexMagic, sizeof(kIndexMagic)) != 0 ||
      header.block_bytes != sizeof(PathBlock) || header.block_steps == 0 ||
      header.zone_shift != kZoneShift)
    throw std::runtime_error("path index: " + path +
                             " is not a readable path index");
  if (header.fingerprint != fingerprint)
    throw std::runtime_error("path index: " + path +
                             " was built from a different container; rebuild "
                             "it with `gfaz index`");

  PathIndex index;
  index.block_steps_ = header.block_steps;
  read_vector(in, index.first_block_, header.num_haplotypes + 1);
  read_vector(in, index.blocks_, header.num_stored_blocks);
  read_vector(in, index.zone_starts_, header.num_zones + 1);
  read_vector(in, index.zone_blocks_, header.num_zone_blocks);
  if (!in || index.first_block_.front() != 0 ||
      index.first_block_.back() != index.blocks_.size() ||
      index.zone_starts_.front() != 0 ||
      index.zone_starts_.back() != index.zone_blocks_.size())
    throw std::runtime_error("path index: " + path + " is truncated");
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
