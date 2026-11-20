#include "handlegraph_adapter/gfaz_handle_graph.hpp"

#include "core/codec/codec.hpp"
#include "core/codec/serialization.hpp"
#include "core/io/gfa_write_utils.hpp"
#include "core/utils/sequence_utils.hpp"
#include "core/utils/threading_utils.hpp"

#include <handlegraph/util.hpp>

#include <algorithm>
#include <atomic>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace gfaz {

using handlegraph::as_integer;
using handlegraph::as_integers;
using handlegraph::number_bool_packing;
using handlegraph::PathMetadata;
using handlegraph::PathSense;
using handlegraph::subrange_t;

namespace {

std::atomic<uint64_t> next_instance_id{1};

uint32_t packed(const handlegraph::handle_t &handle) {
  return static_cast<uint32_t>(as_integer(handle));
}

handlegraph::handle_t unpacked(uint32_t value) {
  return handlegraph::as_handle(static_cast<uint64_t>(value));
}

// The segment number a handle packs, and the handle of an oriented segment.
uint32_t segment_of_handle(const handlegraph::handle_t &handle) {
  return static_cast<uint32_t>(number_bool_packing::unpack_number(handle));
}

handlegraph::handle_t segment_handle(uint32_t segment, bool is_reverse = false) {
  return number_bool_packing::pack(segment, is_reverse);
}

handlegraph::step_handle_t make_step(size_t path, int64_t index) {
  handlegraph::step_handle_t step;
  as_integers(step)[0] = static_cast<int64_t>(path);
  as_integers(step)[1] = index;
  return step;
}

size_t path_of(const handlegraph::step_handle_t &step) {
  return static_cast<size_t>(as_integers(step)[0]);
}

int64_t index_of(const handlegraph::step_handle_t &step) {
  return as_integers(step)[1];
}

// Samples named by the header's RS:Z tag (space-separated), the convention
// GBWTGraph and vg use to mark reference samples in a GFA.
void add_header_reference_samples(const std::string &header,
                                  std::unordered_set<std::string> &samples) {
  const std::string tag = "RS:Z:";
  size_t begin = header.find("\t" + tag);
  if (begin == std::string::npos)
    return;
  begin += 1 + tag.size();
  const size_t end = std::min(header.find('\t', begin), header.size());
  size_t cursor = begin;
  while (cursor < end) {
    const size_t stop = std::min(header.find(' ', cursor), end);
    if (stop > cursor)
      samples.insert(header.substr(cursor, stop - cursor));
    cursor = stop + 1;
  }
}

} // namespace

// One index block of one path, decoded: its nodes and where each one starts.
struct GfazHandleGraph::DecodedBlock {
  uint64_t key = 0; // (path, block), see block_key; 0 is "empty"
  uint32_t step_begin = 0;
  std::vector<NodeId> nodes;
  std::vector<uint64_t> positions;
};

namespace {

uint64_t block_key(size_t path, size_t block) {
  // +1 keeps a valid key different from the empty key 0.
  return ((static_cast<uint64_t>(path) + 1) << 32) | static_cast<uint64_t>(block);
}

} // namespace

// The decoded blocks one thread keeps for one graph. Walking a path touches
// one block at a time; asking for the steps on neighbouring nodes touches the
// same block of every haplotype over and over, which is what the capacity is
// sized for. Slots are recycled in rotation.
class GfazHandleGraph::BlockCache {
public:
  explicit BlockCache(size_t capacity) : slots_(std::max<size_t>(capacity, 1)) {}

  DecodedBlock *find(uint64_t key) {
    const auto it = slot_of_.find(key);
    return it == slot_of_.end() ? nullptr : &slots_[it->second];
  }

  // A slot to decode `key` into; whatever it held is forgotten.
  DecodedBlock &recycle(uint64_t key) {
    DecodedBlock &slot = slots_[next_];
    if (slot.key != 0)
      slot_of_.erase(slot.key);
    slot.key = key;
    slot_of_[key] = next_;
    next_ = (next_ + 1) % slots_.size();
    return slot;
  }

private:
  std::vector<DecodedBlock> slots_;
  std::unordered_map<uint64_t, size_t> slot_of_;
  size_t next_ = 0;
};

GfazHandleGraph::GfazHandleGraph(const CompressedData &data,
                                 const GfazHandleGraphOptions &options)
    : GfazHandleGraph(data, options, nullptr) {}

namespace {

void release(ZstdCompressedBlock &block) {
  std::vector<uint8_t>().swap(block.payload);
}

} // namespace

GfazHandleGraph::GfazHandleGraph(const CompressedData &data,
                                 const GfazHandleGraphOptions &options,
                                 CompressedData *owned)
    : paths_(data,
             IndexedPathsOptions{options.index_path, options.block_steps,
                                 options.num_threads}),
      instance_id_(next_instance_id.fetch_add(1)),
      cached_blocks_(options.cached_blocks) {
  // A step on a node without an S-line cannot be given a handle.
  for (size_t hap = 0; hap < paths_.index.num_haplotypes(); ++hap)
    for (size_t b = 0; b < paths_.index.num_blocks(hap); ++b) {
      const PathBlock &block = paths_.index.block(hap, b);
      if (block.min_node == 0 || block.max_node > paths_.num_nodes())
        throw std::runtime_error(
            "gfaz handle graph: a path visits a node that has no segment");
    }

  // The traversals are decoded now.
  if (owned) {
    release(owned->rules_first_zstd);
    release(owned->rules_second_zstd);
    release(owned->paths_zstd);
    release(owned->walks_zstd);
  }

  number_nodes();
  load_sequences(data);
  if (owned)
    release(owned->segment_sequences_zstd);
  build_adjacency(data, options.num_threads);
  if (owned) {
    release(owned->link_from_ids_zstd);
    release(owned->link_to_ids_zstd);
    release(owned->link_from_orients_zstd);
    release(owned->link_to_orients_zstd);
  }
  name_paths(data, options.reference_samples);
}

GfazHandleGraph::~GfazHandleGraph() = default;

std::unique_ptr<GfazHandleGraph>
GfazHandleGraph::load(const std::string &path, GfazHandleGraphOptions options) {
  if (options.index_path.empty() && std::ifstream(path + ".pidx").good())
    options.index_path = path + ".pidx";
  CompressedData data = deserialize_compressed_data(path);
  return std::unique_ptr<GfazHandleGraph>(
      new GfazHandleGraph(data, options, &data));
}

void GfazHandleGraph::load_sequences(const CompressedData &data) {
  sequences_ = Codec::zstd_decompress_string(data.segment_sequences_zstd);
  const std::vector<uint32_t> &lengths = paths_.segment_lengths;
  sequence_starts_.reserve(lengths.size() / kSequenceStride + 1);
  uint64_t offset = 0;
  for (size_t i = 0; i < lengths.size(); ++i) {
    if (i % kSequenceStride == 0)
      sequence_starts_.push_back(offset);
    offset += lengths[i];
  }
  total_length_ = offset;
  if (offset > sequences_.size())
    throw std::runtime_error(
        "gfaz handle graph: segment sequence column is truncated");
}

uint64_t GfazHandleGraph::sequence_begin(uint32_t segment) const {
  const size_t index = segment - 1;
  const size_t first = index - index % kSequenceStride;
  uint64_t offset = sequence_starts_[index / kSequenceStride];
  for (size_t i = first; i < index; ++i)
    offset += paths_.segment_lengths[i];
  return offset;
}

void GfazHandleGraph::build_adjacency(const CompressedData &data,
                                      int num_threads) {
  const LinkData links = gfa_write_utils::decompress_links(data);
  const uint32_t num_nodes = paths_.num_nodes();
  const size_t num_handles = 2 * static_cast<size_t>(num_nodes);

  // Each link is stored from both ends: a -> b, and flip(b) -> flip(a). Call
  // visit(from, to) with packed handles for every stored direction.
  auto for_each_direction = [&](auto &&visit) {
    for (size_t i = 0; i < links.from_ids.size(); ++i) {
      const uint32_t from_id = links.from_ids[i];
      const uint32_t to_id = links.to_ids[i];
      if (from_id == 0 || to_id == 0 || from_id > num_nodes || to_id > num_nodes)
        continue;
      const uint32_t a = (from_id << 1) | (links.from_orients[i] == '-');
      const uint32_t b = (to_id << 1) | (links.to_orients[i] == '-');
      visit(a, b);
      if ((b ^ 1) != a || (a ^ 1) != b)
        visit(b ^ 1, a ^ 1);
    }
  };

  // Counting pass, then fill (handles start at 2, hence the shift).
  std::vector<uint32_t> offsets(num_handles + 1, 0);
  uint64_t total = 0;
  for_each_direction([&](uint32_t from, uint32_t) {
    ++offsets[from - 2 + 1];
    ++total;
  });
  if (total > std::numeric_limits<uint32_t>::max())
    throw std::runtime_error("gfaz handle graph: too many links");
  for (size_t h = 0; h < num_handles; ++h)
    offsets[h + 1] += offsets[h];
  std::vector<uint32_t> targets(total);
  {
    std::vector<uint32_t> cursor(offsets.begin(), offsets.end() - 1);
    for_each_direction(
        [&](uint32_t from, uint32_t to) { targets[cursor[from - 2]++] = to; });
  }

  // Sort each list and drop repeated links, then close the gaps.
  std::vector<uint32_t> kept(num_handles, 0);
  {
    ScopedOMPThreads omp_scope(num_threads);
#pragma omp parallel for schedule(dynamic, 4096)
    for (long long h = 0; h < static_cast<long long>(num_handles); ++h) {
      uint32_t *begin = targets.data() + offsets[h];
      uint32_t *end = targets.data() + offsets[h + 1];
      std::sort(begin, end);
      kept[h] = static_cast<uint32_t>(std::unique(begin, end) - begin);
    }
  }
  edge_offsets_.assign(num_handles + 1, 0);
  size_t write = 0;
  edge_count_ = 0;
  for (size_t h = 0; h < num_handles; ++h) {
    edge_offsets_[h] = static_cast<uint32_t>(write);
    const uint32_t from = static_cast<uint32_t>(h + 2);
    for (uint32_t i = 0; i < kept[h]; ++i) {
      const uint32_t to = targets[offsets[h] + i];
      targets[write++] = to;
      // Count an edge at the smaller of its two stored directions.
      if (std::make_pair(from, to) <= std::make_pair(to ^ 1, from ^ 1))
        ++edge_count_;
    }
  }
  edge_offsets_[num_handles] = static_cast<uint32_t>(write);
  targets.resize(write);
  targets.shrink_to_fit();
  edge_targets_ = std::move(targets);
}

void GfazHandleGraph::name_paths(
    const CompressedData &data,
    const std::unordered_set<std::string> &reference_samples) {
  std::unordered_set<std::string> references = reference_samples;
  add_header_reference_samples(data.header_line, references);

  // Names follow what vg's GFA import produces, so a path has the same name
  // whether the graph was converted or is read through this adapter.
  const HaplotypeCatalog &catalog = paths_.catalog;
  path_names_.reserve(catalog.size());
  for (size_t hap = 0; hap < catalog.size(); ++hap) {
    const Haplotype &haplotype = catalog[hap];
    PathSense sense;
    std::string sample, locus;
    size_t hap_number, phase_block;
    subrange_t subrange;
    if (!haplotype.is_walk) {
      PathMetadata::parse_path_name(haplotype.name, sense, sample, locus,
                                    hap_number, phase_block, subrange);
      if (sense == PathSense::HAPLOTYPE && references.count(sample)) {
        sense = PathSense::REFERENCE;
      } else if (sense == PathSense::REFERENCE &&
                 hap_number != PathMetadata::NO_HAPLOTYPE &&
                 !references.count(sample)) {
        // A full PanSN name (sample#hap#contig) is a haplotype unless its
        // sample is listed as a reference.
        sense = PathSense::HAPLOTYPE;
        if (phase_block == PathMetadata::NO_PHASE_BLOCK)
          phase_block = 0;
      }
    } else if (haplotype.sample == "*") {
      sense = PathSense::GENERIC;
      sample = PathMetadata::NO_SAMPLE_NAME;
      locus = haplotype.seq_id;
      hap_number = PathMetadata::NO_HAPLOTYPE;
      phase_block = PathMetadata::NO_PHASE_BLOCK;
    } else {
      const bool reference = references.count(haplotype.sample) > 0;
      sense = reference ? PathSense::REFERENCE : PathSense::HAPLOTYPE;
      sample = haplotype.sample;
      locus = haplotype.seq_id;
      hap_number = haplotype.hap_index;
      phase_block = reference ? PathMetadata::NO_PHASE_BLOCK : 0;
    }
    if (haplotype.is_walk) {
      // A walk that starts at 0 is taken to be the whole contig.
      subrange = PathMetadata::NO_SUBRANGE;
      if (haplotype.seq_start > 0) {
        subrange.first = haplotype.seq_start;
        if (haplotype.seq_end >= 0)
          subrange.second = haplotype.seq_end;
      }
    }

    std::string name = PathMetadata::create_path_name(
        sense, sample, locus, hap_number, phase_block, subrange);
    if (!path_by_name_.emplace(name, static_cast<uint32_t>(hap)).second)
      throw std::runtime_error("gfaz handle graph: duplicate path name " +
                               name);
    path_names_.push_back(std::move(name));
  }
}

// ---------------------------------------------------------------------------
// HandleGraph
// ---------------------------------------------------------------------------

// Node ids are assigned the way vg's GFA import assigns them
// (GFAParser::assign_new_sequence_id), so that a graph read here and the same
// GFA read by vg agree: in S-line order, a name made of digits with a positive
// value is the id, until the first name that is not; from there on every
// segment gets the largest id so far plus one.
void GfazHandleGraph::number_nodes() {
  const uint32_t num_nodes = paths_.num_nodes();
  min_node_id_ = num_nodes == 0 ? 0 : 1;
  max_node_id_ = num_nodes;
  const SegmentNames &names = paths_.names;
  if (names.dense())
    return;

  // Ids come out one segment at a time and are kept as runs of consecutive
  // values.
  std::vector<IdRun> runs;
  nid_t previous = 0;
  auto add = [&](uint32_t segment, nid_t id) {
    if (runs.empty() || id != previous + 1)
      runs.push_back({segment, id});
    previous = id;
  };

  // Names that are all positive numbers need no formatting and parsing.
  bool positive_numbers = names.plain_numbers();
  if (positive_numbers) {
    SegmentNames::RunHint hint;
    for (uint32_t segment = 1; segment <= num_nodes && positive_numbers;
         ++segment) {
      const nid_t id = names.number(segment, hint);
      positive_numbers = id > 0;
      if (positive_numbers)
        add(segment, id);
    }
  }
  if (!positive_numbers) {
    runs.clear();
    bool numeric = true;
    nid_t largest = 0;
    std::string name;
    for (uint32_t segment = 1; segment <= num_nodes; ++segment) {
      nid_t id = 0;
      if (numeric) {
        name.clear();
        names.append(name, segment);
        numeric = !name.empty() && name.size() <= 18 &&
                  name.find_first_not_of("0123456789") == std::string::npos;
        if (numeric) {
          id = std::stoll(name);
          numeric = id > 0;
        }
      }
      if (!numeric)
        id = largest + 1;
      largest = std::max(largest, id);
      add(segment, id);
    }
  }
  if (runs.empty() || (runs.size() == 1 && runs[0].id == 1))
    return; // segment n has id n

  // The last id of a run, given where the next run starts.
  auto last_id = [&](uint32_t r) {
    const uint32_t end =
        r + 1 < runs.size() ? runs[r + 1].segment : num_nodes + 1;
    return runs[r].id + static_cast<nid_t>(end - runs[r].segment) - 1;
  };
  std::vector<uint32_t> by_id(runs.size());
  for (uint32_t r = 0; r < by_id.size(); ++r)
    by_id[r] = r;
  std::sort(by_id.begin(), by_id.end(),
            [&](uint32_t a, uint32_t b) { return runs[a].id < runs[b].id; });
  // Two names with one value ("7" and "007") cannot both be that id.
  for (size_t i = 1; i < by_id.size(); ++i)
    if (runs[by_id[i]].id <= last_id(by_id[i - 1]))
      throw std::runtime_error(
          "gfaz handle graph: two segment names give the node id " +
          std::to_string(runs[by_id[i]].id));
  min_node_id_ = runs[by_id.front()].id;
  max_node_id_ = last_id(by_id.back());
  id_runs_ = std::move(runs);
  runs_by_id_ = std::move(by_id);
}

handlegraph::nid_t GfazHandleGraph::id_of_segment(uint32_t segment) const {
  if (id_runs_.empty())
    return static_cast<nid_t>(segment);
  // Lookups come in runs of neighbouring segments: try the run found last.
  thread_local size_t hint = 0;
  size_t r = hint;
  if (r >= id_runs_.size() || id_runs_[r].segment > segment ||
      (r + 1 < id_runs_.size() && id_runs_[r + 1].segment <= segment)) {
    const auto it = std::upper_bound(
        id_runs_.begin(), id_runs_.end(), segment,
        [](uint32_t value, const IdRun &run) { return value < run.segment; });
    r = static_cast<size_t>(it - id_runs_.begin()) - 1;
    hint = r;
  }
  return id_runs_[r].id + static_cast<nid_t>(segment - id_runs_[r].segment);
}

uint32_t GfazHandleGraph::segment_of(nid_t node_id) const {
  if (id_runs_.empty())
    return node_id >= 1 && static_cast<uint64_t>(node_id) <= paths_.num_nodes()
               ? static_cast<uint32_t>(node_id)
               : 0;
  // The last run that starts at or before the id, if the id is inside it.
  const auto it = std::upper_bound(
      runs_by_id_.begin(), runs_by_id_.end(), node_id,
      [&](nid_t value, uint32_t r) { return value < id_runs_[r].id; });
  if (it == runs_by_id_.begin())
    return 0;
  const uint32_t r = *(it - 1);
  const uint32_t end = r + 1 < id_runs_.size() ? id_runs_[r + 1].segment
                                               : paths_.num_nodes() + 1;
  const uint64_t offset = static_cast<uint64_t>(node_id - id_runs_[r].id);
  return offset < end - id_runs_[r].segment
             ? id_runs_[r].segment + static_cast<uint32_t>(offset)
             : 0;
}

bool GfazHandleGraph::has_node(nid_t node_id) const {
  return segment_of(node_id) != 0;
}

handlegraph::handle_t GfazHandleGraph::get_handle(const nid_t &node_id,
                                                  bool is_reverse) const {
  return segment_handle(segment_of(node_id), is_reverse);
}

handlegraph::nid_t GfazHandleGraph::get_id(const handle_t &handle) const {
  return id_of_segment(segment_of_handle(handle));
}

std::vector<handlegraph::oriented_node_range_t> GfazHandleGraph::translate_back(
    const handlegraph::oriented_node_range_t &range) const {
  return {range};
}

std::string
GfazHandleGraph::get_back_graph_node_name(const nid_t &back_node_id) const {
  return paths_.names.name(segment_of(back_node_id));
}

bool GfazHandleGraph::get_is_reverse(const handle_t &handle) const {
  return number_bool_packing::unpack_bit(handle);
}

handlegraph::handle_t GfazHandleGraph::flip(const handle_t &handle) const {
  return number_bool_packing::toggle_bit(handle);
}

size_t GfazHandleGraph::get_length(const handle_t &handle) const {
  return paths_.segment_lengths[segment_of_handle(handle) - 1];
}

std::string GfazHandleGraph::get_sequence(const handle_t &handle) const {
  return get_subsequence(handle, 0, get_length(handle));
}

size_t GfazHandleGraph::get_node_count() const { return paths_.num_nodes(); }

handlegraph::nid_t GfazHandleGraph::min_node_id() const {
  return min_node_id_;
}

handlegraph::nid_t GfazHandleGraph::max_node_id() const {
  return max_node_id_;
}

size_t GfazHandleGraph::get_degree(const handle_t &handle, bool go_left) const {
  const uint32_t from = go_left ? packed(handle) ^ 1 : packed(handle);
  return edge_offsets_[from - 1] - edge_offsets_[from - 2];
}

size_t GfazHandleGraph::get_edge_count() const { return edge_count_; }

size_t GfazHandleGraph::get_total_length() const {
  return total_length_;
}

char GfazHandleGraph::get_base(const handle_t &handle, size_t index) const {
  const uint32_t id = segment_of_handle(handle);
  const uint64_t begin = sequence_begin(id);
  if (!get_is_reverse(handle))
    return sequences_[begin + index];
  return complement_base(
      sequences_[begin + paths_.segment_lengths[id - 1] - 1 - index]);
}

std::string GfazHandleGraph::get_subsequence(const handle_t &handle,
                                             size_t index, size_t size) const {
  const uint32_t id = segment_of_handle(handle);
  const size_t length = get_length(handle);
  index = std::min(index, length);
  size = std::min(size, length - index);
  if (!get_is_reverse(handle))
    return sequences_.substr(sequence_begin(id) + index, size);
  // The reverse strand reads the stored sequence backwards from its end.
  std::string result =
      sequences_.substr(sequence_begin(id) + length - index - size, size);
  reverse_complement_inplace(result);
  return result;
}

bool GfazHandleGraph::follow_edges_impl(
    const handle_t &handle, bool go_left,
    const std::function<bool(const handle_t &)> &iteratee) const {
  // Predecessors of h are the flipped successors of flip(h).
  const uint32_t from = go_left ? packed(handle) ^ 1 : packed(handle);
  for (uint32_t i = edge_offsets_[from - 2]; i < edge_offsets_[from - 1]; ++i) {
    const uint32_t to = go_left ? edge_targets_[i] ^ 1 : edge_targets_[i];
    if (!iteratee(unpacked(to)))
      return false;
  }
  return true;
}

bool GfazHandleGraph::for_each_handle_impl(
    const std::function<bool(const handle_t &)> &iteratee,
    bool parallel) const {
  const long long num_nodes = paths_.num_nodes();
  if (!parallel) {
    for (long long id = 1; id <= num_nodes; ++id)
      if (!iteratee(segment_handle(id)))
        return false;
    return true;
  }
  std::atomic<bool> keep_going{true};
#pragma omp parallel for schedule(dynamic, 1024)
  for (long long id = 1; id <= num_nodes; ++id) {
    if (keep_going.load(std::memory_order_relaxed) &&
        !iteratee(segment_handle(id)))
      keep_going.store(false, std::memory_order_relaxed);
  }
  return keep_going.load();
}

// ---------------------------------------------------------------------------
// PathHandleGraph
// ---------------------------------------------------------------------------

GfazHandleGraph::BlockCache &GfazHandleGraph::thread_cache() const {
  // One cache per thread and per graph. A thread that reads several graphs
  // keeps a cache for each; they are dropped when the thread ends.
  thread_local std::unordered_map<uint64_t, std::unique_ptr<BlockCache>> caches;
  std::unique_ptr<BlockCache> &cache = caches[instance_id_];
  if (!cache)
    cache = std::make_unique<BlockCache>(cached_blocks_);
  return *cache;
}

const GfazHandleGraph::DecodedBlock &
GfazHandleGraph::decoded_block(size_t path, size_t block) const {
  BlockCache &cache = thread_cache();
  const uint64_t key = block_key(path, block);
  if (const DecodedBlock *hit = cache.find(key))
    return *hit;

  DecodedBlock &entry = cache.recycle(key);
  entry.step_begin = paths_.index.block(path, block).step_begin;
  entry.nodes.clear();
  entry.positions.clear();
  paths_.access.for_each_step_in_blocks(path, block, block + 1,
                                        [&](const PathStep &step) {
    entry.nodes.push_back(step.node);
    entry.positions.push_back(step.bp_begin);
  });
  return entry;
}

const GfazHandleGraph::DecodedBlock &
GfazHandleGraph::block_with_step(size_t path, uint32_t step) const {
  return decoded_block(path, paths_.index.block_of_step(path, step));
}

size_t GfazHandleGraph::get_path_count() const { return path_names_.size(); }

bool GfazHandleGraph::has_path(const std::string &path_name) const {
  return path_by_name_.count(path_name) > 0;
}

handlegraph::path_handle_t
GfazHandleGraph::get_path_handle(const std::string &path_name) const {
  const auto it = path_by_name_.find(path_name);
  if (it == path_by_name_.end())
    throw std::runtime_error("gfaz handle graph: no path named " + path_name);
  return handlegraph::as_path_handle(it->second);
}

std::string
GfazHandleGraph::get_path_name(const path_handle_t &path_handle) const {
  return path_names_[as_integer(path_handle)];
}

bool GfazHandleGraph::get_is_circular(const path_handle_t &) const {
  return false;
}

size_t GfazHandleGraph::get_step_count(const path_handle_t &path_handle) const {
  return paths_.index.step_count(as_integer(path_handle));
}

handlegraph::handle_t
GfazHandleGraph::get_handle_of_step(const step_handle_t &step_handle) const {
  const uint32_t step = static_cast<uint32_t>(index_of(step_handle));
  const DecodedBlock &block = block_with_step(path_of(step_handle), step);
  const NodeId node = block.nodes[step - block.step_begin];
  return segment_handle(tquery::abs_node_id(node), node < 0);
}

handlegraph::path_handle_t GfazHandleGraph::get_path_handle_of_step(
    const step_handle_t &step_handle) const {
  return handlegraph::as_path_handle(path_of(step_handle));
}

handlegraph::step_handle_t
GfazHandleGraph::path_begin(const path_handle_t &path_handle) const {
  return make_step(as_integer(path_handle), 0);
}

handlegraph::step_handle_t
GfazHandleGraph::path_end(const path_handle_t &path_handle) const {
  return make_step(as_integer(path_handle),
                   static_cast<int64_t>(get_step_count(path_handle)));
}

handlegraph::step_handle_t
GfazHandleGraph::path_back(const path_handle_t &path_handle) const {
  return make_step(as_integer(path_handle),
                   static_cast<int64_t>(get_step_count(path_handle)) - 1);
}

handlegraph::step_handle_t
GfazHandleGraph::path_front_end(const path_handle_t &path_handle) const {
  return make_step(as_integer(path_handle), -1);
}

bool GfazHandleGraph::has_next_step(const step_handle_t &step_handle) const {
  return index_of(step_handle) + 1 <
         static_cast<int64_t>(paths_.index.step_count(path_of(step_handle)));
}

bool GfazHandleGraph::has_previous_step(const step_handle_t &step_handle) const {
  return index_of(step_handle) > 0;
}

handlegraph::step_handle_t
GfazHandleGraph::get_next_step(const step_handle_t &step_handle) const {
  return make_step(path_of(step_handle), index_of(step_handle) + 1);
}

handlegraph::step_handle_t
GfazHandleGraph::get_previous_step(const step_handle_t &step_handle) const {
  return make_step(path_of(step_handle), index_of(step_handle) - 1);
}

bool GfazHandleGraph::for_each_path_handle_impl(
    const std::function<bool(const path_handle_t &)> &iteratee) const {
  for (size_t path = 0; path < path_names_.size(); ++path)
    if (!iteratee(handlegraph::as_path_handle(path)))
      return false;
  return true;
}

bool GfazHandleGraph::for_each_step_on_handle_impl(
    const handle_t &handle,
    const std::function<bool(const step_handle_t &)> &iteratee) const {
  return for_each_step_position_on_handle(
      handle, [&](const step_handle_t &step, const bool &, const size_t &) {
        return iteratee(step);
      });
}

// ---------------------------------------------------------------------------
// PathPositionHandleGraph
// ---------------------------------------------------------------------------

size_t GfazHandleGraph::get_path_length(const path_handle_t &path_handle) const {
  return paths_.index.bp_length(as_integer(path_handle));
}

size_t GfazHandleGraph::get_position_of_step(const step_handle_t &step) const {
  const size_t path = path_of(step);
  const int64_t index = index_of(step);
  if (index >= static_cast<int64_t>(paths_.index.step_count(path)))
    return paths_.index.bp_length(path);
  const DecodedBlock &block =
      block_with_step(path, static_cast<uint32_t>(index));
  return block.positions[index - block.step_begin];
}

handlegraph::step_handle_t
GfazHandleGraph::get_step_at_position(const path_handle_t &path_handle,
                                      const size_t &position) const {
  const size_t path = as_integer(path_handle);
  if (position >= paths_.index.bp_length(path))
    return path_end(path_handle);
  const DecodedBlock &block =
      decoded_block(path, paths_.index.block_of_bp(path, position));
  // The last step that starts at or before the position.
  const auto it = std::upper_bound(block.positions.begin(),
                                   block.positions.end(), position);
  const size_t offset = static_cast<size_t>(it - block.positions.begin()) - 1;
  return make_step(path, static_cast<int64_t>(block.step_begin + offset));
}

bool GfazHandleGraph::for_each_step_position_on_handle(
    const handle_t &handle,
    const std::function<bool(const step_handle_t &, const bool &,
                             const size_t &)> &iteratee) const {
  const uint32_t node = segment_of_handle(handle);
  const bool handle_reverse = get_is_reverse(handle);

  // Only blocks whose node-id range covers the node can visit it. They go
  // through the block cache, so asking about a run of neighbouring nodes
  // decodes each block once.
  std::vector<std::pair<uint32_t, uint32_t>> blocks;
  paths_.index.for_each_block_overlapping(
      node, node, [&](size_t path, size_t block) {
        blocks.emplace_back(static_cast<uint32_t>(path),
                            static_cast<uint32_t>(block));
      });
  std::sort(blocks.begin(), blocks.end());

  // Collect the visits before calling back: the callback may use the graph and
  // recycle the cached blocks being read here.
  struct Visit {
    step_handle_t step;
    size_t position;
    bool reverse;
  };
  std::vector<Visit> visits;
  for (const auto &[path, block] : blocks) {
    const DecodedBlock &decoded = decoded_block(path, block);
    for (size_t i = 0; i < decoded.nodes.size(); ++i) {
      const NodeId visited = decoded.nodes[i];
      if (tquery::abs_node_id(visited) == node)
        visits.push_back({make_step(path, decoded.step_begin + i),
                          decoded.positions[i],
                          (visited < 0) != handle_reverse});
    }
  }
  for (const Visit &visit : visits)
    if (!iteratee(visit.step, visit.reverse, visit.position))
      return false;
  return true;
}

} // namespace gfaz
