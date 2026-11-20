#include "compute/pav_workflow.hpp"

#include "core/utils/threading_utils.hpp"
#include "compute/packed_uints.hpp"
#include "compute/traversal_query.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace gfaz {
namespace {

using namespace gfaz::tquery;

std::vector<PavRange> read_bed(const std::string &path) {
  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("pav: failed to open BED file: " + path);

  std::vector<PavRange> ranges;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#')
      continue;
    std::stringstream ss(line);
    PavRange r;
    ss >> r.chrom >> r.start >> r.end;
    if (!ss || r.chrom.empty())
      throw std::runtime_error("pav: malformed BED line: " + line);
    ss >> r.name;
    if (r.name.empty())
      r.name = r.chrom + ":" + std::to_string(r.start) + "-" +
               std::to_string(r.end);
    if (r.end < r.start)
      throw std::runtime_error("pav: BED end before start: " + line);
    ranges.push_back(std::move(r));
  }
  if (ranges.empty())
    throw std::runtime_error("pav: BED file contains no ranges");
  return ranges;
}

// The distinct nodes of one group. Small sets are listed as they are found;
// a set that outgrows the list is read back off the bitmap instead.
class NodeSet {
public:
  explicit NodeSet(uint32_t num_nodes)
      : bits_((static_cast<size_t>(num_nodes) >> 6) + 1, 0),
        list_limit_(bits_.size()) {}

  void insert(uint32_t node) {
    uint64_t &word = bits_[node >> 6];
    const uint64_t bit = uint64_t{1} << (node & 63);
    if (word & bit)
      return;
    word |= bit;
    if (listed_.size() < list_limit_)
      listed_.push_back(node);
    else
      overflowed_ = true;
  }

  // Calls use(node) for every node in the set, then empties it.
  template <typename Use> void drain(Use &&use) {
    if (overflowed_) {
      for (size_t w = 0; w < bits_.size(); ++w) {
        uint64_t word = bits_[w];
        bits_[w] = 0;
        for (; word; word &= word - 1)
          use(static_cast<uint32_t>((w << 6) + __builtin_ctzll(word)));
      }
    } else {
      for (uint32_t node : listed_) {
        bits_[node >> 6] = 0;
        use(node);
      }
    }
    listed_.clear();
    overflowed_ = false;
  }

private:
  std::vector<uint64_t> bits_;
  std::vector<uint32_t> listed_;
  size_t list_limit_;
  bool overflowed_ = false;
};

struct TraversalMetadata {
  std::vector<std::string> path_names;
  std::vector<std::string> walk_names;
  std::vector<std::string> group_names;
  // group_of_slice[s] = group id assigned to slice s.
  std::vector<uint32_t> group_of_slice;
  std::vector<std::vector<uint32_t>> groups;
};

TraversalMetadata build_metadata(const CompressedData &data,
                                 GroupingMode grouping) {
  TraversalMetadata meta;
  const size_t num_paths = data.sequence_lengths.size();
  const size_t num_walks = data.walk_lengths.size();
  const size_t total = num_paths + num_walks;

  if (num_paths)
    meta.path_names = load_path_names(data, num_paths, "pav");

  std::vector<std::string> keys;
  keys.reserve(total);
  for (const std::string &name : meta.path_names)
    keys.push_back(path_group_key(name, grouping));

  if (num_walks) {
    const WalkIdentityColumns w = load_walk_identity(data, num_walks, "pav");
    meta.walk_names.reserve(num_walks);
    for (size_t i = 0; i < num_walks; ++i) {
      meta.walk_names.push_back(walk_reference_name(
          w.samples[i], w.haps[i], w.seqs[i], w.starts[i], w.ends[i]));
      keys.push_back(walk_group_key(w.samples[i], w.haps[i], w.seqs[i],
                                    grouping));
    }
  }

  meta.group_of_slice.assign(total, 0);
  std::unordered_map<std::string, uint32_t> key_to_gid;
  key_to_gid.reserve(total * 2 + 1);
  for (uint32_t i = 0; i < static_cast<uint32_t>(total); ++i) {
    const std::string &key = keys[i];
    auto it = key_to_gid.find(key);
    uint32_t gid = 0;
    if (it == key_to_gid.end()) {
      gid = static_cast<uint32_t>(meta.group_names.size());
      key_to_gid.emplace(key, gid);
      meta.group_names.push_back(key);
      meta.groups.emplace_back();
    } else {
      gid = it->second;
    }
    meta.groups[gid].push_back(i);
    meta.group_of_slice[i] = gid;
  }

  return meta;
}

} // namespace

PavResult compute_pav(const CompressedData &data, const PavOptions &options) {
  PavResult result;
  result.ranges = read_bed(options.bed_path);

  const std::vector<uint32_t> segment_lengths = load_segment_lengths(data);
  const uint32_t num_nodes = static_cast<uint32_t>(segment_lengths.size());

  tquery::Rulebook rulebook = tquery::load_rulebook(data);
  const std::vector<int32_t> &rules_first = rulebook.rules_first;
  const std::vector<int32_t> &rules_second = rulebook.rules_second;

  tquery::LoadedTraversals loaded = tquery::load_traversals(data);
  const std::vector<tquery::HapSlice> &slices = loaded.slices;

  TraversalMetadata meta = build_metadata(data, options.grouping);
  result.group_names = meta.group_names;
  if (result.group_names.empty())
    return result;

  const uint32_t min_rule_id = rulebook.min_rule_id;
  const uint32_t max_rule_id = rulebook.max_rule_id;
  const int delta_round = data.delta_round;

  // Bottom-up rule-leaf cache. Built single-threaded; read-only during slice
  // decoding so no synchronisation is needed.
  tquery::RuleLeafCache rule_cache = tquery::make_rule_cache(
      min_rule_id, rules_first, rules_second, "GFAZ_PAV_RULE_CACHE_BYTES",
      static_cast<size_t>(1) << 30);

  // Map BED chrom -> reference slice id.
  std::unordered_map<std::string, uint32_t> path_name_to_slice;
  path_name_to_slice.reserve((meta.path_names.size() + meta.walk_names.size()) *
                                 2 +
                             1);
  for (uint32_t i = 0; i < static_cast<uint32_t>(meta.path_names.size()); ++i)
    path_name_to_slice.emplace(meta.path_names[i], i);
  const uint32_t walk_slice_offset =
      static_cast<uint32_t>(meta.path_names.size());
  for (uint32_t i = 0; i < static_cast<uint32_t>(meta.walk_names.size()); ++i)
    path_name_to_slice.emplace(meta.walk_names[i], walk_slice_offset + i);

  std::unordered_map<std::string, std::vector<uint32_t>> ranges_by_chrom;
  for (uint32_t i = 0; i < static_cast<uint32_t>(result.ranges.size()); ++i) {
    ranges_by_chrom[result.ranges[i].chrom].push_back(i);
  }

  const size_t num_windows = result.ranges.size();
  const size_t num_groups = result.group_names.size();
  result.denominators.assign(num_windows, 0);
  result.numerators.assign(num_windows * num_groups, 0);

  // One entry per distinct BED chrom: its reference traversal and its ranges
  // in sweep order.
  struct Reference {
    uint32_t slice;
    std::vector<uint32_t> range_ids;
  };
  std::vector<Reference> references;
  references.reserve(ranges_by_chrom.size());
  for (auto &entry : ranges_by_chrom) {
    auto pit = path_name_to_slice.find(entry.first);
    if (pit == path_name_to_slice.end())
      throw std::runtime_error("pav: BED reference path not found: " +
                               entry.first);
    std::vector<uint32_t> &range_ids = entry.second;
    std::sort(range_ids.begin(), range_ids.end(), [&](uint32_t a, uint32_t b) {
      return std::tie(result.ranges[a].start, result.ranges[a].end, a) <
             std::tie(result.ranges[b].start, result.ranges[b].end, b);
    });
    references.push_back({pit->second, std::move(range_ids)});
  }

  auto decode_slice = [&](uint32_t s, std::vector<NodeId> &scratch,
                          auto &&visit) {
    auto on_node = [&](NodeId signed_node) {
      const uint32_t node = tquery::abs_node_id(signed_node);
      if (node != 0 && node <= num_nodes)
        visit(node);
    };
    tquery::stream_decoded_nodes(slices[s], delta_round, min_rule_id,
                                 max_rule_id, rules_first, rules_second,
                                 rule_cache, scratch, on_node);
  };

  // Walks a reference and calls emit(range id, node) for every node step that
  // overlaps one of its ranges. A node is reported once per step and range.
  auto sweep_reference = [&](const Reference &ref, std::vector<NodeId> &scratch,
                             auto &&emit) {
    const std::vector<uint32_t> &range_ids = ref.range_ids;
    size_t next_range = 0;
    uint64_t offset = 0;
    decode_slice(ref.slice, scratch, [&](uint32_t node) {
      const uint64_t node_start = offset;
      const uint64_t node_end = offset + segment_lengths[node - 1];
      offset = node_end;
      while (next_range < range_ids.size() &&
             result.ranges[range_ids[next_range]].end <= node_start)
        ++next_range;
      for (size_t j = next_range; j < range_ids.size(); ++j) {
        const PavRange &r = result.ranges[range_ids[j]];
        if (r.start >= node_end)
          break;
        if (r.end > node_start)
          emit(range_ids[j], node);
      }
    });
  };

  ScopedOMPThreads omp_scope(options.num_threads);
  const int T = std::max(1, omp_scope.effective_threads());
  const long long num_references = static_cast<long long>(references.size());

  // --- Which nodes lie in a range. Only these need their groups known. ---
  std::vector<uint8_t> in_range(static_cast<size_t>(num_nodes) + 1, 0);
#pragma omp parallel num_threads(T)
  {
    std::vector<NodeId> scratch;
#pragma omp for schedule(dynamic, 1)
    for (long long r = 0; r < num_references; ++r) {
      sweep_reference(references[r], scratch, [&](uint32_t, uint32_t node) {
        uint8_t marked;
#pragma omp atomic read
        marked = in_range[node];
        if (!marked) {
#pragma omp atomic write
          in_range[node] = 1;
        }
      });
    }
  }

  // --- Node -> groups table for the nodes in a range, filled straight from
  // the traversals: one decode counts the groups on each node, a second one
  // writes them. Nothing is kept per traversal or per group in between. ---
  auto for_each_group_node = [&](auto &&use) {
#pragma omp parallel num_threads(T)
    {
      NodeSet seen(num_nodes);
      std::vector<NodeId> scratch;
#pragma omp for schedule(dynamic, 1)
      for (long long gll = 0; gll < static_cast<long long>(num_groups); ++gll) {
        const uint32_t gid = static_cast<uint32_t>(gll);
        for (uint32_t s : meta.groups[gid])
          decode_slice(s, scratch, [&](uint32_t node) {
            if (in_range[node])
              seen.insert(node);
          });
        seen.drain([&](uint32_t node) { use(gid, node); });
      }
    }
  };

  std::vector<uint32_t> groups_on_node(static_cast<size_t>(num_nodes) + 1, 0);
  for_each_group_node([&](uint32_t, uint32_t node) {
#pragma omp atomic
    ++groups_on_node[node];
  });
  std::vector<uint64_t> node_offsets(static_cast<size_t>(num_nodes) + 2, 0);
  for (uint32_t node = 1; node <= num_nodes; ++node)
    node_offsets[node + 1] = node_offsets[node] + groups_on_node[node];
  // Group ids take the fewest bytes that hold them; this table is where the
  // memory goes.
  PackedUints node_to_group_ids(node_offsets[num_nodes + 1],
                                static_cast<uint32_t>(num_groups - 1));
  // Each group takes the next free slot of a node, counting groups_on_node
  // back down, so a node's groups come in no particular order.
  for_each_group_node([&](uint32_t gid, uint32_t node) {
    uint32_t slot;
#pragma omp atomic capture
    slot = --groups_on_node[node];
    node_to_group_ids.set(node_offsets[node] + slot, gid);
  });
  std::vector<uint32_t>().swap(groups_on_node);
  std::vector<uint8_t>().swap(in_range);

  // --- Sweep each reference again and add every distinct (range, node) to
  // the range's row. A range belongs to one reference, so its row has one
  // writer. ---
#pragma omp parallel num_threads(T)
  {
    std::vector<NodeId> scratch;
    std::vector<std::pair<uint32_t, uint32_t>> range_nodes;
#pragma omp for schedule(dynamic, 1)
    for (long long r = 0; r < num_references; ++r) {
      range_nodes.clear();
      sweep_reference(references[r], scratch,
                      [&](uint32_t range_id, uint32_t node) {
                        range_nodes.emplace_back(range_id, node);
                      });
      std::sort(range_nodes.begin(), range_nodes.end());
      range_nodes.erase(std::unique(range_nodes.begin(), range_nodes.end()),
                        range_nodes.end());
      for (const auto &rn : range_nodes) {
        const uint32_t node = rn.second;
        const uint64_t len = segment_lengths[node - 1];
        result.denominators[rn.first] += len;
        uint64_t *row = result.numerators.data() +
                        static_cast<size_t>(rn.first) * num_groups;
        for (uint64_t k = node_offsets[node]; k < node_offsets[node + 1]; ++k)
          row[node_to_group_ids[k]] += len;
      }
    }
  }

  return result;
}

} // namespace gfaz
