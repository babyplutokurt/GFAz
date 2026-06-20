#include "compute/pav_workflow.hpp"

#include "core/utils/threading_utils.hpp"
#include "compute/traversal_query.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
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

  // Validate every BED chrom resolves to a slice; collect (chrom, slice_id)
  // and a per-slice flag identifying reference targets.
  const size_t num_slices = slices.size();
  std::vector<std::vector<uint32_t> *> slice_to_ref_stream(num_slices, nullptr);
  std::vector<std::vector<uint32_t>> ref_streams; // one per distinct chrom
  ref_streams.reserve(ranges_by_chrom.size());
  std::vector<std::pair<std::string, uint32_t>> chrom_list;
  chrom_list.reserve(ranges_by_chrom.size());

  for (const auto &entry : ranges_by_chrom) {
    auto pit = path_name_to_slice.find(entry.first);
    if (pit == path_name_to_slice.end())
      throw std::runtime_error("pav: BED reference path not found: " +
                               entry.first);
    chrom_list.emplace_back(entry.first, pit->second);
  }
  ref_streams.resize(chrom_list.size());
  for (size_t i = 0; i < chrom_list.size(); ++i) {
    slice_to_ref_stream[chrom_list[i].second] = &ref_streams[i];
  }

  // -------------------------------------------------------------------------
  // Pass 1: parallel over slices, lock-free.
  //   - For every slice produce a sorted-unique list of visited node ids.
  //   - For reference slices additionally emit the ordered node-id stream so
  //     the BED sweep in pass 3 needs no further rule expansion.
  // -------------------------------------------------------------------------
  std::vector<std::vector<uint32_t>> slice_nodes(num_slices);

  {
    ScopedOMPThreads omp_scope(options.num_threads);
    const int T = std::max(1, omp_scope.effective_threads());

#pragma omp parallel num_threads(T)
    {
      std::vector<NodeId> local_decoded;
      std::vector<uint32_t> local_nodes;

#pragma omp for schedule(dynamic, 16)
      for (long long sll = 0; sll < static_cast<long long>(num_slices); ++sll) {
        const uint32_t s = static_cast<uint32_t>(sll);
        std::vector<uint32_t> *ref_stream = slice_to_ref_stream[s];
        local_nodes.clear();

        if (ref_stream) {
          auto visit = [&](NodeId signed_node) {
            const uint32_t node = tquery::abs_node_id(signed_node);
            if (node == 0 || node > num_nodes)
              return;
            local_nodes.push_back(node);
            ref_stream->push_back(node);
          };
          tquery::stream_decoded_nodes(slices[s], delta_round, min_rule_id,
                                       max_rule_id, rules_first, rules_second,
                                       rule_cache, local_decoded, visit);
        } else {
          auto visit = [&](NodeId signed_node) {
            const uint32_t node = tquery::abs_node_id(signed_node);
            if (node == 0 || node > num_nodes)
              return;
            local_nodes.push_back(node);
          };
          tquery::stream_decoded_nodes(slices[s], delta_round, min_rule_id,
                                       max_rule_id, rules_first, rules_second,
                                       rule_cache, local_decoded, visit);
        }

        std::sort(local_nodes.begin(), local_nodes.end());
        local_nodes.erase(
            std::unique(local_nodes.begin(), local_nodes.end()),
            local_nodes.end());
        slice_nodes[s] = std::move(local_nodes);
        local_nodes = std::vector<uint32_t>();
      }
    }
  }

  // We no longer need the rule cache after slice decoding.
  rule_cache.forward = std::vector<std::vector<int32_t>>();
  rule_cache.ready = std::vector<uint8_t>();
  rule_cache.bytes_used = 0;

  // -------------------------------------------------------------------------
  // Pass 2: build per-group sorted-unique node lists, then a
  // CSR-shaped node->groups index. CSR avoids the per-node vector overhead
  // (24 bytes empty * num_nodes) of the previous representation.
  // -------------------------------------------------------------------------
  std::vector<std::vector<uint32_t>> group_nodes(num_groups);
  {
    ScopedOMPThreads omp_scope(options.num_threads);
    const int T = std::max(1, omp_scope.effective_threads());

#pragma omp parallel for schedule(dynamic, 1) num_threads(T)
    for (long long gll = 0; gll < static_cast<long long>(num_groups); ++gll) {
      const uint32_t gid = static_cast<uint32_t>(gll);
      auto &gn = group_nodes[gid];
      size_t total = 0;
      for (uint32_t s : meta.groups[gid])
        total += slice_nodes[s].size();
      gn.reserve(total);
      for (uint32_t s : meta.groups[gid])
        gn.insert(gn.end(), slice_nodes[s].begin(), slice_nodes[s].end());
      std::sort(gn.begin(), gn.end());
      gn.erase(std::unique(gn.begin(), gn.end()), gn.end());
    }
  }

  // Free per-slice node lists; they are no longer needed.
  slice_nodes = std::vector<std::vector<uint32_t>>();

  // Build CSR: node_offsets[node..node+1] indexes into node_to_group_ids.
  std::vector<uint64_t> node_offsets(static_cast<size_t>(num_nodes) + 2, 0);
  for (const auto &gn : group_nodes) {
    for (uint32_t node : gn) {
      if (node != 0 && node <= num_nodes)
        ++node_offsets[node + 1];
    }
  }
  for (size_t i = 1; i < node_offsets.size(); ++i)
    node_offsets[i] += node_offsets[i - 1];

  const uint64_t total_entries = node_offsets[node_offsets.size() - 1];
  std::vector<uint32_t> node_to_group_ids(total_entries);
  {
    std::vector<uint64_t> cursor(node_offsets.begin(),
                                 node_offsets.end() - 1);
    for (uint32_t gid = 0; gid < static_cast<uint32_t>(num_groups); ++gid) {
      for (uint32_t node : group_nodes[gid]) {
        if (node != 0 && node <= num_nodes)
          node_to_group_ids[cursor[node]++] = gid;
      }
    }
  }
  group_nodes = std::vector<std::vector<uint32_t>>();

  // -------------------------------------------------------------------------
  // Pass 3: parallel over distinct BED chroms, sweep using
  // the cached reference node streams. No rule expansion happens here.
  // -------------------------------------------------------------------------
  {
    ScopedOMPThreads omp_scope(options.num_threads);
    const int T = std::max(1, omp_scope.effective_threads());

#pragma omp parallel num_threads(T)
    {
      std::vector<std::pair<uint32_t, uint32_t>> local_window_nodes;

#pragma omp for schedule(dynamic, 1)
      for (long long cll = 0;
           cll < static_cast<long long>(chrom_list.size()); ++cll) {
        const std::string &chrom = chrom_list[cll].first;
        const std::vector<uint32_t> &ref_stream = ref_streams[cll];

        auto rit = ranges_by_chrom.find(chrom);
        if (rit == ranges_by_chrom.end())
          continue;

        std::vector<uint32_t> range_ids = rit->second;
        std::sort(range_ids.begin(), range_ids.end(),
                  [&](uint32_t a, uint32_t b) {
                    return std::tie(result.ranges[a].start,
                                    result.ranges[a].end, a) <
                           std::tie(result.ranges[b].start,
                                    result.ranges[b].end, b);
                  });

        local_window_nodes.clear();
        size_t next_range = 0;
        uint64_t offset = 0;
        for (uint32_t node : ref_stream) {
          if (node == 0 || node > num_nodes)
            continue;
          const uint64_t len = segment_lengths[node - 1];
          const uint64_t node_start = offset;
          const uint64_t node_end = offset + len;
          while (next_range < range_ids.size() &&
                 result.ranges[range_ids[next_range]].end <= node_start) {
            ++next_range;
          }
          for (size_t j = next_range; j < range_ids.size(); ++j) {
            const PavRange &r = result.ranges[range_ids[j]];
            if (r.start >= node_end)
              break;
            if (r.end > node_start && r.start < node_end)
              local_window_nodes.emplace_back(range_ids[j], node);
          }
          offset = node_end;
        }

        std::sort(local_window_nodes.begin(), local_window_nodes.end());
        local_window_nodes.erase(
            std::unique(local_window_nodes.begin(),
                        local_window_nodes.end()),
            local_window_nodes.end());

        for (const auto &wn : local_window_nodes) {
          const uint32_t wid = wn.first;
          const uint32_t node = wn.second;
          const uint64_t len = segment_lengths[node - 1];
#pragma omp atomic
          result.denominators[wid] += len;
          const uint64_t start = node_offsets[node];
          const uint64_t end = node_offsets[node + 1];
          for (uint64_t k = start; k < end; ++k) {
            const uint32_t gid = node_to_group_ids[k];
            const size_t idx =
                static_cast<size_t>(wid) * num_groups + gid;
#pragma omp atomic
            result.numerators[idx] += len;
          }
        }
      }
    }
  }

  return result;
}

} // namespace gfaz
