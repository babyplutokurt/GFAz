#include "compute/similarity_workflow.hpp"

#include "core/utils/threading_utils.hpp"
#include "compute/traversal_query.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
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

// Grouping setup: map every P/W slice to a dense group id and name. Mirrors
// pav_workflow.cpp:build_metadata (a dedup candidate flagged in
// docs/EXTENDING_COMPUTE_ENGINE.md §7).
struct GroupMeta {
  std::vector<std::string> group_names;        // gid -> name
  std::vector<uint32_t> group_of_slice;        // slice -> gid
  std::vector<std::vector<uint32_t>> groups;   // gid -> member slice ids
};

GroupMeta build_groups(const CompressedData &data, GroupingMode grouping) {
  GroupMeta meta;
  const size_t num_paths = data.sequence_lengths.size();
  const size_t num_walks = data.walk_lengths.size();
  const size_t total = num_paths + num_walks;

  std::vector<std::string> keys;
  keys.reserve(total);
  if (num_paths) {
    std::vector<std::string> path_names =
        load_path_names(data, num_paths, "similarity");
    for (const std::string &name : path_names)
      keys.push_back(path_group_key(name, grouping));
  }
  if (num_walks) {
    const WalkIdentityColumns w =
        load_walk_identity(data, num_walks, "similarity");
    for (size_t i = 0; i < num_walks; ++i)
      keys.push_back(
          walk_group_key(w.samples[i], w.haps[i], w.seqs[i], grouping));
  }

  meta.group_of_slice.assign(total, 0);
  std::unordered_map<std::string, uint32_t> key_to_gid;
  key_to_gid.reserve(total * 2 + 1);
  for (uint32_t i = 0; i < static_cast<uint32_t>(total); ++i) {
    auto it = key_to_gid.find(keys[i]);
    uint32_t gid;
    if (it == key_to_gid.end()) {
      gid = static_cast<uint32_t>(meta.group_names.size());
      key_to_gid.emplace(keys[i], gid);
      meta.group_names.push_back(keys[i]);
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

void similarity_to_tsv(const CompressedData &data,
                       const SimilarityOptions &options, std::ostream &out) {
  const std::vector<uint32_t> segment_lengths = load_segment_lengths(data);
  const uint32_t num_nodes = static_cast<uint32_t>(segment_lengths.size());

  Rulebook rulebook = load_rulebook(data);
  const std::vector<int32_t> &rules_first = rulebook.rules_first;
  const std::vector<int32_t> &rules_second = rulebook.rules_second;
  const uint32_t min_rule_id = rulebook.min_rule_id;
  const uint32_t max_rule_id = rulebook.max_rule_id;
  const int delta_round = data.delta_round;

  LoadedTraversals loaded = load_traversals(data);
  const std::vector<HapSlice> &slices = loaded.slices;

  GroupMeta meta = build_groups(data, options.grouping);
  const size_t num_groups = meta.group_names.size();

  // Header (matches `odgi similarity` columns exactly).
  out << "group.a\tgroup.b\tgroup.a.length\tgroup.b.length\tintersection\t";
  if (options.emit_distances)
    out << "jaccard.distance\tcosine.distance\tdice.distance\t"
           "estimated.difference.rate\teuclidean.distance\tmanhattan.distance\n";
  else
    out << "jaccard.similarity\tcosine.similarity\tdice.similarity\t"
           "estimated.identity\n";
  if (num_groups == 0)
    return;

  // Dense upper-triangle (incl. diagonal) intersection accumulator: I[tri(a,b)]
  // for a<=b. The diagonal I[tri(g,g)] equals the group's total length L_g.
  const uint64_t tri_size =
      static_cast<uint64_t>(num_groups) * (num_groups + 1) / 2;
  if (tri_size > (static_cast<uint64_t>(1) << 31))
    throw std::runtime_error(
        "similarity: too many groups (" + std::to_string(num_groups) +
        ") for a dense matrix; use a coarser grouping (-S/-H)");
  auto tri = [num_groups](uint32_t a, uint32_t b) -> uint64_t { // a <= b
    return static_cast<uint64_t>(a) * num_groups -
           static_cast<uint64_t>(a) * (a - 1) / 2 + (b - a);
  };

  const RuleLeafCache rule_cache =
      make_rule_cache(min_rule_id, rules_first, rules_second,
                      "GFAZ_SIMILARITY_RULE_CACHE_BYTES", /*default_budget=*/0);

  // Per node, the groups that visit it and how often: a node-major CSR. It is
  // by far the largest structure here (one entry per group and node the group
  // visits), so it is filled straight from the traversals, which are decoded
  // twice: once to size each node's list, once to fill it. Decoding is cheap
  // next to holding per-group lists as well.
  //
  // A thread counts one group's visits in an array over all nodes and keeps
  // the nodes it touched, so that clearing costs what the group used.
  struct VisitCounter {
    explicit VisitCounter(uint32_t num_nodes) : visits(num_nodes + 1, 0) {}
    std::vector<uint32_t> visits;
    std::vector<uint32_t> touched;
    std::vector<NodeId> scratch;
  };
  // Calls use(gid, counter) once per group with that group's counts and the
  // nodes it visits in increasing order, in parallel over groups.
  auto for_each_group_counted = [&](auto &&use) {
    ScopedOMPThreads omp_scope(options.num_threads);
    const int T = std::max(1, omp_scope.effective_threads());
#pragma omp parallel num_threads(T)
    {
      VisitCounter counter(num_nodes);
#pragma omp for schedule(dynamic, 1)
      for (long long gll = 0; gll < static_cast<long long>(num_groups); ++gll) {
        const uint32_t gid = static_cast<uint32_t>(gll);
        auto visit = [&](NodeId signed_node) {
          const uint32_t node = abs_node_id(signed_node);
          if (node == 0 || node > num_nodes)
            return;
          if (counter.visits[node]++ == 0)
            counter.touched.push_back(node);
        };
        for (uint32_t s : meta.groups[gid])
          stream_decoded_nodes(slices[s], delta_round, min_rule_id, max_rule_id,
                               rules_first, rules_second, rule_cache,
                               counter.scratch, visit);
        // Hand the nodes over in increasing order, so that the caller walks
        // its per-node arrays forwards. A group that visits a good part of
        // the graph is cheaper to read off the array than to sort.
        std::vector<uint32_t> &touched = counter.touched;
        if (touched.size() > num_nodes / 32) {
          touched.clear();
          for (uint32_t node = 1; node <= num_nodes; ++node)
            if (counter.visits[node] != 0)
              touched.push_back(node);
        } else {
          std::sort(touched.begin(), touched.end());
        }
        use(gid, counter);
        for (uint32_t node : touched)
          counter.visits[node] = 0;
        touched.clear();
      }
    }
  };

  // First decode: how many groups visit each node.
  std::vector<uint32_t> groups_on_node(static_cast<size_t>(num_nodes) + 1, 0);
  for_each_group_counted([&](uint32_t, const VisitCounter &counter) {
    for (uint32_t node : counter.touched) {
#pragma omp atomic
      ++groups_on_node[node];
    }
  });
  std::vector<uint64_t> node_off(static_cast<size_t>(num_nodes) + 2, 0);
  for (uint32_t node = 1; node <= num_nodes; ++node)
    node_off[node + 1] = node_off[node] + groups_on_node[node];
  const uint64_t total_entries = node_off[static_cast<size_t>(num_nodes) + 1];

  // Second decode: fill. Each group takes the next free slot of every node it
  // visits, counting groups_on_node back down, so a node's groups come in no
  // particular order.
  std::vector<uint32_t> csr_gid(total_entries);
  std::vector<uint32_t> csr_visits(total_entries);
  for_each_group_counted([&](uint32_t gid, const VisitCounter &counter) {
    for (uint32_t node : counter.touched) {
      uint32_t slot;
#pragma omp atomic capture
      slot = --groups_on_node[node];
      const uint64_t k = node_off[node] + slot;
      csr_gid[k] = gid;
      csr_visits[k] = counter.visits[node];
    }
  });
  std::vector<uint32_t>().swap(groups_on_node);

  // --- Pairwise intersection, per node: with cov_g = length * visits_g,
  // I[a][b] += min(cov_a, cov_b) = length * min(visits_a, visits_b). A node's
  // groups are put in increasing order first, so that for one group the
  // updates run along a row of the matrix. ---
  std::vector<uint64_t> inter(tri_size, 0);
  {
    ScopedOMPThreads omp_scope(options.num_threads);
    const int T = std::max(1, omp_scope.effective_threads());
#pragma omp parallel num_threads(T)
    {
      std::vector<std::pair<uint32_t, uint32_t>> on_node; // (group, visits)
#pragma omp for schedule(dynamic, 256)
      for (long long node = 1; node <= static_cast<long long>(num_nodes);
           ++node) {
        const uint64_t lo = node_off[node];
        const uint64_t hi = node_off[node + 1];
        on_node.clear();
        for (uint64_t i = lo; i < hi; ++i)
          on_node.emplace_back(csr_gid[i], csr_visits[i]);
        std::sort(on_node.begin(), on_node.end());
        const uint64_t length = segment_lengths[node - 1];
        for (size_t i = 0; i < on_node.size(); ++i) {
          const uint32_t va = on_node[i].second;
          uint64_t *row = inter.data() + tri(on_node[i].first, on_node[i].first) -
                          on_node[i].first;
          for (size_t j = i; j < on_node.size(); ++j) {
            const uint64_t add = length * std::min(va, on_node[j].second);
            uint64_t &slot = row[on_node[j].first];
#pragma omp atomic
            slot += add;
          }
        }
      }
    }
  }

  // --- Emit (deterministic: a then b ascending; odgi's columns/formulas). ---
  out << std::fixed << std::setprecision(6);
  for (uint32_t a = 0; a < static_cast<uint32_t>(num_groups); ++a) {
    const uint64_t La = inter[tri(a, a)];
    for (uint32_t b = 0; b < static_cast<uint32_t>(num_groups); ++b) {
      const uint64_t Lb = inter[tri(b, b)];
      const uint64_t I =
          (a <= b) ? inter[tri(a, b)] : inter[tri(b, a)];
      if (!options.all_pairs && I == 0)
        continue;
      const double dI = static_cast<double>(I);
      const double jaccard = dI / static_cast<double>(La + Lb - I);
      // double-precision multiply avoids odgi's uint64*uint64 overflow for
      // genome-scale lengths; identical to odgi for non-overflowing inputs.
      const double cosine =
          dI / std::sqrt(static_cast<double>(La) * static_cast<double>(Lb));
      const double dice = 2.0 * dI / static_cast<double>(La + Lb);
      const double est_identity = 2.0 * jaccard / (1.0 + jaccard);
      out << meta.group_names[a] << '\t' << meta.group_names[b] << '\t' << La
          << '\t' << Lb << '\t' << I << '\t';
      if (options.emit_distances) {
        const uint64_t manhattan = La + Lb - 2 * I;
        const double euclidean = std::sqrt(static_cast<double>(manhattan));
        out << (1.0 - jaccard) << '\t' << (1.0 - cosine) << '\t'
            << (1.0 - dice) << '\t' << (1.0 - est_identity) << '\t' << euclidean
            << '\t' << manhattan << '\n';
      } else {
        out << jaccard << '\t' << cosine << '\t' << dice << '\t' << est_identity
            << '\n';
      }
    }
  }
}

} // namespace gfaz
