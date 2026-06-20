#include "compute/similarity_workflow.hpp"

#include "core/codec/codec.hpp"
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
  const std::vector<uint32_t> segment_lengths =
      Codec::zstd_decompress_uint32_vector(data.segment_seq_lengths_zstd);
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

  // --- Pass 1: per-slice node visit counts (parallel over slices). ---
  const size_t num_slices = slices.size();
  std::vector<std::vector<std::pair<uint32_t, uint32_t>>> slice_visits(
      num_slices); // (node, visit count), node-sorted
  {
    ScopedOMPThreads omp_scope(options.num_threads);
    const int T = std::max(1, omp_scope.effective_threads());
#pragma omp parallel num_threads(T)
    {
      std::vector<NodeId> scratch;
      std::unordered_map<uint32_t, uint32_t> counts;
#pragma omp for schedule(dynamic, 16)
      for (long long sll = 0; sll < static_cast<long long>(num_slices); ++sll) {
        const uint32_t s = static_cast<uint32_t>(sll);
        counts.clear();
        auto visit = [&](NodeId signed_node) {
          const uint32_t node = abs_node_id(signed_node);
          if (node == 0 || node > num_nodes)
            return;
          ++counts[node];
        };
        stream_decoded_nodes(slices[s], delta_round, min_rule_id, max_rule_id,
                             rules_first, rules_second, rule_cache, scratch,
                             visit);
        std::vector<std::pair<uint32_t, uint32_t>> v(counts.begin(),
                                                     counts.end());
        std::sort(v.begin(), v.end());
        slice_visits[s] = std::move(v);
      }
    }
  }

  // --- Pass 2: per-group coverage (parallel over groups). ---
  // cov_g(node) = node_length * (total visits of the group on that node).
  std::vector<std::vector<std::pair<uint32_t, uint64_t>>> group_cov(num_groups);
  {
    ScopedOMPThreads omp_scope(options.num_threads);
    const int T = std::max(1, omp_scope.effective_threads());
#pragma omp parallel num_threads(T)
    {
      std::unordered_map<uint32_t, uint64_t> visits; // node -> total visits
#pragma omp for schedule(dynamic, 1)
      for (long long gll = 0; gll < static_cast<long long>(num_groups); ++gll) {
        const uint32_t gid = static_cast<uint32_t>(gll);
        visits.clear();
        for (uint32_t s : meta.groups[gid])
          for (const auto &nc : slice_visits[s])
            visits[nc.first] += nc.second;
        std::vector<std::pair<uint32_t, uint64_t>> cov;
        cov.reserve(visits.size());
        for (const auto &nv : visits)
          cov.emplace_back(nv.first, nv.second *
                                         static_cast<uint64_t>(
                                             segment_lengths[nv.first - 1]));
        std::sort(cov.begin(), cov.end());
        group_cov[gid] = std::move(cov);
      }
    }
  }
  slice_visits = std::vector<std::vector<std::pair<uint32_t, uint32_t>>>();

  // --- Build node-major CSR of (group, cov), groups ascending per node. ---
  std::vector<uint64_t> node_off(static_cast<size_t>(num_nodes) + 2, 0);
  for (const auto &cov : group_cov)
    for (const auto &nc : cov)
      ++node_off[nc.first + 1];
  for (size_t i = 1; i < node_off.size(); ++i)
    node_off[i] += node_off[i - 1];
  const uint64_t total_entries = node_off[node_off.size() - 1];
  std::vector<uint32_t> csr_gid(total_entries);
  std::vector<uint64_t> csr_cov(total_entries);
  {
    std::vector<uint64_t> cursor(node_off.begin(), node_off.end() - 1);
    for (uint32_t gid = 0; gid < static_cast<uint32_t>(num_groups); ++gid)
      for (const auto &nc : group_cov[gid]) {
        const uint64_t k = cursor[nc.first]++;
        csr_gid[k] = gid;
        csr_cov[k] = nc.second;
      }
  }
  group_cov = std::vector<std::vector<std::pair<uint32_t, uint64_t>>>();

  // --- Pairwise intersection: I[a][b] += min(cov_a, cov_b), per node. ---
  std::vector<uint64_t> inter(tri_size, 0);
  {
    ScopedOMPThreads omp_scope(options.num_threads);
    const int T = std::max(1, omp_scope.effective_threads());
#pragma omp parallel for schedule(dynamic, 256) num_threads(T)
    for (long long node = 1; node <= static_cast<long long>(num_nodes);
         ++node) {
      const uint64_t lo = node_off[node];
      const uint64_t hi = node_off[node + 1];
      for (uint64_t i = lo; i < hi; ++i) {
        const uint32_t ga = csr_gid[i];
        const uint64_t ca = csr_cov[i];
        for (uint64_t j = i; j < hi; ++j) { // csr ascending => ga <= gb
          const uint64_t add = std::min(ca, csr_cov[j]);
          uint64_t &slot = inter[tri(ga, csr_gid[j])];
#pragma omp atomic
          slot += add;
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
