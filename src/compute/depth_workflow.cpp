#include "compute/depth_workflow.hpp"

#include "core/codec/codec.hpp"
#include "core/utils/threading_utils.hpp"
#include "compute/traversal_query.hpp"

#include <cstdint>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace gfaz {

using namespace gfaz::tquery;

void depth_to_tsv(const CompressedData &data, const DepthOptions &options,
                  std::ostream &out) {
  const std::vector<uint32_t> segment_lengths =
      Codec::zstd_decompress_uint32_vector(data.segment_seq_lengths_zstd);
  const uint32_t num_nodes = static_cast<uint32_t>(segment_lengths.size());
  uint64_t graph_length = 0;
  for (uint32_t len : segment_lengths)
    graph_length += len;

  Rulebook rulebook = load_rulebook(data);
  const std::vector<int32_t> &rules_first = rulebook.rules_first;
  const std::vector<int32_t> &rules_second = rulebook.rules_second;
  const uint32_t min_rule_id = rulebook.min_rule_id;
  const uint32_t max_rule_id = rulebook.max_rule_id;
  const int delta_round = data.delta_round;

  LoadedTraversals loaded = load_traversals(data);
  const std::vector<HapSlice> &slices = loaded.slices;
  const size_t num_slices = slices.size();

  const RuleLeafCache rule_cache =
      make_rule_cache(min_rule_id, rules_first, rules_second,
                      "GFAZ_DEPTH_RULE_CACHE_BYTES", /*default_budget=*/0);

  // Shared per-node accumulators: total visits (multiplicity-counted) and, for
  // the per-node table, the number of distinct paths/walks visiting each node.
  // Atomic increments on a wide array keep peak memory at O(num_nodes), not the
  // O(num_nodes * threads) of per-thread copies (the streaming-RAM-win thesis).
  const size_t arr_len = static_cast<size_t>(num_nodes) + 1;
  std::vector<uint64_t> node_total(arr_len, 0);
  std::vector<uint64_t> node_uniq;
  if (options.per_node)
    node_uniq.assign(arr_len, 0);

  {
    ScopedOMPThreads omp_scope(options.num_threads);
    const int T = std::max(1, omp_scope.effective_threads());
#pragma omp parallel num_threads(T)
    {
      std::vector<NodeId> scratch;
      // Per-thread "last path seen on this node" stamp filters repeat visits
      // within one slice so depth.uniq counts distinct paths, not steps.
      std::vector<uint32_t> last_seen;
      if (options.per_node)
        last_seen.assign(arr_len, 0);
      uint32_t stamp = 0;
#pragma omp for schedule(dynamic, 16)
      for (long long sll = 0; sll < static_cast<long long>(num_slices); ++sll) {
        ++stamp;
        const uint32_t s = static_cast<uint32_t>(sll);
        auto visit = [&](NodeId signed_node) {
          const uint32_t node = abs_node_id(signed_node);
          if (node == 0 || node > num_nodes)
            return;
#pragma omp atomic
          node_total[node] += 1;
          if (options.per_node && last_seen[node] != stamp) {
            last_seen[node] = stamp;
#pragma omp atomic
            node_uniq[node] += 1;
          }
        };
        stream_decoded_nodes(slices[s], delta_round, min_rule_id, max_rule_id,
                             rules_first, rules_second, rule_cache, scratch,
                             visit);
      }
    }
  }
  loaded = LoadedTraversals{};

  if (options.per_node) {
    out << "#node.id\tdepth\tdepth.uniq\n";
    for (uint32_t v = 1; v <= num_nodes; ++v)
      out << v << '\t' << node_total[v] << '\t' << node_uniq[v] << '\n';
    return;
  }

  // Summary: step.count = total visits; path.length = total traversed bp.
  uint64_t step_count = 0;
  uint64_t path_length = 0;
  for (uint32_t v = 1; v <= num_nodes; ++v) {
    step_count += node_total[v];
    path_length += node_total[v] * static_cast<uint64_t>(segment_lengths[v - 1]);
  }
  const double mean_node_depth =
      num_nodes ? static_cast<double>(step_count) / num_nodes : 0.0;
  const double mean_graph_depth =
      graph_length ? static_cast<double>(path_length) / graph_length : 0.0;

  out << "#node.count\tgraph.length\tstep.count\tpath.length\tmean.node.depth\t"
         "mean.graph.depth\n";
  out << num_nodes << '\t' << graph_length << '\t' << step_count << '\t'
      << path_length << '\t' << mean_node_depth << '\t' << mean_graph_depth
      << '\n';
}

} // namespace gfaz
