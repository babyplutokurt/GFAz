#include "compute/locate_workflow.hpp"

#include "compute/indexed_paths.hpp"
#include "compute/traversal_query.hpp"
#include "core/utils/threading_utils.hpp"

#include <stdexcept>

namespace gfaz {

void locate_nodes_to_tsv(const CompressedData &data,
                         const LocateOptions &options, std::ostream &out) {
  IndexedPathsOptions paths_options;
  paths_options.index_path = options.index_path;
  paths_options.num_threads = options.num_threads;
  const IndexedPaths paths(data, paths_options);

  const SegmentNameLookup lookup(paths.names, paths.num_nodes());
  std::vector<uint32_t> nodes;
  nodes.reserve(options.nodes.size());
  for (const std::string &name : options.nodes) {
    nodes.push_back(lookup.find(name));
    if (nodes.back() == 0)
      throw std::runtime_error("locate: the graph has no segment named '" +
                               name + "'");
  }

  // One buffer per requested node keeps the output in request order while the
  // lookups run in parallel.
  std::vector<std::string> rows(nodes.size());
  {
    ScopedOMPThreads omp_scope(options.num_threads);
#pragma omp parallel for schedule(dynamic, 1)
    for (long long i = 0; i < static_cast<long long>(nodes.size()); ++i) {
      const uint32_t node = nodes[i];
      std::string &text = rows[i];
      paths.access.for_each_step_on_nodes(
          node, node, [&](size_t hap, const PathStep &step) {
            const Haplotype &haplotype = paths.catalog[hap];
            text += options.nodes[i];
            text += '\t';
            text += haplotype.contig;
            text += step.node < 0 ? "\t-\t" : "\t+\t";
            text += std::to_string(step.step);
            text += '\t';
            text += std::to_string(haplotype.contig_offset + step.bp_begin);
            text += '\t';
            text += std::to_string(haplotype.contig_offset + step.bp_end());
            text += '\n';
          });
    }
  }

  out << "#node\tpath\tstrand\tstep\tstart\tend\n";
  for (const std::string &text : rows)
    out << text;
}

size_t write_path_index(const CompressedData &data,
                        const std::string &index_path, uint32_t block_steps,
                        int num_threads) {
  const std::vector<uint32_t> segment_lengths =
      tquery::load_segment_lengths(data);
  const TraversalSet traversals(data);
  const PathIndex index = PathIndex::build(
      traversals, segment_lengths, PathIndexOptions{block_steps, num_threads});
  index.save(index_path, PathIndex::fingerprint(data));
  return index.total_blocks();
}

} // namespace gfaz
