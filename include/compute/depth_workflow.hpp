#ifndef DEPTH_WORKFLOW_HPP
#define DEPTH_WORKFLOW_HPP

#include "core/defaults.hpp"
#include "core/model/compressed_data.hpp"

#include <ostream>

namespace gfaz {

struct DepthOptions {
  // Emit the per-node depth table (`#node.id depth depth.uniq`, matching
  // `odgi depth -d`) instead of the one-line summary. Default is the summary.
  bool per_node = false;
  int num_threads = kDefaultNumThreads;
};

// Compute node coverage depth directly from the compressed traversals and write
// it to `out`. Default (matching `odgi depth -S`) emits one summary line:
//   #node.count  graph.length  step.count  path.length  mean.node.depth
//   mean.graph.depth
// With options.per_node (matching `odgi depth -d`) emits a per-node table:
//   #node.id  depth  depth.uniq
// where depth = total steps on the node (multiplicity-counted) and depth.uniq =
// number of distinct paths visiting it. Node ids are GFAz's own 1-based ids (so
// the table is byte-identical to odgi only when the GFA segment names are 1..N).
// See docs/workflows/DEPTH_WORKFLOW.md.
void depth_to_tsv(const CompressedData &data, const DepthOptions &options,
                  std::ostream &out);

} // namespace gfaz

#endif // DEPTH_WORKFLOW_HPP
