#ifndef STATS_WORKFLOW_HPP
#define STATS_WORKFLOW_HPP

#include "core/model/compressed_data.hpp"

#include <ostream>

namespace gfaz {

struct StatsOptions {
  // Emit base content (A/C/G/T counts over the segment sequences) instead of the
  // graph-dimension summary. Matches `odgi stats -b`.
  bool base_content = false;
};

// Write a graph summary directly from the compressed container to `out`, in the
// tab-delimited layout of `odgi stats -S`:
//   #length  nodes  edges  paths  steps
// (length = total segment bp, nodes = #segments, edges = #L-lines, paths =
// #P-lines + #W-lines, steps = total node visits across all paths/walks).
// With options.base_content, emits the `odgi stats -b` A/C/G/T tally instead.
// Node identities are GFAz's own 1-based ids; counts are independent of them.
// See docs/workflows/STATS_WORKFLOW.md.
void graph_stats_to_tsv(const CompressedData &data, const StatsOptions &options,
                        std::ostream &out);

} // namespace gfaz

#endif // STATS_WORKFLOW_HPP
