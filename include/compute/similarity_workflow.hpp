#ifndef SIMILARITY_WORKFLOW_HPP
#define SIMILARITY_WORKFLOW_HPP

#include "core/defaults.hpp"
#include "core/model/compressed_data.hpp"
#include "compute/grouping_mode.hpp"

#include <ostream>

namespace gfaz {

struct SimilarityOptions {
  // How P/W traversals collapse into the rows/columns of the matrix. Default
  // groups by sample (PanSN sample#); -H by sample#hap, -p per path/walk (the
  // latter matches `odgi similarity`'s no-delimiter default).
  GroupingMode grouping = GroupingMode::Sample;
  int num_threads = kDefaultNumThreads;
  // Emit dissimilarities (1 - sim) plus euclidean/manhattan distances, matching
  // `odgi similarity -d`. Default emits similarities.
  bool emit_distances = false;
  // Emit every ordered pair including zero-intersection ones (`odgi similarity
  // -a`). Default is sparse: only pairs that co-occur on at least one node.
  bool all_pairs = false;
};

// Compute an all-vs-all group similarity matrix directly from the compressed
// traversals and write it (header + one line per ordered pair) to `out`, in the
// tab-delimited column layout of `odgi similarity`. Multiplicity-aware: a
// group's length and the pairwise intersection count each node visit (matching
// odgi's coverage-histogram definition, not a set/union Jaccard). See
// docs/workflows/SIMILARITY_WORKFLOW.md.
void similarity_to_tsv(const CompressedData &data,
                       const SimilarityOptions &options, std::ostream &out);

} // namespace gfaz

#endif // SIMILARITY_WORKFLOW_HPP
