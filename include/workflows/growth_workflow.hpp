#ifndef GROWTH_WORKFLOW_HPP
#define GROWTH_WORKFLOW_HPP

#include "cli/common.hpp"
#include "model/compressed_data.hpp"

#include <cstdint>
#include <vector>

namespace gfaz {

// How to map P/W-lines to "haplotype" identity before coverage counting.
//   PerPathWalk:  each P-line and each W-line is its own haplotype (GFAz
//                 default; simple, but inflates N when one haplotype is split
//                 into multiple walks/paths).
//   SampleHapSeq: group by (sample, hap, seqid) after stripping PanSN
//                 ":start-end". Matches Panacus default (id() + clear_coords).
//   SampleHap:    group by (sample, hap). Matches Panacus --groupby-haplotype
//                 (-H). Typical HPRC "per-haplotype" growth curve.
//   Sample:       group by sample. Matches Panacus --groupby-sample (-S).
enum class GroupingMode {
  PerPathWalk,
  SampleHapSeq,
  SampleHap,
  Sample,
};

// Pangenome growth result, Panacus-style.
// hist[c] = number of nodes covered by exactly c haplotypes (c in [0, N]).
// growth[k] = expected number of nodes covered by a random size-k subset of
//             the N haplotypes (k in [1, N]); growth[0] is unused.
struct GrowthResult {
  uint32_t num_haplotypes = 0;
  uint32_t num_nodes = 0;
  std::vector<uint64_t> hist;
  std::vector<double> growth;
};

GrowthResult compute_growth(const CompressedData &data,
                            int num_threads = cli::kDefaultNumThreads,
                            GroupingMode mode = GroupingMode::PerPathWalk);

} // namespace gfaz

#endif
