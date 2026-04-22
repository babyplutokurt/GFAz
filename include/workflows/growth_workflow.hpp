#ifndef GROWTH_WORKFLOW_HPP
#define GROWTH_WORKFLOW_HPP

#include "cli/common.hpp"
#include "model/compressed_data.hpp"

#include <cstdint>
#include <vector>

namespace gfaz {

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
                            int num_threads = cli::kDefaultNumThreads);

} // namespace gfaz

#endif
