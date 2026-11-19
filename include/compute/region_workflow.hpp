#ifndef COMPUTE_REGION_WORKFLOW_HPP
#define COMPUTE_REGION_WORKFLOW_HPP

// Extract the subgraph around reference regions as GFA, straight from a
// container. See docs/workflows/REGION_WORKFLOW.md.

#include "core/defaults.hpp"
#include "core/model/compressed_data.hpp"
#include "core/model/gfa_graph.hpp"

#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

namespace gfaz {

struct RegionOptions {
  // "CONTIG:START-END" (0-based, end-exclusive) or "CONTIG" for all of it.
  // CONTIG is a P-line name or sample#hap#seq of a W-line.
  std::vector<std::string> regions;
  // Two visits of a haplotype to the region are joined into one subpath,
  // together with everything between them, when they are at most this many
  // bases apart. This is what pulls in alleles that leave the reference.
  uint64_t max_gap_bp = 100000;
  // Steps per block of the path index built for the query.
  uint32_t index_block_steps = 4096;
  int num_threads = kDefaultNumThreads;
};

// A stretch of one haplotype inside the extracted subgraph.
struct Subpath {
  uint32_t haplotype = 0;
  uint32_t step_begin = 0;
  uint32_t step_end = 0;
  uint64_t bp_begin = 0; // haplotype coordinates
  uint64_t bp_end = 0;
  std::vector<NodeId> nodes;
};

struct RegionSubgraph {
  // node_selected[n] != 0 if node n is part of the subgraph (index 0 unused).
  std::vector<uint8_t> node_selected;
  // Ordered by haplotype, then position.
  std::vector<Subpath> subpaths;
};

// Select the nodes and haplotype stretches of the requested regions.
RegionSubgraph select_regions(const CompressedData &data,
                              const RegionOptions &options);

// select_regions, then write the subgraph as GFA: the selected segments, every
// link between two of them, and one P- or W-line per subpath. Segments are written
// under their original names, as `gfaz decompress` writes them.
void extract_regions_to_gfa(const CompressedData &data,
                            const RegionOptions &options, std::ostream &out);

} // namespace gfaz

#endif // COMPUTE_REGION_WORKFLOW_HPP
