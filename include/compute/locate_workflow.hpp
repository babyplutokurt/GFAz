#ifndef COMPUTE_LOCATE_WORKFLOW_HPP
#define COMPUTE_LOCATE_WORKFLOW_HPP

// Where every haplotype visits given nodes. See
// docs/workflows/LOCATE_WORKFLOW.md.

#include "core/defaults.hpp"
#include "core/model/compressed_data.hpp"

#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

namespace gfaz {

struct LocateOptions {
  // Segments to look up, by their names in the GFA.
  std::vector<std::string> nodes;
  // A saved path index (`gfaz index`) to use; empty builds one for the query.
  std::string index_path;
  int num_threads = kDefaultNumThreads;
};

// Write one line per visit, tab-separated, in the order the nodes were given
// and then by haplotype and position:
//   #node  path  strand  step  start  end
// path is the P-line name or sample#hap#seq of a W-line, strand is + or -,
// step is the 0-based index of the visit in its P- or W-line, and [start, end)
// are the bases the node covers on that sequence.
void locate_nodes_to_tsv(const CompressedData &data,
                         const LocateOptions &options, std::ostream &out);

// Build the path index of a container and save it to index_path. Returns the
// number of blocks written.
size_t write_path_index(const CompressedData &data,
                        const std::string &index_path, uint32_t block_steps,
                        int num_threads);

} // namespace gfaz

#endif // COMPUTE_LOCATE_WORKFLOW_HPP
