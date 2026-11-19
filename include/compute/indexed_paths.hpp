#ifndef COMPUTE_INDEXED_PATHS_HPP
#define COMPUTE_INDEXED_PATHS_HPP

// The bundle a position-aware compute app starts from: decoded traversals,
// their PathIndex, segment lengths, haplotype names, and a PathAccess over
// them. Construct one from a container and use `access` and `catalog`.

#include "compute/haplotype_catalog.hpp"
#include "compute/path_access.hpp"
#include "compute/path_index.hpp"
#include "compute/traversal_set.hpp"
#include "core/defaults.hpp"
#include "core/model/compressed_data.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace gfaz {

struct IndexedPathsOptions {
  // A saved index (`gfaz index`) to load. Empty builds the index in memory,
  // which takes one pass over all traversals.
  std::string index_path;
  // Steps per block when the index is built here.
  uint32_t block_steps = PathIndexOptions().block_steps;
  int num_threads = kDefaultNumThreads;
};

struct IndexedPaths {
  IndexedPaths(const CompressedData &data, const IndexedPathsOptions &options);
  IndexedPaths(const IndexedPaths &) = delete;
  IndexedPaths &operator=(const IndexedPaths &) = delete;

  std::vector<uint32_t> segment_lengths; // segment_lengths[n - 1] for node n
  TraversalSet traversals;
  PathIndex index;
  HaplotypeCatalog catalog;
  PathAccess access;

  uint32_t num_nodes() const {
    return static_cast<uint32_t>(segment_lengths.size());
  }
};

} // namespace gfaz

#endif // COMPUTE_INDEXED_PATHS_HPP
