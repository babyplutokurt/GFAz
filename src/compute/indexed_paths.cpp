#include "compute/indexed_paths.hpp"

#include "compute/traversal_query.hpp"

namespace gfaz {

namespace {

PathIndex load_or_build_index(const CompressedData &data,
                              const TraversalSet &traversals,
                              const std::vector<uint32_t> &segment_lengths,
                              const IndexedPathsOptions &options) {
  if (!options.index_path.empty())
    return PathIndex::load(options.index_path, PathIndex::fingerprint(data));
  return PathIndex::build(
      traversals, segment_lengths,
      PathIndexOptions{options.block_steps, options.num_threads});
}

} // namespace

IndexedPaths::IndexedPaths(const CompressedData &data,
                           const IndexedPathsOptions &options)
    : segment_lengths(tquery::load_segment_lengths(data)),
      names(SegmentNames::decode(data)), traversals(data),
      index(load_or_build_index(data, traversals, segment_lengths, options)),
      catalog(data), access(traversals, index, segment_lengths) {}

} // namespace gfaz
