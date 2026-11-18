#ifndef TRAVERSAL_SINK_HPP
#define TRAVERSAL_SINK_HPP

#include "core/model/gfa_graph.hpp"

#include <cstddef>
#include <vector>

namespace gfaz {

// Receives path/walk traversals from the parser as they are decoded, instead
// of having every traversal stored in the GfaGraph. Lets the compressor keep
// only part of the traversal data in memory.
class TraversalSink {
public:
  virtual ~TraversalSink() = default;

  // Called before any traversal is delivered, and again if the parser has to
  // restart (which discards everything delivered so far).
  virtual void prepare(size_t num_paths, size_t num_walks) = 0;

  // Buffer for the calling thread to decode a traversal into before passing
  // it to consume(). Owned by the sink so it can be released afterwards.
  virtual std::vector<NodeId> &scratch() = 0;

  // Thread-safe; each (is_walk, index) is delivered exactly once per prepare().
  // `sequence` is the parser's scratch buffer and may be modified or moved
  // from. `slot` is the graph's vector for this traversal; the sink may fill
  // it or leave it empty.
  virtual void consume(bool is_walk, size_t index,
                       std::vector<NodeId> &sequence,
                       std::vector<NodeId> &slot) = 0;
};

} // namespace gfaz

#endif // TRAVERSAL_SINK_HPP
