#ifndef PATH_CHUNK_PLANNER_HPP
#define PATH_CHUNK_PLANNER_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gpu_compression {

struct TraversalChunk {
  size_t segment_begin = 0;
  size_t segment_end = 0;
  size_t node_begin = 0;
  size_t node_end = 0;

  size_t num_segments() const { return segment_end - segment_begin; }
  size_t num_nodes() const { return node_end - node_begin; }
};

size_t default_rolling_chunk_bytes();

std::vector<TraversalChunk>
build_traversal_chunks(const std::vector<uint32_t> &lengths,
                       size_t target_chunk_nodes);

} // namespace gpu_compression

#endif // PATH_CHUNK_PLANNER_HPP
