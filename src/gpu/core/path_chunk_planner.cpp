#include "gpu/core/path_chunk_planner.hpp"

#include <algorithm>
#include <cstdlib>

namespace gpu_compression {

size_t default_rolling_chunk_bytes() {
  constexpr size_t kDefaultChunkBytes = 1ull << 30; // 1 GiB
  const char *env_mb = std::getenv("GFAZ_GPU_ROLLING_CHUNK_MB");
  if (!env_mb || *env_mb == '\0') {
    return kDefaultChunkBytes;
  }

  char *end = nullptr;
  unsigned long long value_mb = std::strtoull(env_mb, &end, 10);
  if (end == env_mb || value_mb == 0) {
    return kDefaultChunkBytes;
  }
  return static_cast<size_t>(value_mb) * 1024ull * 1024ull;
}

std::vector<TraversalChunk>
build_traversal_chunks(const std::vector<uint32_t> &lengths,
                       size_t target_chunk_nodes) {
  std::vector<TraversalChunk> chunks;
  chunks.reserve(std::max<size_t>(1, lengths.size() / 1024));

  size_t seg_begin = 0;
  size_t node_begin = 0;
  while (seg_begin < lengths.size()) {
    size_t seg_end = seg_begin;
    size_t node_end = node_begin;

    while (seg_end < lengths.size()) {
      size_t next_len = lengths[seg_end];
      size_t proposed = node_end - node_begin + next_len;
      if (seg_end > seg_begin && proposed > target_chunk_nodes) {
        break;
      }
      node_end += next_len;
      ++seg_end;
      if (node_end - node_begin >= target_chunk_nodes) {
        break;
      }
    }

    if (seg_end == seg_begin) {
      node_end += lengths[seg_end];
      ++seg_end;
    }

    chunks.push_back({seg_begin, seg_end, node_begin, node_end});
    seg_begin = seg_end;
    node_begin = node_end;
  }

  return chunks;
}

} // namespace gpu_compression
