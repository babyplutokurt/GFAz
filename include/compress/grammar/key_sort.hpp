#ifndef KEY_SORT_HPP
#define KEY_SORT_HPP

#include <cstddef>
#include <cstdint>

namespace gfaz {

// Sorts `keys[0, count)` ascending (signed order). Same result as std::sort;
// uses up to `num_threads` threads and a temporary buffer of `count` keys.
//
// Keys are split into cache-sized buckets around splitters taken from a sample
// (so skewed key distributions still split evenly), and each bucket is then
// radix-sorted on the bytes that actually vary within it.
void parallel_sort_keys(int64_t *keys, size_t count, int num_threads);

} // namespace gfaz

#endif // KEY_SORT_HPP
