#include "core/utils/runtime_utils.hpp"
#include "compress/grammar/key_sort.hpp"

#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(_OPENMP) && defined(__GLIBCXX__)
#include <parallel/algorithm>
#endif

namespace gfaz {

namespace {

constexpr size_t kPlainSortThreshold = size_t{1} << 16;
constexpr size_t kTargetBucketKeys = size_t{1} << 17;
constexpr size_t kMaxBuckets = size_t{1} << 14; // bucket index fits uint16_t
constexpr size_t kSamplesPerBucket = 16;

// Byte-wise LSD radix sort of the `count` keys in `src`, using `other` as the
// second buffer. Bytes on which all keys agree are skipped. The sorted keys end
// up in `dest`, which must be `src` or `other`.
void radix_sort_bucket(int64_t *src, int64_t *other, size_t count,
                       int64_t *dest) {
  if (count < 512) {
    if (dest != src)
      std::memcpy(dest, src, count * sizeof(int64_t));
    std::sort(dest, dest + count);
    return;
  }

  uint64_t varying = 0;
  const uint64_t first = static_cast<uint64_t>(src[0]);
  for (size_t i = 1; i < count; ++i)
    varying |= static_cast<uint64_t>(src[i]) ^ first;

  int64_t *from = src;
  int64_t *to = other;
  for (int byte = 0; byte < 8; ++byte) {
    const int shift = byte * 8;
    if (((varying >> shift) & 0xff) == 0)
      continue;
    // Flipping the top bit of the top byte turns unsigned byte order into
    // signed key order.
    const uint64_t flip = byte == 7 ? 0x80 : 0;
    size_t positions[256] = {};
    for (size_t i = 0; i < count; ++i)
      ++positions[((static_cast<uint64_t>(from[i]) >> shift) & 0xff) ^ flip];
    size_t total = 0;
    for (size_t &position : positions) {
      const size_t bucket_count = position;
      position = total;
      total += bucket_count;
    }
    for (size_t i = 0; i < count; ++i) {
      const int64_t key = from[i];
      to[positions[((static_cast<uint64_t>(key) >> shift) & 0xff) ^ flip]++] =
          key;
    }
    std::swap(from, to);
  }
  if (from != dest)
    std::memcpy(dest, from, count * sizeof(int64_t));
}

// floor(i * count / parts) for i <= parts, without overflowing.
size_t scaled_index(size_t i, size_t count, size_t parts) {
  return (count / parts) * i + (count % parts) * i / parts;
}

void plain_sort(int64_t *keys, size_t count, int num_threads) {
#if defined(_OPENMP) && defined(__GLIBCXX__)
  if (num_threads > 1) {
    __gnu_parallel::sort(keys, keys + count, std::less<int64_t>(),
                         __gnu_parallel::default_parallel_tag(
                             static_cast<unsigned>(num_threads)));
    return;
  }
#endif
  (void)num_threads;
  std::sort(keys, keys + count);
}

} // namespace

void parallel_sort_keys(int64_t *keys, size_t count, int num_threads) {
  const int threads = std::max(1, num_threads);
  if (count < kPlainSortThreshold) {
    std::sort(keys, keys + count);
    return;
  }

  size_t buckets = 16;
  while (buckets < kMaxBuckets && buckets * kTargetBucketKeys < count)
    buckets *= 2;

  // Splitters: evenly spaced order statistics of an evenly spaced sample.
  const size_t sample_count = buckets * kSamplesPerBucket;
  std::vector<int64_t> sample(sample_count);
  for (size_t i = 0; i < sample_count; ++i)
    sample[i] = keys[scaled_index(i, count, sample_count)];
  std::sort(sample.begin(), sample.end());
  std::vector<int64_t> splitters(buckets - 1);
  for (size_t j = 0; j + 1 < buckets; ++j)
    splitters[j] = sample[(j + 1) * kSamplesPerBucket];

  // Bucket of a key = number of splitters <= key.
  const int64_t *const split = splitters.data();
  const auto bucket_of = [split, buckets](int64_t key) {
    size_t index = 0;
    for (size_t step = buckets >> 1; step > 0; step >>= 1)
      index += split[index + step - 1] <= key ? step : 0;
    return index;
  };

  const size_t ranges = static_cast<size_t>(threads);
  const auto range_begin = [count, ranges](size_t r) {
    return scaled_index(r, count, ranges);
  };

  std::unique_ptr<uint16_t[]> bucket_ids(new uint16_t[count]);
  gfaz::runtime_utils::advise_huge_pages(bucket_ids.get(),
                                         count * sizeof(uint16_t));
  std::vector<size_t> positions(ranges * buckets, 0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
  for (size_t r = 0; r < ranges; ++r) {
    size_t *histogram = positions.data() + r * buckets;
    for (size_t i = range_begin(r); i < range_begin(r + 1); ++i) {
      const size_t bucket = bucket_of(keys[i]);
      bucket_ids[i] = static_cast<uint16_t>(bucket);
      ++histogram[bucket];
    }
  }

  // Turn the per-range histograms into write positions: bucket by bucket, and
  // within a bucket range by range.
  std::vector<size_t> bucket_start(buckets + 1, 0);
  size_t total = 0;
  for (size_t b = 0; b < buckets; ++b) {
    bucket_start[b] = total;
    for (size_t r = 0; r < ranges; ++r) {
      size_t &position = positions[r * buckets + b];
      const size_t range_count = position;
      position = total;
      total += range_count;
    }
  }
  bucket_start[buckets] = total;

  std::unique_ptr<int64_t[]> scattered(new int64_t[count]);
  gfaz::runtime_utils::advise_huge_pages(scattered.get(),
                                         count * sizeof(int64_t));
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
  for (size_t r = 0; r < ranges; ++r) {
    size_t *position = positions.data() + r * buckets;
    for (size_t i = range_begin(r); i < range_begin(r + 1); ++i)
      scattered[position[bucket_ids[i]]++] = keys[i];
  }
  bucket_ids.reset();

  // Buckets are disjoint key ranges in order, so sorting each one back into
  // its own span of `keys` sorts the whole array. A bucket far larger than
  // intended (one key value dominating) is left for a parallel sort afterwards
  // rather than tying up a single thread.
  const size_t oversized = std::max(kTargetBucketKeys * 8, count / ranges);
  std::vector<size_t> deferred;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 4) num_threads(threads)
#endif
  for (size_t b = 0; b < buckets; ++b) {
    const size_t begin = bucket_start[b];
    const size_t size = bucket_start[b + 1] - begin;
    if (size == 0)
      continue;
    if (size > oversized && threads > 1) {
#ifdef _OPENMP
#pragma omp critical(gfaz_key_sort_deferred)
#endif
      deferred.push_back(b);
      continue;
    }
    radix_sort_bucket(scattered.get() + begin, keys + begin, size,
                      keys + begin);
  }
  for (size_t b : deferred) {
    const size_t begin = bucket_start[b];
    const size_t size = bucket_start[b + 1] - begin;
    std::memcpy(keys + begin, scattered.get() + begin, size * sizeof(int64_t));
    plain_sort(keys + begin, size, threads);
  }
}

} // namespace gfaz
