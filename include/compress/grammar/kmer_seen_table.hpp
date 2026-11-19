#ifndef KMER_SEEN_TABLE_HPP
#define KMER_SEEN_TABLE_HPP

#include "compress/grammar/packed_2mer.hpp"
#include "core/utils/runtime_utils.hpp"

#include <cstdint>
#include <cstdlib>
#include <new>

// Single-thread open-addressing set of canonical 2-mers that also remembers
// whether a 2-mer was inserted more than once. One 8-byte slot per 2-mer, so
// an insert touches one cache line; `prefetch` lets a caller that knows its
// next keys start loading their slots early.
//
// The "seen twice" mark lives inside the key. It needs both node IDs of every
// 2-mer to satisfy |id| < 2^30 (see `ids_fit`): bits 62 and 63 of the packed
// value are then equal, and a slot with bit 62 flipped is a marked entry that
// can never collide with a real key. Zero marks an empty slot; the all-zero
// 2-mer (delta-transformed sequences do contain zeros) is counted on the side.
class KmerSeenTable {
public:
  // Largest absolute node ID the table can hold, exclusive.
  static constexpr uint32_t kIdLimit = uint32_t{1} << 30;
  static bool ids_fit(uint32_t exclusive_max_abs_id) {
    return exclusive_max_abs_id <= kIdLimit;
  }

  KmerSeenTable() = default;
  KmerSeenTable(const KmerSeenTable &) = delete;
  KmerSeenTable &operator=(const KmerSeenTable &) = delete;
  ~KmerSeenTable() { std::free(slots_); }

  // Sizes the table for `expected` distinct 2-mers; it still grows on demand.
  void reserve(size_t expected) {
    size_t capacity = kMinCapacity;
    while (capacity - capacity / 4 < expected)
      capacity <<= 1;
    if (capacity > capacity_)
      rehash(capacity);
  }

  static uint64_t hash(Packed2mer key) {
    // splitmix64 finalizer
    uint64_t x = static_cast<uint64_t>(key);
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return x;
  }

  void prefetch(uint64_t key_hash) const {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(slots_ + (key_hash & mask_), 1);
#else
    (void)key_hash;
#endif
  }

  // Records one occurrence of `key`, whose hash is `key_hash`.
  void insert(Packed2mer key, uint64_t key_hash) {
    if (key == 0) {
      if (zero_count_ < 2)
        ++zero_count_;
      return;
    }
    if (size_ >= grow_at_)
      rehash(capacity_ == 0 ? kMinCapacity : capacity_ * 2);
    const uint64_t wanted = static_cast<uint64_t>(key);
    size_t index = key_hash & mask_;
    for (;;) {
      const uint64_t slot = slots_[index];
      if (slot == 0) {
        slots_[index] = wanted;
        ++size_;
        return;
      }
      const uint64_t difference = slot ^ wanted;
      if (difference == 0) {
        slots_[index] = wanted ^ kMark;
        ++repeated_;
        return;
      }
      if (difference == kMark)
        return;
      index = (index + 1) & mask_;
    }
  }

  size_t size() const { return size_ + (zero_count_ > 0 ? 1 : 0); }
  // Number of distinct 2-mers inserted more than once.
  size_t repeated() const { return repeated_ + (zero_count_ > 1 ? 1 : 0); }

  // Writes every 2-mer once, and those inserted more than once a second time.
  // Returns the number written, which is size() + repeated().
  size_t emit(Packed2mer *out) const {
    size_t written = 0;
    for (unsigned copy = 0; copy < zero_count_; ++copy)
      out[written++] = 0;
    for (size_t index = 0; index < capacity_; ++index) {
      const uint64_t slot = slots_[index];
      if (slot == 0)
        continue;
      // A real key has bits 62 and 63 equal; a marked one does not.
      const bool marked = ((slot >> 62) & 1) != (slot >> 63);
      const Packed2mer key = static_cast<Packed2mer>(marked ? slot ^ kMark : slot);
      out[written++] = key;
      if (marked)
        out[written++] = key;
    }
    return written;
  }

  void release() {
    std::free(slots_);
    slots_ = nullptr;
    capacity_ = mask_ = size_ = repeated_ = grow_at_ = 0;
    zero_count_ = 0;
  }

private:
  static constexpr uint64_t kMark = uint64_t{1} << 62;
  static constexpr size_t kMinCapacity = size_t{1} << 12;

  void rehash(size_t capacity) {
    // calloc: untouched slots stay unmapped zero pages until first use.
    uint64_t *fresh =
        static_cast<uint64_t *>(std::calloc(capacity, sizeof(uint64_t)));
    if (fresh == nullptr)
      throw std::bad_alloc();
    gfaz::runtime_utils::advise_huge_pages(fresh, capacity * sizeof(uint64_t));
    const size_t mask = capacity - 1;
    for (size_t index = 0; index < capacity_; ++index) {
      const uint64_t slot = slots_[index];
      if (slot == 0)
        continue;
      const bool marked = ((slot >> 62) & 1) != (slot >> 63);
      const uint64_t key = marked ? slot ^ kMark : slot;
      size_t target = hash(static_cast<Packed2mer>(key)) & mask;
      while (fresh[target] != 0)
        target = (target + 1) & mask;
      fresh[target] = slot;
    }
    std::free(slots_);
    slots_ = fresh;
    capacity_ = capacity;
    mask_ = mask;
    grow_at_ = capacity - capacity / 4; // 75% load
  }

  uint64_t *slots_ = nullptr;
  size_t capacity_ = 0;
  size_t mask_ = 0;
  size_t size_ = 0;
  size_t repeated_ = 0;
  size_t grow_at_ = 0;
  unsigned zero_count_ = 0; // occurrences of the all-zero 2-mer, capped at 2
};

#endif // KMER_SEEN_TABLE_HPP
