#ifndef RULE_FLAT_MAP_HPP
#define RULE_FLAT_MAP_HPP

#include "compress/grammar/packed_2mer.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>

// Open-addressing hash table mapping canonical Packed2mer -> rule ID.
//
// Node IDs occupy at most 31 bits (int32 with the sign bit encoding
// orientation), so no valid 2-mer ever has INT32_MIN in its high half. That
// makes pack_2mer(INT32_MIN, 0) safe as the vacant-slot marker, which in turn
// allows lock-free parallel construction: threads claim slots by CAS on the
// key alone and the value store needs no synchronization because lookups only
// begin after the building parallel region's barrier.
class RuleFlatMap {
public:
  static constexpr int64_t kEmptyKey =
      static_cast<int64_t>(0x8000000000000000ULL);
  static constexpr uint32_t kNotFound = 0xFFFFFFFFu;

  struct Slot {
    int64_t key;
    uint32_t value;
    uint32_t pad;
  };

  bool empty() const { return capacity_ == 0; }

  void clear() {
    slots_.reset();
    capacity_ = 0;
    mask_ = 0;
  }

  // Bulk-build from n unique canonical 2-mers; keys[i] maps to value_base + i.
  // Safe to call from multiple threads via the internal parallel loop only.
  void build(const Packed2mer *keys, size_t n, uint32_t value_base) {
    size_t cap = 16;
    while (cap < n + n / 2)
      cap <<= 1;
    slots_.reset(new Slot[cap]);
    capacity_ = cap;
    mask_ = cap - 1;

    Slot *slots = slots_.get();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < cap; ++i) {
      slots[i].key = kEmptyKey;
      slots[i].value = 0;
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < n; ++i) {
      size_t idx = hash_key(keys[i]) & mask_;
      for (;;) {
        int64_t expected = kEmptyKey;
        if (__atomic_compare_exchange_n(&slots[idx].key, &expected, keys[i],
                                        false, __ATOMIC_RELAXED,
                                        __ATOMIC_RELAXED)) {
          slots[idx].value = value_base + static_cast<uint32_t>(i);
          break;
        }
        idx = (idx + 1) & mask_;
      }
    }
  }

  uint32_t find(Packed2mer key) const {
    size_t idx = hash_key(key) & mask_;
    const Slot *slots = slots_.get();
    for (;;) {
      const int64_t k = slots[idx].key;
      if (k == key)
        return slots[idx].value;
      if (k == kEmptyKey)
        return kNotFound;
      idx = (idx + 1) & mask_;
    }
  }

private:
  static inline uint64_t hash_key(int64_t key) {
    // splitmix64 finalizer
    uint64_t x = static_cast<uint64_t>(key);
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return x;
  }

  std::unique_ptr<Slot[]> slots_;
  size_t capacity_ = 0;
  size_t mask_ = 0;
};

#endif // RULE_FLAT_MAP_HPP
