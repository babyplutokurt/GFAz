// parallel_sort_keys must give exactly what std::sort gives, for the key
// distributions the rule generator produces and for the degenerate ones.

#include "compress/grammar/key_sort.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

bool check(const std::string &label, std::vector<int64_t> keys, int threads) {
  std::vector<int64_t> expected = keys;
  std::sort(expected.begin(), expected.end());
  gfaz::parallel_sort_keys(keys.data(), keys.size(), threads);
  if (keys == expected)
    return true;
  std::cerr << "FAIL " << label << " (n=" << keys.size()
            << ", threads=" << threads << ")" << std::endl;
  return false;
}

} // namespace

int main() {
  std::mt19937_64 rng(12345);
  const int64_t lowest = std::numeric_limits<int64_t>::min();
  const int64_t highest = std::numeric_limits<int64_t>::max();
  bool ok = true;

  for (size_t n : {size_t{0}, size_t{1}, size_t{1000}, size_t{65535},
                   size_t{65536}, size_t{300000}, size_t{5000000}}) {
    for (int threads : {1, 3, 16}) {
      std::vector<int64_t> keys(n);

      for (auto &key : keys)
        key = static_cast<int64_t>(rng());
      ok &= check("uniform", keys, threads);

      // Packed 2-mers of small signed deltas: both halves near zero, mixed
      // signs, many repeats.
      for (auto &key : keys) {
        const int32_t first = static_cast<int32_t>(rng() % 41) - 20;
        const int32_t second = static_cast<int32_t>(rng() % 2001) - 1000;
        key = (static_cast<int64_t>(first) << 32) |
              static_cast<uint32_t>(second);
      }
      ok &= check("small deltas", keys, threads);

      // Mostly one value, with a thin spread of others.
      for (auto &key : keys)
        key = rng() % 100 == 0 ? static_cast<int64_t>(rng()) : 42;
      ok &= check("one dominant value", keys, threads);

      std::fill(keys.begin(), keys.end(), -7);
      ok &= check("all equal", keys, threads);

      for (size_t i = 0; i < n; ++i)
        keys[i] = static_cast<int64_t>(n - i) * 1000003;
      ok &= check("descending", keys, threads);

      // Two far-apart clusters plus the extreme values.
      for (auto &key : keys)
        key = (rng() & 1) ? lowest + static_cast<int64_t>(rng() % 1000)
                          : highest - static_cast<int64_t>(rng() % 1000);
      if (n >= 2) {
        keys[0] = highest;
        keys[n - 1] = lowest;
      }
      ok &= check("extremes", keys, threads);
    }
  }

  if (!ok)
    return 1;
  std::cout << "PASS key_sort" << std::endl;
  return 0;
}
