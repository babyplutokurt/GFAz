// KmerSeenTable must report each inserted 2-mer once, and a second time when
// it was inserted more than once, for every key the rule scan can produce:
// zero IDs (delta-transformed sequences), negative IDs and IDs just under the
// 2^30 limit.

#include "compress/grammar/kmer_seen_table.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace {

bool check(const std::string &label, const std::vector<Packed2mer> &keys,
           size_t reserve) {
  KmerSeenTable table;
  if (reserve > 0)
    table.reserve(reserve);
  std::map<Packed2mer, size_t> counts;
  for (const Packed2mer key : keys) {
    table.insert(key, KmerSeenTable::hash(key));
    ++counts[key];
  }

  std::vector<Packed2mer> expected;
  size_t repeated = 0;
  for (const auto &[key, count] : counts) {
    expected.push_back(key);
    if (count > 1) {
      expected.push_back(key);
      ++repeated;
    }
  }

  std::vector<Packed2mer> got(table.size() + table.repeated());
  const size_t written = table.emit(got.data());
  std::sort(got.begin(), got.end());
  if (written == got.size() && table.size() == counts.size() &&
      table.repeated() == repeated && got == expected)
    return true;
  std::cerr << "FAIL " << label << " (n=" << keys.size() << ")" << std::endl;
  return false;
}

} // namespace

int main() {
  std::mt19937_64 rng(2024);
  const int32_t limit = static_cast<int32_t>(KmerSeenTable::kIdLimit) - 1;
  bool ok = true;

  ok &= check("empty", {}, 0);
  ok &= check("zero once", {pack_2mer(0, 0)}, 0);
  ok &= check("zero many", {pack_2mer(0, 0), pack_2mer(0, 0), pack_2mer(0, 0)}, 0);

  std::vector<Packed2mer> edges;
  for (const int32_t a : {0, 1, -1, limit, -limit})
    for (const int32_t b : {0, 1, -1, limit, -limit})
      for (int copies = 0; copies < 1 + (a + b) % 3; ++copies)
        edges.push_back(pack_2mer(a, b));
  ok &= check("edge ids", edges, 0);

  for (const size_t n : {size_t{100}, size_t{5000}, size_t{300000}}) {
    for (const int32_t span : {4, 1000, limit}) {
      std::uniform_int_distribution<int32_t> id(-span, span);
      std::vector<Packed2mer> keys(n);
      for (auto &key : keys)
        key = pack_2mer(id(rng), id(rng));
      const std::string label =
          "random n=" + std::to_string(n) + " span=" + std::to_string(span);
      ok &= check(label, keys, 0);
      ok &= check(label + " reserved", keys, n);
    }
  }

  if (!ok)
    return 1;
  std::cout << "PASS kmer_seen_table" << std::endl;
  return 0;
}
