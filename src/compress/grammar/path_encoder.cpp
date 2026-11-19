#include "compress/grammar/path_encoder.hpp"
#include "compress/grammar/packed_2mer.hpp"
#include <algorithm>
#include <cstdint>
#include <limits>
#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

// Table slots are prefetched this many input pairs ahead of the probe.
constexpr size_t kPrefetchDistance = 12;

inline uint32_t abs_id(gfaz::NodeId id) {
  return id < 0 ? 0u - static_cast<uint32_t>(id) : static_cast<uint32_t>(id);
}

// `new_rule_start`: a pair whose left element has an ID at or above this value
// is not looked up. See encode_path_2mer_dispatch for when that is valid.
// `min_new_id`: a pair of two IDs below this value is not looked up either.
template <typename RuleLookup, typename Prefetch, typename MarkRule>
void encode_path_2mer(std::vector<gfaz::NodeId> &path,
                      const CompressionRules2Mer &rules,
                      uint32_t new_rule_start, uint32_t min_new_id,
                      RuleLookup lookup,
                      Prefetch prefetch, MarkRule mark_rule) {
  size_t encoded_size = 0;
  const size_t input_size = path.size();

  for (size_t i = 0; i < input_size; ++i) {
    // Every pair that does get looked up is a pair of adjacent input elements
    // (the write position trails the read position, so path[i + k] is still
    // input). Start loading its slot now.
    if (i + kPrefetchDistance < input_size) {
      const gfaz::NodeId ahead_first = path[i + kPrefetchDistance - 1];
      const gfaz::NodeId ahead_second = path[i + kPrefetchDistance];
      if (std::max(abs_id(ahead_first), abs_id(ahead_second)) >= min_new_id)
        prefetch(canonical_2mer(pack_2mer(ahead_first, ahead_second)));
    }

    const gfaz::NodeId node = path[i];
    path[encoded_size++] = node;

    if (encoded_size < 2)
      continue;

    int32_t first = path[encoded_size - 2];
    int32_t second = path[encoded_size - 1];
    const uint32_t first_abs = abs_id(first);
    if (first_abs >= new_rule_start ||
        std::max(first_abs, abs_id(second)) < min_new_id)
      continue;
    Packed2mer top_kmer = pack_2mer(first, second);
    const Packed2mer canonical = canonical_2mer(top_kmer);
    const uint32_t rule_id = lookup(canonical);
    if (rule_id == RuleFlatMap::kNotFound)
      continue;
    const int32_t oriented_rule_id =
        top_kmer == canonical ? static_cast<int32_t>(rule_id)
                              : -static_cast<int32_t>(rule_id);

    path[encoded_size - 2] = oriented_rule_id;
    --encoded_size;
    mark_rule(rule_id - rules.rules_start_id);
  }

  path.resize(encoded_size);
}

template <typename MarkRule>
void encode_path_2mer_dispatch(std::vector<gfaz::NodeId> &path,
                               const CompressionRules2Mer &rules,
                               uint32_t min_new_id, MarkRule mark_rule) {
  if (!rules.flat_map.empty()) {
    // flat_map is filled by RuleGenerator::generate_rules_2mer_combined, whose
    // rules are the 2-mers of the sequences being encoded. Those sequences
    // hold only IDs below rules_start_id, so a pair that starts with a rule
    // created in this pass can never be a rule itself.
    encode_path_2mer(
        path, rules, rules.rules_start_id, min_new_id,
        [&rules](Packed2mer canonical) { return rules.flat_map.find(canonical); },
        [&rules](Packed2mer canonical) { rules.flat_map.prefetch(canonical); },
        mark_rule);
  } else {
    // Rule sets loaded from a file (add-haplotypes) span several rounds, so a
    // freshly created rule can start a later rule: look up every pair.
    encode_path_2mer(
        path, rules, std::numeric_limits<uint32_t>::max(), 0,
        [&rules](Packed2mer canonical) {
          auto it = rules.kmer_to_rule_id.find(canonical);
          return it == rules.kmer_to_rule_id.end() ? RuleFlatMap::kNotFound
                                                   : it->second;
        },
        [](Packed2mer) {}, mark_rule);
  }
}

// Shared driver: runs `body(index, mark_rule)` over [0, count) in parallel and
// folds the per-thread rule-usage bitmaps into `rules_used`.
template <typename Body>
void encode_range_2mer(size_t count, const CompressionRules2Mer &rules,
                       std::vector<uint8_t> &rules_used, Body body) {
  const size_t num_rules = rules.next_available_id - rules.rules_start_id;
  if (rules_used.size() != num_rules)
    rules_used.assign(num_rules, 0);

  if (count == 0)
    return;

#ifdef _OPENMP
  const int actual_threads = omp_get_max_threads();
  const size_t word_count = (num_rules + 63) / 64;
  std::vector<std::vector<uint64_t>> thread_usage(
      actual_threads, std::vector<uint64_t>(word_count, 0));

#pragma omp parallel
  {
    auto &local_usage = thread_usage[omp_get_thread_num()];
    const auto mark_rule = [&local_usage](size_t rule_offset) {
      local_usage[rule_offset / 64] |= uint64_t{1} << (rule_offset % 64);
    };

#pragma omp for schedule(dynamic)
    for (size_t p = 0; p < count; ++p)
      body(p, mark_rule);
  }

#pragma omp parallel for schedule(static)
  for (size_t word = 0; word < word_count; ++word) {
    uint64_t used_bits = 0;
    for (const auto &local_usage : thread_usage)
      used_bits |= local_usage[word];

    while (used_bits != 0) {
      const unsigned bit = __builtin_ctzll(used_bits);
      const size_t rule_offset = word * 64 + bit;
      if (rule_offset < num_rules)
        rules_used[rule_offset] = 1;
      used_bits &= used_bits - 1;
    }
  }
#else
  const auto mark_rule = [&rules_used](size_t rule_offset) {
    rules_used[rule_offset] = 1;
  };
  for (size_t p = 0; p < count; ++p)
    body(p, mark_rule);
#endif
}

} // namespace

PathEncoder::PathEncoder() {}

// Encoding shrinks a traversal in place. Give the freed tail back right away
// so resident memory follows the encoded size from round to round instead of
// staying at the original traversal size until the last round.
void PathEncoder::release_unused_capacity(std::vector<gfaz::NodeId> &path) {
  constexpr size_t kMinimumReclaimElements =
      (64ull * 1024ull) / sizeof(gfaz::NodeId);
  const size_t unused = path.capacity() - path.size();
  if (unused < kMinimumReclaimElements)
    return;
#if defined(__linux__)
  // Drop the pages behind the tail and keep the allocation: no copy into a
  // smaller buffer, and the address range costs nothing while untouched.
  static const uintptr_t page_size =
      static_cast<uintptr_t>(sysconf(_SC_PAGESIZE));
  const uintptr_t tail_begin =
      (reinterpret_cast<uintptr_t>(path.data() + path.size()) + page_size - 1) &
      ~(page_size - 1);
  const uintptr_t tail_end =
      reinterpret_cast<uintptr_t>(path.data() + path.capacity()) &
      ~(page_size - 1);
  if (tail_end <= tail_begin ||
      madvise(reinterpret_cast<void *>(tail_begin), tail_end - tail_begin,
              MADV_DONTNEED) == 0)
    return;
#endif
  if (unused <= path.size() / 4)
    return;
  std::vector<gfaz::NodeId>(path).swap(path);
}

void PathEncoder::encode_paths_2mer(
    std::vector<std::vector<gfaz::NodeId>> &paths,
    const CompressionRules2Mer &rules, std::vector<uint8_t> &rules_used,
    uint32_t min_new_id) {
  encode_range_2mer(paths.size(), rules, rules_used,
                    [&](size_t p, const auto &mark_rule) {
                      encode_path_2mer_dispatch(paths[p], rules, min_new_id,
                                                mark_rule);
                      release_unused_capacity(paths[p]);
                    });
}

void PathEncoder::encode_indexed_2mer(size_t count,
                                      const CompressionRules2Mer &rules,
                                      std::vector<uint8_t> &rules_used,
                                      const VisitIndex &visit,
                                      uint32_t min_new_id) {
  encode_range_2mer(count, rules, rules_used,
                    [&](size_t p, const auto &mark_rule) {
                      visit(p, [&](std::vector<gfaz::NodeId> &path) {
                        encode_path_2mer_dispatch(path, rules, min_new_id,
                                                  mark_rule);
                      });
                    });
}
