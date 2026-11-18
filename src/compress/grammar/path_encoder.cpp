#include "compress/grammar/path_encoder.hpp"
#include "compress/grammar/packed_2mer.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

template <typename RuleLookup, typename MarkRule>
void encode_path_2mer(std::vector<gfaz::NodeId> &path,
                      const CompressionRules2Mer &rules, RuleLookup lookup,
                      MarkRule mark_rule) {
  size_t encoded_size = 0;

  for (size_t i = 0; i < path.size(); ++i) {
    const gfaz::NodeId node = path[i];
    path[encoded_size++] = node;

    if (encoded_size < 2)
      continue;

    int32_t first = path[encoded_size - 2];
    int32_t second = path[encoded_size - 1];
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
                               MarkRule mark_rule) {
  if (!rules.flat_map.empty()) {
    encode_path_2mer(
        path, rules,
        [&rules](Packed2mer canonical) { return rules.flat_map.find(canonical); },
        mark_rule);
  } else {
    encode_path_2mer(
        path, rules,
        [&rules](Packed2mer canonical) {
          auto it = rules.kmer_to_rule_id.find(canonical);
          return it == rules.kmer_to_rule_id.end() ? RuleFlatMap::kNotFound
                                                   : it->second;
        },
        mark_rule);
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
  if (unused < kMinimumReclaimElements || unused <= path.size() / 4)
    return;
  std::vector<gfaz::NodeId>(path).swap(path);
}

void PathEncoder::encode_paths_2mer(
    std::vector<std::vector<gfaz::NodeId>> &paths,
    const CompressionRules2Mer &rules, std::vector<uint8_t> &rules_used) {
  encode_range_2mer(paths.size(), rules, rules_used,
                    [&](size_t p, const auto &mark_rule) {
                      encode_path_2mer_dispatch(paths[p], rules, mark_rule);
                      release_unused_capacity(paths[p]);
                    });
}

void PathEncoder::encode_indexed_2mer(size_t count,
                                      const CompressionRules2Mer &rules,
                                      std::vector<uint8_t> &rules_used,
                                      const VisitIndex &visit) {
  encode_range_2mer(count, rules, rules_used,
                    [&](size_t p, const auto &mark_rule) {
                      visit(p, [&](std::vector<gfaz::NodeId> &path) {
                        encode_path_2mer_dispatch(path, rules, mark_rule);
                      });
                    });
}
