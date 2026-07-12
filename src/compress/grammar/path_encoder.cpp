#include "compress/grammar/path_encoder.hpp"
#include "compress/grammar/packed_2mer.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

template <typename MarkRule>
void encode_path_2mer(std::vector<gfaz::NodeId> &path,
                      const CompressionRules2Mer &rules,
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

    auto it = rules.kmer_to_rule_id.find(top_kmer);
    int32_t oriented_rule_id = 0;
    if (it != rules.kmer_to_rule_id.end()) {
      oriented_rule_id = static_cast<int32_t>(it->second);
    } else {
      auto reverse_it =
          rules.kmer_to_rule_id.find(reverse_2mer(top_kmer));
      if (reverse_it == rules.kmer_to_rule_id.end())
        continue;
      it = reverse_it;
      oriented_rule_id = -static_cast<int32_t>(it->second);
    }

    path[encoded_size - 2] = oriented_rule_id;
    --encoded_size;
    mark_rule(it->second - rules.rules_start_id);
  }

  path.resize(encoded_size);
}

} // namespace

PathEncoder::PathEncoder() {}

void PathEncoder::encode_paths_2mer(
    std::vector<std::vector<gfaz::NodeId>> &paths,
    const CompressionRules2Mer &rules, std::vector<uint8_t> &rules_used) {
  const size_t num_rules =
      rules.next_available_id - rules.rules_start_id;
  if (rules_used.size() != num_rules)
    rules_used.assign(num_rules, 0);

  if (paths.empty())
    return;

#ifdef _OPENMP
  const int actual_threads = omp_get_max_threads();
  const size_t word_count = (num_rules + 63) / 64;
  std::vector<std::vector<uint64_t>> thread_usage(
      actual_threads, std::vector<uint64_t>(word_count, 0));

#pragma omp parallel
  {
    auto &local_usage = thread_usage[omp_get_thread_num()];

#pragma omp for schedule(dynamic)
    for (size_t p = 0; p < paths.size(); ++p) {
      encode_path_2mer(paths[p], rules, [&](size_t rule_offset) {
        local_usage[rule_offset / 64] |=
            uint64_t{1} << (rule_offset % 64);
      });
    }
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
  for (auto &path : paths) {
    encode_path_2mer(path, rules, [&](size_t rule_offset) {
      rules_used[rule_offset] = 1;
    });
  }
#endif
}
