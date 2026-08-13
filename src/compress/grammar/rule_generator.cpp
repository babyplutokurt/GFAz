#include "compress/grammar/rule_generator.hpp"
#include "core/utils/debug_log.hpp"
#include "robin_hood.h"
#include "core/utils/threading_utils.hpp"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(_OPENMP) && defined(__GLIBCXX__)
#include <parallel/algorithm>
#endif

RuleGenerator::RuleGenerator() {}

// DEPRECATED: This function is kept for legacy compatibility but is not used in
// the current workflow. See generate_rules_2mer_combined below.
CompressionRules2Mer RuleGenerator::generate_rules_2mer(
    const std::vector<std::vector<gfaz::NodeId>> &paths, uint32_t starting_id,
    size_t freq_threshold, int num_threads) {

  CompressionRules2Mer rules;
  rules.rules_start_id = starting_id;
  uint32_t current_rule_id = starting_id;

  // Global sets for final result - using Packed2mer (int64_t) for faster
  // hashing
  robin_hood::unordered_flat_set<Packed2mer> seen;
  robin_hood::unordered_flat_set<Packed2mer> repeated;

  // --- 1. Collection Phase (Parallel with OpenMP) ---
#ifdef _OPENMP
  int actual_threads = resolve_omp_thread_count(num_threads);

  // Pre-allocate storage for all thread-local results
  std::vector<robin_hood::unordered_flat_set<Packed2mer>> thread_seen(
      actual_threads);
  std::vector<robin_hood::unordered_flat_set<Packed2mer>> thread_repeated(
      actual_threads);

// Parallel collection - no critical section, fully parallel
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    auto &local_seen = thread_seen[tid];
    auto &local_repeated = thread_repeated[tid];

// Dynamic scheduling for load balancing with varying path lengths
#pragma omp for schedule(dynamic)
    for (size_t p = 0; p < paths.size(); ++p) {
      const auto &path = paths[p];
      if (path.size() < 2) {
        continue;
      }

      for (size_t i = 0; i <= path.size() - 2; ++i) {
        Packed2mer kmer = pack_2mer(path[i], path[i + 1]);
        Packed2mer canonical = canonical_2mer(kmer);

        // Fast path: already repeated locally, skip
        if (local_repeated.count(canonical)) {
          continue;
        }

        // Check if we've seen this locally before
        if (!local_seen.insert(canonical).second) {
          local_repeated.insert(canonical);
        }
      }
    }
  }
  // End of parallel region - all threads done with collection

  // Sequential merge of all thread results (no blocking during collection)
  for (int t = 0; t < actual_threads; ++t) {
    // Merge thread_repeated[t] → global repeated
    for (const auto &kmer : thread_repeated[t]) {
      repeated.insert(kmer);
    }

    // Merge thread_seen[t]: check cross-thread repetition
    for (const auto &kmer : thread_seen[t]) {
      if (repeated.count(kmer)) {
        continue; // Already globally repeated
      } else if (!seen.insert(kmer).second) {
        repeated.insert(kmer); // Seen by another thread → repeated!
      }
    }
  }
#else
  // Sequential fallback when OpenMP is not available
  for (const auto &path : paths) {
    if (path.size() < 2) {
      continue;
    }

    for (size_t i = 0; i <= path.size() - 2; ++i) {
      Packed2mer kmer = pack_2mer(path[i], path[i + 1]);
      Packed2mer canonical = canonical_2mer(kmer);

      if (repeated.count(canonical)) {
        continue;
      }

      if (!seen.insert(canonical).second) {
        repeated.insert(canonical);
      }
    }
  }
#endif

  // For freq_threshold > 2, we would need the full counting approach
  if (freq_threshold > 2) {
    std::cerr << "Warning: freq_threshold > 2 not fully supported with two-set "
                 "optimization. Using threshold=2."
              << std::endl;
  }

  if (gfaz_debug_enabled()) {
    std::cerr << "\n--- 2-mer Collection Stats (Debug) ---" << std::endl;
    std::cerr << "Unique 2-mers (seen once): " << seen.size() << std::endl;
    std::cerr << "Repeated 2-mers (seen 2+): " << repeated.size() << std::endl;
#ifdef _OPENMP
    std::cerr << "OpenMP threads used: " << actual_threads << std::endl;
#endif
    std::cerr << "--------------------------------------" << std::endl;
  }

  // --- 2. Rule Creation Phase ---
  rules.kmer_to_rule_id.reserve(repeated.size());
  rules.rule_id_to_kmer.reserve(repeated.size());
  for (const auto &kmer : repeated) {
    rules.kmer_to_rule_id[kmer] = current_rule_id;
    rules.rule_id_to_kmer.push_back(
        kmer); // Vector index = rule_id - rules_start_id
    current_rule_id++;
  }

  rules.next_available_id = current_rule_id;

  return rules;
}

CompressionRules2Mer RuleGenerator::generate_rules_2mer_combined(
    const std::vector<std::vector<gfaz::NodeId>> &paths,
    const std::vector<std::vector<gfaz::NodeId>> &walks, uint32_t starting_id,
    size_t freq_threshold, int num_threads) {

  CompressionRules2Mer rules;
  rules.rules_start_id = starting_id;
  uint32_t current_rule_id = starting_id;

  std::vector<Packed2mer> repeated_pairs;

#ifdef _OPENMP
  const int actual_threads = resolve_omp_thread_count(num_threads);
  std::vector<robin_hood::unordered_flat_set<Packed2mer>> thread_seen(
      actual_threads);
  std::vector<robin_hood::unordered_flat_set<Packed2mer>> thread_repeated(
      actual_threads);
  const size_t total_sequences = paths.size() + walks.size();

  const auto t_scan0 = std::chrono::high_resolution_clock::now();
#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    auto &local_seen = thread_seen[tid];
    auto &local_repeated = thread_repeated[tid];

#pragma omp for schedule(dynamic)
    for (size_t p = 0; p < total_sequences; ++p) {
      const auto &seq =
          p < paths.size() ? paths[p] : walks[p - paths.size()];
      if (seq.size() < 2)
        continue;

      for (size_t i = 0; i + 1 < seq.size(); ++i) {
        const Packed2mer canonical =
            canonical_2mer(pack_2mer(seq[i], seq[i + 1]));
        if (local_repeated.count(canonical))
          continue;
        if (!local_seen.insert(canonical).second)
          local_repeated.insert(canonical);
      }
    }
  }

  const auto t_scan1 = std::chrono::high_resolution_clock::now();

  std::vector<size_t> offsets(static_cast<size_t>(actual_threads) + 1, 0);
  for (int t = 0; t < actual_threads; ++t) {
    offsets[static_cast<size_t>(t) + 1] =
        offsets[static_cast<size_t>(t)] + thread_seen[t].size() +
        thread_repeated[t].size();
  }

  std::vector<Packed2mer> merge_keys(offsets.back());
#pragma omp parallel for schedule(static)
  for (int t = 0; t < actual_threads; ++t) {
    size_t pos = offsets[static_cast<size_t>(t)];
    for (Packed2mer kmer : thread_seen[t])
      merge_keys[pos++] = kmer;
    for (Packed2mer kmer : thread_repeated[t])
      merge_keys[pos++] = kmer;
    thread_seen[t].clear();
    thread_repeated[t].clear();
  }
  thread_seen.clear();
  thread_repeated.clear();
  const auto t_collect = std::chrono::high_resolution_clock::now();

#if defined(__GLIBCXX__)
  __gnu_parallel::sort(merge_keys.begin(), merge_keys.end());
#else
  std::sort(merge_keys.begin(), merge_keys.end());
#endif
  const auto t_sort = std::chrono::high_resolution_clock::now();

  // Parallel duplicate extraction over the sorted keys. Chunk starts are
  // advanced to run boundaries so every run of equal keys is scanned by
  // exactly one chunk; sorted order is preserved in the output.
  const size_t num_keys = merge_keys.size();
  const size_t num_chunks =
      std::max<size_t>(1, std::min<size_t>(static_cast<size_t>(actual_threads) * 4,
                                           num_keys / 4096 + 1));
  std::vector<size_t> chunk_begin(num_chunks + 1, num_keys);
  for (size_t c = 0; c < num_chunks; ++c) {
    size_t begin = c * num_keys / num_chunks;
    while (begin > 0 && begin < num_keys &&
           merge_keys[begin] == merge_keys[begin - 1])
      ++begin;
    chunk_begin[c] = begin;
  }

  std::vector<size_t> chunk_count(num_chunks, 0);
#pragma omp parallel for schedule(static)
  for (size_t c = 0; c < num_chunks; ++c) {
    size_t count = 0;
    for (size_t i = chunk_begin[c]; i < chunk_begin[c + 1];) {
      size_t j = i + 1;
      while (j < chunk_begin[c + 1] && merge_keys[j] == merge_keys[i])
        ++j;
      if (j - i >= 2)
        ++count;
      i = j;
    }
    chunk_count[c] = count;
  }

  std::vector<size_t> chunk_offset(num_chunks + 1, 0);
  for (size_t c = 0; c < num_chunks; ++c)
    chunk_offset[c + 1] = chunk_offset[c] + chunk_count[c];

  repeated_pairs.resize(chunk_offset.back());
#pragma omp parallel for schedule(static)
  for (size_t c = 0; c < num_chunks; ++c) {
    size_t pos = chunk_offset[c];
    for (size_t i = chunk_begin[c]; i < chunk_begin[c + 1];) {
      size_t j = i + 1;
      while (j < chunk_begin[c + 1] && merge_keys[j] == merge_keys[i])
        ++j;
      if (j - i >= 2)
        repeated_pairs[pos++] = merge_keys[i];
      i = j;
    }
  }
  merge_keys.clear();
  merge_keys.shrink_to_fit();
  const auto t_dedup = std::chrono::high_resolution_clock::now();

  if (gfaz_debug_enabled()) {
    const auto ms = [](auto a, auto b) {
      return std::chrono::duration<double, std::milli>(b - a).count();
    };
    std::cerr << "    [generate sub] scan=" << ms(t_scan0, t_scan1)
              << " ms, collect=" << ms(t_scan1, t_collect)
              << " ms (keys=" << offsets.back() << ")"
              << ", sort=" << ms(t_collect, t_sort)
              << " ms, dedup=" << ms(t_sort, t_dedup) << " ms" << std::endl;
  }
#else
  robin_hood::unordered_flat_set<Packed2mer> seen;
  robin_hood::unordered_flat_set<Packed2mer> repeated;
  auto process_sequences = [&](const auto &sequences) {
    for (const auto &seq : sequences) {
      for (size_t i = 0; i + 1 < seq.size(); ++i) {
        const Packed2mer canonical =
            canonical_2mer(pack_2mer(seq[i], seq[i + 1]));
        if (repeated.count(canonical))
          continue;
        if (!seen.insert(canonical).second)
          repeated.insert(canonical);
      }
    }
  };
  process_sequences(paths);
  process_sequences(walks);
  repeated_pairs.assign(repeated.begin(), repeated.end());
#endif

  if (freq_threshold > 2) {
    std::cerr << "Warning: freq_threshold > 2 not fully supported with two-set "
                 "optimization. Using threshold=2."
              << std::endl;
  }

#ifdef _OPENMP
  // repeated_pairs is sorted, so rule IDs assigned by index reproduce the
  // exact ID order the serial map build used to produce. The flat map is
  // built in parallel; the robin_hood map stays empty on this path.
  rules.rule_id_to_kmer = std::move(repeated_pairs);
  rules.flat_map.build(rules.rule_id_to_kmer.data(),
                       rules.rule_id_to_kmer.size(), rules.rules_start_id);
  current_rule_id =
      rules.rules_start_id + static_cast<uint32_t>(rules.rule_id_to_kmer.size());
#else
  // Create rules from repeated 2-mers
  rules.kmer_to_rule_id.reserve(repeated_pairs.size());
  rules.rule_id_to_kmer.reserve(repeated_pairs.size());
  for (const auto &kmer : repeated_pairs) {
    rules.kmer_to_rule_id[kmer] = current_rule_id;
    rules.rule_id_to_kmer.push_back(
        kmer); // Vector index = rule_id - rules_start_id
    current_rule_id++;
  }
#endif

  rules.next_available_id = current_rule_id;

  return rules;
}
