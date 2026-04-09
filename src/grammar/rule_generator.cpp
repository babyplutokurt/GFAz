#include "grammar/rule_generator.hpp"
#include "utils/debug_log.hpp"
#include "robin_hood.h"
#include "utils/threading_utils.hpp"
#include <cstdlib>
#include <iostream>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

RuleGenerator::RuleGenerator() {}

// DEPRECATED: This function is kept for legacy compatibility but is not used in
// the current workflow. See generate_rules_2mer_combined below.
CompressionRules2Mer RuleGenerator::generate_rules_2mer(
    const std::vector<std::vector<NodeId>> &paths, uint32_t starting_id,
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
        if (local_seen.count(canonical)) {
          local_repeated.insert(canonical);
        } else {
          local_seen.insert(canonical);
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
      } else if (seen.count(kmer)) {
        repeated.insert(kmer); // Seen by another thread → repeated!
      } else {
        seen.insert(kmer); // First global occurrence
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

      if (seen.count(canonical)) {
        repeated.insert(canonical);
      } else {
        seen.insert(canonical);
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
    const std::vector<std::vector<NodeId>> &paths,
    const std::vector<std::vector<NodeId>> &walks, uint32_t starting_id,
    size_t freq_threshold, int num_threads) {

  CompressionRules2Mer rules;
  rules.rules_start_id = starting_id;
  uint32_t current_rule_id = starting_id;

  // Global sets for final result - using Packed2mer (int64_t) for faster
  // hashing
  robin_hood::unordered_flat_set<Packed2mer> seen;
  robin_hood::unordered_flat_set<Packed2mer> repeated;

  // Helper lambda to process a sequence vector
  auto process_sequences =
      [&](const std::vector<std::vector<NodeId>> &sequences) {
#ifdef _OPENMP
        int actual_threads = resolve_omp_thread_count(num_threads);

        std::vector<robin_hood::unordered_flat_set<Packed2mer>> thread_seen(
            actual_threads);
        std::vector<robin_hood::unordered_flat_set<Packed2mer>> thread_repeated(
            actual_threads);

#pragma omp parallel
        {
          int tid = omp_get_thread_num();
          auto &local_seen = thread_seen[tid];
          auto &local_repeated = thread_repeated[tid];

#pragma omp for schedule(dynamic)

          for (size_t p = 0; p < sequences.size(); ++p) {
            const auto &seq = sequences[p];
            if (seq.size() < 2)
              continue;

            for (size_t i = 0; i <= seq.size() - 2; ++i) {
              Packed2mer kmer = pack_2mer(seq[i], seq[i + 1]);
              Packed2mer canonical = canonical_2mer(kmer);

              if (local_repeated.count(canonical))
                continue;

              if (local_seen.count(canonical)) {
                local_repeated.insert(canonical);
              } else {
                local_seen.insert(canonical);
              }
            }
          }
        }

        // Merge thread results into global sets
        for (int t = 0; t < actual_threads; ++t) {
          for (const auto &kmer : thread_repeated[t]) {
            repeated.insert(kmer);
          }
          for (const auto &kmer : thread_seen[t]) {
            if (repeated.count(kmer))
              continue;
            if (seen.count(kmer)) {
              repeated.insert(kmer);
            } else {
              seen.insert(kmer);
            }
          }
        }
#else
        for (const auto &seq : sequences) {
          if (seq.size() < 2)
            continue;
          for (size_t i = 0; i <= seq.size() - 2; ++i) {
            Packed2mer kmer = pack_2mer(seq[i], seq[i + 1]);
            Packed2mer canonical = canonical_2mer(kmer);
            if (repeated.count(canonical))
              continue;
            if (seen.count(canonical)) {
              repeated.insert(canonical);
            } else {
              seen.insert(canonical);
            }
          }
        }
#endif
      };

  // Process paths
  process_sequences(paths);

  // Process walks (adds to same seen/repeated sets)
  process_sequences(walks);

  if (freq_threshold > 2) {
    std::cerr << "Warning: freq_threshold > 2 not fully supported with two-set "
                 "optimization. Using threshold=2."
              << std::endl;
  }

  // Create rules from repeated 2-mers
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

