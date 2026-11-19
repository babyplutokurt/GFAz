#include "compress/grammar/rule_generator.hpp"
#include "compress/grammar/kmer_seen_table.hpp"
#include "compress/grammar/key_sort.hpp"
#include "core/utils/debug_log.hpp"
#include "robin_hood.h"
#include "core/utils/threading_utils.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string_view>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(_OPENMP) && defined(__GLIBCXX__)
#include <parallel/algorithm>
#endif

RuleGenerator::RuleGenerator() {}

namespace {

inline uint32_t abs_node_id(gfaz::NodeId id) {
  return id < 0 ? 0u - static_cast<uint32_t>(id) : static_cast<uint32_t>(id);
}

// Inserts the 2-mers of `seq` that contain an ID >= min_new_id and returns
// how many were left out. Inserts are independent of each other, so a block
// of 2-mers is hashed and its slots start loading before the inserts.
size_t insert_2mers(KmerSeenTable &table, const std::vector<gfaz::NodeId> &seq,
                    uint32_t min_new_id) {
  if (seq.size() < 2)
    return 0;
  constexpr size_t kBlock = 32;
  Packed2mer keys[kBlock];
  uint64_t hashes[kBlock];
  size_t skipped = 0;
  const size_t pairs = seq.size() - 1;
  for (size_t begin = 0; begin < pairs; begin += kBlock) {
    const size_t block_end = std::min(begin + kBlock, pairs);
    size_t count = 0;
    for (size_t i = begin; i < block_end; ++i) {
      if (std::max(abs_node_id(seq[i]), abs_node_id(seq[i + 1])) < min_new_id)
        continue;
      keys[count] = canonical_2mer(pack_2mer(seq[i], seq[i + 1]));
      hashes[count] = KmerSeenTable::hash(keys[count]);
      table.prefetch(hashes[count]);
      ++count;
    }
    skipped += block_end - begin - count;
    for (size_t k = 0; k < count; ++k)
      table.insert(keys[k], hashes[k]);
  }
  return skipped;
}

} // namespace

struct NextRoundScan::ThreadState {
  KmerSeenTable table;
  size_t skipped = 0;
  bool reserved = false;
};

NextRoundScan::NextRoundScan(uint32_t min_new_id, size_t expected_keys)
    : min_new_id_(min_new_id) {
  int threads = 1;
#ifdef _OPENMP
  threads = std::max(1, omp_get_max_threads());
#endif
  expected_per_thread_ = expected_keys / static_cast<size_t>(threads) / 2;
  threads_.resize(static_cast<size_t>(threads));
  for (auto &state : threads_)
    state = std::make_unique<ThreadState>();
}

NextRoundScan::~NextRoundScan() = default;

bool NextRoundScan::ids_fit(uint32_t id_limit) {
  return KmerSeenTable::ids_fit(id_limit);
}

void NextRoundScan::add(const std::vector<gfaz::NodeId> &sequence) {
  size_t thread = 0;
#ifdef _OPENMP
  thread = static_cast<size_t>(omp_get_thread_num());
#endif
  ThreadState &state = *threads_[thread];
  if (!state.reserved) {
    state.table.reserve(expected_per_thread_);
    state.reserved = true;
  }
  state.skipped += insert_2mers(state.table, sequence, min_new_id_);
}

NextRoundScan::Keys NextRoundScan::finish(uint32_t rules_start_id,
                                          const std::vector<uint32_t> &id_map) {
  Keys result;
  std::vector<size_t> offsets(threads_.size() + 1, 0);
  size_t skipped = 0;
  for (size_t t = 0; t < threads_.size(); ++t) {
    offsets[t + 1] = offsets[t] + threads_[t]->table.size() +
                     threads_[t]->table.repeated();
    skipped += threads_[t]->skipped;
  }
  result.count = offsets.back();
  result.unfiltered_count = result.count + skipped;
  result.keys.reset(new Packed2mer[result.count]);

  const auto final_id = [&](gfaz::NodeId id) -> gfaz::NodeId {
    const uint32_t abs_id = abs_node_id(id);
    if (abs_id < rules_start_id)
      return id;
    const uint32_t mapped = id_map[abs_id - rules_start_id];
    return id > 0 ? static_cast<gfaz::NodeId>(mapped)
                  : -static_cast<gfaz::NodeId>(mapped);
  };
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1)
#endif
  for (size_t t = 0; t < threads_.size(); ++t) {
    Packed2mer *out = result.keys.get() + offsets[t];
    const size_t written = threads_[t]->table.emit(out);
    threads_[t]->table.release();
    for (size_t k = 0; k < written; ++k)
      out[k] = pack_2mer(final_id(unpack_first(out[k])),
                         final_id(unpack_second(out[k])));
  }
  threads_.clear();
  return result;
}

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
    size_t freq_threshold, int num_threads, const TraversalFetch *fetch,
    size_t *scan_keys, const TraversalLength *fetch_length,
    uint32_t min_new_id, NextRoundScan::Keys *collected) {

  CompressionRules2Mer rules;
  rules.rules_start_id = starting_id;
  uint32_t current_rule_id = starting_id;

  std::vector<Packed2mer> repeated_pairs;

#ifdef _OPENMP
  const int actual_threads = resolve_omp_thread_count(num_threads);
  const size_t total_sequences = paths.size() + walks.size();
  const auto sequence_at = [&](size_t p) -> const std::vector<gfaz::NodeId> & {
    return p < paths.size() ? paths[p] : walks[p - paths.size()];
  };

  // The rules are the 2-mers that occur at least twice, in sorted order. The
  // per-thread sets below exist to shrink what has to be sorted: they drop
  // repeats within a thread. Once most 2-mers are distinct (the later rounds)
  // the sets drop little and cost more than sorting the extra keys, so every
  // 2-mer is written straight to the sort buffer instead. Both routes give
  // the same sorted list of keys that occur twice or more. The choice uses the
  // previous round's key count, since 2-mers only get more distinct per round.
  //
  // With min_new_id > 0 only 2-mers that contain an ID >= min_new_id are
  // collected (see the header for why the others cannot repeat).
  const auto may_repeat = [min_new_id](gfaz::NodeId a, gfaz::NodeId b) {
    const uint32_t abs_a = a < 0 ? 0u - static_cast<uint32_t>(a)
                                 : static_cast<uint32_t>(a);
    const uint32_t abs_b = b < 0 ? 0u - static_cast<uint32_t>(b)
                                 : static_cast<uint32_t>(b);
    return std::max(abs_a, abs_b) >= min_new_id;
  };
  size_t total_2mers = 0;
  bool emit_all = false;
  // GFAZ_RULE_SCAN=sets|all pins the route; tests use it to check that both
  // produce the same output.
  const char *forced_scan = std::getenv("GFAZ_RULE_SCAN");
  const bool force_legacy_sets =
      forced_scan != nullptr && std::string_view(forced_scan) == "sets-legacy";
  const bool force_sets =
      force_legacy_sets ||
      (forced_scan != nullptr && std::string_view(forced_scan) == "sets");
  const bool force_all =
      forced_scan != nullptr && std::string_view(forced_scan) == "all";
  // With fetched traversals the offsets need each length up front.
  if (collected == nullptr &&
      (fetch == nullptr || fetch_length != nullptr) && !force_sets &&
      (force_all || (scan_keys != nullptr && *scan_keys > 0))) {
    for (size_t p = 0; p < total_sequences; ++p) {
      const size_t length =
          fetch != nullptr ? (*fetch_length)(p) : sequence_at(p).size();
      total_2mers += length < 2 ? 0 : length - 1;
    }
    emit_all = force_all || total_2mers / 2 <= *scan_keys;
  }

  // The hash route keeps one table per thread. KmerSeenTable needs one probe
  // per 2-mer but only holds IDs below 2^30; beyond that (or with
  // GFAZ_RULE_SCAN=sets-legacy) two robin_hood sets per thread do the same.
  // Every ID in the sequences is below starting_id.
  const bool use_seen_tables = collected == nullptr && !emit_all &&
                               !force_legacy_sets &&
                               KmerSeenTable::ids_fit(starting_id);
  const bool use_legacy_sets =
      collected == nullptr && !emit_all && !use_seen_tables;
  std::vector<KmerSeenTable> thread_tables(use_seen_tables ? actual_threads : 0);
  std::vector<robin_hood::unordered_flat_set<Packed2mer>> thread_seen(
      use_legacy_sets ? actual_threads : 0);
  std::vector<robin_hood::unordered_flat_set<Packed2mer>> thread_repeated(
      use_legacy_sets ? actual_threads : 0);
  size_t num_keys = 0;
  // 2-mers the hash routes left out because of min_new_id. Each of them occurs
  // exactly once, so it would have been one more key.
  size_t skipped_keys = 0;
  // Uninitialized so the pages are first touched by the parallel writers.
  std::unique_ptr<Packed2mer[]> merge_keys;

  const auto t_scan0 = std::chrono::high_resolution_clock::now();
  if (collected != nullptr) {
    merge_keys = std::move(collected->keys);
    num_keys = collected->count;
    skipped_keys = collected->unfiltered_count - collected->count;
  } else if (emit_all) {
    // Threads take fixed-size pieces of the buffer as they go, so 2-mers that
    // are filtered out cost no space; the sort does not care about the order.
    // The buffer is sized for every 2-mer but only written pages get mapped.
    constexpr size_t kPiece = size_t{1} << 16;
    merge_keys.reset(new Packed2mer[
        total_2mers + static_cast<size_t>(actual_threads) * kPiece]);
    std::atomic<size_t> next_piece{0};
    // Unused tail of each thread's last piece: [begin, end).
    std::vector<std::pair<size_t, size_t>> gaps(actual_threads, {0, 0});
#pragma omp parallel
    {
      std::vector<gfaz::NodeId> scratch;
      Packed2mer *out = merge_keys.get();
      Packed2mer *out_end = out;
#pragma omp for schedule(dynamic)
      for (size_t p = 0; p < total_sequences; ++p) {
        const auto &seq =
            fetch != nullptr ? (*fetch)(p, scratch) : sequence_at(p);
        for (size_t i = 0; i + 1 < seq.size(); ++i) {
          if (!may_repeat(seq[i], seq[i + 1]))
            continue;
          if (out == out_end) {
            out = merge_keys.get() + next_piece.fetch_add(kPiece);
            out_end = out + kPiece;
          }
          *out++ = canonical_2mer(pack_2mer(seq[i], seq[i + 1]));
        }
      }
      gaps[omp_get_thread_num()] = {
          static_cast<size_t>(out - merge_keys.get()),
          static_cast<size_t>(out_end - merge_keys.get())};
    }
    // Close the gaps with keys taken from the end of the used range.
    size_t used_end = next_piece.load();
    size_t gap_total = 0;
    for (const auto &gap : gaps)
      gap_total += gap.second - gap.first;
    num_keys = used_end - gap_total;
    std::sort(gaps.begin(), gaps.end());
    size_t last_gap = gaps.size();
    for (size_t g = 0; g < gaps.size(); ++g) {
      for (size_t hole = gaps[g].first;
           hole < gaps[g].second && hole < num_keys; ++hole) {
        // Step back to the last position that holds a key.
        for (;;) {
          --used_end;
          while (last_gap > 0 && gaps[last_gap - 1].first > used_end)
            --last_gap;
          if (last_gap == 0 || used_end >= gaps[last_gap - 1].second)
            break;
        }
        merge_keys[hole] = merge_keys[used_end];
      }
    }
  } else if (use_seen_tables) {
    // Start each table at half the previous round's keys per thread: random
    // inserts make the whole table resident, so a generous start costs memory,
    // while a low one costs a single extra doubling.
    const size_t expected_per_thread =
        scan_keys != nullptr
            ? *scan_keys / static_cast<size_t>(actual_threads) / 2
            : 0;
#pragma omp parallel reduction(+ : skipped_keys)
    {
      KmerSeenTable &table = thread_tables[omp_get_thread_num()];
      table.reserve(expected_per_thread);
      std::vector<gfaz::NodeId> scratch;

#pragma omp for schedule(dynamic)
      for (size_t p = 0; p < total_sequences; ++p) {
        const auto &seq =
            fetch != nullptr ? (*fetch)(p, scratch) : sequence_at(p);
        skipped_keys += insert_2mers(table, seq, min_new_id);
      }
    }
  } else {
#pragma omp parallel reduction(+ : skipped_keys)
    {
      const int tid = omp_get_thread_num();
      auto &local_seen = thread_seen[tid];
      auto &local_repeated = thread_repeated[tid];
      std::vector<gfaz::NodeId> scratch;

#pragma omp for schedule(dynamic)
      for (size_t p = 0; p < total_sequences; ++p) {
        const auto &seq =
            fetch != nullptr ? (*fetch)(p, scratch) : sequence_at(p);
        if (seq.size() < 2)
          continue;

        for (size_t i = 0; i + 1 < seq.size(); ++i) {
          if (!may_repeat(seq[i], seq[i + 1])) {
            ++skipped_keys;
            continue;
          }
          const Packed2mer canonical =
              canonical_2mer(pack_2mer(seq[i], seq[i + 1]));
          if (local_repeated.count(canonical))
            continue;
          if (!local_seen.insert(canonical).second)
            local_repeated.insert(canonical);
        }
      }
    }
  }
  const auto t_scan1 = std::chrono::high_resolution_clock::now();

  if (!emit_all && collected == nullptr) {
    std::vector<size_t> offsets(static_cast<size_t>(actual_threads) + 1, 0);
    for (int t = 0; t < actual_threads; ++t) {
      offsets[static_cast<size_t>(t) + 1] =
          offsets[static_cast<size_t>(t)] +
          (use_seen_tables
               ? thread_tables[t].size() + thread_tables[t].repeated()
               : thread_seen[t].size() + thread_repeated[t].size());
    }

    num_keys = offsets.back();
    merge_keys.reset(new Packed2mer[num_keys]);
#pragma omp parallel for schedule(static)
    for (int t = 0; t < actual_threads; ++t) {
      size_t pos = offsets[static_cast<size_t>(t)];
      if (use_seen_tables) {
        thread_tables[t].emit(merge_keys.get() + pos);
        thread_tables[t].release();
        continue;
      }
      for (Packed2mer kmer : thread_seen[t])
        merge_keys[pos++] = kmer;
      for (Packed2mer kmer : thread_repeated[t])
        merge_keys[pos++] = kmer;
      thread_seen[t].clear();
      thread_repeated[t].clear();
    }
    thread_seen.clear();
    thread_repeated.clear();
  }
  // The next round compares its 2-mer count against this. Report what the
  // count would be without min_new_id, so the filter does not move the point
  // where rounds switch to the emit-all route.
  if (scan_keys != nullptr)
    *scan_keys = emit_all ? total_2mers : num_keys + skipped_keys;
  const auto t_collect = std::chrono::high_resolution_clock::now();

  gfaz::parallel_sort_keys(merge_keys.get(), num_keys, actual_threads);
  const auto t_sort = std::chrono::high_resolution_clock::now();

  // Parallel duplicate extraction over the sorted keys. Chunk starts are
  // advanced to run boundaries so every run of equal keys is scanned by
  // exactly one chunk; sorted order is preserved in the output.
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
  merge_keys.reset();
  const auto t_dedup = std::chrono::high_resolution_clock::now();

  if (gfaz_debug_enabled()) {
    const auto ms = [](auto a, auto b) {
      return std::chrono::duration<double, std::milli>(b - a).count();
    };
    std::cerr << "    [generate sub] scan=" << ms(t_scan0, t_scan1)
              << " ms, collect=" << ms(t_scan1, t_collect)
              << " ms (keys=" << num_keys
              << (emit_all ? ", all 2-mers)"
                           : collected != nullptr ? ", collected in encode)"
                                                  : ")")
              << ", sort=" << ms(t_collect, t_sort)
              << " ms, dedup=" << ms(t_sort, t_dedup) << " ms" << std::endl;
  }
#else
  robin_hood::unordered_flat_set<Packed2mer> seen;
  robin_hood::unordered_flat_set<Packed2mer> repeated;
  auto process_sequences = [&](const auto &sequences) {
    for (const auto &seq : sequences) {
      for (size_t i = 0; i + 1 < seq.size(); ++i) {
        const uint32_t abs_a = static_cast<uint32_t>(std::abs(seq[i]));
        const uint32_t abs_b = static_cast<uint32_t>(std::abs(seq[i + 1]));
        if (std::max(abs_a, abs_b) < min_new_id)
          continue;
        const Packed2mer canonical =
            canonical_2mer(pack_2mer(seq[i], seq[i + 1]));
        if (repeated.count(canonical))
          continue;
        if (!seen.insert(canonical).second)
          repeated.insert(canonical);
      }
    }
  };
  if (fetch != nullptr) {
    std::vector<gfaz::NodeId> scratch;
    std::vector<std::vector<gfaz::NodeId>> one(1);
    for (size_t p = 0; p < paths.size() + walks.size(); ++p) {
      one[0] = (*fetch)(p, scratch);
      process_sequences(one);
    }
  } else {
    process_sequences(paths);
    process_sequences(walks);
  }
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
