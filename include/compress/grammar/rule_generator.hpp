#ifndef RULE_GENERATOR_HPP
#define RULE_GENERATOR_HPP

#include "compress/grammar/packed_2mer.hpp"
#include "compress/grammar/rule_flat_map.hpp"
#include "core/model/gfa_graph.hpp"
#include "robin_hood.h"
#include <functional>
#include <vector>


// Struct to hold the generated compression rules for 2-mers
// Uses Packed2mer (int64_t) as key for faster hashing and operations
// rule_id_to_kmer uses vector for O(1) indexed access (index = rule_id -
// rules_start_id)
// The main compression workflow fills flat_map (parallel-built lookup);
// kmer_to_rule_id remains for callers that assemble rules incrementally
// (add_haplotypes, legacy paths). Encoding prefers flat_map when non-empty.
struct CompressionRules2Mer {
  robin_hood::unordered_flat_map<Packed2mer, uint32_t> kmer_to_rule_id;
  RuleFlatMap flat_map;
  std::vector<Packed2mer> rule_id_to_kmer; // index = rule_id - rules_start_id
  uint32_t rules_start_id;
  uint32_t next_available_id;

  // Helper: number of rules
  size_t num_rules() const { return rule_id_to_kmer.size(); }
};

// Returns traversal `index` (paths first, then walks) for a scan. Traversals
// that are not held in memory are decoded into `scratch` and that is returned.
using TraversalFetch = std::function<const std::vector<gfaz::NodeId> &(
    size_t index, std::vector<gfaz::NodeId> &scratch)>;

// Length of traversal `index` without loading it.
using TraversalLength = std::function<size_t(size_t index)>;

class RuleGenerator {
public:
  RuleGenerator();

  // Generate 2-mer rules from paths
  CompressionRules2Mer
  generate_rules_2mer(const std::vector<std::vector<gfaz::NodeId>> &paths,
                      uint32_t starting_id, size_t freq_threshold,
                      int num_threads = 0);

  // Generate 2-mer rules from both paths and walks (zero-copy).
  // `scan_keys`, when given, carries the number of keys the previous round
  // sorted in and this round's count out; the generator uses it to pick the
  // cheaper way to find repeated 2-mers. It never changes the rules produced.
  //
  // `min_new_id` > 0 says the sequences are the output of a grammar round
  // whose rules got IDs >= min_new_id. A 2-mer of two older IDs was then
  // already adjacent before that round and was not replaced, so it occurred
  // once and still does: only 2-mers containing an ID >= min_new_id are
  // looked at. The rules are the same as with min_new_id = 0.
  CompressionRules2Mer
  generate_rules_2mer_combined(const std::vector<std::vector<gfaz::NodeId>> &paths,
                               const std::vector<std::vector<gfaz::NodeId>> &walks,
                               uint32_t starting_id, size_t freq_threshold,
                               int num_threads = 0,
                               const TraversalFetch *fetch = nullptr,
                               size_t *scan_keys = nullptr,
                               const TraversalLength *fetch_length = nullptr,
                               uint32_t min_new_id = 0);
};

#endif // RULE_GENERATOR_HPP
