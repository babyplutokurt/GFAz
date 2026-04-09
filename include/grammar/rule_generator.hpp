#ifndef RULE_GENERATOR_HPP
#define RULE_GENERATOR_HPP

#include "grammar/packed_2mer.hpp"
#include "model/gfa_graph.hpp"
#include "robin_hood.h"
#include <vector>


// Struct to hold the generated compression rules for 2-mers
// Uses Packed2mer (int64_t) as key for faster hashing and operations
// rule_id_to_kmer uses vector for O(1) indexed access (index = rule_id -
// rules_start_id)
struct CompressionRules2Mer {
  robin_hood::unordered_flat_map<Packed2mer, uint32_t> kmer_to_rule_id;
  std::vector<Packed2mer> rule_id_to_kmer; // index = rule_id - rules_start_id
  uint32_t rules_start_id;
  uint32_t next_available_id;

  // Helper: get kmer for a rule_id (O(1) vector access)
  Packed2mer get_kmer(uint32_t rule_id) const {
    return rule_id_to_kmer[rule_id - rules_start_id];
  }

  // Helper: number of rules
  size_t num_rules() const { return rule_id_to_kmer.size(); }
};

class RuleGenerator {
public:
  RuleGenerator();

  // Generate 2-mer rules from paths
  CompressionRules2Mer
  generate_rules_2mer(const std::vector<std::vector<NodeId>> &paths,
                      uint32_t starting_id, size_t freq_threshold,
                      int num_threads = 0);

  // Generate 2-mer rules from both paths and walks (zero-copy)
  CompressionRules2Mer
  generate_rules_2mer_combined(const std::vector<std::vector<NodeId>> &paths,
                               const std::vector<std::vector<NodeId>> &walks,
                               uint32_t starting_id, size_t freq_threshold,
                               int num_threads = 0);
};

#endif // RULE_GENERATOR_HPP
