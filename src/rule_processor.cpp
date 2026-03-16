#include "rule_processor.hpp"
#include "packed_2mer.hpp"
#include "robin_hood.h"
#include <vector>

RuleProcessor::RuleProcessor() {}

// DEPRECATED: This function is no longer used in the fused compression workflow
// Kept for legacy compatibility
std::vector<uint32_t>
RuleProcessor::compact_rules_2mer(CompressionRules2Mer &rules,
                                  const std::vector<uint8_t> &rules_used) {

  if (rules_used.empty()) {
    rules.next_available_id = rules.rules_start_id;
    rules.kmer_to_rule_id.clear();
    rules.rule_id_to_kmer.clear();
    return {};
  }

  // Build prefix sum for compaction
  std::vector<uint32_t> prefix_sum(rules_used.size() + 1, 0);
  for (size_t i = 0; i < rules_used.size(); ++i) {
    prefix_sum[i + 1] = prefix_sum[i] + (rules_used[i] ? 1 : 0);
  }

  uint32_t total_used_rules = prefix_sum.back();

  // Build new compacted structures
  robin_hood::unordered_flat_map<Packed2mer, uint32_t> new_kmer_to_rule_id;
  std::vector<Packed2mer> new_rule_id_to_kmer;
  new_rule_id_to_kmer.reserve(total_used_rules);

  // Iterate over old vector: index = old_rule_id - rules_start_id
  for (size_t idx = 0; idx < rules.rule_id_to_kmer.size(); ++idx) {
    if (idx < rules_used.size() && rules_used[idx]) {
      Packed2mer kmer = rules.rule_id_to_kmer[idx];
      uint32_t new_rule_id = rules.rules_start_id + prefix_sum[idx];
      new_kmer_to_rule_id[kmer] = new_rule_id;
      new_rule_id_to_kmer.push_back(kmer);
    }
  }

  rules.next_available_id = rules.rules_start_id + total_used_rules;
  rules.rule_id_to_kmer = std::move(new_rule_id_to_kmer);
  rules.kmer_to_rule_id = std::move(new_kmer_to_rule_id);

  // Build id_map: index = old_rule_id - rules_start_id, value = new_rule_id
  std::vector<uint32_t> id_map(rules_used.size(), 0);
  for (size_t i = 0; i < rules_used.size(); ++i) {
    if (rules_used[i]) {
      id_map[i] = rules.rules_start_id + prefix_sum[i];
    }
  }

  return id_map;
}
