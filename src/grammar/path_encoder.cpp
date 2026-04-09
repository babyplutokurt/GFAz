#include "grammar/path_encoder.hpp"
#include "grammar/packed_2mer.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

PathEncoder::PathEncoder() {}

void PathEncoder::encode_paths_2mer(std::vector<std::vector<NodeId>> &paths,
                                    const CompressionRules2Mer &rules,
                                    std::vector<uint8_t> &rules_used) {

  size_t num_rules = rules.next_available_id - rules.rules_start_id;
  // Only initialize if vector is empty or wrong size - don't reset existing
  // data
  if (rules_used.size() != num_rules) {
    rules_used.assign(num_rules, 0);
  }

#ifdef _OPENMP
// Parallel encoding of paths with dynamic scheduling
#pragma omp parallel for schedule(dynamic)

#endif
  for (size_t p = 0; p < paths.size(); ++p) {
    auto &path = paths[p];
    std::vector<NodeId> stack;
    stack.reserve(path.size());

    for (NodeId node : path) {
      stack.push_back(node);

      if (stack.size() >= 2) {
        // Get top 2 nodes
        int32_t first = stack[stack.size() - 2];
        int32_t second = stack[stack.size() - 1];
        Packed2mer top_kmer = pack_2mer(first, second);

        bool rule_found = false;

        // Check for forward rule
        auto it = rules.kmer_to_rule_id.find(top_kmer);
        if (it != rules.kmer_to_rule_id.end()) {
          uint32_t rule_id = it->second;
          int32_t oriented_rule_id = static_cast<int32_t>(rule_id);

          // Replace last 2 nodes with rule ID
          stack.pop_back();
          stack.pop_back();
          stack.push_back(oriented_rule_id);

          // Atomic write to rules_used (benign race - all threads write 1)
          rules_used[rule_id - rules.rules_start_id] = 1;
          rule_found = true;
        }

        // Check for reverse rule
        if (!rule_found) {
          Packed2mer rev_kmer = reverse_2mer(top_kmer);
          auto it_rev = rules.kmer_to_rule_id.find(rev_kmer);
          if (it_rev != rules.kmer_to_rule_id.end()) {
            uint32_t rule_id = it_rev->second;
            int32_t oriented_rule_id = -static_cast<int32_t>(rule_id);

            // Replace last 2 nodes with rule ID
            stack.pop_back();
            stack.pop_back();
            stack.push_back(oriented_rule_id);

            // Atomic write to rules_used (benign race - all threads write 1)
            rules_used[rule_id - rules.rules_start_id] = 1;
            rule_found = true;
          }
        }
      }
    }
    path = std::move(stack);
    path.shrink_to_fit();
  }
}

