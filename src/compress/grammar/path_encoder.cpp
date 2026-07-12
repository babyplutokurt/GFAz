#include "compress/grammar/path_encoder.hpp"
#include "compress/grammar/packed_2mer.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

PathEncoder::PathEncoder() {}

void PathEncoder::encode_paths_2mer(std::vector<std::vector<gfaz::NodeId>> &paths,
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
    size_t encoded_size = 0;

    for (size_t i = 0; i < path.size(); ++i) {
      const gfaz::NodeId node = path[i];
      path[encoded_size++] = node;

      if (encoded_size >= 2) {
        // Get top 2 nodes
        int32_t first = path[encoded_size - 2];
        int32_t second = path[encoded_size - 1];
        Packed2mer top_kmer = pack_2mer(first, second);

        bool rule_found = false;

        // Check for forward rule
        auto it = rules.kmer_to_rule_id.find(top_kmer);
        if (it != rules.kmer_to_rule_id.end()) {
          uint32_t rule_id = it->second;
          int32_t oriented_rule_id = static_cast<int32_t>(rule_id);

          // Replace the last 2 nodes with the rule ID in place.
          path[encoded_size - 2] = oriented_rule_id;
          --encoded_size;

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

            // Replace the last 2 nodes with the rule ID in place.
            path[encoded_size - 2] = oriented_rule_id;
            --encoded_size;

            // Atomic write to rules_used (benign race - all threads write 1)
            rules_used[rule_id - rules.rules_start_id] = 1;
            rule_found = true;
          }
        }
      }
    }
    path.resize(encoded_size);
  }
}

