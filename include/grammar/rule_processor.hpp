#ifndef RULE_PROCESSOR_HPP
#define RULE_PROCESSOR_HPP

#include "grammar/rule_generator.hpp"


class RuleProcessor {
public:
    RuleProcessor();

    // Compacts the 2-mer rule set based on which rules were actually used
    // Returns an id_map: old_id - rules_start_id → new_id
    std::vector<uint32_t> compact_rules_2mer(
        CompressionRules2Mer& rules, 
        const std::vector<uint8_t>& rules_used);
};

#endif // RULE_PROCESSOR_HPP

