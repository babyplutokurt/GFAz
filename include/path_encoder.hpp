#ifndef PATH_ENCODER_HPP
#define PATH_ENCODER_HPP

#include "gfa_parser.hpp"
#include "rule_generator.hpp"

class PathEncoder {
public:
    PathEncoder();

    // Encode paths using 2-mer rules
    void encode_paths_2mer(
        std::vector<std::vector<NodeId>>& paths, 
        const CompressionRules2Mer& rules, 
        std::vector<uint8_t>& rules_used);
};

#endif // PATH_ENCODER_HPP
