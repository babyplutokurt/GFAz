#ifndef PATH_ENCODER_HPP
#define PATH_ENCODER_HPP

#include "grammar/rule_generator.hpp"
#include "model/gfa_graph.hpp"


class PathEncoder {
public:
    PathEncoder();

    // Encode paths using 2-mer rules
    void encode_paths_2mer(
        std::vector<std::vector<gfaz::NodeId>>& paths, 
        const CompressionRules2Mer& rules, 
        std::vector<uint8_t>& rules_used);
};

#endif // PATH_ENCODER_HPP
