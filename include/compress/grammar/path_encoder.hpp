#ifndef PATH_ENCODER_HPP
#define PATH_ENCODER_HPP

#include "compress/grammar/rule_generator.hpp"
#include "core/model/gfa_graph.hpp"

#include <functional>

class PathEncoder {
public:
    PathEncoder();

    // Encode paths using 2-mer rules
    void encode_paths_2mer(
        std::vector<std::vector<gfaz::NodeId>>& paths, 
        const CompressionRules2Mer& rules, 
        std::vector<uint8_t>& rules_used);

    // Encodes one traversal in place.
    using EncodeOne = std::function<void(std::vector<gfaz::NodeId> &)>;
    // Called once per index from a dynamically scheduled parallel loop; must
    // call `encode` on the traversal that belongs to `index`.
    using VisitIndex =
        std::function<void(size_t index, const EncodeOne &encode)>;

    // Same encoding as encode_paths_2mer for traversals that the caller
    // fetches itself (for example from an out-of-core store). `rules_used` is
    // accumulated like in encode_paths_2mer.
    void encode_indexed_2mer(size_t count, const CompressionRules2Mer &rules,
                             std::vector<uint8_t> &rules_used,
                             const VisitIndex &visit);

    // Reallocate a traversal that encoding left much smaller than its
    // capacity, so resident memory follows the encoded size.
    static void release_unused_capacity(std::vector<gfaz::NodeId> &path);
};

#endif // PATH_ENCODER_HPP
