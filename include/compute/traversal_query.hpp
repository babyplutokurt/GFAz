#ifndef TRAVERSAL_QUERY_HPP
#define TRAVERSAL_QUERY_HPP

// The shared read-layer for the compute engine: everything a workflow needs to
// run directly on a CompressedData without materializing the original GFA. Used
// by growth, pav, and deconstruct, and the intended foundation for new compute
// modules. It provides:
//   - grammar rule expansion (build_rule_cache, stream_hap_leaves,
//     stream_decoded_nodes) that turns encoded P/W traversals back into node-id
//     streams, with an optional bounded leaf cache;
//   - haplotype slicing over the flat path/walk arrays (HapSlice, build_slices);
//   - PanSN path-name parsing and grouping keys (path_group_key / walk_group_key,
//     shared so every module groups haplotypes identically);
//   - W-line identity loading (load_walk_identity) and string-column
//     reconstruction (decompress_strings).
// A new module should build on these rather than re-deriving them.

#include "core/model/compressed_data.hpp"
#include "core/model/gfa_graph.hpp"
#include "compute/grouping_mode.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace gfaz {
namespace tquery {

// ---------------------------------------------------------------------------
// Grammar rule expansion
// ---------------------------------------------------------------------------

// Bottom-up cache of fully-expanded rule leaf sequences (forward direction).
// A rule is "ready" iff its leaf sequence fits in budget after recursively
// expanding its children. Reverse expansion is derived by iterating the cached
// leaves in reverse and negating each sign.
struct RuleLeafCache {
  std::vector<std::vector<int32_t>> forward;
  std::vector<uint8_t> ready;
  uint32_t min_rule_id = 0;
  uint32_t max_rule_id = 0;
  size_t bytes_used = 0;
  size_t budget_bytes = 0;
};

// Resolve a rule-cache byte budget from an environment variable, falling back
// to default_bytes when unset/invalid.
size_t resolve_rule_cache_budget(const char *env_name, size_t default_bytes);

// Populate cache.forward/ready for every rule that fits within the budget.
void build_rule_cache(RuleLeafCache &cache,
                      const std::vector<int32_t> &first,
                      const std::vector<int32_t> &second);

// Construct a ready-to-use RuleLeafCache for a rulebook: sets the id range,
// resolves the byte budget from budget_env (falling back to default_budget),
// sizes the cache, and populates it. The shared one-call setup every
// path-iterative module uses before decoding slices.
RuleLeafCache make_rule_cache(uint32_t min_rule_id,
                              const std::vector<int32_t> &rules_first,
                              const std::vector<int32_t> &rules_second,
                              const char *budget_env, size_t default_budget);

inline uint32_t abs_node_id(NodeId node) {
  return static_cast<uint32_t>(node < 0 ? -static_cast<int64_t>(node) : node);
}

template <typename Visitor>
void expand_rule_visit(uint32_t rule_id, bool reverse,
                       const std::vector<int32_t> &first,
                       const std::vector<int32_t> &second, uint32_t min_id,
                       uint32_t max_id, const RuleLeafCache &cache,
                       Visitor &visit) {
  const uint32_t idx = rule_id - min_id;
  if (cache.ready[idx]) {
    const std::vector<int32_t> &leaves = cache.forward[idx];
    if (!reverse) {
      for (int32_t leaf : leaves)
        visit(leaf);
    } else {
      for (auto it = leaves.rbegin(); it != leaves.rend(); ++it)
        visit(static_cast<int32_t>(-*it));
    }
    return;
  }

  const int32_t a = first[idx];
  const int32_t b = second[idx];

  if (!reverse) {
    const uint32_t abs_a = static_cast<uint32_t>(std::abs(a));
    if (abs_a >= min_id && abs_a < max_id)
      expand_rule_visit(abs_a, a < 0, first, second, min_id, max_id, cache,
                        visit);
    else
      visit(a);

    const uint32_t abs_b = static_cast<uint32_t>(std::abs(b));
    if (abs_b >= min_id && abs_b < max_id)
      expand_rule_visit(abs_b, b < 0, first, second, min_id, max_id, cache,
                        visit);
    else
      visit(b);
  } else {
    const uint32_t abs_b = static_cast<uint32_t>(std::abs(b));
    if (abs_b >= min_id && abs_b < max_id)
      expand_rule_visit(abs_b, b >= 0, first, second, min_id, max_id, cache,
                        visit);
    else
      visit(static_cast<int32_t>(-b));

    const uint32_t abs_a = static_cast<uint32_t>(std::abs(a));
    if (abs_a >= min_id && abs_a < max_id)
      expand_rule_visit(abs_a, a >= 0, first, second, min_id, max_id, cache,
                        visit);
    else
      visit(static_cast<int32_t>(-a));
  }
}

template <typename Visitor>
void stream_hap_leaves(const int32_t *encoded, size_t encoded_len,
                       uint32_t min_rule_id, uint32_t max_rule_id,
                       const std::vector<int32_t> &rules_first,
                       const std::vector<int32_t> &rules_second,
                       const RuleLeafCache &cache, Visitor &visit) {
  for (size_t i = 0; i < encoded_len; ++i) {
    const NodeId node = encoded[i];
    const uint32_t abs_id = static_cast<uint32_t>(std::abs(node));
    if (abs_id >= min_rule_id && abs_id < max_rule_id) {
      expand_rule_visit(abs_id, node < 0, rules_first, rules_second,
                        min_rule_id, max_rule_id, cache, visit);
    } else {
      visit(node);
    }
  }
}

// ---------------------------------------------------------------------------
// Haplotype slices over the flat path/walk arrays
// ---------------------------------------------------------------------------

struct HapSlice {
  const int32_t *encoded = nullptr;
  uint32_t enc_len = 0;
  uint32_t orig_len = 0;
};

void build_slices(const std::vector<int32_t> &flat,
                  const std::vector<uint32_t> &lengths,
                  const std::vector<uint32_t> &original_lengths,
                  std::vector<HapSlice> &out);

// Fully decode one haplotype slice (delta_round arbitrary) into signed node
// ids. Used for the general delta path; the streaming visitors below handle
// the common delta_round 0/1 cases without materializing the vector.
void decode_one_haplotype_general(const int32_t *encoded, size_t encoded_len,
                                  uint32_t original_len, int delta_round,
                                  uint32_t min_rule_id, uint32_t max_rule_id,
                                  const std::vector<int32_t> &rules_first,
                                  const std::vector<int32_t> &rules_second,
                                  const RuleLeafCache &cache,
                                  std::vector<NodeId> &decoded);

// Stream the decoded (delta-undone) signed node ids of a slice to a visitor.
template <typename Visitor>
void stream_decoded_nodes(const HapSlice &slice, int delta_round,
                          uint32_t min_rule_id, uint32_t max_rule_id,
                          const std::vector<int32_t> &rules_first,
                          const std::vector<int32_t> &rules_second,
                          const RuleLeafCache &cache,
                          std::vector<NodeId> &decoded, Visitor &visit) {
  if (delta_round == 0) {
    auto leaf_visit = [&](int32_t leaf) { visit(leaf); };
    stream_hap_leaves(slice.encoded, slice.enc_len, min_rule_id, max_rule_id,
                      rules_first, rules_second, cache, leaf_visit);
  } else if (delta_round == 1) {
    int32_t prev = 0;
    auto leaf_visit = [&](int32_t leaf) {
      prev += leaf;
      visit(prev);
    };
    stream_hap_leaves(slice.encoded, slice.enc_len, min_rule_id, max_rule_id,
                      rules_first, rules_second, cache, leaf_visit);
  } else {
    decode_one_haplotype_general(slice.encoded, slice.enc_len, slice.orig_len,
                                 delta_round, min_rule_id, max_rule_id,
                                 rules_first, rules_second, cache, decoded);
    for (NodeId node : decoded)
      visit(node);
  }
}

// ---------------------------------------------------------------------------
// PanSN path-name parsing and grouping keys
// ---------------------------------------------------------------------------

struct PansnParts {
  std::string sample;
  std::string hap;
  std::string seq;
  bool has_hap = false;
  bool has_seq = false;
};

void strip_pansn_coords_inplace(std::string &s);
PansnParts parse_pansn_path_name(const std::string &name);
std::string path_group_key(const std::string &name, GroupingMode mode);
std::string walk_group_key(const std::string &sample, uint32_t hap,
                           const std::string &seq, GroupingMode mode);
std::string walk_reference_name(const std::string &sample, uint32_t hap,
                                const std::string &seq, int64_t start,
                                int64_t end);

// ---------------------------------------------------------------------------
// Column reconstruction helpers
// ---------------------------------------------------------------------------

void reconstruct_strings(const std::string &concat,
                         const std::vector<uint32_t> &lengths,
                         std::vector<std::string> &out, const char *label);

std::vector<std::string> decompress_strings(const ZstdCompressedBlock &strings,
                                            const ZstdCompressedBlock &lengths,
                                            const char *label);

// ---------------------------------------------------------------------------
// W-line identity columns
// ---------------------------------------------------------------------------

// The per-walk identity columns (sample, seqid, hap, start, end) decoded from a
// CompressedData. Shared so every compute module reconstructs walk identity the
// same way.
struct WalkIdentityColumns {
  std::vector<std::string> samples;
  std::vector<std::string> seqs;
  std::vector<uint32_t> haps;
  std::vector<int64_t> starts;
  std::vector<int64_t> ends;
};

// Decode and validate the W-line identity columns; throws "<label>: walk
// metadata count mismatch" if any column does not have num_walks entries.
WalkIdentityColumns load_walk_identity(const CompressedData &data,
                                       size_t num_walks, const char *label);

} // namespace tquery
} // namespace gfaz

#endif // TRAVERSAL_QUERY_HPP
