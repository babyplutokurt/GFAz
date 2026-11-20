#ifndef COMPUTE_TRAVERSAL_SET_HPP
#define COMPUTE_TRAVERSAL_SET_HPP

// Random-access reader over the encoded P/W traversals of a container.
//
// tquery::stream_decoded_nodes decodes a haplotype from its first element.
// TraversalSet decodes any span of top-level elements, given the delta state at
// the start of the span, so a caller that keeps checkpoints (see PathIndex) can
// start in the middle of a haplotype. Haplotypes are numbered [paths... walks...],
// the same order as tquery::LoadedTraversals::slices.

#include "compute/traversal_query.hpp"
#include "core/model/compressed_data.hpp"
#include "core/model/gfa_graph.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace gfaz {

// Running prefix sums that undo the delta transform one value at a time. A
// value-initialized state is the state before the first element of a haplotype.
struct DeltaState {
  static constexpr int kMaxRounds = 4;
  std::array<int32_t, kMaxRounds> sums{};
};

class TraversalSet {
public:
  // Decodes the rulebook and both traversal blocks. Throws if the container
  // uses more delta rounds than DeltaState can hold.
  explicit TraversalSet(const CompressedData &data);

  size_t size() const { return traversals_.slices.size(); }
  size_t num_paths() const { return num_paths_; }
  bool is_walk(size_t hap) const { return hap >= num_paths_; }

  // Number of top-level encoded elements, and of decoded steps.
  uint32_t encoded_length(size_t hap) const {
    return traversals_.slices[hap].enc_len;
  }
  uint32_t step_count(size_t hap) const {
    return traversals_.slices[hap].orig_len;
  }

  int delta_rounds() const { return delta_rounds_; }
  const tquery::Rulebook &rules() const { return rules_; }

  // Decode elements [enc_begin, enc_end) of a haplotype and pass every signed
  // node id to visit(NodeId). `state` must be the delta state at enc_begin and
  // is left at the state for enc_end.
  template <typename Visitor>
  void decode(size_t hap, uint32_t enc_begin, uint32_t enc_end,
              DeltaState &state, Visitor &&visit) const {
    const int32_t *encoded = traversals_.slices[hap].encoded;
    const int rounds = delta_rounds_;
    auto leaf_visit = [&](int32_t value) {
      for (int r = 0; r < rounds; ++r) {
        state.sums[r] += value;
        value = state.sums[r];
      }
      visit(static_cast<NodeId>(value));
    };
    for (uint32_t i = enc_begin; i < enc_end; ++i) {
      const int32_t element = encoded[i];
      const uint32_t abs_id = static_cast<uint32_t>(std::abs(element));
      if (abs_id >= rules_.min_rule_id && abs_id < rules_.max_rule_id)
        tquery::expand_rule_leaves(abs_id, element < 0, rules_, leaf_visit);
      else
        leaf_visit(element);
    }
  }

  // Decode a whole haplotype.
  template <typename Visitor> void decode_all(size_t hap, Visitor &&visit) const {
    DeltaState state;
    decode(hap, 0, encoded_length(hap), state, visit);
  }

private:
  tquery::Rulebook rules_;
  tquery::LoadedTraversals traversals_;
  size_t num_paths_ = 0;
  int delta_rounds_ = 0;
};

} // namespace gfaz

#endif // COMPUTE_TRAVERSAL_SET_HPP
