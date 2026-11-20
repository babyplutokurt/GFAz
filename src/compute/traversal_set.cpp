#include "compute/traversal_set.hpp"

#include <stdexcept>
#include <string>

namespace gfaz {

TraversalSet::TraversalSet(const CompressedData &data)
    : rules_(tquery::load_rulebook(data)),
      traversals_(tquery::load_traversals(data)),
      num_paths_(data.sequence_lengths.size()),
      delta_rounds_(data.delta_round) {
  if (delta_rounds_ < 0 || delta_rounds_ > DeltaState::kMaxRounds)
    throw std::runtime_error(
        "traversal set: unsupported delta round count " +
        std::to_string(delta_rounds_) + " (maximum " +
        std::to_string(DeltaState::kMaxRounds) + ")");
}

} // namespace gfaz
