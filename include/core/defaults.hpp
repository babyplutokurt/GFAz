#ifndef GFAZ_CORE_DEFAULTS_HPP
#define GFAZ_CORE_DEFAULTS_HPP

// Shared default parameters for the compressor and compute-engine libraries.
// Kept in the core foundation so library headers never have to reach up into
// the CLI layer (cli/common.hpp) just to name a default.

namespace gfaz {

constexpr int kDefaultRounds = 8;
constexpr int kDefaultDeltaRound = 1;
constexpr int kDefaultFreqThreshold = 2;
constexpr int kDefaultNumThreads = 0;

} // namespace gfaz

#endif // GFAZ_CORE_DEFAULTS_HPP
