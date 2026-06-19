#ifndef PAV_WORKFLOW_HPP
#define PAV_WORKFLOW_HPP

#include "core/defaults.hpp"
#include "core/model/compressed_data.hpp"
#include "compute/grouping_mode.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace gfaz {

struct PavOptions {
  std::string bed_path;
  int num_threads = kDefaultNumThreads;
  GroupingMode grouping = GroupingMode::PerPathWalk;
  bool matrix_output = false;
  bool emit_binary = false;
  double binary_threshold = 0.0;
};

struct PavRange {
  std::string chrom;
  uint64_t start = 0;
  uint64_t end = 0;
  std::string name;
};

struct PavResult {
  std::vector<PavRange> ranges;
  std::vector<std::string> group_names;
  std::vector<uint64_t> denominators;
  // Row-major: range index * group_names.size() + group index.
  std::vector<uint64_t> numerators;
};

PavResult compute_pav(const CompressedData &data, const PavOptions &options);

} // namespace gfaz

#endif
