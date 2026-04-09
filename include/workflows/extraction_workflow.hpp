#ifndef EXTRACTION_WORKFLOW_HPP
#define EXTRACTION_WORKFLOW_HPP

#include "workflows/compression_workflow.hpp"

#include <cstdint>
#include <string>
#include <vector>


struct WalkLookupKey {
  std::string sample_id;
  uint32_t hap_index = 0;
  std::string seq_id;
  int64_t seq_start = -1;
  int64_t seq_end = -1;
};

std::vector<std::string>
extract_path_lines_by_name(const CompressedData &data,
                           const std::vector<std::string> &path_names,
                           int num_threads = 0);

std::string extract_path_line_by_name(const CompressedData &data,
                                      const std::string &path_name,
                                      int num_threads = 0);

std::string extract_walk_line(const CompressedData &data,
                              const std::string &sample_id,
                              uint32_t hap_index,
                              const std::string &seq_id,
                              int64_t seq_start,
                              int64_t seq_end,
                              int num_threads = 0);

std::vector<std::string>
extract_walk_lines(const CompressedData &data,
                   const std::vector<WalkLookupKey> &walk_keys,
                   int num_threads = 0);

std::vector<std::string>
extract_walk_lines_by_name(const CompressedData &data,
                           const std::vector<std::string> &walk_names,
                           int num_threads = 0);

std::string extract_walk_line_by_name(const CompressedData &data,
                                      const std::string &walk_name,
                                      int num_threads = 0);

#endif

