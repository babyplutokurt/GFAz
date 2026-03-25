#ifndef EXTRACTION_WORKFLOW_HPP
#define EXTRACTION_WORKFLOW_HPP

#include "compression_workflow.hpp"

#include <cstdint>
#include <string>

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

std::string extract_walk_line_by_name(const CompressedData &data,
                                      const std::string &walk_name,
                                      int num_threads = 0);

#endif
