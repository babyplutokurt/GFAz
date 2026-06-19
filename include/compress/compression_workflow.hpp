#ifndef COMPRESSION_WORKFLOW_HPP
#define COMPRESSION_WORKFLOW_HPP

#include "core/model/compressed_data.hpp"
#include "core/utils/threading_utils.hpp"

gfaz::CompressedData compress_gfa(const std::string &gfa_file_path, int num_rounds,
                            size_t freq_threshold, int delta_round,
                            int num_threads = 0, bool show_stats = false);

#endif
