#ifndef DECOMPRESSION_WORKFLOW_HPP
#define DECOMPRESSION_WORKFLOW_HPP

#include "core/model/compressed_data.hpp"
#include "core/model/gfa_graph.hpp"


void decompress_gfa(const gfaz::CompressedData &data, gfaz::GfaGraph &output_graph,
                    int num_threads = 0);

#endif
