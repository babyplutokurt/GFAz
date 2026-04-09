#ifndef DECOMPRESSION_WORKFLOW_HPP
#define DECOMPRESSION_WORKFLOW_HPP

#include "model/compressed_data.hpp"
#include "model/gfa_graph.hpp"


void decompress_gfa(const CompressedData &data, GfaGraph &output_graph,
                    int num_threads = 0);

#endif
