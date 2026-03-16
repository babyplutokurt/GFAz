#ifndef DECOMPRESSION_WORKFLOW_HPP
#define DECOMPRESSION_WORKFLOW_HPP

#include "compression_workflow.hpp"

void decompress_gfa(const CompressedData &data, GfaGraph &output_graph,
                    int num_threads = 0);

#endif
