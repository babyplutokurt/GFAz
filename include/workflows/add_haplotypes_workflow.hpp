#ifndef ADD_HAPLOTYPES_WORKFLOW_HPP
#define ADD_HAPLOTYPES_WORKFLOW_HPP

#include "workflows/compression_workflow.hpp"

#include <string>


void add_haplotypes(CompressedData &data, const std::string &haplotypes_path,
                    int num_threads = 0);

#endif

