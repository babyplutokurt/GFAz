#ifndef GFA_WRITER_HPP
#define GFA_WRITER_HPP

#include "model/compressed_data.hpp"
#include "model/gfa_graph.hpp"
#include <string>

void write_gfa(const GfaGraph &graph, const std::string &output_path);
void write_gfa_from_compressed_data(const CompressedData &data,
                                    const std::string &output_path,
                                    int num_threads = 0);

#endif

