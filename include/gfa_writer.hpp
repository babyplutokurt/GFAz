#ifndef GFA_WRITER_HPP
#define GFA_WRITER_HPP

#include "gfa_parser.hpp"
#include <string>

void write_gfa(const GfaGraph &graph, const std::string &output_path);

#endif
