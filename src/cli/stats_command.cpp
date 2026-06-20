#include "cli/commands.hpp"

#include <getopt.h>
#include <iostream>
#include <string>

#include "cli/common.hpp"
#include "compute/stats_workflow.hpp"
#include "core/codec/serialization.hpp"

namespace gfaz::cli {

int do_stats(int argc, char *argv[]) {
  gfaz::StatsOptions options;
  std::string input_path;

  static struct option long_options[] = {
      {"idx", required_argument, 0, 'i'},
      {"input", required_argument, 0, 'i'},
      {"summarize", no_argument, 0, 'S'},
      {"base-content", no_argument, 0, 'b'},
      {"help", no_argument, 0, 'h'},
      {0, 0, 0, 0}};

  int opt;
  optind = 1;
  while ((opt = getopt_long(argc, argv, "i:Sbh", long_options, nullptr)) != -1) {
    switch (opt) {
    case 'i':
      input_path = optarg;
      break;
    case 'S':
      // Summarize is the default; the flag exists for `odgi stats -S` parity.
      options.base_content = false;
      break;
    case 'b':
      options.base_content = true;
      break;
    case 'h':
      print_stats_help();
      return 0;
    default:
      print_stats_help();
      return 1;
    }
  }

  if (input_path.empty() && optind < argc)
    input_path = argv[optind++];
  if (input_path.empty()) {
    std::cerr << "Error: Expected -i <input.gfaz>\n";
    print_stats_help();
    return 1;
  }

  try {
    const gfaz::CompressedData data =
        gfaz::deserialize_compressed_data(input_path);
    gfaz::graph_stats_to_tsv(data, options, std::cout);
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}

} // namespace gfaz::cli
