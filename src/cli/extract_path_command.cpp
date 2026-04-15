#include "cli/commands.hpp"

#include <getopt.h>
#include <iostream>
#include <string>
#include <vector>

#include "cli/common.hpp"
#include "codec/serialization.hpp"
#include "workflows/extraction_workflow.hpp"

namespace gfaz::cli {

int do_extract_path(int argc, char *argv[]) {
  int num_threads = kDefaultNumThreads;

  static struct option long_options[] = {{"threads", required_argument, 0, 'j'},
                                         {"help", no_argument, 0, 'h'},
                                         {0, 0, 0, 0}};

  int opt;
  optind = 1;
  while ((opt = getopt_long(argc, argv, "j:h", long_options, nullptr)) != -1) {
    switch (opt) {
    case 'j':
      num_threads = std::stoi(optarg);
      break;
    case 'h':
      print_extract_path_help();
      return 0;
    default:
      print_extract_path_help();
      return 1;
    }
  }

  if (optind + 1 >= argc) {
    std::cerr << "Error: Expected <input.gfaz> and at least one <path_name>\n";
    print_extract_path_help();
    return 1;
  }

  const std::string input_path = argv[optind];
  std::vector<std::string> path_names;
  for (int i = optind + 1; i < argc; ++i)
    path_names.push_back(argv[i]);

  try {
    const gfaz::CompressedData data = gfaz::deserialize_compressed_data(input_path);
    for (const auto &line :
         extract_path_lines_by_name(data, path_names, num_threads)) {
      std::cout << line;
    }
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}

} // namespace gfaz::cli
