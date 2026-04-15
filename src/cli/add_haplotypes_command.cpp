#include "cli/commands.hpp"

#include <getopt.h>
#include <iostream>
#include <string>

#include "cli/common.hpp"
#include "codec/serialization.hpp"
#include "workflows/add_haplotypes_workflow.hpp"

namespace gfaz::cli {

int do_add_haplotypes(int argc, char *argv[]) {
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
      print_add_haplotypes_help();
      return 0;
    default:
      print_add_haplotypes_help();
      return 1;
    }
  }

  if (optind + 1 >= argc) {
    std::cerr << "Error: Expected <input.gfaz> and <paths_or_walks.gfa>\n";
    print_add_haplotypes_help();
    return 1;
  }

  const std::string input_path = argv[optind];
  const std::string haplotypes_path = argv[optind + 1];
  std::string output_path;
  if (optind + 2 < argc) {
    output_path = argv[optind + 2];
  } else if (input_path.size() > 5 &&
             input_path.substr(input_path.size() - 5) == ".gfaz") {
    output_path =
        input_path.substr(0, input_path.size() - 5) + ".updated.gfaz";
  } else {
    output_path = input_path + ".updated.gfaz";
  }

  try {
    gfaz::CompressedData data = gfaz::deserialize_compressed_data(input_path);
    add_haplotypes(data, haplotypes_path, num_threads);
    gfaz::serialize_compressed_data(data, output_path);
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}

} // namespace gfaz::cli
