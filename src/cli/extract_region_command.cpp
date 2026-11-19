#include "cli/commands.hpp"

#include <getopt.h>
#include <iostream>
#include <string>

#include "cli/common.hpp"
#include "compute/region_workflow.hpp"
#include "core/codec/serialization.hpp"

namespace gfaz::cli {

int do_extract_region(int argc, char *argv[]) {
  gfaz::RegionOptions options;
  std::string input_path;

  static struct option long_options[] = {
      {"input", required_argument, 0, 'i'},
      {"region", required_argument, 0, 'r'},
      {"max-gap", required_argument, 0, 'g'},
      {"index-block-steps", required_argument, 0, 'b'},
      {"threads", required_argument, 0, 't'},
      {"help", no_argument, 0, 'h'},
      {0, 0, 0, 0}};

  int opt;
  optind = 1;
  while ((opt = getopt_long(argc, argv, "i:r:g:b:t:j:h", long_options,
                            nullptr)) != -1) {
    switch (opt) {
    case 'i':
      input_path = optarg;
      break;
    case 'r':
      options.regions.push_back(optarg);
      break;
    case 'g':
      options.max_gap_bp = std::stoull(optarg);
      break;
    case 'b':
      options.index_block_steps = static_cast<uint32_t>(std::stoul(optarg));
      break;
    case 't':
    case 'j':
      options.num_threads = std::stoi(optarg);
      break;
    case 'h':
      print_extract_region_help();
      return 0;
    default:
      print_extract_region_help();
      return 1;
    }
  }

  if (input_path.empty() && optind < argc)
    input_path = argv[optind++];
  if (input_path.empty() || options.regions.empty()) {
    std::cerr << "Error: Expected -i <input.gfaz> and at least one -r <region>\n";
    print_extract_region_help();
    return 1;
  }

  try {
    const gfaz::CompressedData data =
        gfaz::deserialize_compressed_data(input_path);
    gfaz::extract_regions_to_gfa(data, options, std::cout);
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}

} // namespace gfaz::cli
