#include "cli/commands.hpp"

#include <cstdio>
#include <getopt.h>
#include <iostream>
#include <string>

#include "cli/common.hpp"
#include "codec/serialization.hpp"
#include "workflows/growth_workflow.hpp"

namespace gfaz::cli {

int do_growth(int argc, char *argv[]) {
  static struct option long_options[] = {{"help", no_argument, 0, 'h'},
                                         {0, 0, 0, 0}};

  int opt;
  optind = 1;
  while ((opt = getopt_long(argc, argv, "h", long_options, nullptr)) != -1) {
    switch (opt) {
    case 'h':
      print_growth_help();
      return 0;
    default:
      print_growth_help();
      return 1;
    }
  }

  if (optind >= argc) {
    std::cerr << "Error: Expected <input.gfaz>\n";
    print_growth_help();
    return 1;
  }

  const std::string input_path = argv[optind];

  try {
    const gfaz::CompressedData data =
        gfaz::deserialize_compressed_data(input_path);
    const gfaz::GrowthResult result = gfaz::compute_growth(data);

    std::cout << "# gfaz growth (count=node, coverage>=1, quorum>=0)\n";
    std::cout << "# num_haplotypes=" << result.num_haplotypes
              << " num_nodes=" << result.num_nodes << "\n";
    std::cout << "k\tgrowth\n";
    for (uint32_t k = 1; k <= result.num_haplotypes; ++k) {
      std::printf("%u\t%.4f\n", k, result.growth[k]);
    }
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}

} // namespace gfaz::cli
