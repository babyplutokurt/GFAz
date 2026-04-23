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
  int num_threads = kDefaultNumThreads;
  gfaz::GroupingMode grouping = gfaz::GroupingMode::PerPathWalk;

  static struct option long_options[] = {
      {"threads", required_argument, 0, 'j'},
      {"group-by", required_argument, 0, 'G'},
      {"help", no_argument, 0, 'h'},
      {0, 0, 0, 0}};

  int opt;
  optind = 1;
  while ((opt = getopt_long(argc, argv, "j:G:h", long_options, nullptr)) !=
         -1) {
    switch (opt) {
    case 'j':
      num_threads = std::stoi(optarg);
      break;
    case 'G': {
      const std::string v = optarg;
      if (v == "path" || v == "per-line") {
        grouping = gfaz::GroupingMode::PerPathWalk;
      } else if (v == "sample-hap-seq" || v == "panacus") {
        grouping = gfaz::GroupingMode::SampleHapSeq;
      } else if (v == "sample-hap" || v == "haplotype") {
        grouping = gfaz::GroupingMode::SampleHap;
      } else if (v == "sample") {
        grouping = gfaz::GroupingMode::Sample;
      } else {
        std::cerr << "Error: unknown --group-by value '" << v
                  << "'. Expected 'path', 'sample-hap-seq', 'sample-hap', "
                     "or 'sample'.\n";
        print_growth_help();
        return 1;
      }
      break;
    }
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
    const gfaz::GrowthResult result =
        gfaz::compute_growth(data, num_threads, grouping);

    std::cout << "# gfaz growth (count=node, coverage>=1, quorum>=0)\n";
    const char *mode_label = "path";
    switch (grouping) {
    case gfaz::GroupingMode::SampleHapSeq:
      mode_label = "sample-hap-seq";
      break;
    case gfaz::GroupingMode::SampleHap:
      mode_label = "sample-hap";
      break;
    case gfaz::GroupingMode::Sample:
      mode_label = "sample";
      break;
    case gfaz::GroupingMode::PerPathWalk:
      mode_label = "path";
      break;
    }
    std::cout << "# group-by=" << mode_label << "\n";
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
