#include "cli/commands.hpp"

#include <getopt.h>
#include <iostream>
#include <string>

#include "cli/common.hpp"
#include "core/codec/serialization.hpp"
#include "compute/similarity_workflow.hpp"

namespace gfaz::cli {

int do_similarity(int argc, char *argv[]) {
  gfaz::SimilarityOptions options;
  std::string input_path;
  bool saw_sample = false;
  bool saw_haplotype = false;
  bool saw_perpath = false;

  static struct option long_options[] = {
      {"idx", required_argument, 0, 'i'},
      {"input", required_argument, 0, 'i'},
      {"group-by-sample", no_argument, 0, 'S'},
      {"group-by-haplotype", no_argument, 0, 'H'},
      {"per-path", no_argument, 0, 'p'},
      {"distances", no_argument, 0, 'd'},
      {"all", no_argument, 0, 'a'},
      {"threads", required_argument, 0, 't'},
      {"help", no_argument, 0, 'h'},
      {0, 0, 0, 0}};

  int opt;
  optind = 1;
  while ((opt = getopt_long(argc, argv, "i:SHpadt:j:h", long_options,
                            nullptr)) != -1) {
    switch (opt) {
    case 'i':
      input_path = optarg;
      break;
    case 'S':
      saw_sample = true;
      options.grouping = gfaz::GroupingMode::Sample;
      break;
    case 'H':
      saw_haplotype = true;
      options.grouping = gfaz::GroupingMode::SampleHap;
      break;
    case 'p':
      saw_perpath = true;
      options.grouping = gfaz::GroupingMode::PerPathWalk;
      break;
    case 'd':
      options.emit_distances = true;
      break;
    case 'a':
      options.all_pairs = true;
      break;
    case 't':
    case 'j':
      options.num_threads = std::stoi(optarg);
      break;
    case 'h':
      print_similarity_help();
      return 0;
    default:
      print_similarity_help();
      return 1;
    }
  }

  if (saw_sample + saw_haplotype + saw_perpath > 1) {
    std::cerr << "Error: select only one grouping option: -S, -H, or -p\n";
    return 1;
  }
  if (input_path.empty() && optind < argc)
    input_path = argv[optind++];
  if (input_path.empty()) {
    std::cerr << "Error: Expected -i <input.gfaz>\n";
    print_similarity_help();
    return 1;
  }

  try {
    const gfaz::CompressedData data =
        gfaz::deserialize_compressed_data(input_path);
    gfaz::similarity_to_tsv(data, options, std::cout);
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}

} // namespace gfaz::cli
