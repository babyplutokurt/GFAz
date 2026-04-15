#include "cli/commands.hpp"

#include <getopt.h>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "cli/common.hpp"
#include "codec/serialization.hpp"
#include "workflows/extraction_workflow.hpp"

namespace gfaz::cli {

int do_extract_walk(int argc, char *argv[]) {
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
      print_extract_walk_help();
      return 0;
    default:
      print_extract_walk_help();
      return 1;
    }
  }

  if (optind + 5 >= argc) {
    std::cerr << "Error: Expected <input.gfaz> and at least one walk identifier "
                 "tuple\n";
    print_extract_walk_help();
    return 1;
  }

  const std::string input_path = argv[optind];
  const int remaining = argc - (optind + 1);
  if (remaining % 5 != 0) {
    std::cerr << "Error: Walk identifiers must be provided in groups of 5: "
                 "<sample_id> <hap_index> <seq_id> <seq_start> <seq_end>\n";
    print_extract_walk_help();
    return 1;
  }

  try {
    std::vector<WalkLookupKey> walk_keys;
    for (int i = optind + 1; i < argc; i += 5) {
      WalkLookupKey walk_key;
      walk_key.sample_id = argv[i];
      walk_key.hap_index = static_cast<uint32_t>(std::stoul(argv[i + 1]));
      walk_key.seq_id = argv[i + 2];
      walk_key.seq_start =
          (std::string(argv[i + 3]) == "*") ? -1 : std::stoll(argv[i + 3]);
      walk_key.seq_end =
          (std::string(argv[i + 4]) == "*") ? -1 : std::stoll(argv[i + 4]);
      walk_keys.push_back(std::move(walk_key));
    }

    const gfaz::CompressedData data = gfaz::deserialize_compressed_data(input_path);
    for (const auto &line : extract_walk_lines(data, walk_keys, num_threads)) {
      std::cout << line;
    }
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}

} // namespace gfaz::cli
