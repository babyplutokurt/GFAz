#include "cli/commands.hpp"

#include <getopt.h>
#include <iostream>
#include <string>

#include "cli/common.hpp"
#include "compute/locate_workflow.hpp"
#include "core/codec/serialization.hpp"

namespace gfaz::cli {

int do_locate(int argc, char *argv[]) {
  gfaz::LocateOptions options;
  std::string input_path;
  bool index_given = false;

  static struct option long_options[] = {
      {"input", required_argument, 0, 'i'},
      {"node", required_argument, 0, 'n'},
      {"index", required_argument, 0, 'x'},
      {"threads", required_argument, 0, 't'},
      {"help", no_argument, 0, 'h'},
      {0, 0, 0, 0}};

  int opt;
  optind = 1;
  while ((opt = getopt_long(argc, argv, "i:n:x:t:j:h", long_options,
                            nullptr)) != -1) {
    switch (opt) {
    case 'i':
      input_path = optarg;
      break;
    case 'n':
      options.nodes.push_back(static_cast<uint32_t>(std::stoul(optarg)));
      break;
    case 'x':
      options.index_path = optarg;
      index_given = true;
      break;
    case 't':
    case 'j':
      options.num_threads = std::stoi(optarg);
      break;
    case 'h':
      print_locate_help();
      return 0;
    default:
      print_locate_help();
      return 1;
    }
  }

  if (input_path.empty() && optind < argc)
    input_path = argv[optind++];
  if (input_path.empty() || options.nodes.empty()) {
    std::cerr << "Error: Expected -i <input.gfaz> and at least one -n <node>\n";
    print_locate_help();
    return 1;
  }
  if (!index_given)
    options.index_path = existing_path_index(input_path);

  try {
    const gfaz::CompressedData data =
        gfaz::deserialize_compressed_data(input_path);
    gfaz::locate_nodes_to_tsv(data, options, std::cout);
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}

int do_index(int argc, char *argv[]) {
  std::string input_path;
  std::string output_path;
  uint32_t block_steps = 4096;
  int num_threads = 0;

  static struct option long_options[] = {
      {"input", required_argument, 0, 'i'},
      {"output", required_argument, 0, 'o'},
      {"block-steps", required_argument, 0, 'b'},
      {"threads", required_argument, 0, 't'},
      {"help", no_argument, 0, 'h'},
      {0, 0, 0, 0}};

  int opt;
  optind = 1;
  while ((opt = getopt_long(argc, argv, "i:o:b:t:j:h", long_options,
                            nullptr)) != -1) {
    switch (opt) {
    case 'i':
      input_path = optarg;
      break;
    case 'o':
      output_path = optarg;
      break;
    case 'b':
      block_steps = static_cast<uint32_t>(std::stoul(optarg));
      break;
    case 't':
    case 'j':
      num_threads = std::stoi(optarg);
      break;
    case 'h':
      print_index_help();
      return 0;
    default:
      print_index_help();
      return 1;
    }
  }

  if (input_path.empty() && optind < argc)
    input_path = argv[optind++];
  if (input_path.empty()) {
    std::cerr << "Error: Expected -i <input.gfaz>\n";
    print_index_help();
    return 1;
  }
  if (output_path.empty())
    output_path = default_path_index(input_path);

  try {
    const gfaz::CompressedData data =
        gfaz::deserialize_compressed_data(input_path);
    const size_t blocks =
        gfaz::write_path_index(data, output_path, block_steps, num_threads);
    std::cerr << "Wrote path index " << output_path << " (" << blocks
              << " blocks)\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}

} // namespace gfaz::cli
