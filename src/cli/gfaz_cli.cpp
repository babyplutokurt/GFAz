#include <string>

#include <iostream>

#include "cli/commands.hpp"
#include "cli/common.hpp"

int main(int argc, char *argv[]) {
  if (argc < 2) {
    gfaz::cli::print_usage();
    return 1;
  }

  std::string command = argv[1];

  if (command == "-h" || command == "--help" || command == "help") {
    gfaz::cli::print_usage();
    return 0;
  }

  if (command == "compress") {
    return gfaz::cli::do_compress(argc - 1, argv + 1);
  } else if (command == "decompress") {
    return gfaz::cli::do_decompress(argc - 1, argv + 1);
  } else if (command == "extract-path") {
    return gfaz::cli::do_extract_path(argc - 1, argv + 1);
  } else if (command == "extract-walk") {
    return gfaz::cli::do_extract_walk(argc - 1, argv + 1);
  } else if (command == "add-haplotypes") {
    return gfaz::cli::do_add_haplotypes(argc - 1, argv + 1);
  } else {
    std::cerr << "Unknown command: " << command << std::endl;
    gfaz::cli::print_usage();
    return 1;
  }
}
