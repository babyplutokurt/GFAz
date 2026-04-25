#include "cli/commands.hpp"

#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <string>

#include "cli/common.hpp"
#include "codec/serialization.hpp"
#include "workflows/pav_workflow.hpp"

namespace gfaz::cli {

int do_pav(int argc, char *argv[]) {
  gfaz::PavOptions options;
  std::string input_path;
  bool saw_sample = false;
  bool saw_haplotype = false;

  static struct option long_options[] = {
      {"idx", required_argument, 0, 'i'},
      {"input", required_argument, 0, 'i'},
      {"bed-file", required_argument, 0, 'b'},
      {"group-by-sample", no_argument, 0, 'S'},
      {"group-by-haplotype", no_argument, 0, 'H'},
      {"matrix-output", no_argument, 0, 'M'},
      {"binary-values", required_argument, 0, 'B'},
      {"threads", required_argument, 0, 't'},
      {"help", no_argument, 0, 'h'},
      {0, 0, 0, 0}};

  int opt;
  optind = 1;
  while ((opt = getopt_long(argc, argv, "i:b:SHMB:t:j:h", long_options,
                            nullptr)) != -1) {
    switch (opt) {
    case 'i':
      input_path = optarg;
      break;
    case 'b':
      options.bed_path = optarg;
      break;
    case 'S':
      saw_sample = true;
      options.grouping = gfaz::GroupingMode::Sample;
      break;
    case 'H':
      saw_haplotype = true;
      options.grouping = gfaz::GroupingMode::SampleHap;
      break;
    case 'M':
      options.matrix_output = true;
      break;
    case 'B':
      options.emit_binary = true;
      options.binary_threshold = std::stod(optarg);
      break;
    case 't':
    case 'j':
      options.num_threads = std::stoi(optarg);
      break;
    case 'h':
      print_pav_help();
      return 0;
    default:
      print_pav_help();
      return 1;
    }
  }

  if (saw_sample && saw_haplotype) {
    std::cerr << "Error: select only one grouping option: -S or -H\n";
    return 1;
  }
  if (options.emit_binary &&
      (options.binary_threshold < 0.0 || options.binary_threshold > 1.0)) {
    std::cerr << "Error: --binary-values threshold must be in [0,1]\n";
    return 1;
  }
  if (input_path.empty() && optind < argc)
    input_path = argv[optind++];
  if (input_path.empty()) {
    std::cerr << "Error: Expected -i <input.gfaz>\n";
    print_pav_help();
    return 1;
  }
  if (options.bed_path.empty()) {
    std::cerr << "Error: Expected -b <ranges.bed>\n";
    print_pav_help();
    return 1;
  }

  try {
    const gfaz::CompressedData data =
        gfaz::deserialize_compressed_data(input_path);
    const gfaz::PavResult result = gfaz::compute_pav(data, options);
    const size_t num_groups = result.group_names.size();

    std::cout << "chrom\tstart\tend\tname";
    if (options.matrix_output) {
      for (const std::string &name : result.group_names)
        std::cout << '\t' << name;
      std::cout << '\n';
      for (size_t w = 0; w < result.ranges.size(); ++w) {
        const auto &r = result.ranges[w];
        std::cout << r.chrom << '\t' << r.start << '\t' << r.end << '\t'
                  << r.name;
        for (size_t g = 0; g < num_groups; ++g) {
          const uint64_t den = result.denominators[w];
          const uint64_t num = result.numerators[w * num_groups + g];
          const double pav = den == 0 ? 0.0 : static_cast<double>(num) /
                                              static_cast<double>(den);
          if (options.emit_binary)
            std::cout << '\t' << (pav >= options.binary_threshold ? 1 : 0);
          else
            std::cout << '\t' << std::setprecision(5) << pav;
        }
        std::cout << '\n';
      }
    } else {
      std::cout << "\tgroup\tpav\n";
      for (size_t w = 0; w < result.ranges.size(); ++w) {
        const auto &r = result.ranges[w];
        for (size_t g = 0; g < num_groups; ++g) {
          const uint64_t den = result.denominators[w];
          const uint64_t num = result.numerators[w * num_groups + g];
          const double pav = den == 0 ? 0.0 : static_cast<double>(num) /
                                              static_cast<double>(den);
          std::cout << r.chrom << '\t' << r.start << '\t' << r.end << '\t'
                    << r.name << '\t' << result.group_names[g] << '\t';
          if (options.emit_binary)
            std::cout << (pav >= options.binary_threshold ? 1 : 0);
          else
            std::cout << std::setprecision(5) << pav;
          std::cout << '\n';
        }
      }
    }
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}

} // namespace gfaz::cli
