#include "cli/commands.hpp"

#include <getopt.h>
#include <iostream>
#include <string>

#include "cli/common.hpp"
#include "core/codec/serialization.hpp"
#include "compute/deconstruct_workflow.hpp"

namespace gfaz::cli {

int do_deconstruct(int argc, char *argv[]) {
  gfaz::DeconstructOptions options;
  std::string input_path;
  bool saw_sample = false;
  bool saw_haplotype = false;
  const char *legacy_mode = nullptr;

  static struct option long_options[] = {
      {"idx", required_argument, 0, 'i'},
      {"input", required_argument, 0, 'i'},
      {"reference", required_argument, 0, 'r'},
      {"path", required_argument, 0, 'r'},
      {"path-prefix", required_argument, 0, 'P'},
      {"reference-prefix", required_argument, 0, 'P'},
      {"group-by-sample", no_argument, 0, 'S'},
      {"group-by-haplotype", no_argument, 0, 'H'},
      {"per-path", no_argument, 0, 'p'},
      {"snarl", no_argument, 0, 1000},
      {"vg-compat", no_argument, 0, 1001},
      {"vg-compact", no_argument, 0, 1001},
      {"linear", no_argument, 0, 1002},
      {"max-site-length", required_argument, 0, 'm'},
      {"no-gt", no_argument, 0, 'G'},
      {"threads", required_argument, 0, 't'},
      {"help", no_argument, 0, 'h'},
      {0, 0, 0, 0}};

  int opt;
  optind = 1;
  while ((opt = getopt_long(argc, argv, "i:r:P:SHpm:Gt:j:h", long_options,
                            nullptr)) != -1) {
    switch (opt) {
    case 'i':
      input_path = optarg;
      break;
    case 'r':
      options.reference_names.emplace_back(optarg);
      break;
    case 'P':
      options.reference_prefixes.emplace_back(optarg);
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
      options.grouping = gfaz::GroupingMode::PerPathWalk;
      break;
    case 1000: // --snarl: leaf-superbubble superset (legacy)
      options.use_snarls = true;
      options.vg_compat = false;
      legacy_mode = "--snarl";
      break;
    case 1001: // --vg-compat / --vg-compact (now the default)
      options.use_snarls = true;
      options.vg_compat = true;
      break;
    case 1002: // --linear (legacy)
      options.use_snarls = false;
      options.vg_compat = false;
      legacy_mode = "--linear";
      break;
    case 'm':
      options.max_site_length =
          static_cast<uint64_t>(std::stoull(optarg));
      break;
    case 'G':
      options.emit_gt = false;
      break;
    case 't':
    case 'j':
      options.num_threads = std::stoi(optarg);
      break;
    case 'h':
      print_deconstruct_help();
      return 0;
    default:
      print_deconstruct_help();
      return 1;
    }
  }

  if (saw_sample && saw_haplotype) {
    std::cerr << "Error: select only one grouping option: -S or -H\n";
    return 1;
  }
  if (legacy_mode) {
    std::cerr << "Warning: " << legacy_mode
              << " is a legacy mode and will be removed in a future release; "
                 "the default now matches `vg deconstruct`.\n";
  }
  if (input_path.empty() && optind < argc)
    input_path = argv[optind++];
  if (input_path.empty()) {
    std::cerr << "Error: Expected -i <input.gfaz>\n";
    print_deconstruct_help();
    return 1;
  }
  if (options.reference_names.empty() && options.reference_prefixes.empty()) {
    std::cerr << "Error: Expected at least one -r <reference-name> or "
                 "-P <reference-prefix>\n";
    print_deconstruct_help();
    return 1;
  }

  try {
    const gfaz::CompressedData data =
        gfaz::deserialize_compressed_data(input_path);
    gfaz::deconstruct_to_vcf(data, options, std::cout);
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}

} // namespace gfaz::cli
