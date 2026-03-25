#include <cstdlib>
#include <getopt.h>
#include <iostream>
#include <string>

#include "add_haplotypes_workflow.hpp"
#include "compression_workflow.hpp"
#include "decompression_workflow.hpp"
#include "extraction_workflow.hpp"
#include "gfa_parser.hpp"
#include "gfa_writer.hpp"
#include "serialization.hpp"

#ifdef ENABLE_CUDA
#include "gpu/compression_workflow_gpu.hpp"
#include "gpu/decompression_workflow_gpu.hpp"
#include "gpu/gfa_graph_gpu.hpp"
#include "gpu/serialization_gpu.hpp"
#endif

namespace {
constexpr int kDefaultRounds = 8;
constexpr int kDefaultDeltaRound = 1;
constexpr int kDefaultFreqThreshold = 2;
constexpr int kDefaultNumThreads = 0;
}

void print_usage() {
  std::cout << R"(
gfaz - GFA Compression Tool (2-mer with reordering)

USAGE:
    gfaz compress [OPTIONS] <input.gfa> [output.gfaz|output.gfaz_gpu]
    gfaz decompress [OPTIONS] <input.gfaz|input.gfaz_gpu> [output.gfa]
    gfaz extract-path [OPTIONS] <input.gfaz> <path_name>
    gfaz extract-walk [OPTIONS] <input.gfaz> <walk_name>
    gfaz add-haplotypes [OPTIONS] <input.gfaz> <paths_or_walks.gfa> [output.gfaz]

SUBCOMMANDS:
    compress      Compress a GFA file to GFAZ format
    decompress    Decompress a GFAZ file to GFA format
    extract-path  Extract a single P-line to stdout
    extract-walk  Extract a single W-line to stdout
    add-haplotypes  Append path-only or walk-only haplotypes using the existing rulebook

OPTIONS (compress):
    -r, --rounds <N>        Number of compression rounds (default: 8)
    -d, --delta <N>         Delta encoding rounds (default: 1)
    -t, --threshold <N>     Frequency threshold (default: 2)
    -j, --threads <N>       Number of threads (default: 0 = auto)
    -g, --gpu               Use GPU backend (if available)
                             Note: GPU mode ignores --delta/--threshold/--threads
    -h, --help              Show this help message

OPTIONS (decompress):
    -j, --threads <N>       Number of threads (default: 0 = auto)
    -g, --gpu               Use GPU backend (if available)
                             Note: GPU mode ignores --threads
    -h, --help              Show this help message

BEHAVIOR:
    - Without output path:
      CPU compress -> <input>.gfaz
      GPU compress -> <input>.gfaz_gpu
      Decompress removes .gfaz_gpu or .gfaz suffix when present
    - In CPU-only builds, --gpu falls back to CPU with a warning.

EXAMPLES:
    gfaz compress input.gfa                      # -> input.gfa.gfaz
    gfaz compress --gpu input.gfa                # -> input.gfa.gfaz_gpu
    gfaz compress input.gfa output.gfaz          # -> output.gfaz
    gfaz compress -r 8 -d 1 input.gfa            # With options
    
    gfaz decompress input.gfaz                   # -> input.gfa (removes .gfaz)
    gfaz decompress --gpu input.gfaz_gpu         # -> input.gfa (removes .gfaz_gpu)
    gfaz decompress input.gfaz output.gfa        # -> output.gfa

)";
}

void print_extract_path_help() {
  std::cout << R"(
gfaz extract-path - Extract a single path line from a GFAZ file

USAGE:
    gfaz extract-path [OPTIONS] <input.gfaz> <path_name> [path_name ...]

OPTIONS:
    -j, --threads <N>       Number of threads (default: 0 = auto)
    -h, --help              Show this help message

OUTPUT:
    Writes the reconstructed P-lines to stdout, in the same order as requested.

NOTE:
    The current CPU .gfaz format does not store original segment names, so
    segment references are emitted as numeric IDs.

)";
}

void print_extract_walk_help() {
  std::cout << R"(
gfaz extract-walk - Extract a single walk line from a GFAZ file

USAGE:
    gfaz extract-walk [OPTIONS] <input.gfaz> <walk_name> [walk_name ...]

OPTIONS:
    -j, --threads <N>       Number of threads (default: 0 = auto)
    -h, --help              Show this help message

OUTPUT:
    Writes the reconstructed W-lines to stdout, in the same order as requested.

NOTE:
    Walk lookup matches the walk name stored in the W-line sample_id field.
    If more than one walk has the same name, extraction fails as ambiguous.
    The current CPU .gfaz format does not store original segment names, so
    segment references are emitted as numeric IDs.

)";
}

void print_add_haplotypes_help() {
  std::cout << R"(
gfaz add-haplotypes - Append path-only or walk-only haplotypes to a CPU GFAZ file

USAGE:
    gfaz add-haplotypes [OPTIONS] <input.gfaz> <paths_or_walks.gfa> [output.gfaz]

OPTIONS:
    -j, --threads <N>       Number of threads (default: 0 = auto)
    -h, --help              Show this help message

BEHAVIOR:
    - The append file must contain only H/P lines or only H/W lines.
    - The existing rulebook is reused; no new grammar rules are generated.
    - Appended path/walk names must be unique.
    - Because the CPU .gfaz format does not store original segment names,
      appended paths/walks must use numeric segment IDs.
    - If no output path is provided, writes <input>.updated.gfaz.
    - If delta encoding causes appended IDs to collide with the stored rule
      region, the command fails without writing output.

)";
}

void print_compress_help() {
  std::cout << R"(
gfaz compress - Compress a GFA file to GFAZ format

USAGE:
    gfaz compress [OPTIONS] <input.gfa> [output.gfaz|output.gfaz_gpu]

OPTIONS:
    -r, --rounds <N>        Number of compression rounds (default: 8)
    -d, --delta <N>         Delta encoding rounds (default: 1)
    -t, --threshold <N>     Frequency threshold (default: 2)
    -j, --threads <N>       Number of threads (default: 0 = auto)
    -g, --gpu               Use GPU backend (if available)
                             Note: GPU mode ignores --delta/--threshold/--threads
    -h, --help              Show this help message

EXAMPLES:
    gfaz compress input.gfa                      # -> input.gfa.gfaz
    gfaz compress --gpu input.gfa                # -> input.gfa.gfaz_gpu
    gfaz compress -r 4 -d 1 -t 3 input.gfa out.gfaz
    gfaz compress --gpu input.gfa out.gfaz_gpu

In CPU-only builds, --gpu prints a warning and uses CPU backend.

)";
}

void print_decompress_help() {
  std::cout << R"(
gfaz decompress - Decompress a GFAZ file to GFA format

USAGE:
    gfaz decompress [OPTIONS] <input.gfaz|input.gfaz_gpu> [output.gfa]

OPTIONS:
    -j, --threads <N>       Number of threads (default: 0 = auto)
    -g, --gpu               Use GPU backend (if available)
                             Note: GPU mode ignores --threads
    -h, --help              Show this help message

EXAMPLES:
    gfaz decompress input.gfaz                   # -> input.gfa
    gfaz decompress --gpu input.gfaz_gpu         # -> input.gfa
    gfaz decompress input.gfaz output.gfa

In CPU-only builds, --gpu prints a warning and uses CPU backend.

)";
}

int do_compress(int argc, char *argv[]) {
  // Default options
  int rounds = kDefaultRounds;
  int delta_round = kDefaultDeltaRound;
  int freq_threshold = kDefaultFreqThreshold;
  int num_threads = kDefaultNumThreads;
  bool use_gpu = false;

  static struct option long_options[] = {
      {"rounds", required_argument, 0, 'r'},
      {"delta", required_argument, 0, 'd'},
      {"threshold", required_argument, 0, 't'},
      {"threads", required_argument, 0, 'j'},
      {"gpu", no_argument, 0, 'g'},
      {"help", no_argument, 0, 'h'},
      {0, 0, 0, 0}};

  int opt;
  optind = 1; // Reset getopt
  while ((opt = getopt_long(argc, argv, "r:d:t:j:gh", long_options, nullptr)) !=
         -1) {
    switch (opt) {
    case 'r':
      rounds = std::stoi(optarg);
      break;
    case 'd':
      delta_round = std::stoi(optarg);
      break;
    case 't':
      freq_threshold = std::stoi(optarg);
      break;
    case 'j':
      num_threads = std::stoi(optarg);
      break;
    case 'g':
      use_gpu = true;
      break;
    case 'h':
      print_compress_help();
      return 0;
    default:
      print_compress_help();
      return 1;
    }
  }

  // Get positional arguments
  if (optind >= argc) {
    std::cerr << "Error: No input file specified\n";
    print_compress_help();
    return 1;
  }

  std::string input_path = argv[optind];
  std::string output_path;
  bool output_provided = (optind + 1 < argc);

  if (output_provided) {
    output_path = argv[optind + 1];
  } else {
    output_path = input_path + (use_gpu ? ".gfaz_gpu" : ".gfaz");
  }

#ifndef ENABLE_CUDA
  if (use_gpu) {
    std::cerr << "Warning: GPU backend requested, but this is a CPU-only build. "
                 "Falling back to CPU backend."
              << std::endl;
    use_gpu = false;
    if (!output_provided) {
      output_path = input_path + ".gfaz";
    }
  }
#endif

#ifdef ENABLE_CUDA
  if (use_gpu) {
    if (delta_round != kDefaultDeltaRound || freq_threshold != kDefaultFreqThreshold ||
        num_threads != kDefaultNumThreads) {
      std::cerr << "Note: GPU backend ignores --delta, --threshold, and --threads."
                << std::endl;
    }
  }
#endif

  std::cout << "=== GFAZ Compress ===" << std::endl;
  std::cout << "Input:  " << input_path << std::endl;
  std::cout << "Output: " << output_path << std::endl;
  std::cout << "Backend: " << (use_gpu ? "GPU" : "CPU") << std::endl;
  std::cout << "Rounds: " << rounds << std::endl;
#ifdef ENABLE_CUDA
  if (!use_gpu) {
#endif
    std::cout << "Delta:  " << delta_round << std::endl;
    std::cout << "Threshold: " << freq_threshold << std::endl;
#ifdef ENABLE_CUDA
  }
#endif
  if (num_threads == 0) {
    std::cout << "Threads: auto (" << resolve_omp_thread_count(0) << ")"
              << std::endl;
  } else {
    std::cout << "Threads: " << num_threads << std::endl;
  }
  std::cout << std::endl;

  try {
#ifdef ENABLE_CUDA
    if (use_gpu) {
      gpu_compression::CompressedData_gpu compressed_data_gpu =
          gpu_compression::compress_gfa_gpu(input_path, rounds);
      serialize_compressed_data_gpu(compressed_data_gpu, output_path);
    } else {
#endif
      CompressedData compressed_data = compress_gfa(
          input_path, rounds, freq_threshold, delta_round, num_threads);
      serialize_compressed_data(compressed_data, output_path);
#ifdef ENABLE_CUDA
    }
#endif

    std::cout << "\nCompression complete!" << std::endl;
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}

int do_decompress(int argc, char *argv[]) {
  int num_threads = kDefaultNumThreads;
  bool use_gpu = false;

  static struct option long_options[] = {{"threads", required_argument, 0, 'j'},
                                         {"gpu", no_argument, 0, 'g'},
                                         {"help", no_argument, 0, 'h'},
                                         {0, 0, 0, 0}};

  int opt;
  optind = 1;
  while ((opt = getopt_long(argc, argv, "j:gh", long_options, nullptr)) != -1) {
    switch (opt) {
    case 'j':
      num_threads = std::stoi(optarg);
      break;
    case 'g':
      use_gpu = true;
      break;
    case 'h':
      print_decompress_help();
      return 0;
    default:
      print_decompress_help();
      return 1;
    }
  }

  if (optind >= argc) {
    std::cerr << "Error: No input file specified\n";
    print_decompress_help();
    return 1;
  }

  std::string input_path = argv[optind];
  std::string output_path;

  if (optind + 1 < argc) {
    output_path = argv[optind + 1];
  } else {
    // Remove .gfaz_gpu or .gfaz suffix if present
    if (input_path.size() > 9 &&
        input_path.substr(input_path.size() - 9) == ".gfaz_gpu") {
      output_path = input_path.substr(0, input_path.size() - 9);
    } else if (input_path.size() > 5 &&
               input_path.substr(input_path.size() - 5) == ".gfaz") {
      output_path = input_path.substr(0, input_path.size() - 5);
    } else {
      output_path = input_path + ".decompressed";
    }
  }

#ifndef ENABLE_CUDA
  if (use_gpu) {
    std::cerr << "Warning: GPU backend requested, but this is a CPU-only build. "
                 "Falling back to CPU backend."
              << std::endl;
    use_gpu = false;
  }
#endif

#ifdef ENABLE_CUDA
  if (use_gpu && num_threads != kDefaultNumThreads) {
    std::cerr << "Note: GPU backend ignores --threads for decompression."
              << std::endl;
  }
#endif

  std::cout << "=== GFAZ Decompress ===" << std::endl;
  std::cout << "Input:  " << input_path << std::endl;
  std::cout << "Output: " << output_path << std::endl;
  std::cout << "Backend: " << (use_gpu ? "GPU" : "CPU") << std::endl;
  if (num_threads == 0) {
    std::cout << "Threads: auto (" << resolve_omp_thread_count(0) << ")"
              << std::endl;
  } else {
    std::cout << "Threads: " << num_threads << std::endl;
  }
  std::cout << std::endl;

  try {
#ifdef ENABLE_CUDA
    if (use_gpu) {
      gpu_compression::CompressedData_gpu data_gpu =
          deserialize_compressed_data_gpu(input_path);
      GfaGraph_gpu graph_gpu = gpu_decompression::decompress_to_gpu_layout(data_gpu);
      GfaGraph graph = convert_from_gpu_layout(graph_gpu);
      write_gfa(graph, output_path);
    } else {
#endif
      CompressedData data = deserialize_compressed_data(input_path);
      GfaGraph graph;
      decompress_gfa(data, graph, num_threads);
      write_gfa(graph, output_path);
#ifdef ENABLE_CUDA
    }
#endif

    std::cout << "\nDecompression complete!" << std::endl;
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}

int do_extract_path(int argc, char *argv[]) {
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
      print_extract_path_help();
      return 0;
    default:
      print_extract_path_help();
      return 1;
    }
  }

  if (optind + 1 >= argc) {
    std::cerr << "Error: Expected <input.gfaz> and at least one <path_name>\n";
    print_extract_path_help();
    return 1;
  }

  const std::string input_path = argv[optind];
  std::vector<std::string> path_names;
  for (int i = optind + 1; i < argc; ++i)
    path_names.push_back(argv[i]);

  try {
    const CompressedData data = deserialize_compressed_data(input_path);
    for (const auto &line :
         extract_path_lines_by_name(data, path_names, num_threads)) {
      std::cout << line;
    }
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}

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

  if (optind + 1 >= argc) {
    std::cerr << "Error: Expected <input.gfaz> and at least one <walk_name>\n";
    print_extract_walk_help();
    return 1;
  }

  const std::string input_path = argv[optind];
  std::vector<std::string> walk_names;
  for (int i = optind + 1; i < argc; ++i)
    walk_names.push_back(argv[i]);

  try {
    const CompressedData data = deserialize_compressed_data(input_path);
    for (const auto &line :
         extract_walk_lines_by_name(data, walk_names, num_threads)) {
      std::cout << line;
    }
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}

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
    CompressedData data = deserialize_compressed_data(input_path);
    add_haplotypes(data, haplotypes_path, num_threads);
    serialize_compressed_data(data, output_path);
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    print_usage();
    return 1;
  }

  std::string command = argv[1];

  if (command == "-h" || command == "--help" || command == "help") {
    print_usage();
    return 0;
  }

  if (command == "compress") {
    return do_compress(argc - 1, argv + 1);
  } else if (command == "decompress") {
    return do_decompress(argc - 1, argv + 1);
  } else if (command == "extract-path") {
    return do_extract_path(argc - 1, argv + 1);
  } else if (command == "extract-walk") {
    return do_extract_walk(argc - 1, argv + 1);
  } else if (command == "add-haplotypes") {
    return do_add_haplotypes(argc - 1, argv + 1);
  } else {
    std::cerr << "Unknown command: " << command << std::endl;
    print_usage();
    return 1;
  }
}
