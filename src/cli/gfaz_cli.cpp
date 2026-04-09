#include <cerrno>
#include <cstdlib>
#include <chrono>
#include <filesystem>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

#include "workflows/add_haplotypes_workflow.hpp"
#include "workflows/compression_workflow.hpp"
#include "workflows/decompression_workflow.hpp"
#include "workflows/extraction_workflow.hpp"
#include "io/gfa_parser.hpp"
#include "io/gfa_writer.hpp"
#include "codec/serialization.hpp"
#include "utils/debug_log.hpp"

#ifdef ENABLE_CUDA
#include "gpu/compression_workflow_gpu.hpp"
#include "gpu/decompression_workflow_gpu.hpp"
#include "gpu/gfa_graph_gpu.hpp"
#include "gpu/gfa_writer_gpu.hpp"

#endif

namespace {
constexpr int kDefaultRounds = 8;
constexpr int kDefaultDeltaRound = 1;
constexpr int kDefaultFreqThreshold = 2;
constexpr int kDefaultNumThreads = 0;

using Clock = std::chrono::steady_clock;
constexpr int kOptGpuChunkMb = 1000;
constexpr int kOptGpuLegacy = 1001;
constexpr int kOptGpuTraversalsPerChunk = 1002;
constexpr int kOptDebug = 1003;

std::string format_size(uintmax_t bytes) {
  std::ostringstream oss;
  if (bytes >= 1024ULL * 1024ULL * 1024ULL)
    oss << std::fixed << std::setprecision(2)
        << (static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0)) << " GB";
  else if (bytes >= 1024ULL * 1024ULL)
    oss << std::fixed << std::setprecision(2)
        << (static_cast<double>(bytes) / (1024.0 * 1024.0)) << " MB";
  else if (bytes >= 1024ULL)
    oss << std::fixed << std::setprecision(2)
        << (static_cast<double>(bytes) / 1024.0) << " KB";
  else
    oss << bytes << " B";
  return oss.str();
}

uintmax_t file_size_or_zero(const std::string &path) {
  std::error_code ec;
  const auto size = std::filesystem::file_size(path, ec);
  return ec ? 0 : size;
}

void configure_debug(bool enabled) {
  if (!enabled)
    return;

  setenv("GFA_COMPRESSION_DEBUG", "1", 1);
#ifdef ENABLE_CUDA
  gpu_compression::set_gpu_compression_debug(true);
  gpu_decompression::set_gpu_decompression_debug(true);
#endif
}

bool parse_ull_arg(const char *name, const char *value,
                   unsigned long long &parsed) {
  errno = 0;
  char *end = nullptr;
  parsed = std::strtoull(value, &end, 10);
  if (errno != 0 || end == value || (end && *end != '\0') || parsed == 0) {
    std::cerr << "Error: Invalid value for " << name << ": " << value
              << std::endl;
    return false;
  }
  return true;
}
}

void print_usage() {
  std::cout << R"(
gfaz - GFA Compression Tool (2-mer with reordering)

USAGE:
    gfaz compress [OPTIONS] <input.gfa> [output.gfaz]
    gfaz decompress [OPTIONS] <input.gfaz> [output.gfa]
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
    --gpu-chunk-mb <N>      Rolling GPU chunk size in MiB
    --gpu-legacy            Use the old whole-graph GPU compression path
                             Note: GPU mode ignores --delta/--threshold/--threads
    -s, --stats             Show size/statistics summary
    --debug                 Show internal debug/timing output
    -h, --help              Show this help message

OPTIONS (decompress):
    -j, --threads <N>       Number of threads (default: 0 = auto)
    -l, --legacy            Use the legacy CPU path:
                             CompressedData -> GfaGraph -> write_gfa
    -g, --gpu               Use GPU backend (if available)
    --gpu-traversals <N>    Traversals expanded per rolling GPU chunk
    --gpu-legacy            Use the old whole-graph GPU decompression path
                             Note: GPU mode ignores --threads
    -s, --stats             Show size/statistics summary
    --debug                 Show internal debug/timing output
    -h, --help              Show this help message

BEHAVIOR:
    - Without output path:
      CPU/GPU compress -> <input>.gfaz
      Decompress removes .gfaz suffix when present
    - In CPU-only builds, --gpu falls back to CPU with a warning.
    - CPU and GPU backends read and write the same .gfaz format.

EXAMPLES:
    gfaz compress input.gfa                      # -> input.gfa.gfaz
    gfaz compress --gpu input.gfa                # -> input.gfa.gfaz
    gfaz compress input.gfa output.gfaz          # -> output.gfaz
    gfaz compress -r 8 -d 1 input.gfa            # With options
    
    gfaz decompress input.gfaz                   # -> input.gfa (removes .gfaz)
    gfaz decompress --gpu input.gfaz             # -> input.gfa
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
    The current .gfaz format reconstructs segment names canonically, so
    segment references are emitted as numeric IDs.

)";
}

void print_extract_walk_help() {
  std::cout << R"(
gfaz extract-walk - Extract a single walk line from a GFAZ file

USAGE:
    gfaz extract-walk [OPTIONS] <input.gfaz>
                      <sample_id> <hap_index> <seq_id> <seq_start> <seq_end>
                      [<sample_id> <hap_index> <seq_id> <seq_start> <seq_end> ...]

OPTIONS:
    -j, --threads <N>       Number of threads (default: 0 = auto)
    -h, --help              Show this help message

OUTPUT:
    Writes the reconstructed W-lines to stdout, in the same order as requested.

NOTE:
    Walk lookup uses the full W-line identifier tuple:
    (sample_id, hap_index, seq_id, seq_start, seq_end).
    Use '*' for seq_start / seq_end values from the original W-line.
    The current .gfaz format reconstructs segment names canonically, so
    segment references are emitted as numeric IDs.

)";
}

void print_add_haplotypes_help() {
  std::cout << R"(
gfaz add-haplotypes - Append path-only or walk-only haplotypes to a GFAZ file

USAGE:
    gfaz add-haplotypes [OPTIONS] <input.gfaz> <paths_or_walks.gfa> [output.gfaz]

OPTIONS:
    -j, --threads <N>       Number of threads (default: 0 = auto)
    -h, --help              Show this help message

BEHAVIOR:
    - The append file must contain only H/P lines or only H/W lines.
    - The existing rulebook is reused; no new grammar rules are generated.
    - Appended path/walk names must be unique.
    - Because the .gfaz format reconstructs segment names canonically,
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
    gfaz compress [OPTIONS] <input.gfa> [output.gfaz]

OPTIONS:
    -r, --rounds <N>        Number of compression rounds (default: 8)
    -d, --delta <N>         Delta encoding rounds (default: 1)
    -t, --threshold <N>     Frequency threshold (default: 2)
    -j, --threads <N>       Number of threads (default: 0 = auto)
    -g, --gpu               Use GPU backend (if available)
    --gpu-chunk-mb <N>      Rolling GPU chunk size in MiB
    --gpu-legacy            Use the old whole-graph GPU compression path
                             Note: GPU mode ignores --delta/--threshold/--threads
    -s, --stats             Show size/statistics summary
    --debug                 Show internal debug/timing output
    -h, --help              Show this help message

EXAMPLES:
    gfaz compress input.gfa                      # -> input.gfa.gfaz
    gfaz compress --gpu input.gfa                # -> input.gfa.gfaz
    gfaz compress -r 4 -d 1 -t 3 input.gfa out.gfaz
    gfaz compress --gpu input.gfa out.gfaz
    gfaz compress --gpu --gpu-chunk-mb 512 input.gfa
    gfaz compress --gpu --gpu-legacy input.gfa

In CPU-only builds, --gpu prints a warning and uses CPU backend.

)";
}

void print_decompress_help() {
  std::cout << R"(
gfaz decompress - Decompress a GFAZ file to GFA format

USAGE:
    gfaz decompress [OPTIONS] <input.gfaz> [output.gfa]

OPTIONS:
    -j, --threads <N>       Number of threads (default: 0 = auto)
    -l, --legacy            Use the legacy CPU path:
                             CompressedData -> GfaGraph -> write_gfa
    -g, --gpu               Use GPU backend (if available)
    --gpu-traversals <N>    Traversals expanded per rolling GPU chunk
    --gpu-legacy            Use the old whole-graph GPU decompression path
                             Note: GPU mode ignores --threads
    -s, --stats             Show size/statistics summary
    --debug                 Show internal debug/timing output
    -h, --help              Show this help message

EXAMPLES:
    gfaz decompress input.gfaz                   # -> input.gfa
    gfaz decompress --gpu input.gfaz             # -> input.gfa
    gfaz decompress input.gfaz output.gfa
    gfaz decompress --gpu --gpu-traversals 256 input.gfaz
    gfaz decompress --gpu --gpu-legacy input.gfaz

In CPU-only builds, --gpu prints a warning and uses CPU backend.
By default, CPU decompression writes GFA directly from CompressedData with
lower peak path/walk memory. Use --legacy to force the old in-memory path.

)";
}

int do_compress(int argc, char *argv[]) {
  // Default options
  int rounds = kDefaultRounds;
  int delta_round = kDefaultDeltaRound;
  int freq_threshold = kDefaultFreqThreshold;
  int num_threads = kDefaultNumThreads;
  bool use_gpu = false;
  bool use_gpu_legacy = false;
  bool show_stats = false;
  bool debug = false;
  unsigned long long gpu_chunk_mb = 0;

  static struct option long_options[] = {
      {"rounds", required_argument, 0, 'r'},
      {"delta", required_argument, 0, 'd'},
      {"threshold", required_argument, 0, 't'},
      {"threads", required_argument, 0, 'j'},
      {"gpu", no_argument, 0, 'g'},
      {"gpu-chunk-mb", required_argument, 0, kOptGpuChunkMb},
      {"gpu-legacy", no_argument, 0, kOptGpuLegacy},
      {"stats", no_argument, 0, 's'},
      {"debug", no_argument, 0, kOptDebug},
      {"help", no_argument, 0, 'h'},
      {0, 0, 0, 0}};

  int opt;
  optind = 1; // Reset getopt
  while ((opt = getopt_long(argc, argv, "r:d:t:j:gsh", long_options, nullptr)) !=
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
    case kOptGpuChunkMb:
      if (!parse_ull_arg("--gpu-chunk-mb", optarg, gpu_chunk_mb)) {
        return 1;
      }
      break;
    case kOptGpuLegacy:
      use_gpu_legacy = true;
      break;
    case 's':
      show_stats = true;
      break;
    case kOptDebug:
      debug = true;
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
    output_path = input_path + ".gfaz";
  }

  if (!use_gpu && (gpu_chunk_mb > 0 || use_gpu_legacy)) {
    std::cerr << "Error: --gpu-chunk-mb and --gpu-legacy require --gpu\n";
    return 1;
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
    if (use_gpu_legacy && gpu_chunk_mb > 0) {
      std::cerr << "Note: --gpu-chunk-mb is ignored with --gpu-legacy."
                << std::endl;
    }
  }
#endif

  std::cout << "=== GFAZ Compress ===" << std::endl;
  std::cout << "Input:  " << input_path << std::endl;
  std::cout << "Output: " << output_path << std::endl;
  std::cout << "Backend: " << (use_gpu ? "GPU" : "CPU") << std::endl;
  std::cout << "Stats: " << (show_stats ? "on" : "off") << std::endl;
  std::cout << "Debug: " << (debug ? "on" : "off") << std::endl;
  std::cout << "Rounds: " << rounds << std::endl;
#ifdef ENABLE_CUDA
  if (use_gpu) {
    std::cout << "Mode:   "
              << (use_gpu_legacy ? "legacy whole-device"
                                 : "rolling scheduler")
              << std::endl;
    if (!use_gpu_legacy) {
      std::cout << "GPU Chunk: "
                << (gpu_chunk_mb > 0 ? std::to_string(gpu_chunk_mb) + " MiB"
                                     : "default")
                << std::endl;
    }
  } else {
#else
  if (!use_gpu) {
#endif
    std::cout << "Delta:  " << delta_round << std::endl;
    std::cout << "Threshold: " << freq_threshold << std::endl;
  }
  if (num_threads == 0) {
    std::cout << "Threads: auto (" << resolve_omp_thread_count(0) << ")"
              << std::endl;
  } else {
    std::cout << "Threads: " << num_threads << std::endl;
  }
  std::cout << std::endl;

  try {
    configure_debug(debug);
    const uintmax_t input_size = file_size_or_zero(input_path);
    const auto start = Clock::now();
    double workflow_ms = 0.0;
    double serialize_ms = 0.0;
#ifdef ENABLE_CUDA
    if (use_gpu) {
      const auto workflow_start = Clock::now();
      gpu_compression::GpuCompressionOptions gpu_options;
      gpu_options.force_full_device_legacy = use_gpu_legacy;
      gpu_options.force_rolling_scheduler = !use_gpu_legacy;
      if (gpu_chunk_mb > 0) {
        gpu_options.rolling_chunk_bytes = static_cast<size_t>(gpu_chunk_mb) *
                                          1024ull * 1024ull;
      }
      CompressedData compressed_data_gpu =
          gpu_compression::compress_gfa_gpu(input_path, rounds, gpu_options);
      const auto workflow_end = Clock::now();
      workflow_ms =
          std::chrono::duration<double, std::milli>(workflow_end - workflow_start)
              .count();
      const auto serialize_start = Clock::now();
      serialize_compressed_data(compressed_data_gpu, output_path);
      const auto serialize_end = Clock::now();
      serialize_ms =
          std::chrono::duration<double, std::milli>(serialize_end -
                                                    serialize_start)
              .count();
    } else {
#endif
      const auto workflow_start = Clock::now();
      CompressedData compressed_data = compress_gfa(
          input_path, rounds, freq_threshold, delta_round, num_threads,
          show_stats);
      const auto workflow_end = Clock::now();
      workflow_ms =
          std::chrono::duration<double, std::milli>(workflow_end - workflow_start)
              .count();
      const auto serialize_start = Clock::now();
      serialize_compressed_data(compressed_data, output_path);
      const auto serialize_end = Clock::now();
      serialize_ms =
          std::chrono::duration<double, std::milli>(serialize_end -
                                                    serialize_start)
              .count();
#ifdef ENABLE_CUDA
    }
#endif
    const auto end = Clock::now();
    if (gfaz_debug_enabled()) {
      const double total_ms =
          std::chrono::duration<double, std::milli>(end - start).count();
      std::cerr << "\n[CLI Compress] workflow:  " << std::fixed
                << std::setprecision(2) << workflow_ms << " ms" << std::endl;
      std::cerr << "[CLI Compress] serialize: " << std::fixed
                << std::setprecision(2) << serialize_ms << " ms" << std::endl;
      std::cerr << "[CLI Compress] total:     " << std::fixed
                << std::setprecision(2) << total_ms << " ms" << std::endl;
    }
    const uintmax_t output_size = file_size_or_zero(output_path);

    std::cout << "\nCompression complete!" << std::endl;
    if (show_stats) {
      const double elapsed_s =
          std::chrono::duration<double>(end - start).count();
      std::cout << "Stats:" << std::endl;
      std::cout << "  Time: " << std::fixed << std::setprecision(3) << elapsed_s
                << " s" << std::endl;
      if (input_size > 0) {
        const double mib = static_cast<double>(input_size) / (1024.0 * 1024.0);
        const double mibps = (elapsed_s > 0.0) ? (mib / elapsed_s) : 0.0;
        std::cout << "  Input: " << format_size(input_size) << std::endl;
        std::cout << "  Output: " << format_size(output_size) << std::endl;
        if (output_size > 0) {
          std::cout << "  Ratio: " << std::fixed << std::setprecision(2)
                    << static_cast<double>(input_size) /
                           static_cast<double>(output_size)
                    << "x" << std::endl;
        }
        std::cout << "  Throughput: " << std::fixed << std::setprecision(2)
                  << mibps << " MiB/s" << std::endl;
      }
    }
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}

int do_decompress(int argc, char *argv[]) {
  int num_threads = kDefaultNumThreads;
  bool use_gpu = false;
  bool use_legacy = false;
  bool use_gpu_legacy = false;
  bool show_stats = false;
  bool debug = false;
  unsigned long long gpu_traversals_per_chunk = 128;

  static struct option long_options[] = {{"threads", required_argument, 0, 'j'},
                                         {"legacy", no_argument, 0, 'l'},
                                         {"gpu", no_argument, 0, 'g'},
                                         {"gpu-traversals", required_argument, 0,
                                          kOptGpuTraversalsPerChunk},
                                         {"gpu-legacy", no_argument, 0,
                                          kOptGpuLegacy},
                                         {"stats", no_argument, 0, 's'},
                                         {"debug", no_argument, 0, kOptDebug},
                                         {"help", no_argument, 0, 'h'},
                                         {0, 0, 0, 0}};

  int opt;
  optind = 1;
  while ((opt = getopt_long(argc, argv, "j:lgsh", long_options, nullptr)) != -1) {
    switch (opt) {
    case 'j':
      num_threads = std::stoi(optarg);
      break;
    case 'l':
      use_legacy = true;
      break;
    case 'g':
      use_gpu = true;
      break;
    case kOptGpuTraversalsPerChunk:
      if (!parse_ull_arg("--gpu-traversals", optarg,
                         gpu_traversals_per_chunk)) {
        return 1;
      }
      if (gpu_traversals_per_chunk >
          std::numeric_limits<uint32_t>::max()) {
        std::cerr << "Error: --gpu-traversals exceeds uint32 range"
                  << std::endl;
        return 1;
      }
      break;
    case kOptGpuLegacy:
      use_gpu_legacy = true;
      break;
    case 's':
      show_stats = true;
      break;
    case kOptDebug:
      debug = true;
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

  if (!use_gpu && (gpu_traversals_per_chunk != 128 || use_gpu_legacy)) {
    std::cerr << "Error: --gpu-traversals and --gpu-legacy require --gpu\n";
    return 1;
  }

  if (use_gpu && use_legacy) {
    std::cerr << "Error: --legacy is the CPU legacy mode. Use --gpu-legacy for "
                 "the old whole-graph GPU decompression path.\n";
    return 1;
  }

  if (optind + 1 < argc) {
    output_path = argv[optind + 1];
  } else {
    // Remove .gfaz suffix if present
    if (input_path.size() > 5 &&
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
  if (use_gpu && use_gpu_legacy && gpu_traversals_per_chunk != 128) {
    std::cerr << "Note: --gpu-traversals is ignored with --gpu-legacy."
              << std::endl;
  }
#endif

  std::cout << "=== GFAZ Decompress ===" << std::endl;
  std::cout << "Input:  " << input_path << std::endl;
  std::cout << "Output: " << output_path << std::endl;
  std::cout << "Backend: " << (use_gpu ? "GPU" : "CPU") << std::endl;
  std::cout << "Stats: " << (show_stats ? "on" : "off") << std::endl;
  std::cout << "Debug: " << (debug ? "on" : "off") << std::endl;
#ifdef ENABLE_CUDA
  if (use_gpu) {
    std::cout << "Mode:   "
              << (use_gpu_legacy ? "legacy whole-device"
                                 : "rolling traversal expansion")
              << std::endl;
    if (!use_gpu_legacy) {
      std::cout << "GPU Traversals/Chunk: " << gpu_traversals_per_chunk
                << std::endl;
    }
  } else {
#else
  if (!use_gpu) {
#endif
    std::cout << "Mode:   " << (use_legacy ? "legacy in-memory" : "streaming direct-writer")
              << std::endl;
  }
  if (num_threads == 0) {
    std::cout << "Threads: auto (" << resolve_omp_thread_count(0) << ")"
              << std::endl;
  } else {
    std::cout << "Threads: " << num_threads << std::endl;
  }
  std::cout << std::endl;

  try {
    configure_debug(debug);
    const uintmax_t input_size = file_size_or_zero(input_path);
    const auto start = Clock::now();
#ifdef ENABLE_CUDA
    if (use_gpu) {
      gpu_decompression::GpuDecompressionOptions gpu_options;
      gpu_options.traversals_per_chunk =
          static_cast<uint32_t>(gpu_traversals_per_chunk);
      gpu_options.use_legacy_full_decompression = use_gpu_legacy;
      CompressedData data_gpu = deserialize_compressed_data(input_path);
      write_gfa_from_compressed_data_gpu(data_gpu, output_path, gpu_options);
    } else {
#endif
      CompressedData data = deserialize_compressed_data(input_path);
      if (use_legacy) {
        GfaGraph graph;
        decompress_gfa(data, graph, num_threads);
        write_gfa(graph, output_path);
      } else {
        write_gfa_from_compressed_data(data, output_path, num_threads);
      }
#ifdef ENABLE_CUDA
    }
#endif
    const auto end = Clock::now();
    const uintmax_t output_size = file_size_or_zero(output_path);

    std::cout << "\nDecompression complete!" << std::endl;
    if (show_stats) {
      const double elapsed_s =
          std::chrono::duration<double>(end - start).count();
      std::cout << "Stats:" << std::endl;
      std::cout << "  Time: " << std::fixed << std::setprecision(3) << elapsed_s
                << " s" << std::endl;
      if (input_size > 0) {
        const double mib = static_cast<double>(input_size) / (1024.0 * 1024.0);
        const double mibps = (elapsed_s > 0.0) ? (mib / elapsed_s) : 0.0;
        std::cout << "  Input: " << format_size(input_size) << std::endl;
        std::cout << "  Output: " << format_size(output_size) << std::endl;
        std::cout << "  Throughput: " << std::fixed << std::setprecision(2)
                  << mibps << " MiB/s" << std::endl;
      }
    }
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

    const CompressedData data = deserialize_compressed_data(input_path);
    for (const auto &line : extract_walk_lines(data, walk_keys, num_threads)) {
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
