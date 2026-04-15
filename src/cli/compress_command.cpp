#include "cli/commands.hpp"

#include <chrono>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <string>

#include "cli/common.hpp"
#include "codec/serialization.hpp"
#include "utils/debug_log.hpp"
#include "utils/runtime_utils.hpp"
#include "workflows/compression_workflow.hpp"

#ifdef ENABLE_CUDA
#include "gpu/compression/compression_workflow_gpu.hpp"
#endif

namespace gfaz::cli {

int do_compress(int argc, char *argv[]) {
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
      {"gpu-rolling-input-chunk-mb", required_argument, 0,
       kOptGpuRollingInputChunkMb},
      {"gpu-chunk-mb", required_argument, 0, kOptGpuRollingInputChunkMb},
      {"gpu-legacy", no_argument, 0, kOptGpuLegacy},
      {"stats", no_argument, 0, 's'},
      {"debug", no_argument, 0, kOptDebug},
      {"help", no_argument, 0, 'h'},
      {0, 0, 0, 0}};

  int opt;
  optind = 1;
  while ((opt = getopt_long(argc, argv, "r:d:t:j:gsh", long_options, nullptr)) !=
         -1) {
    switch (opt) {
    case 'r':
      rounds = std::stoi(optarg);
      break;
    case 'd':
      delta_round = std::stoi(optarg);
      if (delta_round < 1) {
        std::cerr << "Warning: --delta must be >= 1, clamping to 1" << std::endl;
        delta_round = 1;
      }
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
    case kOptGpuRollingInputChunkMb:
      if (!parse_ull_arg("--gpu-rolling-input-chunk-mb", optarg,
                         gpu_chunk_mb)) {
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

  if (optind >= argc) {
    std::cerr << "Error: No input file specified\n";
    print_compress_help();
    return 1;
  }

  std::string input_path = argv[optind];
  std::string output_path;
  const bool output_provided = (optind + 1 < argc);

  if (output_provided) {
    output_path = argv[optind + 1];
  } else {
    output_path = input_path + ".gfaz";
  }

  if (!use_gpu && (gpu_chunk_mb > 0 || use_gpu_legacy)) {
    std::cerr << "Error: --gpu-rolling-input-chunk-mb and --gpu-legacy "
                 "require --gpu\n";
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
    if (delta_round != kDefaultDeltaRound ||
        freq_threshold != kDefaultFreqThreshold) {
      std::cerr << "Note: GPU backend ignores --delta and --threshold."
                << std::endl;
    }
    if (use_gpu_legacy && gpu_chunk_mb > 0) {
      std::cerr << "Note: --gpu-rolling-input-chunk-mb is ignored with "
                   "--gpu-legacy."
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
      std::cout << "GPU Rolling Input Chunk: "
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
      gpu_options.num_threads = num_threads;
      gpu_options.force_full_device_legacy = use_gpu_legacy;
      gpu_options.force_rolling_scheduler = !use_gpu_legacy;
      if (gpu_chunk_mb > 0) {
        gpu_options.rolling_input_chunk_bytes =
            static_cast<size_t>(gpu_chunk_mb) * 1024ull * 1024ull;
      }
      gfaz::CompressedData compressed_data_gpu =
          gpu_compression::compress_gfa_gpu(input_path, rounds, gpu_options);
      const auto workflow_end = Clock::now();
      workflow_ms =
          std::chrono::duration<double, std::milli>(workflow_end - workflow_start)
              .count();
      const auto serialize_start = Clock::now();
      gfaz::serialize_compressed_data(compressed_data_gpu, output_path);
      const auto serialize_end = Clock::now();
      serialize_ms =
          std::chrono::duration<double, std::milli>(serialize_end -
                                                    serialize_start)
              .count();
    } else {
#endif
      const auto workflow_start = Clock::now();
      gfaz::CompressedData compressed_data = compress_gfa(
          input_path, rounds, freq_threshold, delta_round, num_threads,
          show_stats);
      const auto workflow_end = Clock::now();
      workflow_ms =
          std::chrono::duration<double, std::milli>(workflow_end - workflow_start)
              .count();
      const auto serialize_start = Clock::now();
      gfaz::serialize_compressed_data(compressed_data, output_path);
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

} // namespace gfaz::cli
