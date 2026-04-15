#include "cli/commands.hpp"

#include <chrono>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <string>

#include "cli/common.hpp"
#include "codec/serialization.hpp"
#include "io/gfa_writer.hpp"
#include "utils/debug_log.hpp"
#include "utils/runtime_utils.hpp"
#include "workflows/decompression_workflow.hpp"

#ifdef ENABLE_CUDA
#include "gpu/decompression/decompression_workflow_gpu.hpp"
#include "gpu/io/gfa_writer_gpu.hpp"
#endif

namespace gfaz::cli {

int do_decompress(int argc, char *argv[]) {
  int num_threads = kDefaultNumThreads;
  bool use_gpu = false;
  bool use_legacy = false;
  bool use_gpu_legacy = false;
  bool show_stats = false;
  bool debug = false;
  unsigned long long gpu_max_expanded_chunk_mb =
      gpu_decompression::kDefaultRollingOutputChunkBytes /
      (1024ull * 1024ull);

  static struct option long_options[] = {{"threads", required_argument, 0, 'j'},
                                         {"legacy", no_argument, 0, 'l'},
                                         {"gpu", no_argument, 0, 'g'},
                                         {"gpu-rolling-output-chunk-mb",
                                          required_argument, 0,
                                          kOptGpuRollingOutputChunkMb},
                                         {"gpu-max-expanded-chunk-mb",
                                          required_argument, 0,
                                          kOptGpuRollingOutputChunkMb},
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
    case kOptGpuRollingOutputChunkMb:
      if (!parse_ull_arg("--gpu-rolling-output-chunk-mb", optarg,
                         gpu_max_expanded_chunk_mb)) {
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

  if (!use_gpu &&
      (gpu_max_expanded_chunk_mb !=
           gpu_decompression::kDefaultRollingOutputChunkBytes /
               (1024ull * 1024ull) ||
       use_gpu_legacy)) {
    std::cerr << "Error: --gpu-rolling-output-chunk-mb and --gpu-legacy "
                 "require --gpu\n";
    return 1;
  }

  if (use_gpu && use_legacy) {
    std::cerr << "Error: --legacy is the CPU legacy mode. Use --gpu-legacy for "
                 "the old whole-graph GPU decompression path.\n";
    return 1;
  }

  if (optind + 1 < argc) {
    output_path = argv[optind + 1];
  } else if (input_path.size() > 5 &&
             input_path.substr(input_path.size() - 5) == ".gfaz") {
    output_path = input_path.substr(0, input_path.size() - 5);
  } else {
    output_path = input_path + ".decompressed";
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
  if (use_gpu && use_gpu_legacy &&
      (gpu_max_expanded_chunk_mb !=
           gpu_decompression::kDefaultRollingOutputChunkBytes /
               (1024ull * 1024ull))) {
    std::cerr << "Note: --gpu-rolling-output-chunk-mb is ignored with "
                 "--gpu-legacy."
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
      std::cout << "GPU Rolling Output Chunk: "
                << gpu_max_expanded_chunk_mb << " MiB" << std::endl;
    }
  } else {
#else
  if (!use_gpu) {
#endif
    std::cout << "Mode:   "
              << (use_legacy ? "legacy in-memory" : "streaming direct-writer")
              << std::endl;
  }
  if (num_threads == 0) {
    std::cout << "Threads: auto (" << resolve_omp_thread_count(0) << ")"
              << std::endl;
  } else if (num_threads < 0) {
    std::cout << "Threads: inherit OpenMP (" << resolve_omp_thread_count(-1)
              << ")" << std::endl;
  } else {
    std::cout << "Threads: " << num_threads << std::endl;
  }
  std::cout << std::endl;

  try {
    configure_debug(debug);
    const uintmax_t input_size = file_size_or_zero(input_path);
    const auto start = Clock::now();
    double deserialize_ms = 0.0;
    double workflow_ms = 0.0;
#ifdef ENABLE_CUDA
    if (use_gpu) {
      const auto deserialize_start = Clock::now();
      gpu_decompression::GpuDecompressionOptions gpu_options;
      gpu_options.rolling_output_chunk_bytes =
          static_cast<size_t>(gpu_max_expanded_chunk_mb) * 1024ull * 1024ull;
      gpu_options.use_legacy_full_decompression = use_gpu_legacy;
      gfaz::CompressedData data_gpu = gfaz::deserialize_compressed_data(input_path);
      const auto deserialize_end = Clock::now();
      deserialize_ms = std::chrono::duration<double, std::milli>(
                           deserialize_end - deserialize_start)
                           .count();
      const auto workflow_start = Clock::now();
      write_gfa_from_compressed_data_gpu(data_gpu, output_path, gpu_options);
      const auto workflow_end = Clock::now();
      workflow_ms = std::chrono::duration<double, std::milli>(workflow_end -
                                                              workflow_start)
                        .count();
    } else {
#endif
      const auto deserialize_start = Clock::now();
      gfaz::CompressedData data = gfaz::deserialize_compressed_data(input_path);
      const auto deserialize_end = Clock::now();
      deserialize_ms = std::chrono::duration<double, std::milli>(
                           deserialize_end - deserialize_start)
                           .count();
      if (use_legacy) {
        const auto workflow_start = Clock::now();
        gfaz::GfaGraph graph;
        decompress_gfa(data, graph, num_threads);
        write_gfa(graph, output_path);
        const auto workflow_end = Clock::now();
        workflow_ms = std::chrono::duration<double, std::milli>(workflow_end -
                                                                workflow_start)
                          .count();
      } else {
        const auto workflow_start = Clock::now();
        write_gfa_from_compressed_data(data, output_path, num_threads);
        const auto workflow_end = Clock::now();
        workflow_ms = std::chrono::duration<double, std::milli>(workflow_end -
                                                                workflow_start)
                          .count();
      }
#ifdef ENABLE_CUDA
    }
#endif
    const auto end = Clock::now();
    if (gfaz_debug_enabled()) {
      const double total_ms =
          std::chrono::duration<double, std::milli>(end - start).count();
      const auto snapshot = gfaz::runtime_utils::read_process_memory_snapshot();
      std::cerr << "\n[CLI Decompress] deserialize: " << std::fixed
                << std::setprecision(2) << deserialize_ms << " ms"
                << std::endl;
      std::cerr << "[CLI Decompress] workflow:    " << std::fixed
                << std::setprecision(2) << workflow_ms << " ms" << std::endl;
      std::cerr << "[CLI Decompress] total:       " << std::fixed
                << std::setprecision(2) << total_ms << " ms" << std::endl;
      std::cerr << "[CLI Decompress][Memory] complete | "
                << gfaz::runtime_utils::format_memory_snapshot(snapshot)
                << std::endl;
    }
    const uintmax_t output_size = file_size_or_zero(output_path);

    std::cout << "\nDecompression complete!" << std::endl;
    if (show_stats) {
      const double elapsed_s =
          std::chrono::duration<double>(end - start).count();
      std::cout << "Stats:" << std::endl;
      std::cout << "  Time: " << std::fixed << std::setprecision(3) << elapsed_s
                << " s" << std::endl;
      if (input_size > 0 || output_size > 0) {
        const double output_mib =
            static_cast<double>(output_size) / (1024.0 * 1024.0);
        const double mibps =
            (elapsed_s > 0.0) ? (output_mib / elapsed_s) : 0.0;
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

} // namespace gfaz::cli
