#ifndef GFAZ_CLI_COMMON_HPP
#define GFAZ_CLI_COMMON_HPP

#include "core/defaults.hpp"

#include <chrono>
#include <cstdint>
#include <string>

namespace gfaz::cli {

// Shared defaults now live in core/defaults.hpp; re-exported here so existing
// gfaz::cli::kDefault* call sites keep working.
using gfaz::kDefaultDeltaRound;
using gfaz::kDefaultFreqThreshold;
using gfaz::kDefaultNumThreads;
using gfaz::kDefaultRounds;

using Clock = std::chrono::steady_clock;

constexpr int kOptGpuRollingInputChunkMb = 1000;
constexpr int kOptGpuLegacy = 1001;
constexpr int kOptDebug = 1003;
constexpr int kOptGpuRollingOutputChunkMb = 1004;

// Default GPU rolling-output chunk size (MiB) used by `decompress --gpu`. Kept
// here (CPU-visible) so the CLI default and its "--gpu-only" validation compile
// in CPU-only builds; a static_assert under ENABLE_CUDA keeps it in sync with
// gpu_decompression::kDefaultRollingOutputChunkBytes.
constexpr unsigned long long kDefaultGpuRollingOutputChunkMb = 1024;

std::string format_size(uintmax_t bytes);
uintmax_t file_size_or_zero(const std::string &path);
void configure_debug(bool enabled);
bool parse_ull_arg(const char *name, const char *value,
                   unsigned long long &parsed);

void print_usage();
void print_compress_help();
void print_decompress_help();
void print_extract_path_help();
void print_extract_walk_help();
void print_add_haplotypes_help();
void print_growth_help();
void print_pav_help();
void print_similarity_help();
void print_deconstruct_help();

} // namespace gfaz::cli

#endif
