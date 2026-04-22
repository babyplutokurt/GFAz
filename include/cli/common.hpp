#ifndef GFAZ_CLI_COMMON_HPP
#define GFAZ_CLI_COMMON_HPP

#include <chrono>
#include <cstdint>
#include <string>

namespace gfaz::cli {

constexpr int kDefaultRounds = 8;
constexpr int kDefaultDeltaRound = 1;
constexpr int kDefaultFreqThreshold = 2;
constexpr int kDefaultNumThreads = 0;

using Clock = std::chrono::steady_clock;

constexpr int kOptGpuRollingInputChunkMb = 1000;
constexpr int kOptGpuLegacy = 1001;
constexpr int kOptDebug = 1003;
constexpr int kOptGpuRollingOutputChunkMb = 1004;

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

} // namespace gfaz::cli

#endif
