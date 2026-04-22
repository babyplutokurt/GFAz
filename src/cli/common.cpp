#include "cli/common.hpp"

#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "utils/debug_log.hpp"

#ifdef ENABLE_CUDA
#include "gpu/compression/compression_workflow_gpu.hpp"
#include "gpu/decompression/decompression_workflow_gpu.hpp"
#endif

namespace gfaz::cli {

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

void print_usage() {
  std::cout << R"(
gfaz - GFA Compression Tool (2-mer with reordering)

USAGE:
    gfaz compress [OPTIONS] <input.gfa> [output.gfaz]
    gfaz decompress [OPTIONS] <input.gfaz> [output.gfa]
    gfaz extract-path [OPTIONS] <input.gfaz> <path_name> [path_name ...]
    gfaz extract-walk [OPTIONS] <input.gfaz>
                      <sample_id> <hap_index> <seq_id> <seq_start> <seq_end>
                      [<sample_id> <hap_index> <seq_id> <seq_start> <seq_end> ...]
    gfaz add-haplotypes [OPTIONS] <input.gfaz> <paths_or_walks.gfa> [output.gfaz]
    gfaz growth [OPTIONS] <input.gfaz>

SUBCOMMANDS:
    compress      Compress a GFA file to GFAZ format
    decompress    Decompress a GFAZ file to GFA format
    extract-path  Extract a single P-line to stdout
    extract-walk  Extract a single W-line to stdout
    add-haplotypes  Append path-only or walk-only haplotypes using the existing rulebook
    growth        Compute pangenome growth curve (Panacus-equivalent, count=node)

OPTIONS (compress):
    -r, --rounds <N>        Number of compression rounds (default: 8)
    -d, --delta <N>         Delta encoding rounds, >= 0 (default: 1)
    -t, --threshold <N>     Frequency threshold (default: 2)
    -j, --threads <N>       Threads: >0 explicit, 0 auto, <0 inherit OpenMP
    -g, --gpu               Use GPU backend (if available)
    --gpu-rolling-input-chunk-mb <N>
                            Rolling GPU input chunk size in MiB
    --gpu-legacy            Use the old whole-graph GPU compression path
                             Note: GPU mode ignores --delta/--threshold
    -s, --stats             Show size/statistics summary
    --debug                 Show internal debug/timing output
    -h, --help              Show this help message

OPTIONS (decompress):
    -j, --threads <N>       Threads: >0 explicit, 0 auto, <0 inherit OpenMP
    -l, --legacy            Use the legacy CPU path:
                             CompressedData -> GfaGraph -> write_gfa
    -g, --gpu               Use GPU backend (if available)
    --gpu-rolling-output-chunk-mb <N>
                            Rolling GPU output chunk size in MiB
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
    - CPU decompression defaults to streaming direct-writer mode.
    - GPU decompression defaults to rolling traversal expansion.

EXAMPLES:
    gfaz compress input.gfa                      # -> input.gfa.gfaz
    gfaz compress --gpu input.gfa                # -> input.gfa.gfaz
    gfaz compress input.gfa output.gfaz          # -> output.gfaz
    gfaz compress -r 8 -d 1 input.gfa            # CPU tuned compression
    gfaz compress --gpu --gpu-rolling-input-chunk-mb 512 input.gfa

    gfaz decompress input.gfaz                   # -> input.gfa (removes .gfaz)
    gfaz decompress --gpu input.gfaz             # -> input.gfa
    gfaz decompress input.gfaz output.gfa        # -> output.gfa
    gfaz decompress --gpu --gpu-rolling-output-chunk-mb 128 input.gfaz
    gfaz decompress --gpu --gpu-legacy input.gfaz

)";
}

void print_extract_path_help() {
  std::cout << R"(
gfaz extract-path - Extract a single path line from a GFAZ file

USAGE:
    gfaz extract-path [OPTIONS] <input.gfaz> <path_name> [path_name ...]

OPTIONS:
    -j, --threads <N>       Threads: >0 explicit, 0 auto, <0 inherit OpenMP
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
    -j, --threads <N>       Threads: >0 explicit, 0 auto, <0 inherit OpenMP
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
    -j, --threads <N>       Threads: >0 explicit, 0 auto, <0 inherit OpenMP
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

void print_growth_help() {
  std::cout << R"(
gfaz growth - Compute pangenome growth curve from a GFAZ file

USAGE:
    gfaz growth [OPTIONS] <input.gfaz>

OPTIONS:
    -h, --help              Show this help message

OUTPUT:
    Tab-separated table of expected number of distinct nodes covered by a
    random subset of size k, for k = 1..N (N = number of paths + walks).
    Equivalent to Panacus 'growth' with count=node, coverage>=1, quorum>=0.

NOTES:
    Single-threaded, decode-one path/walk at a time. Uses the existing
    grammar rule expansion + inverse-delta on each haplotype, builds a
    per-node coverage histogram, and evaluates the closed-form growth curve.

)";
}

void print_compress_help() {
  std::cout << R"(
gfaz compress - Compress a GFA file to GFAZ format

USAGE:
    gfaz compress [OPTIONS] <input.gfa> [output.gfaz]

OPTIONS:
    -r, --rounds <N>        Number of compression rounds (default: 8)
    -d, --delta <N>         Delta encoding rounds, >= 0 (default: 1)
    -t, --threshold <N>     Frequency threshold (default: 2)
    -j, --threads <N>       Threads: >0 explicit, 0 auto, <0 inherit OpenMP
    -g, --gpu               Use GPU backend (if available)
    --gpu-rolling-input-chunk-mb <N>
                            Rolling GPU input chunk size in MiB
    --gpu-legacy            Use the old whole-graph GPU compression path
                             Note: GPU mode ignores --delta/--threshold
    -s, --stats             Show size/statistics summary
    --debug                 Show internal debug/timing output
    -h, --help              Show this help message

EXAMPLES:
    gfaz compress input.gfa                      # -> input.gfa.gfaz
    gfaz compress --gpu input.gfa                # -> input.gfa.gfaz
    gfaz compress -r 4 -d 1 -t 3 input.gfa out.gfaz
    gfaz compress --gpu input.gfa out.gfaz
    gfaz compress --gpu --gpu-rolling-input-chunk-mb 512 input.gfa
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
    --gpu-rolling-output-chunk-mb <N>
                            Rolling GPU output chunk size in MiB
    --gpu-legacy            Use the old whole-graph GPU decompression path
                             Note: GPU mode ignores --threads
    -s, --stats             Show size/statistics summary
    --debug                 Show internal debug/timing output
    -h, --help              Show this help message

EXAMPLES:
    gfaz decompress input.gfaz                   # -> input.gfa
    gfaz decompress --gpu input.gfaz             # -> input.gfa
    gfaz decompress input.gfaz output.gfa
    gfaz decompress --gpu --gpu-rolling-output-chunk-mb 128 input.gfaz
    gfaz decompress --gpu --gpu-legacy input.gfaz

In CPU-only builds, --gpu prints a warning and uses CPU backend.
By default, CPU decompression writes GFA directly from CompressedData with
lower peak path/walk memory. Use --legacy to force the old in-memory path.

)";
}

} // namespace gfaz::cli
