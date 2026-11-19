#ifndef COMPRESSION_WORKFLOW_HPP
#define COMPRESSION_WORKFLOW_HPP

#include "core/model/compressed_data.hpp"
#include "core/utils/threading_utils.hpp"

#include <cstddef>
#include <string>

struct CompressOptions {
  // Upper bound on the path/walk traversal data held in memory while
  // compressing. Traversals beyond it are kept in a temp file and streamed
  // through the first grammar rounds; they return to memory as the grammar
  // shrinks them. The output is identical either way. 0 keeps everything in
  // memory.
  size_t traversal_memory_budget_bytes = 0;
  // Directory for the temp file; empty means $TMPDIR, then /tmp. Put it on
  // fast local storage.
  std::string tmp_dir;
  // A record that references a segment no S-line defines is an error. Set
  // this to accept the input as older versions did: unknown names are dropped
  // with a warning; numbers above N in a graph named 1..N are kept.
  bool allow_missing_segments = false;
};

gfaz::CompressedData compress_gfa(const std::string &gfa_file_path, int num_rounds,
                            size_t freq_threshold, int delta_round,
                            int num_threads = 0, bool show_stats = false);

gfaz::CompressedData compress_gfa_with_options(const std::string &gfa_file_path,
                                         int num_rounds, size_t freq_threshold,
                                         int delta_round, int num_threads,
                                         bool show_stats,
                                         const CompressOptions &options);

#endif
