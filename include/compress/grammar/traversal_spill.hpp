#ifndef TRAVERSAL_SPILL_HPP
#define TRAVERSAL_SPILL_HPP

#include "compress/io/traversal_sink.hpp"
#include "core/model/gfa_graph.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace gfaz {

// Out-of-core store for path/walk traversals during CPU compression.
//
// At most `memory_budget_bytes` of traversal data stays in the GfaGraph's
// vectors; the overflow lives in an unlinked temp file and is loaded one
// traversal at a time by the grammar passes. As the grammar shrinks the
// traversals, spilled ones move back into memory whenever they fit.
//
// A traversal is either in memory (its graph vector holds it) or spilled (its
// graph vector is empty and a record points into the temp file). Spilled
// records are rewritten by every encode pass, so they all carry the same
// pending rule-ID remap, which is applied when a record is loaded.
class TraversalSpill final : public TraversalSink {
public:
  TraversalSpill(size_t memory_budget_bytes, std::string tmp_dir,
                 int delta_round);
  ~TraversalSpill() override;

  TraversalSpill(const TraversalSpill &) = delete;
  TraversalSpill &operator=(const TraversalSpill &) = delete;

  // --- TraversalSink (parse time) ---
  // Delta-transforms each traversal, then keeps it in memory or spills it.
  void prepare(size_t num_paths, size_t num_walks) override;
  void consume(bool is_walk, size_t index, std::vector<NodeId> &sequence,
               std::vector<NodeId> &slot) override;
  // Per-thread traversal buffer (valid inside parallel regions started after
  // prepare()/begin_rewrite()).
  std::vector<NodeId> &scratch() override;
  // Free the per-thread buffers. They are sized by the largest traversal
  // seen, so drop them between passes rather than carry that size forward.
  void release_thread_buffers();

  // --- State ---
  size_t num_paths() const { return num_paths_; }
  size_t num_walks() const { return num_walks_; }
  size_t size() const { return records_.size(); }
  // Traversal lengths as parsed, before any grammar encoding.
  const std::vector<uint32_t> &original_lengths() const {
    return original_lengths_;
  }
  // Largest absolute symbol produced by the delta transform(s).
  uint32_t max_abs() const { return max_abs_.load(); }
  bool is_spilled(size_t i) const { return records_[i].bytes != 0; }
  // Number of steps of spilled traversal `i`.
  size_t spilled_length(size_t i) const { return records_[i].steps; }
  size_t spilled_count() const { return spilled_count_.load(); }
  size_t spilled_steps() const { return spilled_steps_.load(); }
  size_t memory_bytes() const { return memory_bytes_.load(); }
  size_t peak_file_bytes() const { return peak_file_bytes_; }
  size_t total_written_bytes() const { return total_written_bytes_.load(); }

  // --- Grammar passes (thread-safe per distinct index) ---
  // Decode spilled traversal `i` into `out`, applying the pending remap.
  void load(size_t i, std::vector<NodeId> &out);

  // Between begin_rewrite() and end_rewrite() every spilled traversal must be
  // passed to store() exactly once: it goes back to memory (`slot`) if the
  // budget allows, otherwise into the next temp-file generation.
  void begin_rewrite();
  void store(size_t i, std::vector<NodeId> &sequence,
             std::vector<NodeId> &slot);
  void end_rewrite();

  // An in-memory traversal shrank by `elements`.
  void note_memory_shrink(size_t elements) {
    memory_bytes_.fetch_sub(elements * sizeof(NodeId));
  }

  // Remap to apply to every currently spilled record when it is next loaded
  // (same semantics as compression_utils::remap_rule_ids).
  void set_pending_remap(uint32_t rules_start_id, std::vector<uint32_t> id_map);

  // Load every spilled traversal into its slot regardless of the budget.
  void materialize_all(std::vector<std::vector<NodeId>> &paths,
                       std::vector<std::vector<NodeId>> &walks);

  // I/O errors inside parallel regions are recorded instead of thrown; call
  // this from serial code to surface them.
  void throw_if_failed() const;

private:
  struct Record {
    uint64_t offset = 0;
    uint64_t bytes = 0; // 0 = in memory
    uint32_t steps = 0;
  };

  struct ThreadBuffers {
    std::vector<NodeId> sequence;
    std::vector<uint8_t> io;
  };
  ThreadBuffers &thread_buffers();
  void ensure_thread_buffers();

  bool try_reserve_memory(size_t elements);
  void write_record(int fd, std::atomic<uint64_t> &cursor, size_t i,
                    const std::vector<NodeId> &sequence);
  void fail(const std::string &message);
  int open_temp_file();

  size_t memory_budget_bytes_;
  std::string tmp_dir_;
  int delta_round_;

  size_t num_paths_ = 0;
  size_t num_walks_ = 0;
  std::vector<Record> records_;
  std::vector<uint32_t> original_lengths_;
  std::vector<ThreadBuffers> thread_buffers_;

  // Two temp files: records are read from `read_fd_` and, during a rewrite,
  // written to `write_fd_`; end_rewrite() swaps them and empties the old one.
  int read_fd_ = -1;
  int write_fd_ = -1;
  std::atomic<uint64_t> read_cursor_{0};
  std::atomic<uint64_t> write_cursor_{0};
  bool rewriting_ = false;

  std::atomic<size_t> memory_bytes_{0};
  std::atomic<size_t> spilled_count_{0};
  std::atomic<size_t> spilled_steps_{0};
  std::atomic<size_t> total_written_bytes_{0};
  std::atomic<uint32_t> max_abs_{0};
  size_t peak_file_bytes_ = 0;

  uint32_t remap_start_id_ = 0;
  std::vector<uint32_t> remap_;

  std::atomic<bool> failed_{false};
  mutable std::mutex error_mutex_;
  std::string error_message_;
};

} // namespace gfaz

#endif // TRAVERSAL_SPILL_HPP
