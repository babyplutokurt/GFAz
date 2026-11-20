#ifndef GFAZ_HANDLE_GRAPH_HPP
#define GFAZ_HANDLE_GRAPH_HPP

// A read-only libhandlegraph graph backed by a .gfaz container.
//
// Anything written against handlegraph::PathPositionHandleGraph (vg, odgi and
// other libhandlegraph tools) can run on it without first converting the graph
// to its own format. Paths stay grammar-compressed in memory and are decoded a
// block at a time through the PathIndex; node sequences and an adjacency list
// are decoded once, when first used.
//
// Cost model, so callers know what to expect:
//   - node, sequence and edge queries: O(1), plain arrays;
//   - walking a path with get_next_step / get_previous_step: amortized O(1),
//     one block decode per `block_steps` steps (decoded blocks are cached per
//     thread);
//   - a step by position, or the position of a step: one block decode;
//   - the steps on a node (for_each_step_on_handle): a scan of every block
//     the index lists for the node's zone of ids, decoding the ones not in the
//     thread's cache. Neighbouring nodes share blocks, so a local neighbourhood
//     is cheap; repeat-rich zones list many blocks and are slow.
//
// Node ids follow the GFA the way vg's GFA import does: segment names that are
// numbers are the node ids, and other names get new ids; the names themselves
// are reported through NamedNodeBackTranslation. Built only with
// -DGFAZ_BUILD_HANDLEGRAPH=ON; see docs/HANDLEGRAPH_ADAPTER.md.

#include "compute/indexed_paths.hpp"
#include "handlegraph_adapter/packed_sequences.hpp"
#include "core/defaults.hpp"
#include "core/model/compressed_data.hpp"

#include <handlegraph/named_node_back_translation.hpp>
#include <handlegraph/path_position_handle_graph.hpp>

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace gfaz {

struct GfazHandleGraphOptions {
  // A saved path index (`gfaz index`) to load; empty builds one at load.
  std::string index_path;
  // Steps per index block when the index is built at load. Smaller blocks make
  // random access and for_each_step_on_handle cheaper and the index larger.
  uint32_t block_steps = 1024;
  // Decoded blocks each thread keeps. Should exceed the number of haplotypes
  // crossing one locus, so that asking for the steps on a run of neighbouring
  // nodes decodes each block once. One block is about 12 bytes per step.
  size_t cached_blocks = 1024;
  // Samples whose W-lines (and PanSN-named P-lines) are reference paths, in
  // addition to those listed in the header's RS:Z tag.
  std::unordered_set<std::string> reference_samples;
  int num_threads = kDefaultNumThreads;
};

class GfazHandleGraph : public handlegraph::PathPositionHandleGraph,
                        public handlegraph::NamedNodeBackTranslation {
public:
  using handle_t = handlegraph::handle_t;
  using path_handle_t = handlegraph::path_handle_t;
  using step_handle_t = handlegraph::step_handle_t;
  using nid_t = handlegraph::nid_t;

  GfazHandleGraph(const CompressedData &data,
                  const GfazHandleGraphOptions &options = {});
  // Read a .gfaz file. When options.index_path is empty the saved index
  // <path>.pidx is used if it exists and its blocks are no longer than
  // options.block_steps (`gfaz index -b 1024` for the default). Sequences, edges and paths are each read
  // when first asked for (see Part), so a caller that only wants names and
  // counts does not wait for them; an error in one is thrown at that point.
  static std::unique_ptr<GfazHandleGraph>
  load(const std::string &path, GfazHandleGraphOptions options = {});

  ~GfazHandleGraph() override;

  // HandleGraph
  bool has_node(nid_t node_id) const override;
  handle_t get_handle(const nid_t &node_id,
                      bool is_reverse = false) const override;
  nid_t get_id(const handle_t &handle) const override;
  bool get_is_reverse(const handle_t &handle) const override;
  handle_t flip(const handle_t &handle) const override;
  size_t get_length(const handle_t &handle) const override;
  std::string get_sequence(const handle_t &handle) const override;
  size_t get_node_count() const override;
  nid_t min_node_id() const override;
  nid_t max_node_id() const override;
  size_t get_degree(const handle_t &handle, bool go_left) const override;
  size_t get_edge_count() const override;
  size_t get_total_length() const override;
  char get_base(const handle_t &handle, size_t index) const override;
  std::string get_subsequence(const handle_t &handle, size_t index,
                              size_t size) const override;

  // PathHandleGraph
  size_t get_path_count() const override;
  bool has_path(const std::string &path_name) const override;
  path_handle_t get_path_handle(const std::string &path_name) const override;
  std::string get_path_name(const path_handle_t &path_handle) const override;
  bool get_is_circular(const path_handle_t &path_handle) const override;
  size_t get_step_count(const path_handle_t &path_handle) const override;
  using handlegraph::PathHandleGraph::get_step_count;
  handle_t get_handle_of_step(const step_handle_t &step_handle) const override;
  path_handle_t
  get_path_handle_of_step(const step_handle_t &step_handle) const override;
  step_handle_t path_begin(const path_handle_t &path_handle) const override;
  step_handle_t path_end(const path_handle_t &path_handle) const override;
  step_handle_t path_back(const path_handle_t &path_handle) const override;
  step_handle_t path_front_end(const path_handle_t &path_handle) const override;
  bool has_next_step(const step_handle_t &step_handle) const override;
  bool has_previous_step(const step_handle_t &step_handle) const override;
  step_handle_t get_next_step(const step_handle_t &step_handle) const override;
  step_handle_t
  get_previous_step(const step_handle_t &step_handle) const override;

  // PathPositionHandleGraph
  size_t get_path_length(const path_handle_t &path_handle) const override;
  size_t get_position_of_step(const step_handle_t &step) const override;
  step_handle_t get_step_at_position(const path_handle_t &path,
                                     const size_t &position) const override;
  bool for_each_step_position_on_handle(
      const handle_t &handle,
      const std::function<bool(const step_handle_t &, const bool &,
                               const size_t &)> &iteratee) const override;

  // NamedNodeBackTranslation: a node is one whole GFA segment, so a range
  // translates to itself and the name is the segment's.
  std::vector<handlegraph::oriented_node_range_t> translate_back(
      const handlegraph::oriented_node_range_t &range) const override;
  std::string get_back_graph_node_name(const nid_t &back_node_id) const override;

  // The container's 1-based segment number of a node id, or 0 if there is no
  // such node. This is the node number GFAz's own API uses.
  uint32_t segment_of(nid_t node_id) const;

protected:
  bool follow_edges_impl(
      const handle_t &handle, bool go_left,
      const std::function<bool(const handle_t &)> &iteratee) const override;
  bool for_each_handle_impl(const std::function<bool(const handle_t &)> &iteratee,
                            bool parallel = false) const override;
  bool for_each_path_handle_impl(
      const std::function<bool(const path_handle_t &)> &iteratee) const override;
  bool for_each_step_on_handle_impl(
      const handle_t &handle,
      const std::function<bool(const step_handle_t &)> &iteratee) const override;

private:
  struct DecodedBlock;
  class BlockCache;
  BlockCache &thread_cache() const;

  // The parts that take time to read from the container. Each is read at
  // most once, on its own thread. Asking for the edges reads only them;
  // asking for sequences or paths starts all three side by side, since a
  // caller that walks paths or reads bases goes on to use the rest.
  enum Part { kSequences, kEdges, kPaths, kNumParts };
  class Loader;
  struct PathStore;
  void require(Part part) const {
    if (!ready_[part].load(std::memory_order_acquire))
      wait_for(part);
  }
  void wait_for(Part part) const;
  void load_part(Part part, const CompressedData &data, CompressedData *owned);

  // `owned` is the container when this graph keeps it and reads from it
  // later, emptying each column once it is decoded; null reads everything
  // now and leaves the caller's container untouched.
  GfazHandleGraph(const CompressedData &data,
                  const GfazHandleGraphOptions &options,
                  std::unique_ptr<CompressedData> owned);

  void number_nodes();
  nid_t id_of_segment(uint32_t segment) const;
  // Where a segment's sequence starts in sequences_.
  uint64_t sequence_begin(uint32_t segment) const;
  void load_sequences(const CompressedData &data);
  void build_adjacency(const CompressedData &data);
  void load_paths(const CompressedData &data, CompressedData *owned);
  void name_paths(const CompressedData &data,
                  const std::unordered_set<std::string> &reference_samples);
  const PackedSequences &sequences() const {
    require(kSequences);
    return sequences_;
  }
  const PathStore &path_store() const {
    require(kPaths);
    return *path_store_;
  }
  // The decoded block holding a step, from this thread's cache.
  const DecodedBlock &block_with_step(size_t path, uint32_t step) const;
  const DecodedBlock &decoded_block(size_t path, size_t block) const;

  GfazHandleGraphOptions options_;
  uint64_t instance_id_ = 0; // tells cached blocks of different graphs apart

  std::vector<uint32_t> segment_lengths_; // segment_lengths_[n - 1], segment n
  SegmentNames names_;
  uint32_t num_nodes() const {
    return static_cast<uint32_t>(segment_lengths_.size());
  }

  // Node ids when they are not simply the segment numbers (see number_nodes),
  // as runs: the segments from id_runs_[r].segment up to the next run have the
  // consecutive ids from id_runs_[r].id. runs_by_id_ lists the runs by id.
  // Both are empty when segment n has id n.
  struct IdRun {
    uint32_t segment;
    nid_t id;
  };
  std::vector<IdRun> id_runs_;
  std::vector<uint32_t> runs_by_id_;
  nid_t min_node_id_ = 0;
  nid_t max_node_id_ = 0;

  // The sequences of all segments back to back. sequence_starts_[k] is where
  // segment k * kSequenceStride + 1 starts; the segments in between are found
  // by adding up their lengths.
  static constexpr uint32_t kSequenceStride = 32;
  PackedSequences sequences_;
  std::vector<uint64_t> sequence_starts_;
  uint64_t total_length_ = 0;

  // A handle packs a segment number and an orientation, whatever the node ids
  // are. Successors of every oriented node, as packed handles: the list of
  // handle h is edge_targets_[edge_offsets_[h - 2], edge_offsets_[h - 1]) (the
  // smallest packed handle is 2: segment 1 forward).
  std::vector<uint32_t> edge_offsets_;
  std::vector<uint32_t> edge_targets_;
  size_t edge_count_ = 0;

  std::vector<std::string> path_names_;
  std::unordered_map<std::string, uint32_t> path_by_name_;

  // Decoded traversals and their index.
  std::unique_ptr<PathStore> path_store_;

  // Set once a part is loaded. The loader is declared last so that its
  // threads end before anything they write to goes away.
  mutable std::atomic<bool> ready_[kNumParts] = {};
  std::unique_ptr<Loader> loader_;
};

} // namespace gfaz

#endif // GFAZ_HANDLE_GRAPH_HPP
