#ifndef GFAZ_HANDLE_GRAPH_HPP
#define GFAZ_HANDLE_GRAPH_HPP

// A read-only libhandlegraph graph backed by a .gfaz container.
//
// Anything written against handlegraph::PathPositionHandleGraph (vg, odgi and
// other libhandlegraph tools) can run on it without first converting the graph
// to its own format. Paths stay grammar-compressed in memory and are decoded a
// block at a time through the PathIndex; node sequences and an adjacency list
// are decoded once at load.
//
// Cost model, so callers know what to expect:
//   - node, sequence and edge queries: O(1), plain arrays;
//   - walking a path with get_next_step / get_previous_step: amortized O(1),
//     one block decode per `block_steps` steps (decoded blocks are cached per
//     thread);
//   - a step by position, or the position of a step: one block decode;
//   - the steps on a node (for_each_step_on_handle): one block decode per
//     haplotype block whose node-id range covers the node. Cheap for a few
//     nodes, slow as an inner loop over every node of a large graph.
//
// Node ids are the container's 1-based ids. Built only with
// -DGFAZ_BUILD_HANDLEGRAPH=ON; see docs/HANDLEGRAPH_ADAPTER.md.

#include "compute/indexed_paths.hpp"
#include "core/defaults.hpp"
#include "core/model/compressed_data.hpp"

#include <handlegraph/path_position_handle_graph.hpp>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace gfaz {

struct GfazHandleGraphOptions {
  // A saved path index (`gfaz index`) to load; empty builds one at load.
  std::string index_path;
  // Steps per index block when the index is built at load. Smaller blocks make
  // random access and for_each_step_on_handle cheaper and the index larger.
  uint32_t block_steps = 1024;
  // Samples whose W-lines (and PanSN-named P-lines) are reference paths, in
  // addition to those listed in the header's RS:Z tag.
  std::unordered_set<std::string> reference_samples;
  int num_threads = kDefaultNumThreads;
};

class GfazHandleGraph : public handlegraph::PathPositionHandleGraph {
public:
  using handle_t = handlegraph::handle_t;
  using path_handle_t = handlegraph::path_handle_t;
  using step_handle_t = handlegraph::step_handle_t;
  using nid_t = handlegraph::nid_t;

  GfazHandleGraph(const CompressedData &data,
                  const GfazHandleGraphOptions &options = {});
  // Read a .gfaz file. The saved index <path>.pidx is used when present and
  // options.index_path is empty.
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

  // The underlying GFAz objects, for callers that want the native API too.
  const IndexedPaths &paths() const { return paths_; }

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

  void load_sequences(const CompressedData &data);
  void build_adjacency(const CompressedData &data, int num_threads);
  void name_paths(const CompressedData &data,
                  const std::unordered_set<std::string> &reference_samples);
  // The decoded block holding a step, from this thread's cache.
  const DecodedBlock &block_with_step(size_t path, uint32_t step) const;
  const DecodedBlock &decoded_block(size_t path, size_t block) const;

  IndexedPaths paths_;
  uint64_t instance_id_ = 0; // tells cached blocks of different graphs apart

  // Node n occupies sequences_[sequence_offsets_[n - 1], sequence_offsets_[n]).
  std::string sequences_;
  std::vector<uint64_t> sequence_offsets_;

  // Successors of every oriented node, as packed handles: the list of handle h
  // is edge_targets_[edge_offsets_[h - 2], edge_offsets_[h - 1]) (the smallest
  // packed handle is 2: node 1 forward).
  std::vector<uint32_t> edge_offsets_;
  std::vector<uint32_t> edge_targets_;
  size_t edge_count_ = 0;

  std::vector<std::string> path_names_;
  std::unordered_map<std::string, uint32_t> path_by_name_;
};

} // namespace gfaz

#endif // GFAZ_HANDLE_GRAPH_HPP
