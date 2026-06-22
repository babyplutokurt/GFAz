#ifndef WORKFLOWS_SNARL_FINDER_HPP
#define WORKFLOWS_SNARL_FINDER_HPP

// Topology-based snarl (superbubble) enumeration over the .gfaz container.
//
// Unlike the linear-anchor heuristic (reference-unique nodes), this builds the
// graph's bidirected node-side adjacency directly from the stored L-line links
// and finds superbubbles with the per-entrance BFS of Onodera et al. (2013).
// Bubbles are projected onto a decoded reference node stream so each becomes a
// VCF site with a genomic position; overlapping bubbles are reduced to a
// non-overlapping top-level chain along the reference.
//
// Only the graph topology is needed here -- no path/walk traversals are
// decoded. Allele observation (which haplotype spells what through a snarl) is
// done separately by streaming each path once; see deconstruct_workflow.cpp.

#include "core/model/compressed_data.hpp"
#include "core/model/gfa_graph.hpp"

#include <cstdint>
#include <limits>
#include <vector>

namespace gfaz {

// Bidirected node-side graph in CSR form. Each segment id n (1-based) yields two
// vertices, one per orientation: vid(+n) is the "leaving forward" side and
// vid(-n) the "leaving reverse" side. An L-line (f,fo -> t,to) contributes the
// arc vid(sf)->vid(st) and its reverse complement vid(-st)->vid(-sf), where
// sf/st are the signed (orientation-encoded) node ids.
struct DoubledGraph {
  std::vector<uint64_t> adj_start; // size num_vertices()+1 (CSR row pointers)
  std::vector<uint32_t> adj;       // arc targets (vertex indices)
  uint32_t num_nodes = 0;

  static constexpr uint32_t kInvalid = std::numeric_limits<uint32_t>::max();

  size_t num_vertices() const { return static_cast<size_t>(num_nodes) * 2; }

  // signed node id (sign = orientation, |id| in [1,num_nodes]) -> vertex index.
  static uint32_t vid(NodeId s) {
    const uint32_t a = static_cast<uint32_t>(s < 0 ? -static_cast<int64_t>(s) : s);
    return (a - 1u) * 2u + (s < 0 ? 1u : 0u);
  }
  // vertex index -> signed node id.
  static NodeId signed_of(uint32_t v) {
    const NodeId n = static_cast<NodeId>(v / 2u) + 1;
    return (v & 1u) ? -n : n;
  }
  // reverse-complement of a vertex (flip orientation bit).
  static uint32_t rc(uint32_t v) { return v ^ 1u; }

  uint64_t outdeg(uint32_t v) const { return adj_start[v + 1] - adj_start[v]; }
  // Every arc has a reverse-complement twin, so in-arcs to v equal out-arcs of
  // its complement.
  uint64_t indeg(uint32_t v) const { return outdeg(rc(v)); }
};

// Build the doubled node-side graph from the stored L-line links. Returns an
// edge-less graph (no arcs) when the container has no links (num_links == 0);
// callers should treat that as "no snarls callable".
DoubledGraph build_doubled_graph_from_links(const CompressedData &data,
                                            uint32_t num_nodes);

// A snarl that the reference participates in, described by its boundaries both
// as positions on the decoded reference stream and as oriented nodes.
struct ReferenceSnarl {
  uint32_t start_ref_index = 0; // entrance boundary: index into ref_nodes
  uint32_t end_ref_index = 0;   // exit boundary: index into ref_nodes
  NodeId start_node = 0;        // == ref_nodes[start_ref_index]
  NodeId end_node = 0;         // == ref_nodes[end_ref_index]
};

// Find superbubbles whose entrance is a reference node-side and whose exit
// rejoins the reference downstream, then keep only a non-overlapping top-level
// chain (outermost wins) ordered by reference position. `ref_nodes` is the
// reference traversal as signed oriented node ids.
std::vector<ReferenceSnarl>
find_reference_snarls(const DoubledGraph &g,
                      const std::vector<NodeId> &ref_nodes);

// Undirected segment graph (one vertex per segment, one edge per L-line) in CSR
// form, used to compute the global biconnected decomposition that yields vg-like
// top-level snarls. Self-loops and out-of-range links are dropped; every kept
// link is one undirected edge with a unique id, stored in both directions.
struct SegmentGraph {
  std::vector<uint64_t> adj_start; // size num_nodes+1 (segments indexed 0-based)
  std::vector<uint32_t> adj;       // neighbor segment (1-based id)
  std::vector<uint32_t> adj_edge;  // edge id parallel to `adj`
  uint32_t num_nodes = 0;
  uint32_t num_edges = 0;
};

SegmentGraph build_segment_graph_from_links(const CompressedData &data,
                                            uint32_t num_nodes);

// Top-level snarl decomposition matching `vg deconstruct`'s default granularity.
//
// vg emits exactly one VCF record per *top-level* snarl (it does not recurse into
// children without -a) and drops a snarl whose reference traversal is ambiguous
// (cyclic). We approximate this in two stages. First, the biconnected
// decomposition of the node-end graph identifies the clean *chain regions* the
// reference threads through: each maximal block the reference crosses exactly
// once (acyclic, every interior reference index inside the block), which drops the
// tangled palindrome/satellite regions vg also skips. Second -- and crucially for
// cyclic graphs -- each block is decomposed into its internal per-entrance
// superbubble chain rather than being emitted whole: a biconnected block is a
// chain region, not a single snarl. For a linear chromosome a block has no
// internal cut vertex and is already one bubble, so this is a no-op; for a
// circular genome the entire backbone is a single block (the wrap-around link
// leaves no cut vertices), and the superbubble decomposition is what recovers the
// individual bubbles instead of collapsing them into one chromosome-spanning
// record. A block with no clean superbubble inside it is still emitted whole.
//
// Needs both the doubled node-side graph `g` (for superbubble detection) and the
// undirected node-end segment graph `sg` (for the biconnected decomposition).
// Returns clean snarls only, reduced to a non-overlapping chain ordered by
// reference position.
std::vector<ReferenceSnarl>
find_reference_snarls_top_level(const DoubledGraph &g, const SegmentGraph &sg,
                                const std::vector<NodeId> &ref_nodes);

} // namespace gfaz

#endif // WORKFLOWS_SNARL_FINDER_HPP
