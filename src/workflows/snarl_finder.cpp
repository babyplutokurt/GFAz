#include "workflows/snarl_finder.hpp"

#include "codec/codec.hpp"

#include <algorithm>
#include <array>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace gfaz {

namespace {

// |signed node id| as an unsigned segment id (0 stays 0).
inline uint32_t abs_seg(NodeId node) {
  return static_cast<uint32_t>(node < 0 ? -static_cast<int64_t>(node) : node);
}

// Signed (orientation-encoded) node id for one endpoint of an L-line.
inline NodeId signed_endpoint(uint32_t id, char orient) {
  const NodeId n = static_cast<NodeId>(id);
  return orient == '-' ? -n : n;
}

// Per-entrance superbubble detection (Onodera, Sadakane & Shibuya 2013).
//
// Starting from entrance vertex `s`, expand the frontier in the doubled graph.
// A vertex becomes "ready" once all of its in-arcs have been seen from inside
// the growing region; the bubble closes when exactly one ready vertex remains
// and the frontier is otherwise empty -- that vertex is the exit. Any tip,
// cycle back to the entrance, or revisit (non-DAG) aborts the search.
//
// Returns the exit vertex, or DoubledGraph::kInvalid if `s` is not the entrance
// of a superbubble.
uint32_t detect_superbubble(const DoubledGraph &g, uint32_t s) {
  // count of in-arcs traversed for each frontier vertex not yet ready
  std::unordered_map<uint32_t, uint64_t> seen;
  std::unordered_set<uint32_t> visited; // vertices already expanded (incl. s)
  std::vector<uint32_t> ready;          // vertices with all in-arcs seen

  visited.insert(s);

  auto touch = [&](uint32_t u) -> bool {
    // returns false to signal abort
    if (u == s)
      return false; // arc back to entrance -> cycle
    if (visited.count(u))
      return false; // revisit -> not a DAG superbubble
    uint64_t &c = seen[u];
    ++c;
    if (c == g.indeg(u)) {
      seen.erase(u);
      ready.push_back(u);
    }
    return true;
  };

  const uint64_t s_begin = g.adj_start[s];
  const uint64_t s_end = g.adj_start[s + 1];
  for (uint64_t e = s_begin; e < s_end; ++e) {
    if (!touch(g.adj[e]))
      return DoubledGraph::kInvalid;
  }

  while (!ready.empty()) {
    if (ready.size() == 1 && seen.empty()) {
      const uint32_t t = ready[0];
      return (t == s) ? DoubledGraph::kInvalid : t;
    }
    const uint32_t v = ready.back();
    ready.pop_back();
    visited.insert(v);
    const uint64_t v_begin = g.adj_start[v];
    const uint64_t v_end = g.adj_start[v + 1];
    if (v_begin == v_end)
      return DoubledGraph::kInvalid; // tip inside the bubble
    for (uint64_t e = v_begin; e < v_end; ++e) {
      if (!touch(g.adj[e]))
        return DoubledGraph::kInvalid;
    }
  }
  return DoubledGraph::kInvalid;
}

// Generalized bidirected snarl detection (fallback for inversions / non-DAG
// snarls that detect_superbubble rejects).
//
// detect_superbubble only finds acyclic superbubbles, so an inversion -- where
// the snarl interior is traversed on the opposite strand and the exit node's
// reverse side feeds back into the interior -- is silently missed. vg recovers
// these through a full Cactus decomposition; here we instead test, for each
// downstream reference node-side reachable from the entrance, whether it forms a
// separating pair with the entrance: a single source (the entrance node, both
// sides) and single sink (the exit node, both sides) bounding a finite interior
// that is closed under both forward and backward reachability. This captures
// inversions (and inverted chains) without materializing a Cactus graph, and is
// only consulted when detect_superbubble fails, so it can never regress the
// cases that already match vg.
//
// `ref_index_of` maps a vertex to the ascending reference indices where the
// reference leaves through that node-side; `entrance_ref_index` is i (the
// reference index of the entrance). Returns the exit vertex (a downstream
// reference node-side) or DoubledGraph::kInvalid.
uint32_t detect_inversion_snarl(
    const DoubledGraph &g, uint32_t e, uint32_t entrance_ref_index,
    const std::unordered_map<uint32_t, std::vector<uint32_t>> &ref_index_of,
    size_t node_budget) {
  using DG = DoubledGraph;
  const uint32_t i = entrance_ref_index;

  // Smallest reference index strictly downstream of i carried by vertex v, or
  // UINT32_MAX if v never lies downstream on the reference. (A vertex may carry
  // several indices when the reference repeats a node-side.)
  auto downstream_ref_index = [&](uint32_t v) -> uint32_t {
    auto it = ref_index_of.find(v);
    if (it == ref_index_of.end())
      return DG::kInvalid;
    for (uint32_t idx : it->second) // ascending
      if (idx > i)
        return idx;
    return DG::kInvalid;
  };

  // Collect candidate exits: downstream reference node-sides (forward) reachable
  // from e within the node budget. Expansion ignores the entrance node entirely
  // (both of its sides bound the snarl).
  //
  // Critically, this fallback runs for *every* reference branch point that is
  // not a clean superbubble -- tips and tangles are common in real graphs -- so
  // it must bail cheaply when there is no inversion to find. We fold in an
  // inversion *signature* check: a genuine inversion requires some node to be
  // reachable in *both* orientations (the node spelled forward by one allele and
  // reverse by another). While growing the region we watch for any handle whose
  // reverse complement is also in the region; if the bounded region closes
  // without ever exhibiting that signature, there is no inversion here and we
  // skip the (more expensive) separating-pair tests entirely.
  const uint32_t rc_e = DG::rc(e);
  std::unordered_set<uint32_t> reachable;
  std::vector<uint32_t> stack{e};
  reachable.insert(e);
  bool inversion_signature = false;
  while (!stack.empty()) {
    if (reachable.size() > node_budget)
      return DG::kInvalid; // region too large -- treat as not an inversion snarl
    const uint32_t v = stack.back();
    stack.pop_back();
    if (v == rc_e)
      continue; // other side of the entrance boundary node -- do not expand
    for (uint64_t a = g.adj_start[v]; a < g.adj_start[v + 1]; ++a) {
      const uint32_t w = g.adj[a];
      if (reachable.insert(w).second) {
        stack.push_back(w);
        // Signature: an interior node reached on both strands. Ignore the
        // entrance node itself (its two sides are the boundary, not interior).
        if (!inversion_signature && w != e && w != rc_e &&
            reachable.count(DG::rc(w)))
          inversion_signature = true;
      }
    }
  }
  if (!inversion_signature)
    return DG::kInvalid; // no reversal in this region -- not an inversion snarl

  // Order candidate exits by reference position (nearest first) so we return the
  // minimal (innermost) separating snarl.
  std::vector<std::pair<uint32_t, uint32_t>> candidates; // (ref index, vertex)
  for (uint32_t v : reachable) {
    if (v == e || v == rc_e)
      continue;
    const uint32_t idx = downstream_ref_index(v);
    if (idx != DG::kInvalid)
      candidates.emplace_back(idx, v);
  }
  std::sort(candidates.begin(), candidates.end());

  // Directional closure used for the separating-pair test. Grows the region from
  // `src` following `forward` (out-arcs) or backward (in-arcs) edges. The four
  // boundary node-sides in `terminals` are reached but never expanded. Returns
  // false on escape: reaching a reference node-side beyond the snarl span
  // ([i, exit_idx] forward; [i, exit_idx] backward) or cycling back to `src`.
  // The non-terminal expanded handles are written to `interior`.
  auto closure = [&](uint32_t src, bool forward,
                     const std::array<uint32_t, 4> &terminals,
                     uint32_t lo_idx, uint32_t hi_idx,
                     std::unordered_set<uint32_t> &interior) -> bool {
    auto is_terminal = [&](uint32_t v) {
      return v == terminals[0] || v == terminals[1] || v == terminals[2] ||
             v == terminals[3];
    };
    std::unordered_set<uint32_t> seen;
    std::vector<uint32_t> st{src};
    seen.insert(src);
    while (!st.empty()) {
      if (seen.size() > node_budget)
        return false;
      const uint32_t v = st.back();
      st.pop_back();
      if (v != src && !is_terminal(v)) {
        // Escape check: a non-boundary reference node-side outside the span
        // means this pair does not bound the region.
        auto it = ref_index_of.find(v);
        if (it != ref_index_of.end())
          for (uint32_t idx : it->second)
            if (idx < lo_idx || idx > hi_idx)
              return false;
        interior.insert(v);
      }
      if (is_terminal(v))
        continue; // boundary: reached but not expanded
      if (forward) {
        for (uint64_t a = g.adj_start[v]; a < g.adj_start[v + 1]; ++a) {
          const uint32_t w = g.adj[a];
          if (w == src)
            return false; // cycle through the source boundary
          if (seen.insert(w).second)
            st.push_back(w);
        }
      } else {
        const uint32_t rcv = DG::rc(v); // predecessors of v = rc(out-arcs of rc(v))
        for (uint64_t a = g.adj_start[rcv]; a < g.adj_start[rcv + 1]; ++a) {
          const uint32_t w = DG::rc(g.adj[a]);
          if (w == src)
            return false;
          if (seen.insert(w).second)
            st.push_back(w);
        }
      }
    }
    return true;
  };

  // Only the nearest candidate exits can form the minimal (innermost) snarl; a
  // hard cap keeps the per-entrance cost bounded on pathological tangles.
  constexpr size_t kMaxCandidates = 16;
  const size_t tested = std::min(candidates.size(), kMaxCandidates);
  for (size_t ci = 0; ci < tested; ++ci) {
    const uint32_t exit_idx = candidates[ci].first;
    const uint32_t t = candidates[ci].second;
    if (t == rc_e)
      continue;
    const uint32_t rc_t = DG::rc(t);
    // Forward closure from the entrance; boundaries are both sides of the
    // entrance and exit nodes.
    std::unordered_set<uint32_t> fwd_interior;
    if (!closure(e, /*forward=*/true, {t, rc_t, rc_e, DG::kInvalid}, i, exit_idx,
                 fwd_interior))
      continue;
    if (fwd_interior.count(t) || !reachable.count(t))
      continue; // exit must be a boundary reached forward
    // Backward closure from the exit; symmetric boundary set.
    std::unordered_set<uint32_t> bwd_interior;
    if (!closure(t, /*forward=*/false, {e, rc_e, rc_t, DG::kInvalid}, i, exit_idx,
                 bwd_interior))
      continue;
    // A true separating pair: the interior seen from each side must coincide.
    if (fwd_interior.size() != bwd_interior.size())
      continue;
    bool same = true;
    for (uint32_t v : fwd_interior)
      if (!bwd_interior.count(v)) {
        same = false;
        break;
      }
    if (same)
      return t;
  }
  return DG::kInvalid;
}

} // namespace

DoubledGraph build_doubled_graph_from_links(const CompressedData &data,
                                            uint32_t num_nodes) {
  DoubledGraph g;
  g.num_nodes = num_nodes;
  const size_t nv = g.num_vertices();
  g.adj_start.assign(nv + 1, 0);
  if (num_nodes == 0 || data.num_links == 0)
    return g;

  const size_t num_links = data.num_links;
  const std::vector<uint32_t> from_ids =
      Codec::decompress_delta_varint_uint32(data.link_from_ids_zstd, num_links);
  const std::vector<uint32_t> to_ids =
      Codec::decompress_delta_varint_uint32(data.link_to_ids_zstd, num_links);
  const std::vector<char> from_or =
      Codec::decompress_orientations(data.link_from_orients_zstd, num_links);
  const std::vector<char> to_or =
      Codec::decompress_orientations(data.link_to_orients_zstd, num_links);

  auto in_range = [&](uint32_t id) { return id != 0u && id <= num_nodes; };

  // Pass 1: count out-arcs per vertex (each link contributes a forward arc and
  // its reverse-complement twin).
  for (size_t i = 0; i < num_links; ++i) {
    if (!in_range(from_ids[i]) || !in_range(to_ids[i]))
      continue;
    const NodeId sf = signed_endpoint(from_ids[i], from_or[i]);
    const NodeId st = signed_endpoint(to_ids[i], to_or[i]);
    ++g.adj_start[DoubledGraph::vid(sf) + 1];
    ++g.adj_start[DoubledGraph::vid(-st) + 1];
  }
  for (size_t v = 1; v <= nv; ++v)
    g.adj_start[v] += g.adj_start[v - 1];

  g.adj.resize(g.adj_start[nv]);
  std::vector<uint64_t> cursor(g.adj_start.begin(), g.adj_start.end());

  // Pass 2: scatter arc targets.
  for (size_t i = 0; i < num_links; ++i) {
    if (!in_range(from_ids[i]) || !in_range(to_ids[i]))
      continue;
    const NodeId sf = signed_endpoint(from_ids[i], from_or[i]);
    const NodeId st = signed_endpoint(to_ids[i], to_or[i]);
    const uint32_t vf = DoubledGraph::vid(sf);
    const uint32_t vt = DoubledGraph::vid(st);
    g.adj[cursor[vf]++] = vt;
    g.adj[cursor[DoubledGraph::vid(-st)]++] = DoubledGraph::vid(-sf);
  }
  return g;
}

namespace {
// Node-end vertex indexing: each segment s (1-based) owns two end-vertices,
//   end5(s) = 2*(s-1)   (5' end)   end3(s) = 2*(s-1)+1   (3' end).
inline uint32_t end5(uint32_t s) { return (s - 1u) * 2u; }
inline uint32_t end3(uint32_t s) { return (s - 1u) * 2u + 1u; }
inline uint32_t seg_of_end(uint32_t v) { return v / 2u + 1u; }
} // namespace

SegmentGraph build_segment_graph_from_links(const CompressedData &data,
                                            uint32_t num_nodes) {
  // The biconnected decomposition runs on the *node-end* graph (a.k.a. the
  // bidirected "interval" graph): two vertices per segment (its 5' and 3' ends),
  // a "black" edge joining each segment's own two ends, and one "grey" edge per
  // L-line joining the departure end of one segment to the arrival end of the
  // other. Black edges are what let an inversion -- where a single interior node
  // is traversed on the opposite strand -- close into a real cycle, so it lands
  // in one biconnected block instead of looking like an articulation point (which
  // is what happens if you collapse orientation onto a plain segment graph).
  SegmentGraph sg;
  sg.num_nodes = num_nodes;
  const size_t nv = static_cast<size_t>(num_nodes) * 2;
  sg.adj_start.assign(nv + 1, 0);
  if (num_nodes == 0)
    return sg;

  const size_t num_links = data.num_links;
  std::vector<uint32_t> from_ids, to_ids;
  std::vector<char> from_or, to_or;
  if (num_links) {
    from_ids =
        Codec::decompress_delta_varint_uint32(data.link_from_ids_zstd, num_links);
    to_ids =
        Codec::decompress_delta_varint_uint32(data.link_to_ids_zstd, num_links);
    from_or = Codec::decompress_orientations(data.link_from_orients_zstd, num_links);
    to_or = Codec::decompress_orientations(data.link_to_orients_zstd, num_links);
  }

  auto in_range = [&](uint32_t id) { return id != 0u && id <= num_nodes; };
  // Grey-edge endpoints for one link: departure end of `from` -> arrival end of
  // `to`. '+' departs the 3' end and arrives at the 5' end; '-' is mirrored.
  auto link_ends = [&](size_t i, uint32_t &p, uint32_t &q) {
    const uint32_t f = from_ids[i], t = to_ids[i];
    p = (from_or[i] == '-') ? end5(f) : end3(f);
    q = (to_or[i] == '-') ? end3(t) : end5(t);
  };

  // Pass 1: degrees (stored shifted: degree of vertex v at adj_start[v+1]).
  // Every segment contributes a black edge; each in-range link a grey edge
  // (self-loops included -- a node-end self grey edge is a valid bidirected
  // cycle, unlike a plain self-loop).
  for (uint32_t s = 1; s <= num_nodes; ++s) {
    ++sg.adj_start[end5(s) + 1];
    ++sg.adj_start[end3(s) + 1];
  }
  uint32_t kept = 0;
  for (size_t i = 0; i < num_links; ++i) {
    if (!in_range(from_ids[i]) || !in_range(to_ids[i]))
      continue;
    uint32_t p, q;
    link_ends(i, p, q);
    ++sg.adj_start[p + 1];
    ++sg.adj_start[q + 1];
    ++kept;
  }
  for (size_t v = 1; v <= nv; ++v)
    sg.adj_start[v] += sg.adj_start[v - 1];
  sg.adj.resize(sg.adj_start[nv]);
  sg.adj_edge.resize(sg.adj_start[nv]);
  sg.num_edges = num_nodes + kept;

  // Pass 2: scatter. Edge ids: black edges [0, num_nodes), grey edges after.
  std::vector<uint64_t> cursor(sg.adj_start.begin(), sg.adj_start.end());
  auto add_edge = [&](uint32_t a, uint32_t b, uint32_t eid) {
    const uint64_t pa = cursor[a]++;
    const uint64_t pb = cursor[b]++;
    sg.adj[pa] = b;
    sg.adj_edge[pa] = eid;
    sg.adj[pb] = a;
    sg.adj_edge[pb] = eid;
  };
  for (uint32_t s = 1; s <= num_nodes; ++s)
    add_edge(end5(s), end3(s), s - 1); // black edge id = s-1
  uint32_t geid = num_nodes;
  for (size_t i = 0; i < num_links; ++i) {
    if (!in_range(from_ids[i]) || !in_range(to_ids[i]))
      continue;
    uint32_t p, q;
    link_ends(i, p, q);
    add_edge(p, q, geid++);
  }
  return sg;
}

std::vector<ReferenceSnarl>
find_reference_snarls_top_level(const SegmentGraph &sg,
                                const std::vector<NodeId> &ref_nodes) {
  std::vector<ReferenceSnarl> out;
  const size_t L = ref_nodes.size();
  if (L < 2 || sg.num_nodes == 0)
    return out;

  // segment id (1-based) -> ascending reference indices it occupies.
  std::unordered_map<uint32_t, std::vector<uint32_t>> ref_idx;
  ref_idx.reserve(L * 2 + 1);
  for (uint32_t i = 0; i < L; ++i) {
    const uint32_t s = abs_seg(ref_nodes[i]);
    if (s != 0)
      ref_idx[s].push_back(i);
  }

  // --- Iterative Hopcroft-Tarjan biconnected components over node-ends ---
  // disc/low are 1-based timestamps over node-end vertices; 0 means unvisited.
  const size_t nv = static_cast<size_t>(sg.num_nodes) * 2;
  std::vector<uint32_t> disc(nv, 0);
  std::vector<uint32_t> low(nv, 0);
  uint32_t timer = 0;

  struct Frame {
    uint32_t u;          // current node-end vertex
    uint32_t parent_eid; // edge used to enter u (UINT32_MAX at root)
    uint64_t it;         // cursor into u's adjacency
  };
  std::vector<Frame> stack;
  std::vector<uint32_t> edge_stack; // edge ids on the current DFS path
  // Parallel to edge_stack: the segment of the node-end reached when the edge was
  // pushed. A block's segment set is the union of these plus the segment of the
  // articulation node-end bounding it.
  std::vector<uint32_t> edge_to_seg;
  edge_to_seg.reserve(64);

  std::vector<ReferenceSnarl> candidates;

  // Finish one biconnected block: `segs` is its segment set (including the
  // bounding articulation node). Build a clean top-level snarl from it, if any.
  auto emit_block = [&](const std::unordered_set<uint32_t> &segs) {
    std::vector<uint32_t> ridx;
    for (uint32_t s : segs) {
      auto it = ref_idx.find(s);
      if (it != ref_idx.end())
        for (uint32_t idx : it->second)
          ridx.push_back(idx);
    }
    if (ridx.size() < 2)
      return;
    std::sort(ridx.begin(), ridx.end());
    const uint32_t a = ridx.front();
    const uint32_t b = ridx.back();
    if (a >= b)
      return;
    // Clean (vg-emittable) iff the reference crosses the block exactly once: no
    // reference segment recurs within [a,b], and every reference index in [a,b]
    // belongs to this block. A cyclic/ambiguous reference traversal (palindrome,
    // satellite) fails this and is dropped, matching vg.
    std::unordered_set<uint32_t> span_segs;
    span_segs.reserve((b - a + 1) * 2 + 1);
    bool clean = true;
    for (uint32_t i = a; i <= b && clean; ++i) {
      const uint32_t s = abs_seg(ref_nodes[i]);
      if (s == 0 || !segs.count(s)) {
        clean = false;
        break;
      }
      if (!span_segs.insert(s).second) {
        clean = false; // reference segment repeats -> cyclic traversal
        break;
      }
    }
    if (!clean)
      return;
    candidates.push_back(
        ReferenceSnarl{a, b, ref_nodes[a], ref_nodes[b]});
  };

  // DFS roots: both ends of every reference segment (same component via their
  // black edge, but seeding both is harmless and robust to isolated ends).
  for (const auto &kv : ref_idx) {
    const uint32_t s = kv.first;
    if (s == 0 || s > sg.num_nodes)
      continue;
    for (uint32_t root : {end5(s), end3(s)}) {
      if (disc[root] != 0)
        continue;
      disc[root] = low[root] = ++timer;
      stack.push_back(Frame{root, DoubledGraph::kInvalid, sg.adj_start[root]});
      while (!stack.empty()) {
      Frame &f = stack.back();
      const uint32_t u = f.u;
      if (f.it < sg.adj_start[u + 1]) {
        const uint64_t pos = f.it++;
        const uint32_t w = sg.adj[pos];
        const uint32_t eid = sg.adj_edge[pos];
        if (eid == f.parent_eid)
          continue; // don't traverse the edge we came in on
        if (disc[w] == 0) {
          edge_stack.push_back(eid);
          edge_to_seg.push_back(seg_of_end(w));
          disc[w] = low[w] = ++timer;
          stack.push_back(Frame{w, eid, sg.adj_start[w]});
        } else if (disc[w] < disc[u]) {
          // Back edge to an ancestor.
          edge_stack.push_back(eid);
          edge_to_seg.push_back(seg_of_end(w));
          if (disc[w] < low[u])
            low[u] = disc[w];
        }
      } else {
        const uint32_t pe = f.parent_eid;
        const uint32_t low_u = low[u];
        stack.pop_back();
        if (!stack.empty()) {
          const uint32_t p = stack.back().u;
          if (low_u < low[p])
            low[p] = low_u;
          if (low_u >= disc[p]) {
            // p is an articulation point (or root): pop the block bounded by the
            // tree edge (p,u) == pe, collecting its segment set (bounding node p
            // plus every endpoint reached by an edge of the block).
            std::unordered_set<uint32_t> segs;
            segs.insert(seg_of_end(p));
            size_t edge_count = 0;
            while (!edge_stack.empty()) {
              const uint32_t e = edge_stack.back();
              const uint32_t s = edge_to_seg.back();
              edge_stack.pop_back();
              edge_to_seg.pop_back();
              segs.insert(s);
              ++edge_count;
              if (e == pe)
                break;
            }
            if (edge_count >= 2)
              emit_block(segs);
          }
        }
      }
      }
    }
  }

  // Reduce to a non-overlapping top-level chain along the reference (outermost
  // wins on ties). BCC spans are edge-disjoint but their reference projections
  // can still nest/overlap when the reference is locally repetitive; the greedy
  // keeps the earliest-starting, widest block and drops anything it covers.
  std::sort(candidates.begin(), candidates.end(),
            [](const ReferenceSnarl &x, const ReferenceSnarl &y) {
              if (x.start_ref_index != y.start_ref_index)
                return x.start_ref_index < y.start_ref_index;
              return x.end_ref_index > y.end_ref_index;
            });
  int64_t last_end = -1;
  for (const ReferenceSnarl &s : candidates) {
    if (static_cast<int64_t>(s.start_ref_index) >= last_end) {
      out.push_back(s);
      last_end = static_cast<int64_t>(s.end_ref_index);
    }
  }
  return out;
}

std::vector<ReferenceSnarl>
find_reference_snarls(const DoubledGraph &g,
                      const std::vector<NodeId> &ref_nodes) {
  std::vector<ReferenceSnarl> out;
  const size_t L = ref_nodes.size();
  if (L < 2 || g.num_nodes == 0)
    return out;

  // vertex -> ascending list of reference indices where the reference leaves
  // through that node-side.
  std::unordered_map<uint32_t, std::vector<uint32_t>> ref_index_of;
  ref_index_of.reserve(L * 2 + 1);
  for (uint32_t i = 0; i < L; ++i) {
    if (ref_nodes[i] == 0)
      continue;
    ref_index_of[DoubledGraph::vid(ref_nodes[i])].push_back(i);
  }

  std::vector<ReferenceSnarl> candidates;
  for (uint32_t i = 0; i < L; ++i) {
    if (ref_nodes[i] == 0)
      continue;
    const uint32_t v = DoubledGraph::vid(ref_nodes[i]);
    if (v >= g.num_vertices() || g.outdeg(v) < 2)
      continue; // not a branch point -> no bubble starts here
    uint32_t t = detect_superbubble(g, v);
    if (t == DoubledGraph::kInvalid) {
      // Fall back to the general bidirected test, which recovers inversions and
      // other non-DAG snarls the superbubble algorithm rejects. Bounded so it
      // never scans far past a local site.
      static constexpr size_t kInversionNodeBudget = 4096;
      t = detect_inversion_snarl(g, v, i, ref_index_of, kInversionNodeBudget);
    }
    if (t == DoubledGraph::kInvalid)
      continue;
    auto it = ref_index_of.find(t);
    if (it == ref_index_of.end())
      continue; // exit is not on the reference (shouldn't happen for a ref bubble)
    // first reference index strictly greater than i
    const std::vector<uint32_t> &idxs = it->second;
    auto jt = std::upper_bound(idxs.begin(), idxs.end(), i);
    if (jt == idxs.end())
      continue;
    const uint32_t j = *jt;
    if (j <= i)
      continue;
    candidates.push_back(
        ReferenceSnarl{i, j, ref_nodes[i], ref_nodes[j]});
  }

  // Reduce to a non-overlapping top-level chain along the reference: sort by
  // start ascending then end descending (outermost first), greedily keep
  // snarls whose span does not overlap the previously kept one. Shared
  // boundaries (chained bubbles) are allowed.
  std::sort(candidates.begin(), candidates.end(),
            [](const ReferenceSnarl &a, const ReferenceSnarl &b) {
              if (a.start_ref_index != b.start_ref_index)
                return a.start_ref_index < b.start_ref_index;
              return a.end_ref_index > b.end_ref_index;
            });

  int64_t last_end = -1;
  for (const ReferenceSnarl &s : candidates) {
    if (static_cast<int64_t>(s.start_ref_index) >= last_end) {
      out.push_back(s);
      last_end = static_cast<int64_t>(s.end_ref_index);
    }
  }
  return out;
}

} // namespace gfaz
