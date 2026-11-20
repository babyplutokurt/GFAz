# libhandlegraph adapter

`gfaz::GfazHandleGraph` (`include/handlegraph_adapter/gfaz_handle_graph.hpp`)
is a read-only `handlegraph::PathPositionHandleGraph` backed by a `.gfaz`
container. Tools written against
[libhandlegraph](https://github.com/vgteam/libhandlegraph) (vg, odgi and
others) can run their algorithms on a `.gfaz` without first converting it to
their own graph format.

```cpp
#include "handlegraph_adapter/gfaz_handle_graph.hpp"

std::unique_ptr<gfaz::GfazHandleGraph> graph =
    gfaz::GfazHandleGraph::load("graph.gfaz");
const handlegraph::PathPositionHandleGraph &g = *graph;   // use as any handle graph
```

## Building

The adapter needs the libhandlegraph headers, so it is off by default:

```bash
cmake -S . -B build -DGFAZ_BUILD_HANDLEGRAPH=ON \
      -DGFAZ_HANDLEGRAPH_INCLUDE_DIR=/path/to/include \
      -DGFAZ_HANDLEGRAPH_LIBRARY=/path/to/libhandlegraph.a   # only for the test
cmake --build build --target gfaz_handlegraph
```

A host project links `gfaz_handlegraph`, `gfaz_compute` and `gfaz_core`, in
that order, before its own `libhandlegraph`.

## What is in memory

| Part | Representation |
|---|---|
| Paths and walks | The container's grammar-compressed traversals, plus a `PathIndex` (see [workflows/LOCATE_WORKFLOW.md](workflows/LOCATE_WORKFLOW.md)) |
| Node sequences | One decoded string and an offset per node |
| Edges | Sorted successor lists for both orientations of every node |
| Path names | One string per path |

Paths are never expanded as a whole. A step is found by decoding the index
block that holds it; each thread keeps the last few decoded blocks.

## Cost of each operation

| Operation | Cost |
|---|---|
| Node, sequence, edge and degree queries | O(1), plain arrays |
| Walking a path (`get_next_step`, `get_previous_step`, `for_each_step_in_path`) | Amortized O(1): one block decode per `block_steps` steps |
| `get_step_at_position`, `get_position_of_step`, `get_path_length` | One block decode, or none if the block is cached |
| `for_each_step_on_handle`, `for_each_step_position_on_handle`, `steps_of_handle` | One block decode per haplotype block whose node-id range covers the node |

The last row is the one to watch. On a graph numbered along the genome it is
about one block per haplotype, which is fine for a handful of nodes and slow as
the inner loop of an algorithm that asks for the steps on every node. A smaller
`block_steps` (default 1024 here, 4096 for the CLI) makes each decode cheaper
and the index larger. GBWT-based graphs answer that query natively.

The graph cannot be modified. Algorithms that need a mutable graph must convert
first.

## Path names and senses

Names follow what vg's GFA import produces, so a path is called the same thing
whether the graph was converted or is read through the adapter:

- A P-line keeps its name. A full PanSN name (`sample#hap#contig`) is a
  haplotype with phase block 0 unless its sample is a reference sample.
- A W-line is a haplotype named `sample#hap#contig#0`, with a `[start-end]`
  subrange if it does not start at 0; for a reference sample it is a reference
  path `sample#hap#contig`.
- Reference samples come from the header's `RS:Z:` tag and from
  `GfazHandleGraphOptions::reference_samples`.

rGFA tags on S-lines are not turned into paths.

## Node ids

Node ids are assigned by the rule vg's GFA import uses
(`GFAParser::assign_new_sequence_id`), so a `.gfaz` read through the adapter
and the same GFA read by vg number their nodes alike:

- Segment names that are positive numbers are the node ids, whether or not
  they run `1..N`.
- From the first name that is not such a number, every segment gets the
  largest id so far plus one. A graph named `s1, s2, ...` therefore has the ids
  `1..N` in S-line order.

The graph also implements `handlegraph::NamedNodeBackTranslation`:
`get_back_graph_node_name(id)` returns the segment's name in the GFA. vg looks
for that interface (`algorithms::find_translation`) when it writes segment
names.

A handle packs the segment's position in the file, not its id, so every query
other than `get_handle(id)` costs the same either way; when the ids are not
`1..N`, `get_handle(id)` is a binary search. `segment_of(id)` gives the
position, which is the node number the native API (`paths()`) uses.

## Validation

`tests/cpp/test_handle_graph.cpp` compares every interface method with a
brute-force answer from the decompressed GFA: sequences on both strands, the
edge set and degrees, every step of every path forwards and backwards with its
position, position lookups, and the steps on sampled nodes. It runs on the
fixtures and on a generated multi-block graph at several block sizes
(`tests/regression/test_handle_graph.py`; skipped when the adapter is not
built).
