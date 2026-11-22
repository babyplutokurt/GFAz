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
| Node sequences | 2 bits per base, with a side list for bases other than `ACGT`; a start offset for every 32nd node |
| Edges | Sorted successor lists for both orientations of every node |
| Node ids | Nothing when they are `1..N`; otherwise one entry per run of consecutive ids |
| Path names | One string per path |

Paths are never expanded as a whole. A step is found by decoding the index
block that holds it; each thread keeps the last few decoded blocks.

`GfazHandleGraph::load` reads names, lengths and node ids at once and the
three large parts when they are first used:

| Part | First used by |
|---|---|
| Sequences | `get_sequence`, `get_subsequence`, `get_base` |
| Edges | `follow_edges`, `get_degree`, `get_edge_count` |
| Paths (decoded traversals and the index) | anything that takes or returns a step, `get_step_count`, `get_path_length` |

Asking for the edges reads only the edges. Asking for sequences or paths
starts all three side by side on their own threads, because a caller that
walks paths or reads bases goes on to use the rest. So listing path names
reads none of them (`vg paths -L` on HPRC chr6: 0.06 s), counting nodes and
edges reads the edges (`vg stats -z -l`: 0.21 s, 0.32 GB), and a full workload
loads as fast as before: HPRC v1.1 (92.9 M nodes, 4.6 B steps) in about 9 s
with a peak of 8.4 GB at 16 threads, or 5.3 s and 8.0 GB with a saved
1024-step index. An error in a part (a truncated column, a stale index) is
thrown where that part is first used. Each compressed column is freed once
decoded.

A graph constructed from a `CompressedData` the caller keeps reads everything
in the constructor and frees nothing.

## Cost of each operation

| Operation | Cost |
|---|---|
| Node, sequence, edge and degree queries | O(1); a sequence is unpacked on request |
| Walking a path (`get_next_step`, `get_previous_step`, `for_each_step_in_path`) | Amortized O(1): one block decode per `block_steps` steps |
| `get_step_at_position`, `get_position_of_step`, `get_path_length` | One block decode, or none if the block is cached |
| `for_each_step_on_handle`, `for_each_step_position_on_handle`, `steps_of_handle` | One block decode per block listed for the node's zone of 256 ids; in a repeat zone, a read of the node's stored visits |

The last row is the one to watch. The index lists, for each zone of 256 node
ids, the blocks that visit a node in it: about 100 blocks for a typical zone of
HPRC chr6. A smaller `block_steps` (default 1024 here, 4096 for the CLI) makes
each decode cheaper and the index larger. `load` therefore uses the saved index
`<path>.pidx` only when its blocks are no longer than `block_steps`, and builds
one otherwise; write it with `gfaz index -b 1024` for this adapter.
(`vg deconstruct` on chr6 takes 85 s with 1024-step blocks and 103 s with
4096-step ones.) An index named in `GfazHandleGraphOptions::index_path` is
always used. GBWT-based graphs answer the query natively.

Zones inside repeats are different: on chr6, 4% of the zones (satellite
arrays) are each listed by about 50,000 blocks, which no block cache holds, and
one lookup there used to decode some 50 million steps. For every zone listed by
more than `repeat_zone_blocks` blocks (default 256) the adapter keeps a
`RepeatIndex` (`include/compute/repeat_index.hpp`): every visit of every node
of the zone as (path, step), 8 bytes a visit. `for_each_step_on_handle` on such
a node reads the list and decodes nothing; the position variant decodes only
the blocks that hold a visit. The index is built from the paths the first time
the steps on a node of such a zone are asked for, so commands that stay out of
repeats do not pay for it. On chr6 it takes about 2 s to build and holds 144 million visits (36% of
all steps) in 1.1 GB. With it `vg call` on chr6 went from 1,161 s to 88 s at 16
threads, with the same VCF.

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
position, which is the node number GFAz's own API uses.

## Validation

`tests/cpp/test_handle_graph.cpp` compares every interface method with a
brute-force answer from the decompressed GFA: sequences on both strands, the
edge set and degrees, every step of every path forwards and backwards with its
position, position lookups, and the steps on sampled nodes. It runs on the
fixtures and on a generated multi-block graph at several block sizes
(`tests/regression/test_handle_graph.py`; skipped when the adapter is not
built).
