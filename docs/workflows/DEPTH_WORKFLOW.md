# `gfaz depth` — node coverage depth

`gfaz depth` computes node coverage directly from a compressed `.gfaz`,
reproducing `odgi depth -S` (summary) and `-d` (per-node table). It streams the
P/W traversals into a single shared per-node counter, so peak memory is
O(num_nodes) rather than the materialized-graph footprint of odgi.

```
gfaz depth -i graph.gfaz [-S | -d] [-t N]
```

## Output

Default (matches `odgi depth -S`), tab-delimited to stdout:

```
#node.count  graph.length  step.count  path.length  mean.node.depth  mean.graph.depth
```

| Column | Meaning |
|---|---|
| `node.count` | number of segments |
| `graph.length` | total segment bp |
| `step.count` | total node visits across all paths/walks |
| `path.length` | total traversed bp (Σ over steps of node length) |
| `mean.node.depth` | `step.count / node.count` |
| `mean.graph.depth` | `path.length / graph.length` |

With `-d` (matches `odgi depth -d`) it emits a per-node table:

```
#node.id  depth  depth.uniq
```

where `depth` = total steps on the node (multiplicity-counted, so a path visiting
a node twice adds two) and `depth.uniq` = number of **distinct** paths/walks
visiting the node. Rows are emitted in node-id order (thread-count invariant).

## Concordance with odgi

The summary is a pure aggregate and agrees with `odgi depth -S` byte-for-byte.
The `-d` table is keyed by node id; gfaz uses its own 1-based ids, so it matches
`odgi depth -d` byte-for-byte **only when the GFA segment names are already
1..N** (true for the test fixture — the same node-id caveat as `gfaz
deconstruct`'s AT/ID). On graphs with arbitrary segment names the depth *values*
are correct but the ids are gfaz's own.

`tests/concordance/test_depth_vs_odgi.py` checks both modes against committed
odgi goldens (`tests/golden/depth_fixture.{summarize,per_node}.golden`) with the
body sorted (odgi's `-d` row order is thread-dependent). `tests/regression/test_depth.py`
pins hand-computed values and the `-t 1` vs `-t 4` determinism.

Correspondence: `gfaz depth` == `odgi depth -S`; `gfaz depth -d` == `odgi
depth -d`.

## Algorithm

Streaming, reusing the shared traversal layer (see
[../EXTENDING_COMPUTE_ENGINE.md](../EXTENDING_COMPUTE_ENGINE.md)):

1. `load_traversals` + `load_rulebook` + `make_rule_cache`.
2. Parallel over slices, `stream_decoded_nodes` into a single shared
   `node_total[]` (atomic `+= 1` per visit). For `-d`, a per-thread "last path
   seen" stamp drives a second shared `node_uniq[]` (atomic `+= 1` once per path
   per node) for the distinct-path count.
3. Summary derives `step.count` and `path.length` from `node_total[]` and the
   segment lengths.

A single shared counter (not per-thread copies) keeps peak memory at
O(num_nodes); atomics scatter across ~100M nodes with negligible contention (the
same pattern as `gfaz growth`). Determinism comes from integer-exact atomics plus
node-id-ordered emission.

## Benchmark

`scripts/benchmark/compare_depth_perf.py --gfa graph.gfa --gfaz graph.gfaz
--concordance` times gfaz vs `odgi build` + `odgi depth -d` (wall + peak RSS) and
verifies value concordance on graphs whose node ids coincide.

## Out of scope (v1)

- `odgi depth`'s default per-path table (`#path start end mean.depth`) and the
  `-D`/`-a`/`-v` vector forms; the gfaz default is the summary.
- Per-sample / per-haplotype grouping (odgi's `-d` is path-based).
