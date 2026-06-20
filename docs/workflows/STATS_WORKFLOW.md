# `gfaz stats` — graph dimension summary

`gfaz stats` reports the basic dimensions of a pangenome directly from a
compressed `.gfaz`, reproducing `odgi stats -S` (and `-b`) without materializing
the graph. It is a metadata-only query: the summary needs no traversal decode, so
it is effectively instant even on HPRC-scale inputs.

```
gfaz stats -i graph.gfaz [-S | -b]
```

## Output

Default (matches `odgi stats -S`), tab-delimited to stdout:

```
#length  nodes  edges  paths  steps
```

| Column | Meaning | Source |
|---|---|---|
| `length` | total segment bp | Σ segment lengths |
| `nodes`  | number of segments | #S-lines |
| `edges`  | number of links | #L-lines (`num_links`) |
| `paths`  | number of paths | #P-lines + #W-lines (odgi counts walks as paths) |
| `steps`  | total node visits | Σ pre-grammar traversal lengths |

With `-b` (matches `odgi stats -b`) it emits the base content instead, one row
per canonical base:

```
A  <count>
C  <count>
G  <count>
T  <count>
```

counted over the (uppercased) segment sequences.

## Concordance with odgi

These are pure aggregates, independent of node identity, so gfaz and odgi agree
**byte-for-byte**. `tests/concordance/test_stats_vs_odgi.py` checks the summary
and base content against committed odgi goldens
(`tests/golden/stats_fixture.{summarize,base}.golden`); this never invokes odgi.
`tests/regression/test_stats.py` additionally pins the hand-computed values on
`similarity_fixture.gfa`.

Correspondence: `gfaz stats` == `odgi stats -S`; `gfaz stats -b` == `odgi
stats -b`. (odgi drops W-lines on `odgi build`, so the test fixture is path-only;
gfaz counts both P- and W-lines.)

## Notes

- The summary reads only the compressed container metadata (segment lengths,
  link count, path/walk counts, pre-grammar traversal lengths) — no traversal
  decode, no graph object.
- Node counts are independent of node identity; gfaz uses its own 1-based node
  ids and never recovers the original GFA segment names.
- `-b` reports only A/C/G/T (matching odgi's four-row output); other characters
  (e.g. `N`) are not tallied.

## Parity status vs `odgi stats`

Implemented modes are **byte-exact** vs odgi. The rest is not implemented (v1);
this table is the to-do list if we extend parity later.

| `odgi stats` flag | Output | gfaz | Notes / what it would take |
|---|---|---|---|
| `-S` summarize | `#length nodes edges paths steps` | ✅ byte-exact (gfaz default) | metadata-only |
| `-b` base content | A/C/G/T counts | ✅ byte-exact | segment-seq tally; only A/C/G/T reported (no `N`) |
| `-W` weakly-connected components | per-component dims | ❌ | needs a union-find over the L-line edge set (have `build_segment_graph_from_links` in `snarl_finder.hpp` to build on) |
| `-L` self-loops | count | ❌ | scan L-lines for `from == to` |
| `-N` nondeterministic edges | edge sets | ❌ | group edges by (node-side, next base); needs segment first/last bases |
| `-a` pangenome sequence-class counts | private/core/shell bp | ❌ | overlaps `gfaz growth` core/variable + planned novelty metrics |
| `-D <delim>` group summary | per-group dims | ❌ | reuse `path_group_key` to bucket paths |
| `-f` file size / sorting-goodness eval | misc | ❌ | low value here |

Note: `gfaz stats` (no flag) emits the `-S` summary; `odgi stats` with no flag
emits nothing useful (it requires a mode flag), so there is no default-mode
divergence to worry about.
