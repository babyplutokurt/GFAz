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

## Out of scope (v1)

`odgi stats`'s topology metrics (`-W` weakly-connected components, `-L`
self-loops, `-N` nondeterministic edges, `-a` pangenome sequence-class counts)
are not implemented; the class counts overlap with `gfaz growth` / planned
novelty metrics.
