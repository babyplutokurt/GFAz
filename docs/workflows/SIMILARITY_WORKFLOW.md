# `gfaz similarity` — all-vs-all group similarity matrix

`gfaz similarity` computes a pairwise similarity matrix between haplotype groups
directly from a compressed `.gfaz`, reproducing `odgi similarity`'s output
without materializing the graph. It is GFAz's first **path-pair-iterative** app:
N² group comparisons against memory proportional to traversal coverage, not the
full graph object.

```
gfaz similarity -i graph.gfaz [-S | -H | -p] [-d] [-a] [-t N] > matrix.tsv
```

## Output

Tab-delimited to stdout, one line per ordered group pair, in `odgi similarity`'s
exact column layout:

```
group.a  group.b  group.a.length  group.b.length  intersection \
  jaccard.similarity  cosine.similarity  dice.similarity  estimated.identity
```

With `-d` the four similarity columns become `*.distance` /
`estimated.difference.rate` plus `euclidean.distance` and `manhattan.distance`.

By default only pairs that co-occur on ≥1 node are emitted (sparse); `-a` emits
every ordered pair including zero-intersection ones. Self-pairs are always
included (their intersection is the group's own length). Rows are emitted in
deterministic `(a, b)` group order (thread-count invariant).

## Definitions (multiplicity-aware, matching odgi)

For a group `g` (its member P/W traversals) and node `n` with length `len(n)`:

```
cov_g(n) = len(n) * (number of steps of g on n)      # repeated visits count
L_g      = Σ_n cov_g(n)                               # group.length
I(a,b)   = Σ_n min(cov_a(n), cov_b(n))                # intersection
```

```
jaccard = I / (La + Lb − I)        cosine = I / √(La·Lb)
dice    = 2I / (La + Lb)           estimated.identity = 2·jaccard / (1 + jaccard)
euclidean = √(La + Lb − 2I)        manhattan = La + Lb − 2I
```

This is a coverage-histogram intersection, **not** a set/union Jaccard: a node
traversed twice (or by two paths in the group) contributes twice. That matches
`odgi similarity` exactly.

## Grouping

`-S` group by sample (PanSN `sample#`, default) · `-H` by `sample#hap` · `-p`
each P/W line on its own. Correspondence to `odgi similarity`: gfaz `-p` ==
odgi default (no `-D`); gfaz `-S` == odgi `-D '#' -p 1`.

## Concordance with odgi

Because the output is in terms of group names and bp (no node ids), gfaz and
odgi agree on values **exactly** (printed `%.6f`), unlike `deconstruct` where
gfaz's 1-based node ids differ from vg's. `tests/concordance/test_similarity_vs_odgi.py`
checks header-verbatim + sorted-body equality against committed odgi goldens
(`tests/golden/similarity_fixture.*.golden`) for `-p`, `-S`, `-S -d`, `-S -a`.
`tests/regression/test_similarity.py` additionally pins hand-computed values.

One intentional divergence: cosine is computed in double precision
(`I/√((double)La·(double)Lb)`), avoiding odgi's `uint64*uint64` overflow for
genome-scale group lengths; identical for non-overflowing inputs.

## Algorithm

Streaming, reusing the shared traversal layer (see
[../EXTENDING_COMPUTE_ENGINE.md](../EXTENDING_COMPUTE_ENGINE.md)):

1. `load_traversals` + `load_rulebook`; build per-slice group ids
   (`path_group_key` / `walk_group_key`).
2. Pass 1 (parallel over slices): `stream_decoded_nodes` → per-slice node visit
   counts.
3. Pass 2 (parallel over groups): merge member slices → per-group `cov_g(n)`.
4. Build a node-major CSR of `(group, cov)` (groups ascending per node).
5. Pairwise (parallel over nodes): `I[a][b] += min(cov_a, cov_b)`; the diagonal
   yields `L_g`. Emit.

Memory is O(Σ_g distinct nodes per group) for the coverage/CSR plus a tiny N×N
group matrix — far below a materialized graph. Determinism comes from per-thread
merges, integer-exact atomic accumulation, and sorted emission.

## Benchmark

`scripts/benchmark/compare_similarity_perf.py --gfa graph.gfa --gfaz graph.gfaz
--grouping sample --concordance` times gfaz vs `odgi build` + `odgi similarity`
(wall + peak RSS) and verifies value concordance.

## Limitations (v1)

- Multiplicity-aware only (matches odgi); a set/union `--set` Jaccard variant is
  a future flag.
- No `--mask` node masking; grouping is PanSN-based (`-S/-H/-p`), not arbitrary
  `-D`/`-p N` delimiter positions.
