# GFAz PAV Workflow

This document describes how `gfaz pav` computes presence/absence variants
(PAVs) over a compressed GFA, why it is structured the way it is, and what
benchmarks against `odgi pav` look like in practice. It also lists the next
optimisation directions we plan to pursue.

## What PAV computes

For each BED range `(chrom, start, end)` on a reference path/walk and each
query group (a set of paths/walks), the PAV ratio is

```
                Σ (length of node n) for unique n in [start,end) ∩ group
PAV_ratio  =  ─────────────────────────────────────────────────────────────
                Σ (length of node n) for unique n in [start,end)
```

A node counts at most once per group regardless of how many times the group
traverses it. The ratio is a per-group fraction of the reference window's
node-length covered by that group.

This matches `odgi pav`'s semantics so results are directly comparable.

## Architecture

### Public API (no change since the rewrite)

```
PavResult compute_pav(const CompressedData &data,
                      const PavOptions   &options);
```

`compute_pav` reads a `CompressedData` (the in-memory form of a `.gfaz` file)
and returns a flat row-major matrix of numerators/denominators plus group
labels. The CLI (`pav_command.cpp`) handles arg parsing and printing only.

### Internal pipeline

`compute_pav` runs three passes after a one-shot setup, all sharing a
read-only rule-leaf cache built up front:

```
                    +-----------------------------+
                    |  Setup                      |
                    |  - decompress lengths,      |
                    |    rules, paths, walks      |
                    |  - build slices             |
                    |  - build group metadata     |
                    |  - resolve BED chroms       |
                    |  - build RuleLeafCache      |  (single-threaded, RO afterwards)
                    +--------------+--------------+
                                   |
                                   v
            +----------------------+----------------------+
            |  Pass 1 — parallel slice decode             |
            |  - one decode per slice                     |
            |  - lock-free: each slice writes to its own  |
            |    slice_nodes[s] (sorted unique node ids)  |
            |  - reference slices additionally emit an    |
            |    ordered node-id stream into ref_streams  |
            +----------------------+----------------------+
                                   |
                                   v
            +----------------------+----------------------+
            |  Pass 2 — group-set assembly                |
            |  - per-group concat + sort + unique         |
            |    over slice_nodes                         |
            |  - flatten to CSR                           |
            |    (node_offsets, node_to_group_ids)        |
            |  - free slice_nodes and group_nodes         |
            +----------------------+----------------------+
                                   |
                                   v
            +----------------------+----------------------+
            |  Pass 3 — chrom sweep                       |
            |  - parallel over distinct BED chroms        |
            |  - for each chrom:                          |
            |    walk its cached ref_stream, compute      |
            |    node offsets on the fly, intersect with  |
            |    sorted BED ranges, dedupe (window,node)  |
            |  - per (window,node): atomic += length to   |
            |    denominator and to numerators[w*G+gid]   |
            |    for every gid in node_to_groups[node]    |
            +---------------------------------------------+
```

Key properties that fall out of this layout:

1. Each slice is decoded **exactly once** for the entire computation. The
   reference walk's ordered stream is captured as a free side effect of the
   pass 1 decode that the group-membership pass already performs.
2. The hot inner loops are lock-free. Pass 1 writes to per-slice slots; pass
   3 uses `#pragma omp atomic` only on the final accumulation, which is
   contention-free in the common case (different threads handle different
   chroms, hence different `wid` ranges).
3. Memory layout is CSR for the node→groups index, eliminating the
   `vector<vector<uint32_t>>` per-node header overhead.
4. The rule-leaf cache turns the dominant per-leaf cost from recursive
   grammar descent into a flat array iteration whenever a rule's expansion
   fits the budget.

### Rule-leaf cache

`RuleLeafCache` stores the forward leaf sequence of each cacheable rule.
Built bottom-up before pass 1; read-only thereafter. Reverse expansions
iterate the cached array in reverse with sign-negation, so we only store one
direction.

Budget-aware: a rule is cached only if its accumulated leaf list (including
recursively expanded child rules) fits within the remaining byte budget.
Rules that don't fit fall back to recursive expansion at decode time. The
budget defaults to 1 GiB and can be tuned via `GFAZ_PAV_RULE_CACHE_BYTES`.

### Grouping modes

| Flag                          | Mode             | Key |
|-------------------------------|------------------|-----|
| (none)                        | `PerPathWalk`    | full path/walk name |
| `-S` / `--group-by-sample`    | `Sample`         | PanSN sample (first `#` field) |
| `-H` / `--group-by-haplotype` | `SampleHap`      | `sample#hap` if present, else `sample` |

PanSN parsing strips trailing `:start-end` coordinate suffixes so subpath
names produced by `odgi build` (e.g. `HG00438#1#chr5:0-149820500`) collapse
correctly into their parent sample/haplotype.

## File map

```
src/cli/pav_command.cpp            # arg parsing + output formatting
src/workflows/pav_workflow.cpp     # compute_pav implementation
include/workflows/pav_workflow.hpp # PavOptions / PavResult / compute_pav
tests/regression/test_cli_commands.py  # pav fixture regressions
tests/fixtures/pav_*.bed           # test BEDs
scripts/benchmark/pav_compare_gfaz_odgi.py  # path-only A/B benchmark driver
```

## Configuration knobs

| Env var                          | Default | Effect |
|----------------------------------|---------|--------|
| `GFAZ_PAV_RULE_CACHE_BYTES`      | `1 GiB` | Budget for the rule-leaf cache. `0` disables caching. |
| `GFAZ_NUM_THREADS`               | half of CPUs, capped at 8 | Used when `-t 0` is passed. |
| `OMP_NUM_THREADS`                | —       | Honoured if `GFAZ_NUM_THREADS` is unset. |

Threads are also controllable per-invocation via `-t / -j N`.

## Benchmark commands

The same workflow applies to any GFA. Below: full pipeline for a path-only
GFA and for a walk-bearing GFA (HPRC-style).

### Tooling locations (example layout used in development)

```
GFaz_BIN=/home/kurty/Release/gfa_compression/build/bin/gfaz
ODGI_BIN=/home/kurty/Project/gfaz_odgi/odgi/bin/odgi
TIME_BIN=/usr/bin/time
THREADS=32
```

### Path-only GFA — end-to-end

```bash
GFA=/path/to/graph.paths_only.gfa
WORK=/data1/work/kurty/gfaz/example_pav
mkdir -p "$WORK"

GFAZ=$WORK/$(basename "$GFA" .gfa).gfaz
OG=$WORK/$(basename "$GFA" .gfa).og
BED=$WORK/$(basename "$GFA" .gfa).whole_paths.bed
GFAZ_OUT=$WORK/gfaz.pav.tsv
ODGI_OUT=$WORK/odgi.pav.tsv

# 1. Compress GFA -> GFAz
"$TIME_BIN" -v "$GFaz_BIN" compress "$GFA" "$GFAZ" \
    2> "$WORK/gfaz_compress.time.log"

# 2. Build whole-path BED from P-lines (one line per path).
awk 'BEGIN{OFS="\t"} $1=="P"{print $2, 0, 1000000000, $2}' "$GFA" > "$BED"
wc -l "$BED"

# 3a. Run GFAz pav.
"$TIME_BIN" -v "$GFaz_BIN" pav \
    -i "$GFAZ" -b "$BED" -S -M -t "$THREADS" \
    > "$GFAZ_OUT" 2> "$WORK/gfaz_pav.time.log"

# 3b. Build odgi index, then run odgi pav.
"$TIME_BIN" -v "$ODGI_BIN" build -g "$GFA" -o "$OG" -t "$THREADS" \
    2> "$WORK/odgi_build.time.log"

"$TIME_BIN" -v "$ODGI_BIN" pav \
    -i "$OG" -b "$BED" -S -M -t "$THREADS" \
    > "$ODGI_OUT" 2> "$WORK/odgi_pav.time.log"

# 4. Compare wall-clock and RSS.
grep -E 'Elapsed|Maximum resident' "$WORK/"*.time.log

# 5. Sort and diff outputs (byte-identical for path-only GFAs in practice).
diff <(sort "$GFAZ_OUT") <(sort "$ODGI_OUT") | head -20
```

There is a script that automates the path-only flow:

```bash
python3 scripts/benchmark/pav_compare_gfaz_odgi.py \
    /path/to/graph.paths_only.gfa \
    --gfaz-bin "$GFaz_BIN" \
    --odgi-bin "$ODGI_BIN" \
    --threads 32
```

It writes `*.pav_compare.tsv` with timing/RSS plus output hashes. It rejects
W-line input intentionally, because BED resolution semantics differ between
the two tools when walks are present.

### Walk-bearing GFA (HPRC) — end-to-end

`odgi build` from a raw GFA silently drops W-lines. The supported pipeline
is to compress with GFAz first, then build odgi from the GFAz round-trip
(which materialises walks as P-lines under PanSN subpath names) so both
tools see the same path table.

```bash
GFA=/data1/work/kurty/gfa/hprc-v1.1-mc-chm13.gfa
GFAZ=/data1/work/kurty/gfaz/hprc-v1.1-mc-chm13.gfaz   # already produced
WORK=/data1/work/kurty/gfaz/hprc_pav_compare_clean
mkdir -p "$WORK"

OG=$WORK/hprc-v1.1-mc-chm13.from_gfaz.og
BED=$WORK/hprc-v1.1-mc-chm13.shared_whole_walks.bed
GFAZ_OUT=$WORK/hprc-v1.1-mc-chm13.gfaz.pav.tsv
ODGI_OUT=$WORK/hprc-v1.1-mc-chm13.odgi.pav.tsv

# 1. Compress (skip if already done).
"$TIME_BIN" -v "$GFaz_BIN" compress "$GFA" "$GFAZ" \
    2> "$WORK/gfaz_compress.time.log"

# 2. Build odgi from the GFAz so walks become first-class paths.
"$TIME_BIN" -v "$ODGI_BIN" build -g "$GFAZ" -o "$OG" -t "$THREADS" \
    2> "$WORK/odgi_build.time.log"

# 3. Generate the BED from odgi's path list. This guarantees the chrom
#    column matches both tools' path/walk naming.
"$ODGI_BIN" paths -i "$OG" -L \
    | awk 'BEGIN{OFS="\t"} {print $1, 0, 1000000000, $1}' > "$BED"
wc -l "$BED"

# 4a. GFAz pav (uses original .gfaz, not the .og).
"$TIME_BIN" -v "$GFaz_BIN" pav \
    -i "$GFAZ" -b "$BED" -S -M -t "$THREADS" \
    > "$GFAZ_OUT" 2> "$WORK/gfaz_pav.time.log"

# 4b. odgi pav.
"$TIME_BIN" -v "$ODGI_BIN" pav \
    -i "$OG" -b "$BED" -S -M -t "$THREADS" \
    > "$ODGI_OUT" 2> "$WORK/odgi_pav.time.log"

# 5. Timing + RSS summary.
grep -E 'Elapsed|Maximum resident' "$WORK/"*.time.log

# 6. Sorted diff (column-by-column equality is not guaranteed yet — see
#    "Known divergences" below).
(head -n1 "$ODGI_OUT" && tail -n +2 "$ODGI_OUT" | sort -k1,1 -k2,2n -k3,3n -k4,4) > "$WORK/odgi.sorted.tsv"
(head -n1 "$GFAZ_OUT" && tail -n +2 "$GFAZ_OUT" | sort -k1,1 -k2,2n -k3,3n -k4,4) > "$WORK/gfaz.sorted.tsv"
diff -u "$WORK/odgi.sorted.tsv" "$WORK/gfaz.sorted.tsv" | head -40
```

## Reference benchmark numbers

Captured on this machine with 32 threads. Datasets, BEDs, and binaries as
in the commands above.

### chr1 — path-only, 90 paths, 2,262 BED ranges

| Tool                | Time      | Max RSS     |
|---------------------|-----------|-------------|
| GFAz pav (current)  | 9.06 s    | 9.30 GB     |
| GFAz pav (previous) | 84.06 s   | 4.30 GB     |
| odgi pav            | 49 min 41 s | 29.91 GB  |

Speedup vs. odgi: ~329×. Speedup vs. previous GFAz: ~9.3×. Memory regressed
vs. previous GFAz (4.3 → 9.3 GB) — see "Future work" for the planned fix.

### HPRC `hprc-v1.1-mc-chm13.gfa` — 30,640 walks, 30,640 BED ranges

| Tool                | Time      | Max RSS     |
|---------------------|-----------|-------------|
| GFAz pav (current)  | 1 min 16 s | 71.99 GB   |
| GFAz pav (previous) | 16 min 45 s | 38.26 GB  |
| odgi pav            | 10 min 53 s | 227.21 GB |

Speedup vs. odgi: ~8.6×. Speedup vs. previous GFAz: ~13.2×. Memory regressed
vs. previous GFAz (38.3 → 72 GB) — same root cause as on chr1.

For context, odgi additionally pays an ~4 min `odgi build` step (RSS 123 GB)
that GFAz does not need; GFAz operates directly on the `.gfaz`.

## Known divergences vs. odgi

A handful of HPRC matrix rows differ in the 4th–5th decimal between GFAz
and odgi when grouping with `-S`. The set of (chrom,start,end,name,group)
keys matches; the numerator values differ slightly on a small subset of
rows. Likely causes still under investigation:

1. odgi's `-S` and `-H` use a field-count split on `#`. GFAz uses a PanSN
   parser that also strips `:start-end` coordinate suffixes. For names
   shaped `sample#contig` (no haplotype field) the two can disagree on
   which token is the sample.
2. Order in which steps map to groups. odgi unions step-derived group ids
   per node; GFAz unions slice-derived group ids per group. Both should be
   set-equivalent, but the two implementations may diverge on slices that
   map to the same group via different paths.

Chr1 path-only outputs are byte-identical between the tools, so the
divergence is specific to mixed P+W or PanSN subpath naming.

## Future work

In rough priority order; (1) is the active next change.

1. **Memory reduction (next change).** Remove the per-slice
   `slice_nodes[s]` retention. In pass 1, push each slice's deduped node
   set straight into a per-thread, per-group accumulator and drop the
   slice's local buffer. Merge per-group across threads before pass 2.
   Free `ref_streams[c]` immediately after pass 3 sweeps it (use a
   `vector<unique_ptr<vector<uint32_t>>>` so individual chroms can be
   reclaimed inside the parallel loop). Expected: chr1 RSS back near
   ~5 GB, HPRC RSS back near ~30 GB, no speed regression.

2. **Pin down the numeric divergence vs. odgi.** Build a single-window BED
   on a name where the two disagree, hand-trace which group is misclassified
   under `-S`, and fix the parser (or odgi's behaviour, depending on which
   is canonical for downstream tooling).

3. **`-p / --path-groups`** support. Accept odgi's two-column TSV
   (`path.name<TAB>group.name`) so users can map paths to arbitrary groups.

4. **Auxiliary per-slice node-set sidecar.** Optionally precompute a
   Roaring bitmap per path/walk during compression and persist it next to
   the `.gfaz`. Pass 1 of `pav` then becomes "OR a few bitmaps" with no
   grammar decoding at all. Roaring bitmaps for HPRC ≈ tens of MB total,
   smaller than the rule cache. Opt-in so non-PAV users don't pay storage.

5. **`--stats` instrumentation.** Print time spent per phase (rule cache
   build, slice decode, group assembly, BED sweep) and counts (slices
   decoded, distinct chroms, ref re-decodes — should now be 0). Helps
   future tuning be data-driven instead of speculative.

6. **Streaming output.** For matrix output on huge BEDs, format and emit
   rows as soon as their numerator row is finalised so peak memory is
   bounded by `num_groups`, not `num_windows * num_groups`. Currently we
   keep the full numerator array.

7. **Per-walk reference offset cache on disk.** Optional: cache the decoded
   reference offset stream in a sidecar so repeated PAV queries against
   the same `.gfaz` skip pass 1 entirely for previously-seen reference
   walks. Pure win for repeated workloads, no help for one-shot.

8. **GPU offload of pass 1.** Slice decoding is embarrassingly parallel
   and dominated by integer arithmetic over compact arrays. Worth
   exploring once the CPU pipeline is fully tuned and we have a clear
   roofline.
