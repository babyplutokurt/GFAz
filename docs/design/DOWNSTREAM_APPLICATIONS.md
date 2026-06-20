# GFAz Downstream Applications Roadmap

Tracking candidate applications where GFAz can act as a memory-saving compute
engine for GFA downstream tasks, not just a file compressor.

## Design Premise

- Keep the current random-access design: all grammar rules + all encoded
  paths/walks resident (~8 GB for a 360 GB GFA).
- Per-path / per-walk decode is cheap because the 2-mer grammar already reaches
  ~80× ratio; Zstd adds only ~1.2–1.3× on top, so decoding one traversal is
  close to a memcpy plus a recursive rule expansion.
- Sweet spot: workloads that are **path-iterative** (stream one path at a time)
  or **path-pair-iterative** (nested loop over paths, O(2) resident). These
  give asymptotic memory wins over tools that materialize the full graph.
- Explicitly out of scope for now: read alignment (GBZ territory),
  topology-heavy graph ops (sort, unchop, bubble decomp), and locus/region
  extraction (needs a node→paths inverse index we have not built yet).

## Workload Taxonomy

### Path-iterative (one path at a time, small accumulator)

| Workload | State size | Existing tool / memory pain |
|---|---|---|
| Pangenome growth curves / Heaps' law | O(nodes) bitset | Panacus (~51 GB on HPRC); odgi `pangenome-growth` (148–182 GB) |
| Core / variable / private node classification | O(nodes × groups) | Panacus, odgi |
| Node coverage / depth histograms | O(nodes) counter | odgi `depth`, Panacus |
| Haplotype k-mer profiles (for subsampling) | fixed read k-mer set | PanGenie-index (~97 GB on 394 haps) |
| Per-sample FASTA / haplotype extraction | O(1) | `vg paths --extract-fasta`, odgi `paths -f` |
| PanSN-filter projection (e.g., dump HG002#*#* as GFA) | O(1) | ad-hoc scripts |
| Haplotype-vs-reference distance | one fixed ref path | no clean tool today |

### Path-pair-iterative (outer × inner, two paths resident)

| Workload | Why it fits |
|---|---|
| All-vs-all haplotype Jaccard / Dice / shared-node matrix | N² decodes, reader state stays O(2 paths) |
| IBD / shared-run detection across haplotypes | pure pairwise scan |
| Phylogenetic distance matrices from pangenomes | same |
| Pairwise LD / co-occurrence statistics | same |
| Haplotype clustering distance input | same |

### Deferred (need new data structures)

| Workload | Missing piece |
|---|---|
| Locus / region extraction (impg-style) | node→paths inverse index |
| PWAS / per-node genotype vectors | node→paths inverse index |
| Reference-coordinate region queries | ref path bp→node offset index |

## Shipped

| # | Target | Status | Notes |
|---|---|---|---|
| 1 | Pangenome growth (Panacus-equivalent) | ✅ shipped | `gfaz growth` — multithreaded (OpenMP), node-counted, all grouping modes (path / sample-hap-seq / sample-hap / sample). See [GROWTH_WORKFLOW.md](../workflows/GROWTH_WORKFLOW.md). Core/variable + bp + grammar push-down remain follow-ups. |
| 1b | PAV over BED ranges (odgi-equivalent) | ✅ shipped | `gfaz pav` — long/matrix/binary output, record/sample/sample#hap grouping. See [PAV_WORKFLOW.md](../workflows/PAV_WORKFLOW.md). |
| 1c | Deconstruct GFA → VCF (vg-equivalent) | ✅ shipped | `gfaz deconstruct` — default matches `vg deconstruct` at ~99.99% concordance, 17–24× faster, ~6.7× less RAM (HPRC-v1.1 whole-genome). See [DECONSTRUCT_WORKFLOW.md](../workflows/DECONSTRUCT_WORKFLOW.md). |
| 2 | All-vs-all similarity matrix (odgi-equivalent) | ✅ shipped | `gfaz similarity` — **first path-pair-iterative app**; matches `odgi similarity` *exactly* (values agree at %.6f, not just position-level, since output is group-name/bp only). Full parity: similarities, `-d` distances, `-a` all-pairs, `-S/-H/-p` grouping. See [SIMILARITY_WORKFLOW.md](../workflows/SIMILARITY_WORKFLOW.md). |
| 3 | Graph stats (odgi-equivalent) | ✅ shipped (subset) | `gfaz stats` — `-S` `#length nodes edges paths steps` summary + `-b` A/C/G/T base content, byte-identical to `odgi stats`. Metadata-only (no traversal decode). Other `odgi stats` flags (`-W`/`-L`/`-N`/`-a`/`-D`) not yet done — full parity table in [STATS_WORKFLOW.md](../workflows/STATS_WORKFLOW.md). |
| 3b | Node coverage depth (odgi-equivalent) | ✅ shipped (subset) | `gfaz depth` — `-S` summary + `-d` per-node `depth`/`depth.uniq` table, matching `odgi depth`. Streams into one shared O(num_nodes) counter (no materialized graph). odgi's default per-path table and `-v`/`-D`/`-a`/window/position modes not yet done — full parity table in [DEPTH_WORKFLOW.md](../workflows/DEPTH_WORKFLOW.md). |

`growth`/`pav`/`deconstruct` are **path-iterative** (stream one path at a time);
`similarity` opens the **path-pair-iterative** regime (N² comparisons, coverage-
vector resident). All are built on the shared traversal layer documented in
[../EXTENDING_COMPUTE_ENGINE.md](../EXTENDING_COMPUTE_ENGINE.md) — read that
before starting any new app.

## 2026 external scan

A quick landscape check (June 2026) — the planned directions still have no direct
streaming-at-scale competitor, and the field is moving toward exactly the
matrix/feature representations GFAz is positioned to emit:

- **Allele-centric / pan-graph-matrix representations** are an active direction
  for scalable analysis ([arXiv 2512.21320](https://arxiv.org/html/2512.21320v1)),
  reinforcing the matrix/feature-table framing in
  [COMPUTE_ENGINE_DIRECTION.md](COMPUTE_ENGINE_DIRECTION.md).
- **Graph/path comparison tools** — gretl (graph QC metrics) and PANCAT (variation-
  graph diffs via edit distance) — show demand for path-similarity/QC outputs.
- **Low-coverage pangenome genotyping** continues to grow, keeping the
  haplotype-scoring/subsampling direction relevant.
- General catalog: [awesome-pangenomes](https://github.com/colindaven/awesome-pangenomes).

Takeaway: no change of course needed — the existing picks are well-aimed; what
follows just ranks them for execution.

## Recommended next builds (ranked)

Scored on fit-to-GFAz (does it match the path-iterative / path-pair streaming
model?), effort, paper value, and what it reuses from the extension surface.

The rank-1 flagship (all-vs-all similarity matrix) and the rank-2 companions
(`stats` / `depth`) are now **shipped** (see Shipped above). Remaining candidates:

| Rank | App | Regime | Fit | Effort | Paper value | Reuses |
|---|---|---|---|---|---|---|
| 1 (quick win) | Per-haplotype FASTA / GFA extraction by PanSN selector | path-iterative | ★★★ | low | ★ | `stream_decoded_nodes` + segment sequences; the "samtools view" of pangenomes |
| 2 (PAV extension) | Novelty / panel metrics (private/accessory/core bp, non-panel burden) | path-iterative | ★★ | medium | ★★ | panel node bitset + stream; Phase 3 in [COMPUTE_ENGINE_DIRECTION.md](COMPUTE_ENGINE_DIRECTION.md) |
| — (deferred) | All-vs-all IBD / shared-run detection | path-pair | ★★ | high | ★★★ | builds on `similarity`'s pair machinery; needs careful match definitions |

**Recommended order and rationale:**

1. **`extraction`** — low-effort, reuses `stream_decoded_nodes` + segment
   sequences, and broadens adoption (the "samtools view" of pangenomes, unlocking
   interactive users); baseline `vg paths --extract-fasta` / `odgi paths -f`.
   Note `gfaz extract-path` / `extract-walk` already cover single-line extraction,
   so this is mostly a PanSN-selector + FASTA wrapper on top.
2. **Then novelty/panel metrics** as the PAV-engine extension once multi-reference
   panel input is wanted.
3. **IBD / shared-runs** is the natural deeper follow-up to `similarity` in the
   path-pair regime, but needs careful long-match/positional definitions
   (see [COMPUTE_ENGINE_DIRECTION.md](COMPUTE_ENGINE_DIRECTION.md) §"Why Not
   Start With IBD").

(`Haplotype k-mer scoring` for PanGenie/Giraffe subsampling remains the highest
*downstream* impact but is gated on the open profiling question below — confirm
the bottleneck is panel loading before committing.)

## Strategic Framing

GFAz wins where the workload is **path-iterative or path-pair-iterative**. The
two regimes are the load-bearing distinction:

- **path-iterative** — stream one path at a time into a small accumulator
  (O(nodes) or O(windows)). All three shipped apps (`growth`, `pav`,
  `deconstruct`) live here.
- **path-pair-iterative** — nested loop over paths, only O(2 paths) resident
  (all-vs-all distances, IBD, LD). **Not yet demonstrated** — the flagship next
  build (the distance matrix) is what opens it.

Framing for the paper / docs:

- **GBZ** is the format for read mappers (Giraffe, GBWT-style bidirectional
  extension).
- **odgi** is for topology-heavy ops that need the full graph materialized.
- **GFAz** is for per-haplotype analytics at population scale — growth curves,
  pairwise distances, haplotype scoring/subsampling, extraction — with
  laptop-class memory on HPRC-scale inputs.

## Open Questions

- PanGenie memory profile: is the win from loading the panel, or from the k-mer
  index step? Need to profile before committing to (3).
- Block-level chunking of paths: not needed for current 8 GB resident target,
  but revisit if we want truly lazy decode on cloud/Lambda deployments.
- Walk-by-name lookup is currently O(N) linear scan
  (`src/compress/extraction_workflow.cpp:308`). Fine for the 4 targets above,
  but fix before any interactive / per-query service.
