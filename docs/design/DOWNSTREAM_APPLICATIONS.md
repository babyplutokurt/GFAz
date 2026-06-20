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

All three shipped apps are **path-iterative** (stream one path at a time). They
are built on the shared traversal layer documented in
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

| Rank | App | Regime | Fit | Effort | Paper value | Reuses |
|---|---|---|---|---|---|---|
| **1 (flagship)** | All-vs-all haplotype distance/similarity matrix (Jaccard / Dice / shared-node) | **path-pair** | ★★★ | medium | ★★★ | `stream_decoded_nodes` per pair → node-set/shared-count accumulator (O(2 paths) resident) |
| 2 (quick win) | Per-haplotype FASTA / GFA extraction by PanSN selector | path-iterative | ★★★ | low | ★ | `stream_decoded_nodes` + segment sequences; the "samtools view" of pangenomes |
| 3 (companions) | `depth` / `stats` (node-window coverage histograms, path/metadata summaries) | path-iterative | ★★★ | low | ★ | same coverage accumulator as `growth`/`pav` |
| 4 (PAV extension) | Novelty / panel metrics (private/accessory/core bp, non-panel burden) | path-iterative | ★★ | medium | ★★ | panel node bitset + stream; Phase 3 in [COMPUTE_ENGINE_DIRECTION.md](COMPUTE_ENGINE_DIRECTION.md) |

**Recommended order and rationale:**

1. **Flagship: the all-vs-all distance matrix.** It opens the **path-pair-iterative
   regime** — the one workload class GFAz has not yet demonstrated — where N²
   decodes run against O(2 paths) resident memory. No materialized-graph tool does
   this streamingly at HPRC scale, so it is the strongest standalone paper claim
   and the natural input to clustering/phylogeny. Medium effort: the per-pair
   accumulator is small; the work is choosing the metric set and an efficient
   pair-iteration schedule.
2. **Ship `extraction` and `depth`/`stats` alongside it** — both are low-effort,
   reuse machinery that already exists, and broaden adoption (extraction unlocks
   interactive users; depth/stats round out the odgi/vg-equivalent surface).
3. **Then novelty/panel metrics** as the PAV-engine extension once multi-reference
   panel input is wanted.

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
