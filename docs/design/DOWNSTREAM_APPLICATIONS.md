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

## Prioritized First Implementations

| # | Target | Status | Notes |
|---|---|---|---|
| 1 | Pangenome growth (Panacus-equivalent) | ✅ shipped | `gfaz growth` — multithreaded (OpenMP), node-counted, all grouping modes (path / sample-hap-seq / sample-hap / sample). See [GROWTH_WORKFLOW.md](../workflows/GROWTH_WORKFLOW.md). Core/variable + bp + grammar push-down remain follow-ups. |
| 1b | PAV over BED ranges (odgi-equivalent) | ✅ shipped | `gfaz pav` — long/matrix/binary output, record/sample/sample#hap grouping. See [PAV_WORKFLOW.md](../workflows/PAV_WORKFLOW.md). |
| 1c | Deconstruct GFA → VCF (vg-equivalent) | ✅ shipped | `gfaz deconstruct` — default matches `vg deconstruct` at ~99.99% concordance, 17–24× faster. See [DECONSTRUCT_WORKFLOW.md](../workflows/DECONSTRUCT_WORKFLOW.md). |
| 2 | All-vs-all haplotype distance matrix (Jaccard / shared-node) | planned | N² work, O(1) memory — nothing on the market does this streamingly at scale. |
| 3 | Haplotype k-mer scoring for PanGenie / Giraffe subsampling | planned | Highest downstream impact; validate that panel loading (not k-mer index build) is the real bottleneck first. |
| 4 | Per-haplotype FASTA / GFA extraction CLI + Python API | planned | Trivial to build; becomes the "samtools view" of pangenomes; unlocks interactive users. |

## Strategic Framing

GFAz wins where the workload is **path-iterative or path-pair-iterative**.
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
  (`src/workflows/extraction_workflow.cpp:308`). Fine for the 4 targets above,
  but fix before any interactive / per-query service.
