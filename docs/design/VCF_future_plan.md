> **Status (2026-06):** Item 1 below — `gfaz deconstruct` — is **shipped**. Its
> default reproduces `vg deconstruct` at ~99.99% position concordance while running
> 17–24× faster; see [DECONSTRUCT_WORKFLOW.md](../workflows/DECONSTRUCT_WORKFLOW.md). This file
> now tracks the *remaining* VCF roadmap (items 2–5). The deconstruct entry is kept
> for historical context.

  ✅ 1. gfaz deconstruct — GFA → VCF (the flagship) — **SHIPPED**

  Pick a reference path (e.g. GRCh38's chr1 walk). Walk the graph's "bubbles" (places where haplotypes diverge then reconverge), and for each one emit a VCF row: REF allele = reference's sub-traversal, ALT =
   the alternative sub-traversals, and per-sample genotypes = which allele each haplotype took.

  - Why GFAz wins: the standard tool, vg deconstruct, is a known memory/time bottleneck. You already iterate all haplotype traversals cheaply at laptop-class RAM (your 360 GB → 40s benchmark). Bubble-finding
   + per-sample allele assignment is exactly the path-iterative shape pav already has.
  - Impact: instantly makes every .gfaz consumable by the entire variant-analysis ecosystem. This is the highest-leverage feature.
  - Effort: medium-high — the genuinely new piece is snarl/bubble enumeration relative to a reference path.
  - Outcome: delivered with default top-level-snarl (vg-compat) mode; legacy `--snarl`/`--linear` modes deprecated.

  2. gfaz genotype / allele-frequency export

  A lighter cousin of deconstruct: given variant sites (or just per-node), report allele frequency and which haplotypes carry which allele — emitted as a VCF INFO (AF) or a genotype matrix. This is pav
  generalized into VCF's allele vocabulary.
  - Effort: low-medium — mostly a reformatting + aggregation layer over existing traversal membership logic.
  
  3. Genotype-matrix compression (GFAz codec applied to VCF itself)

  A VCF's genotype matrix (samples × sites) is extremely repetitive because of linkage disequilibrium — neighboring variants travel together in haplotype blocks, so rows look like your repeated subpaths.
  Your 2-mer grammar + columnar zstd could compress genotype matrices the way specialized tools (GTC, BGT, GTShark, xsi) do.
  - Why interesting: reuses your codec engine on a second, huge market — but it's a somewhat separate data model from graphs.
  - Effort: medium — new front-end parser, but the codec core transfers. 

  4. gfaz construct — VCF → GFA

  The reverse: build/augment a pangenome from a reference + VCF, then compress. Useful for round-tripping, but the input is already "variant-only," so the compression upside is smaller than deconstruct's
  ecosystem upside.

  5. Region-targeted VCF

  Combine your existing BED + extract-path plumbing with deconstruct to emit VCF for just a locus — fast clinical/region queries without touching the whole graph.