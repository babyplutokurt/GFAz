# Deconstruct Workflow (GFA → VCF)

> **Status:** Implemented. `gfaz deconstruct` derives a VCF directly from
> compressed paths/walks in a `.gfaz` container, without materializing the
> original GFA — the same path-iterative, low-memory approach used by `growth`
> and `pav`. Workflow: `src/workflows/deconstruct_workflow.cpp` (shared decode
> machinery in `src/workflows/traversal_query.cpp`); CLI:
> `src/cli/deconstruct_command.cpp`; tests:
> `tests/regression/test_deconstruct.py`.
>
> There are **three site-finding modes**. The **default is vg-compat**
> (top-level snarls), which reproduces `vg deconstruct` at ~99.99% position
> concordance on full human chromosomes while running 17–24× faster (see §11.4);
> producing output identical to `vg` is the goal of this workflow. The other two
> modes — `--snarl` (leaf-superbubble superset) and `--linear` (the flat
> reference-anchor heuristic documented in §4–§6) — are **legacy and will be
> removed in a future release**; they print a deprecation warning on stderr.
> §11 documents the topology-based modes. This document is the reference for all
> three.

## 1. Purpose

VCF (Variant Call Format) is the universal interchange format for genetic
variation. Almost every downstream tool — `bcftools`, GATK, `plink`,
imputation, GWAS, clinical pipelines — consumes VCF. A pangenome `.gfaz`
encodes the *full* variation as graph traversals, but that variation is not
directly consumable by those tools.

`gfaz deconstruct` bridges the two: given a `.gfaz` and a chosen **reference
path**, it walks the bubbles where haplotypes diverge from the reference and
emits one VCF record per variant site, with per-sample genotypes. It is the
GFAz-native equivalent of `vg deconstruct`, computed on the compressed
representation.

### Scope decisions (v1)

These were chosen deliberately to ship a correct, useful first version:

- **Flat, biallelic-friendly decomposition.** Anchor-based, reference-relative
  sites. Multiple ALT alleles per site are allowed, but there is **no nested
  snarl recursion** (no `LV`/`PS` snarl-tree annotations). Nested decomposition
  is deferred to v2.
- **Per-sample genotypes, ploidy from haplotypes.** One VCF sample column per
  sample; the sample's haplotype paths/walks are combined into a phased
  genotype (`0|1`). Because each haplotype is a distinct traversal, genotypes
  are **inherently phased** (`|`).

## 2. Background: what a VCF row means here

```
#CHROM   POS    ID   REF   ALT   QUAL  FILTER  INFO              FORMAT  HG002   HG005
GRCh38   1000   .    A     G     .     .       AC=1;AN=4;AF=0.25 GT      0|1     0|0
GRCh38   2500   .    ATC   A     .     .       AC=2;AN=4;AF=0.5  GT      0|1     1|0
```

- `CHROM` = reference path/contig name; `POS` = 1-based genomic coordinate on
  that reference.
- `REF` = the DNA the reference spells through the bubble; `ALT` = the distinct
  DNA spellings other haplotypes take through the same bubble.
- Row 2 is a deletion: REF `ATC`, ALT `A` (left-anchored — see §5.3).
- `GT` per sample = which allele index each of that sample's haplotypes took.
  `0` = REF, `1` = first ALT, `.` = the haplotype does not span the site.

## 3. Inputs and outputs

### CLI

```bash
gfaz deconstruct -i input.gfaz -r <reference-name> [options] > out.vcf
```

| Flag | Meaning |
|---|---|
| `-i, --input <file>` | Input `.gfaz`. |
| `-r, --reference <name>` | Reference path/walk name (repeatable, or a PanSN sample → multiple contigs). Required. |
| `-S, --group-by-sample` | One VCF column per sample (default). |
| `-H, --group-by-haplotype` | One VCF column per (sample, hap). |
| `--per-path` | One VCF column per path/walk (haploid). |
| *(default)* | Emit one record per **top-level snarl** via the global biconnected decomposition, matching `vg deconstruct`'s default granularity (cyclic-reference snarls dropped). Equivalent to the former `--vg-compat`. See §11. |
| `--vg-compat` (alias `--vg-compact`) | Explicit selection of the default mode (no-op; kept for backward compatibility). |
| `--snarl` | **Legacy.** Find sites by graph topology but report the leaf-bubble *superset* (superbubbles + inversions from the stored L-line links), rather than top-level snarls. See §11. |
| `--linear` | **Legacy.** Use the flat reference-anchor heuristic (§4–§6) instead of graph topology. |
| `-t, --threads <N>` | Thread count (mirrors `pav`). |
| `--max-site-length <bp>` | Sites whose reference span exceeds this are emitted as a single `CPX`-flagged record instead of enumerating every allele (megasite guard, §4.2/§6). `0` disables the guard. |
| `--no-gt` | Emit site + `INFO` (AC/AN/AF) only, skip `GT` columns. |
| `-h, --help` | Usage. |

Output is a VCFv4.2 stream on stdout.

### Reference selection

The reference defines the coordinate system (`CHROM`, `POS`). It is identified
by path/walk name exactly as `pav` maps a BED `chrom` to a slice
(`path_name_to_slice` in `pav_workflow.cpp`). If `-r` names a PanSN sample
rather than a single contig, each contig of that sample becomes its own
`CHROM`. Every reference contig is processed independently.

## 4. Algorithm

### 4.1 Definitions

- **Anchor:** a node that appears **exactly once** in the reference traversal.
  Multi-occurrence nodes (cycles/repeats) are excluded as candidate sync points
  in v1 to keep site boundaries unambiguous.
- **Breakpoint:** an anchor that **no haplotype bridges over**. A haplotype
  *bridges* anchor `a_k` if, in its own traversal, it goes directly from some
  anchor left of `a_k` (in reference order) to some anchor right of `a_k`
  without visiting `a_k` itself — i.e. a deletion of, or a long alternate route
  around, `a_k`. Breakpoints are the anchors that survive this filter.
- **Site:** the interval between two **consecutive breakpoints** (not merely
  adjacent anchors). The reference spells some (possibly empty) intermediate
  node stretch between them — that is allele 0 (REF).
- **Allele:** a haplotype's node stretch between the same two breakpoints. If it
  differs from the reference stretch, it is an ALT allele.

> **Why breakpoints, not raw anchor pairs.** Defining sites by globally-computed
> breakpoints *before* extracting any allele makes every allele unambiguous by
> construction: a haplotype that spans a site passes cleanly through both its
> source and sink breakpoints (that is the definition of a breakpoint), so two
> haplotypes can never disagree on a site's `[source, sink]`. A haplotype that
> skips an anchor simply demotes that anchor from breakpoint status, widening
> the site to the next surviving breakpoints on each side — which is the correct
> behavior, since variation flanking a deletion is in linkage and cannot be
> separated without nested snarls (deferred to v2). This is close to how `vg`
> defines top-level snarls, so v1 approximates the top of the snarl tree and v2
> nesting is an extension rather than a rewrite.

### 4.2 Passes (mirrors the `pav` pass structure)

**Pass 0 — setup (single-threaded).**
- Decompress rules; `delta_decode_int32` on `rules_first`/`rules_second`
  (identical to `compute_pav` lines 483–488).
- Build the bottom-up **`RuleLeafCache`** (reused verbatim from
  `pav_workflow.cpp`).
- Decompress segment sequences: `seg_concat =
  zstd_decompress_string(segment_sequences_zstd)` and `seg_lengths =
  zstd_decompress_uint32_vector(segment_seq_lengths_zstd)`. Build a prefix-sum
  `seg_offset[]` so node `N`'s forward DNA is
  `seg_concat[seg_offset[N-1] .. seg_offset[N]]`.
- Build slices (`build_slices`) and grouping metadata (`build_metadata`,
  `GroupingMode`) — reused from `pav`.

**Pass 1 — reference profile (single-threaded per contig).**
- Decode the reference slice to its ordered `(node, orientation)` stream using
  `stream_decoded_nodes`.
- Compute each node's genomic offset by accumulating `seg_lengths` along the
  stream (the `offset += len` sweep from `pav` pass 3).
- Count node occurrences; mark nodes occurring exactly once as **anchors**.
  Build `anchor_to_ref_index` and `anchor_genomic_pos`. Record the reference's
  intermediate node stretch (and its orientation) for each anchor interval.

**Pass 2a — breakpoint marking (parallel over haplotypes).**
For each non-reference slice, decode to its node stream and extract its **ordered
anchor subsequence** (the anchors it visits, in the order it visits them):
- For each pair of anchors adjacent *in this haplotype's subsequence*, if they
  are not also adjacent in reference order, every reference anchor strictly
  between them is **bridged** — set it in a thread-local `skipped[]` bitset.
- If the haplotype's anchor subsequence is **not monotonically increasing** in
  reference index (inversion/rearrangement), or visits an anchor more than once,
  its region is **complex**: flag it and exclude it from breakpoint marking (it
  must not create spurious bridges). See §6.

Merge the thread-local bitsets with a bitwise OR. This pass is lock-free and the
same shape as `pav` pass 1. A haplotype that simply does not reach a region
bridges nothing, so partial haplotypes never force a merge — they become `.`
genotypes later.

**Pass 2b — site definition (trivial, single-threaded).**
`breakpoints = anchors where !skipped`. Final **sites** = the intervals between
consecutive breakpoints. O(#anchors). Each site's `[source, sink]` is fixed and
shared by every haplotype that spans it — there is no boundary reconciliation.

**Pass 3 — allele observation (parallel over haplotypes).**
With sites now fixed, walk each haplotype again. For each site it spans cleanly
(visits both source and sink breakpoints consecutively in its own traversal),
record its allele = the node stretch between them. Compute an **allele
signature** (hash of the oriented node-id stretch) to dedupe without spelling
DNA. Emit `(site_id, signature, node_stretch, slice_id)` into a thread-local
buffer. (Pass 2a may cache each haplotype's decoded anchor positions to avoid a
second decode; this is a memory-vs-recompute knob, like the rule-cache budget.)

**Pass 3.5 — site assembly (merge).**
- Per site, assign allele indices: REF stretch = `0`; distinct ALT signatures =
  `1..n` in sorted order.
- Build per-slice → allele-index membership.
- Collapse slices to VCF columns via the chosen `GroupingMode`: for each sample
  column, gather the allele index from each of its haplotypes that span the
  site → phased `GT` (`|`-joined). Haplotypes absent from the site → `.`.
- **Megasite guard:** if a site exceeds `--max-site-length` (bp span) or a
  distinct-allele cap, emit it as a single record flagged `INFO=...;CPX` (or a
  symbolic `<CPX>` ALT) with a stderr warning, instead of exploding into a
  pathological multiallelic record. v2 nesting is what recovers resolution here.

**Pass 4 — emit (sorted by POS).**
- For each site with ≥1 ALT: spell REF/ALT DNA (§5), compute `POS` with the
  left-anchor base, compute `INFO` (AC/AN/AF/NS), write the row.
- Rows are emitted per contig in ascending `POS`.

## 5. VCF field construction

### 5.1 Spelling DNA from a node stretch

Concatenate segment DNA along the oriented stretch. **A node traversed in
reverse orientation (negative node id) contributes the reverse complement of
its sequence.** No reverse-complement helper currently exists in the codebase
(confirmed: no `reverse_complement`/`revcomp` in `include/` or `src/`) — one
must be added, e.g. `gfaz::reverse_complement(std::string_view)` in a small
sequence-utility header, handling `A/C/G/T/N` and lowercase.

### 5.2 Field-by-field mapping

| VCF field | Source |
|---|---|
| `##fileformat` | `VCFv4.2` |
| `##contig=<ID=...,length=...>` | reference contig name + sum of `seg_lengths` along the reference |
| `CHROM` | reference contig name |
| `POS` | 1-based genomic position of the left-anchor base (§5.3) |
| `ID` | `.` (v2 may emit a snarl-style `>a>b` id) |
| `REF` | left-anchor base + reference intermediate DNA |
| `ALT` | comma-joined distinct ALT spellings (left-anchor base + alt intermediate DNA) |
| `QUAL` | `.` |
| `FILTER` | `.` |
| `INFO` | `AC` (per-ALT allele count), `AN` (total called alleles), `AF` (AC/AN), `NS` (samples with data); optional `AT` (vg-style allele traversal string) behind a flag in v2 |
| `FORMAT` | `GT` (omitted entirely under `--no-gt`) |
| sample columns | phased `GT`, `|`-joined across the sample's haplotypes |

### 5.3 Left-anchor base (indel convention)

A site bounded by source anchor `a_i` and sink anchor `a_{i+1}` may have an
empty reference intermediate (pure insertion) or empty alt intermediate (pure
deletion). VCF disallows empty REF/ALT, so we **left-pad** with the last base of
the source anchor node:

- `POS` = genomic coordinate of that last base of `a_i`.
- `REF` = `last_base(a_i)` + `ref_intermediate_DNA`.
- `ALT_k` = `last_base(a_i)` + `alt_intermediate_DNA_k`.

For pure SNPs (single differing node, equal length) this still produces a valid
record; trimming to a minimal SNP representation is an optional normalization
step (or left to `bcftools norm`).

## 6. Edge cases and v1 limitations

| Case | v1 handling |
|---|---|
| Haplotype does not span a site (missing an anchor) | `GT` = `.` for those haplotypes |
| Node occurs multiple times in the reference | excluded as an anchor; bubble merges into the surrounding unique-anchor interval |
| Anchor skip (haplotype skips a reference anchor) | the skipped anchor is demoted from breakpoint status (pass 2a), automatically widening the site to the next surviving breakpoints; alleles are then defined against fixed breakpoints (pass 3) with no reconciliation needed |
| Cascading skips collapse a large region into one site (megasite) | bounded by the `--max-site-length` / distinct-allele guard: oversized sites are emitted as a single `CPX`-flagged record rather than a multiallelic explosion |
| Haplotype visits anchors out of reference order, or visits one anchor twice | region flagged **complex**; the haplotype is excluded from breakpoint marking so it cannot create spurious bridges, and its genotype there is `.` (or `CPX`) |
| Reverse-oriented traversal of a region (inversion) | if both bounding anchors are visited in reversed order relative to the reference, reverse-complement the stretch and proceed; **inconsistent/partial-orientation snarls are flagged and skipped in v1** (documented limitation) |
| Empty segment sequence / `*` overlap | contributes zero-length DNA; handled naturally by the offset table |
| No variation | header-only VCF |
| Variable ploidy across samples | ploidy follows the number of that sample's haplotypes present; documented, not normalized |
| Variant at the very start of the contig (no preceding base) | fall back to right-anchoring the base after the sink, per VCF practice; documented |

Known v1 limitations, explicitly out of scope: nested snarl decomposition,
`LV`/`PS` annotations, complex inversions, multi-reference coordinate
reconciliation, and minimal-allele normalization (defer to `bcftools norm`).

## 7. Code plan

### New files
- `include/workflows/deconstruct_workflow.hpp`
  - `struct DeconstructOptions { std::vector<std::string> reference_names;
    GroupingMode grouping = Sample; int num_threads; bool emit_gt = true; };`
  - `void deconstruct_to_vcf(const CompressedData&, const DeconstructOptions&,
    std::ostream&);` (streaming emit; large outputs should not be buffered).
- `src/workflows/deconstruct_workflow.cpp` — the four passes above.
- `src/cli/deconstruct_command.cpp` — `do_deconstruct`, arg parsing (mirrors
  `pav_command.cpp`), VCF header emission.
- `include/utils/sequence_utils.hpp` (+ `.cpp`) — `reverse_complement(...)`.

### Shared-helper extraction (refactor)
`RuleLeafCache`, `build_rule_cache`, `expand_rule_visit`, `stream_hap_leaves`,
`stream_decoded_nodes`, `build_slices`, `decode_one_haplotype_general` currently
live in the anonymous namespace of `pav_workflow.cpp`. `deconstruct` needs all
of them. Extract into a shared header (e.g.
`include/workflows/traversal_decode.hpp` + `.cpp`) and have both `pav` and
`deconstruct` include it. This is a mechanical, behavior-preserving refactor and
should be validated against the existing `pav` regression before deconstruct is
layered on.

The PanSN grouping helpers (`parse_pansn_path_name`, `path_group_key`,
`walk_group_key`, `build_metadata`) should likewise be shared rather than
duplicated.

### Wiring
- `include/cli/commands.hpp`: declare `int do_deconstruct(int, char**)`.
- `src/cli/gfaz_cli.cpp`: dispatch `"deconstruct"`.
- `src/cli/common.cpp`: add `print_deconstruct_help` and a usage line.
- `CMakeLists.txt`: add the new sources.

## 8. Performance and memory

- Reuses the `RuleLeafCache` (default 1 GiB budget,
  `GFAZ_PAV_RULE_CACHE_BYTES`-tunable) so grammar expansion stays bounded — the
  same envelope that keeps `pav` at laptop-class memory on HPRC-scale graphs.
- Pass 2 is parallel and lock-free; allele signatures avoid spelling DNA for
  every haplotype (DNA is spelled once per *distinct* allele at emit time).
- Segment DNA is decompressed once and shared read-only across threads.
- Expected envelope: comparable to `pav`, plus the resident segment-sequence
  buffer (already required for any sequence-spelling operation).

## 9. Validation plan

1. **Shared-helper refactor:** existing `pav` regression must pass unchanged.
2. **Round-trip sanity on `chrY.gfaz`:** small fixture; manually inspect a
   handful of SNP/indel rows.
3. **Cross-check against `vg deconstruct`** on a small graph for which both can
   run: compare site positions and allele sets (allowing for v1's flat vs.
   nested differences and normalization).
4. **`bcftools view`/`bcftools norm`** must parse the output without error;
   normalized SNP/indel records should be well-formed.
5. **Genotype consistency:** for a sample whose haplotype equals the reference,
   all its `GT` should be `0|0`.

## 10. Open questions for v2

- ~~Nested snarl decomposition (vg parity).~~ **Done** via `--vg-compat` (§11):
  one record per top-level snarl from a global biconnected decomposition,
  reproducing `vg deconstruct` at ~99.99% position concordance.
- ~~Inversion-aware allele representation.~~ **Done**: `--snarl`/`--vg-compat`
  capture inversions (node-end graph; §11.2).
- `LV`/`PS` nested snarl-tree annotations (vg `-a` style child recursion).
- `AT` allele-traversal `INFO` field (node-string per allele).
- Direct BCF / bgzipped output and tabix indexing.
- Multi-reference / graph-coordinate `CHROM` reconciliation.

## 11. Topology-based site finding (`--snarl` / `--vg-compat`)

The default algorithm (§4) projects sites from the reference's *linear* anchor
structure. The topology modes instead derive sites from the **graph structure**
recovered from the stored L-line links, matching how `vg deconstruct` defines
sites. Both reuse the same streaming allele-observation machinery, so the
low-memory, path-iterative envelope is unchanged — only *which* intervals become
sites differs. Code: `src/workflows/snarl_finder.cpp` +
`deconstruct_workflow.cpp::deconstruct_contig_snarl`.

### 11.1 `--snarl` (leaf-bubble superset)

For each reference branch point, find the minimal enclosing **superbubble**
(Onodera–Sadakane–Shibuya per-entrance detection on the bidirected node-side
graph), with a bounded separating-pair fallback (`detect_inversion_snarl`) that
recovers inversions and other non-DAG snarls the superbubble test rejects. The
result is reduced to a non-overlapping chain along the reference. This reports a
**superset** of leaf sites: every clean local bubble, including well-formed sites
inside cyclic/repetitive reference regions (e.g. chrY satellites) that vg's
global decomposition collapses or skips. Use it when you want maximal site
resolution rather than vg parity.

### 11.2 `--vg-compat` (top-level snarls, vg parity)

`vg deconstruct` emits exactly **one VCF record per top-level snarl** (it does
not recurse into child snarls without `-a`) and drops a snarl whose reference
traversal is ambiguous (cyclic). `--vg-compat` reproduces this with a single
**global biconnected-component (BCC) decomposition**:

- **Node-end graph.** Each segment contributes two vertices (its 5′ and 3′ ends)
  joined by a *black* edge; each L-line contributes one *grey* edge between the
  departure end of one segment and the arrival end of the other. The black edges
  are what make a single-node **inversion** close into one biconnected block —
  collapsing orientation onto a plain segment graph would make the inverted
  interior node look like an articulation point and miss the site.
- **Top-level snarls = non-trivial biconnected blocks** the reference threads
  through. A block's reference span is `[min, max]` of the reference indices of
  the nodes it touches; the boundaries are the reference nodes at those indices.
  Nested bubbles and chains of leaf bubbles enclosed by one block collapse into a
  single record — exactly vg's top-level granularity.
- **Cyclic-reference suppression.** A block is emitted only if the reference
  crosses it once cleanly: no reference segment recurs within `[min, max]` and
  every reference index in that span belongs to the block. Tangled
  palindrome/satellite regions, where the reference re-enters the block, fail
  this test and are dropped — matching the regions vg also skips.

The decomposition is an iterative Hopcroft–Tarjan pass, **O(V+E)**, run once per
contig over topology — *not* in the hot per-haplotype streaming loop — so it adds
negligible cost and preserves the speed/memory advantage.

### 11.3 Allele observation (shared by both modes)

Once sites (snarl boundaries) are fixed, each sample slice is streamed **once**;
a small state machine captures only the interior node stretch between a snarl's
entrance and exit boundary node-sides (forward, or reverse-complemented for an
inversion). Whole haplotypes are never retained. Alleles are then assembled per
snarl exactly as in §4.2 Pass 3.5 (REF = reference interior, ALT = distinct
sample spellings, phased `GT`, AC/AN/AF/NS, megasite guard).

### 11.4 Validation against `vg deconstruct`

Measured on the HPRC chrY/chr1/chr6 smoothed pangenomes (16 threads;
`scripts/benchmark/compare_deconstruct_perf.py`), comparing `--vg-compat` output
to `vg deconstruct` position-by-position:

| Dataset | vg records | `--vg-compat` records | Δ | POS concordance | speed | RSS |
|---|---|---|---|---|---|---|
| chrY  | 9,543      | 9,539      | −0.04% | 9,538/9,543 (99.9%) | 0.2 s vs 11 s (55×) | 219 MB vs 1.15 GB |
| chr6  | 1,302,385  | 1,304,121  | +0.13% | 99.99%              | 20.8 s vs 364 s (17.5×) | 12.5 vs 17.2 GiB |
| chr1  | 1,593,956  | 1,593,899  | −0.004% | 99.99%             | 28.5 s vs 696 s (24.4×) | 16.9 vs 28.2 GiB |

At shared positions, `REF` and the `ALT` allele *set* are identical except for a
handful of records (6 on chr6, 15 on chr1); the residual differences are
cosmetic alt-allele *ordering* and a few large complex SVs in the pericentromere
that gfaz's reference-anchored boundaries don't capture. The default `--snarl`
superset is correspondingly larger (chr1 +17%, chr6 +3.2%, chrY 29,946).
