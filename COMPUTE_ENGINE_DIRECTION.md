# GFAz Compute Engine Direction

This note captures the strongest application direction for extending GFAz from a
GFA compressor into a compressed pangenome compute engine.

## Recommendation

The most promising top-tier direction is:

> Compute reference-window and annotation-window haplotype matrices directly from
> compressed `.gfaz` files.

Candidate command names:

- `gfaz matrix`
- `gfaz pav`
- `gfaz genotype-matrix`

The core output is a matrix over genomic windows, annotation intervals, or graph
features:

```text
window_or_feature x sample/haplotype -> value
```

Useful values include:

- presence/absence ratio
- copy-number-like coverage ratio
- reference-covered bp
- non-reference or novel bp
- private/accessory/core status
- binary presence/absence after thresholding

This direction is stronger than generic all-vs-all path similarity because it is
closer to high-impact biological workflows: PAV/SV analysis, GWAS feature
generation, pangenome QC, breeding/population genomics, and annotation-level
comparisons.

## Why This Fits GFAz

GFAz stores compressed path and walk traversals. Its advantage is not arbitrary
graph topology queries; it is fast streaming of embedded haplotypes without
materializing the full graph.

The winning computation pattern is:

```text
compressed traversal stream -> small accumulator -> biological statistic
```

`gfaz growth` already demonstrates this pattern. A matrix/PAV engine generalizes
it from one global statistic to many reference windows or annotation features.

For example:

```bash
gfaz matrix graph.gfaz \
  --reference CHM13#0#chr8 \
  --bed genes_or_windows.bed \
  --group-by sample \
  --metric pav,copy,novelty \
  --binary 0.8 \
  > matrix.tsv
```

Conceptual algorithm:

1. Load `.gfaz` metadata, segment lengths, rulebook, and compressed traversals.
2. Decode the selected reference path/walk once.
3. Build a temporary `node -> window_id(s)` map for requested BED windows or
   annotation intervals.
4. Stream every path/walk from compressed traversal storage.
5. For each streamed node visit, update the relevant window/group accumulator.
6. Emit sparse or dense matrix output.

This does not require a persisted odgi-style inverse index:

```text
node -> all path/walk steps touching node
```

Instead, it builds only the temporary index needed for the requested reference
windows. Memory is proportional to the number of nodes/windows/groups involved,
not the full materialized graph object.

## Metrics

For a reference window `W`, let `N(W)` be the reference path's node intervals
covering `W`.

For a haplotype/group `H`:

```text
reference_bp(W) = bp length of W on the reference path
covered_bp(H,W) = bp from N(W) touched by H, usually counting each node once
copy_bp(H,W)    = bp from N(W) touched by H, counting repeated visits
```

Then:

```text
PAV ratio       = covered_bp(H,W) / reference_bp(W)
copy ratio      = copy_bp(H,W) / reference_bp(W)
absence         = PAV ratio < threshold
presence        = PAV ratio >= threshold
```

Additional panel-relative metrics:

```text
panel_nodes(P)       = nodes touched by selected reference/panel paths
novel_bp(H)          = bp in H on nodes not in panel_nodes(P)
panel_covered_bp(H)  = bp in H on nodes in panel_nodes(P)
novel_fraction(H)    = novel_bp(H) / total_bp(H)
```

These metrics are path-centric and can be computed by streaming traversals.

## Why Not Start With IBD, LD, Or Phylogeny

These are scientifically interesting but riskier as the flagship direction.

### IBD / Shared-Run Detection

Graph-based haplotype matching is an active compressed-index research area,
especially around GBWT/PBWT-like methods. It requires careful definitions of
long matches, positional matches, repeats, orientation, gap tolerance, and
homology. GFAz can support selected-pair or reference-vs-all shared-run scans,
but full all-vs-all IBD is not the simplest first compute-engine story.

### LD / Co-Occurrence

LD is usually defined over variant alleles across samples. Raw graph nodes are
not guaranteed to be normalized alleles. They may be unitigs, reference chunks,
repeats, duplicated sequence, or artifacts of graph construction. A meaningful
LD command first needs a feature/variant matrix. Therefore, matrix generation is
the correct prerequisite.

### Phylogenetic Distance

Phylogenetic tools normally expect aligned characters, variants, or a distance
matrix with a clear evolutionary interpretation. GFAz can produce clustering
input or simple graph-path distances, but the stronger claim is that GFAz
generates scalable pangenome-derived matrices from which downstream tools can
compute trees or clusters.

## Relation To Existing Tools

### odgi

ODGI provides a broad graph toolkit: build, view, sort, extract, depth, PAV,
stats, visualization, and more. Its strength is the materialized ODGI graph data
structure and handlegraph-style APIs, including node-to-step traversal queries.

GFAz should not try to replace odgi for topology-heavy operations. The stronger
position is:

```text
odgi: graph operations and visualization on a materialized graph
GFAz: compressed traversal analytics without full graph materialization
```

`odgi pav` is especially relevant because it validates PAV matrix generation as
a real user need. GFAz can target the same biological output while avoiding
full graph decompression/materialization.

### Panacus

Panacus demonstrates that scalable pangenome graph analytics is publishable and
useful. It focuses on growth/core curves and coverage distributions from GFA.
GFAz already targets this space with `growth`.

The extension proposed here is to move from global pangenome summaries to
window/annotation-level sample matrices.

### vg / GBZ / GBWT

vg and GBZ/GBWT are strong for read mapping, haplotype-aware indexing,
genotyping, and path queries in compressed haplotype indexes. GFAz should avoid
competing directly with GBWT as a mapper/haplotype index.

The complementary GFAz role is direct analytics over compressed GFA traversals:

```text
large GFA -> .gfaz -> path-derived matrices/statistics
```

## Proposed Roadmap

### Phase 1: Shared Traversal Streaming Layer

Refactor the reusable parts of `growth_workflow.cpp` into a traversal streaming
utility:

- load/decompress rulebook
- build P/W encoded slices
- expand grammar rules
- apply inverse delta
- expose segment length lookup
- expose grouping keys for P and W records

Target event model:

```text
record_id, record_type, group_id, signed_node_id, abs_node_id, node_length
```

Optional later fields:

```text
step_index, path_offset_start, path_offset_end, orientation
```

### Phase 2: `gfaz matrix` / `gfaz pav`

Initial scope:

- one reference path/walk
- BED windows on that reference
- P and W traversal support
- group by record, sample, sample#hap, or sample#hap#seq
- dense TSV output
- metrics: `pav`, `copy`, `binary`

Implementation sketch:

1. Decode reference traversal.
2. Convert BED intervals to node-overlap intervals along the reference.
3. Build temporary `node_id -> window contribution(s)`.
4. Stream query traversals and update `(window_id, group_id)` accumulators.
5. Normalize by reference window length and emit.

### Phase 3: Novelty And Panel Metrics

Add:

- `gfaz novelty`
- multi-reference or panel input
- per-haplotype non-panel bp
- per-window non-reference burden
- private/shared/core classification by group

This phase uses the same streaming layer with node bitsets/count arrays.

### Phase 4: Supporting Analytics

Add lower-risk companion commands:

- `gfaz depth`: node/window coverage and histograms
- `gfaz stats`: path/walk length and metadata summaries
- `gfaz similarity`: sparse path/group similarity matrix
- `gfaz shared-runs`: selected-pair or reference-vs-all shared node runs

These are useful, but the matrix/PAV engine should remain the flagship.

## Publishable Claim

The central claim should be:

> GFAz is a compressed pangenome analytics engine: it computes biologically
> useful path-derived matrices directly from compressed GFA, using memory
> proportional to traversal accumulators rather than a full materialized graph.

This is stronger than saying GFAz is a smaller GFA file. It positions GFAz as a
compute substrate for pangenome-scale downstream analysis.

## References

- ODGI: understanding pangenome graphs. Bioinformatics 2022.
  https://academic.oup.com/bioinformatics/article/38/13/3319/6585331
- `odgi pav` command documentation.
  https://odgi.readthedocs.io/en/latest/rst/commands/odgi_pav.html
- Panacus: fast and exact pangenome growth and core size estimation.
  Bioinformatics 2024.
  https://academic.oup.com/bioinformatics/article/40/12/btae720/7914008
- GBZ file format for pangenome graphs. Bioinformatics 2022.
  https://academic.oup.com/bioinformatics/article/38/22/5012/6731924
- Haplotype Matching with GBWT for Pangenome Graphs. bioRxiv 2025.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11838520/
- Genotyping structural variants in pangenome graphs using the vg toolkit.
  Genome Biology 2020.
  https://genomebiology.biomedcentral.com/articles/10.1186/s13059-020-1941-7
- Graph pangenome captures missing heritability and empowers tomato breeding.
  Nature 2022.
  https://pubmed.ncbi.nlm.nih.gov/35676474/
