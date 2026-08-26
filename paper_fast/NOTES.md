# FAST '27 submission notes

## Hard requirements (from the CFP)

| Item | Value |
|:--|:--|
| Venue | 25th USENIX Conference on File and Storage Technologies, Feb 23 2027 |
| **Fall deadline** | **Tue 15 Sep 2026, 23:59 AoE** |
| Author response | 17–19 Nov 2026 · Notification 8 Dec 2026 · Final files 26 Jan 2027 |
| Length | **12 pages max, excluding references** (short-paper track is 6) |
| Format | US letter, 2 col, 10pt Times on 12pt leading, text block 7in × 9in |
| Template | `usenix-2020-09.sty` (vendored in `source/`) |
| Review | **Double-blind.** No names, no affiliations, no de-anonymizing URLs. Own prior work cited in the third person. No "reference removed for blind review." |
| Disclosure | Prior arXiv/tech-report/talk versions permitted but must be disclosed to the chairs on the submission form |
| Artifacts | Not required, but the PC "values a paper more highly" with them |

The CFP's own definition of a good paper — worth re-reading before each revision:
addresses a significant problem; presents a compelling solution; **demonstrates
the benefits *and drawbacks*** of the solution; draws conclusions with sound
experimental method; clearly describes what was done; clearly articulates the
advance over previous work.

Relevant topic lines: *data deduplication and compression*; *data layouts and
file formats*; *big data, analytics, and data lake storage*; *storage for AI and
scientific workloads*; *HPC data management systems*.

## Positioning: how this differs from the ICS paper

The ICS paper's thesis is "GFA compresses better and faster." That thesis is
finished and is **cited here as third-party prior work** (`gfaz2026`), not
re-argued. Compression ratio and compressor throughput appear only as calibration
in §4 and are explicitly disclaimed as non-results.

This paper's thesis is a storage-systems one: **the materialization step between
storage and computation is not required by the computation.** Its arguments are
the five computability properties (§3), the execution engine (§5), the
graph-free variant projection (§6), and an evaluation that measures bytes moved
rather than only speedup (§7).

The load-bearing new claim, and the one to protect in revision: *because the
compressed form is what the query reads, a better ratio makes queries faster
rather than slower.* §7.6 is the experiment that tests it.

## Anonymization

- System renamed to **Loom** — one macro, `\sys` in `main.tex`. Reusing the
  public name of the prior system would identify the authors by implication.
- We are **not** claiming the container is someone else's work. `ref.bib` carries
  its real author list. The CFP's "consider it as written by a third party"
  governs phrasing, not authorship: avoid "our previous work [12]", write "a
  state-of-the-art container [12]". Reviewers guessing is expected and fine.
- Double-blind does **not** forbid "we". It forbids identifying who "we" are, so
  "we re-engineered the CPU backend" is allowed; "our earlier paper [12]" is not.
- Container extension anonymized to `\cont` (`.pgz`).
- The GitHub URL from the ICS intro is **removed**. If we want an artifact link,
  use an anonymous mirror (anonymous.4open.science or a scrubbed Zenodo DOI).
- Check before submitting: no "Temple", no author names, no ORCID, no
  acknowledgements block (the NIH grant number identifies the group — restore it
  at camera-ready only).

## Open experiments

`\needsexp{...}` marks each one in the source; it renders in red in draft mode.
Setting `\draftmodefalse` makes the build **fail** on any remaining marker, so a
stray placeholder cannot ship. Ranked by how much a reviewer will miss it:

1. **Cold-cache end-to-end, incl. expand-to-NVMe (§7.5).** The most important
   one. Converts the paper from a speedup claim into a storage claim, and
   pre-empts the first question any FAST reviewer asks. Need wall-clock, bytes
   read, bytes written, peak RSS for three configurations on chr1 / hprc-v1.1 /
   HGSVC3.
2. **Ratio-vs-query-time sweep (§7.6).** Tests the central mechanism under
   control by varying grammar rounds on a fixed graph. Without it, the claim
   rests on a cross-graph correlation that confounds ratio with graph identity.
3. **GBZ comparison (§7.7).** GBZ is the closest intellectual competitor. Best
   version: run `vg deconstruct` over a GBZ rather than GFA text, which isolates
   compressed-index-resident from text-resident.
4. **Rule-leaf cache sweep (§7.8).** Cheap to run, and the flat line for
   `growth` matters as much as the knee for `pav`.
5. **Full thread-scaling curves (§7.4).** Currently only 1- and 16-thread
   endpoints. A 1/2/4/8/16/32 figure is easy and looks much stronger.
6. **Matched re-measurement of the CPU backend (§4).** Needed only for the one
   sentence claiming the backend improvement, but needed before that sentence
   can print a number. See below.

### The backend-improvement claim

The improvement is real and unpublished, so it is ours to claim, but the
headline factor currently available is confounded and must not be printed
as-is:

| | ICS '26 | README today | Apparent |
|:--|--:|--:|--:|
| Peak compression | 385 MB/s | 1355 MiB/s | 3.5x |
| Peak decompression | 1.8 GB/s | 5426 MiB/s | 3.0x |
| v1.1 compression | 231 MB/s | 291 MiB/s | 1.26x |
| v2.1 decompression | 1559 MB/s | 5325 MiB/s | 3.4x |

`paper_compute_engine/source/4_evaluation.tex:95` states the published runs used
**16 threads**; the README numbers use **32** on the same 16-core/32-thread
part, so roughly half the apparent gain is SMT rather than optimization. The
methodologies also differ: the published figures primed the page cache and
averaged five runs, the README uses end-to-end CLI wall time including parse and
serialization, medians of 3-9. Re-run both configurations at one thread count
under one methodology and report the matched factor per dataset. Expect
something nearer 2-3x, with whole-genome decompression improving far more than
v1.1 compression.

Where the claim goes: §4 only, plus the load-bearing use in §7.5 (at these rates
the container is read and entropy-decoded faster than NVMe can stream the
expanded GFA, which is the sharpest answer to "why not just expand to fast local
storage"). **Not** a contribution bullet and **not** in the abstract: as a
headline it is an engineering speedup with no new idea described in this paper,
and it pulls the framing back toward the ICS story.

## Length

First full draft builds clean at **16 pages**, of which body ≈ 14.3 and
references ≈ 1.7. **Over the 12-page body limit by ~2.3 pages before any figure
is added** — budget another ~1.5 pages for the figures the experiments above
will produce, so plan on cutting ~4.

Trim candidates, cheapest first:

- §6 VCF conventions paragraph → 3 sentences (~0.3 pg).
- §5 analysis descriptions for `similarity`/`depth`/`stats` → one sentence each;
  Table 3 already carries the detail (~0.4 pg).
- §2.1 GFA format recap → compress to one paragraph; §3 and §4 restate what
  matters anyway (~0.3 pg).
- §9 related work → tighten; the column-store paragraph can lose two sentences
  (~0.3 pg).
- §1 has two paragraphs of contract/framing before "The workload"; one would do
  (~0.25 pg).
- Merge Tables 5/6/7 into one multi-panel table (~0.5 pg) — same move that
  worked for §7 of the ICS draft.
- Algorithm 1 can drop to a 4-line inline description if space gets tight, but
  keep Algorithm 2; the state machine is harder to convey in prose.

Do the cutting **after** the experiments land, not before — which section gives
way depends on which figures earn their space.

## Figures still to draw

- **Fig. 1 (§1 or §2):** the materialize-to-analyze tax. Two pipelines side by
  side: GFA → parse → in-memory graph → fold → answer, vs. container → stream →
  fold → answer, annotated with bytes at each stage for hprc-v2.1. This is the
  paper's thesis in one picture and it currently has none.
- **Fig. 2 (§5):** the decode-and-fold loop, showing the visitor inlined into
  the decoder and the rulebook shared read-only across threads.
- Experiment figures for §7.4–§7.8 as those runs complete.

## Build

```
cd source && latexmk -pdf main.tex
```

`algorithm.sty` / `algorithmic.sty` are not in this machine's BasicTeX install;
they were added to the user tree with
`tlmgr init-usertree && tlmgr --usermode install algorithms algorithmicx`.

### The abstract's throughput figures

`0_abstract.tex` carries `\needsexp{X}` / `\needsexp{Y}` for compression and
decompression rates. Before filling them in:

1. They must be the matched-thread, single-methodology numbers from §4, not the
   README's 32-thread CLI wall times against the ICS paper's 16-thread warm-cache
   averages.
2. They must hold on the **whole-genome** graphs, not the single-chromosome peak.
   Decompression is safe everywhere (2292-5426 MiB/s on HPRC). Compression is not:
   v1.1 is 291 MiB/s and v2.0 is 555 MiB/s, both sub-GiB/s. If that survives
   re-measurement, claim the GiB/s rate for decompression only and move the
   compression figure to §4.
3. A reviewer will check the abstract against `tab:datasets`. A peak quoted where
   the text implies whole-genome is the kind of thing that gets caught.
