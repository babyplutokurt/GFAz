# `gfaz extract-region`

Writes the subgraph around one or more reference regions as GFA, directly from
a `.gfaz`, without decompressing the whole graph.

```bash
gfaz extract-region -i graph.gfaz -r 'CHM13#0#chr6:30000000-30100000' > mhc.gfa
```

The output opens in any GFA tool (Bandage, odgi, vg), which is the point: a
locus is small even when the pangenome is not.

## Regions

`-r CONTIG:START-END`, 0-based and end-exclusive, or `-r CONTIG` for the whole
sequence. `-r` may be repeated.

- For a P-line, `CONTIG` is the path name and coordinates count from the start
  of the path.
- For a W-line, `CONTIG` is `sample#hap#seq` and coordinates are positions on
  that sequence: a walk `W s 1 chr1 5000 9000 ...` covers `s#1#chr1:5000-9000`.
  Several walks can lie on one contig; each contributes the part it covers.

## What is kept

1. The nodes the reference visits inside the regions are marked.
2. Every haplotype is searched for visits to a marked node. Two visits at most
   `--max-gap` bases apart (default 100000) are joined, and everything between
   them is kept. This is what brings in alleles that leave the reference: an
   insertion is the stretch between two reference anchors.
3. Each joined stretch becomes one output line. A W-line carries the stretch's
   coordinates; a partial P-line is named `NAME:START-END` in path coordinates.
4. The segments are the nodes on those stretches, with their tags. The links
   are every stored link between two kept segments.

A haplotype that starts or ends on a non-reference allele at the edge of the
region is cut at its first and last marked node. J- and C-lines are not
written.

Segments carry their original GFA names, as `gfaz decompress` writes them, so
a region lines up with the decompressed graph and with the GFA it came from.

## How it runs

The command builds a `PathIndex` (see
[../EXTENDING_COMPUTE_ENGINE.md](../EXTENDING_COMPUTE_ENGINE.md)): one parallel
pass over the traversals that records, for every block of about 4096 steps,
where it starts and which node-id range it visits. The region is then answered
by decoding only the blocks whose range reaches the marked nodes. Building the
index dominates the run time; the lookup itself is milliseconds.

Measured at 16 threads for a 100 kb window with 90 haplotypes:

| Graph | Time | Peak RSS |
|---|---|---|
| HPRC chr6 (4.1 GB GFA) | 0.6 s | 0.55 GB |
| HPRC v1.1 (50.8 GB GFA) | 11.2 s | 11.2 GB |

## Validation

`tests/regression/test_extract_region.py` checks every case against a
brute-force recomputation from the decompressed GFA, across index block sizes
and thread counts.
