# GFAz

GFAz compresses pangenome graphs in GFA format into a small `.gfaz` file, and
runs analyses directly on that file without decompressing it.

- **Small.** Up to 84x smaller than the GFA on whole-genome HPRC graphs: up to
  20x smaller than Gzip and 13x smaller than Zstd.
- **Fast.** Gigabytes of GFA per second on 16 CPU threads, in both directions.
- **Usable compressed.** `gfaz` computes growth curves, presence/absence,
  depth, VCFs and region extracts from the `.gfaz`, and
  [vg can read it as a graph](docs/VG_INTEGRATION.md).
- **Lossless.** Decompression gives back the GFA records, including the
  original segment names.

## Example: HPRC v2.1

The HPRC v2.1 Minigraph-Cactus graph is 396 GB of GFA. GFAz stores it in under
5 GB.

| Step | Time | Throughput | Peak memory |
|---|---|---|---|
| `gfaz compress` | 122 s | 3.2 GB/s | 161 GB |
| `gfaz compress --memory-budget 2G` | 125 s | 3.2 GB/s | 30 GB |
| `gfaz decompress` to `/dev/null` | 50 s | 7.9 GB/s | 25 GB |
| `gfaz decompress` to an NVMe file | 180 s | 2.2 GB/s | 27 GB |

16 threads on an AMD Ryzen Threadripper PRO 9955WX with 512 GB of memory, input
in the page cache. Time is the wall time of the whole command, and throughput
is the GFA size over that time. `--memory-budget` gives the same output, byte
for byte. Seven more graphs, from 0.1 GB to 384 GB, are in
[docs/WORKFLOW.md](docs/WORKFLOW.md#measured-cpu-throughput-and-memory).

An experimental GPU backend reaches up to 4.8 GB/s compression and 9.4 GB/s
decompression. It reads and writes the same `.gfaz` format.

### Analyses on the compressed file

At 16 threads, against tools that read the uncompressed GFA (`vg deconstruct`,
Panacus, `odgi`), `gfaz` runs up to 614x faster with up to 27x lower peak
memory where both complete. It finishes whole-genome analyses on the HPRC v2.x
graphs, where `vg` and `odgi` cannot load the input on a 512 GB machine.

Outputs match the baselines: `growth` reproduces Panacus's curve exactly, `pav`
matrices are identical to `odgi`'s, and `deconstruct` reproduces more than
99.99% of `vg`'s sites at the same position with the same reference allele.
Per-dataset results and the comparisons against Gzip, Zstd, sqz and GBZ are in
the paper (see [Citation](#citation)).

## Build

```bash
conda activate gfa
git submodule update --init --recursive
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
```

The command-line tool is `build/bin/gfaz`. For the GPU backend add
`-DENABLE_CUDA=ON -DCUDA_PATH=/usr/local/cuda-12.8`; other options are in
[docs/BUILD_GUIDE.md](docs/BUILD_GUIDE.md).

## Quick start

```bash
gfaz compress graph.gfa                 # writes graph.gfa.gfaz
gfaz decompress graph.gfa.gfaz          # writes the GFA back
```

To compress a large graph on a machine with less memory, keep at most 2 GiB of
path and walk data in memory and put the rest in a temporary file on fast local
storage:

```bash
gfaz compress --memory-budget 2G --tmp-dir /path/on/nvme graph.gfa
```

## Commands

Every command except `compress` reads the `.gfaz`. `gfaz <command> --help`
lists the options.

| Command | What it does | Details |
|---|---|---|
| `compress` | GFA to `.gfaz` | [WORKFLOW](docs/WORKFLOW.md) |
| `decompress` | `.gfaz` back to GFA | [WORKFLOW](docs/WORKFLOW.md) |
| `extract-path`, `extract-walk` | Print chosen P-lines or W-lines | |
| `add-haplotypes` | Append paths or walks to an existing `.gfaz` | |
| `stats` | Graph size summary, as `odgi stats` | [STATS](docs/workflows/STATS_WORKFLOW.md) |
| `growth` | Pangenome growth curve, as Panacus | [GROWTH](docs/workflows/GROWTH_WORKFLOW.md) |
| `pav` | Presence/absence over BED ranges, as `odgi pav` | [PAV](docs/workflows/PAV_WORKFLOW.md) |
| `similarity` | All-vs-all similarity matrix, as `odgi similarity` | [SIMILARITY](docs/workflows/SIMILARITY_WORKFLOW.md) |
| `depth` | Node coverage depth, as `odgi depth` | [DEPTH](docs/workflows/DEPTH_WORKFLOW.md) |
| `deconstruct` | VCF against a reference path, as `vg deconstruct` | [DECONSTRUCT](docs/workflows/DECONSTRUCT_WORKFLOW.md) |
| `extract-region` | Subgraph around a reference region, as GFA | [REGION](docs/workflows/REGION_WORKFLOW.md) |
| `locate` | Where every path and walk visits a node | [LOCATE](docs/workflows/LOCATE_WORKFLOW.md) |
| `index` | Save the path index `extract-region` and `locate` use | [LOCATE](docs/workflows/LOCATE_WORKFLOW.md) |

Some examples:

```bash
# Growth curve, grouping haplotypes by sample
gfaz growth -i graph.gfaz -j 16 -G sample

# VCF against a reference, one record per top-level snarl
gfaz deconstruct -i graph.gfaz -r chr1 -t 16 > graph.vcf

# The subgraph around a region, with every haplotype that passes through it
gfaz extract-region -i graph.gfaz -r 'CHM13#0#chr6:30000000-30100000' > mhc.gfa
```

Things to know:

- **Missing segments.** `compress` stops if a path, walk or link names a
  segment that has no S-line. `--allow-missing-segments` drops those
  references with a warning instead.
- **GPU.** `--gpu` on `compress` or `decompress` uses the GPU backend. In a
  CPU-only build it falls back to the CPU with a warning.
- **In-memory decompression.** `decompress` streams the GFA out by default.
  `--legacy` rebuilds the whole graph in memory first.

## From Python and from other tools

- **Python.** The `gfa_compression` module compresses, decompresses and reads
  paths in place. See [docs/PYTHON.md](docs/PYTHON.md).

  ```python
  import gfa_compression as gfac

  gfac.serialize(gfac.compress_file("graph.gfa"), "graph.gfaz")
  reader = gfac.PathReader("graph.gfaz")
  ```

- **vg.** vg accepts a `.gfaz` wherever it accepts a graph. See
  [docs/VG_INTEGRATION.md](docs/VG_INTEGRATION.md).
- **libhandlegraph.** `GfazHandleGraph` presents a `.gfaz` as a read-only
  handle graph to any libhandlegraph tool. See
  [docs/HANDLEGRAPH_ADAPTER.md](docs/HANDLEGRAPH_ADAPTER.md).

## Tests

```bash
conda activate gfa
python3 tests/run_all.py            # PASS / SKIP / FAIL summary
ctest --test-dir build              # the same, through CTest
```

A suite that cannot run, such as the GPU tests in a CPU-only build, is reported
as SKIP. See [tests/README.md](tests/README.md).

## Documentation

[docs/README.md](docs/README.md) is the index: build options, the file format
and compression workflow, one page per command, and how to add a new analysis
to the compute engine.

## Limitations

- The GPU backend is experimental and needs a CUDA build.
- Decompression keeps every record, but may change the order and spacing of
  records compared with the original file.

## Citation

If you use GFAz in your work, please cite:

```bibtex
@inproceedings{yang2026gfaz,
  title={GFAz: State-of-the-Art Graphical Fragment Assembly Compression},
  author={Yang, Taolue and Liu, Youyuan and Jiang, Bo and Shi, Xinghua and Jin, Sian},
  booktitle={Proceedings of the 40th ACM International Conference on Supercomputing},
  pages={650--661},
  year={2026}
}
```

## License

MIT
