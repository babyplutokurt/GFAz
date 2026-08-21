# GFAz: State-of-the-Art Graphical Fragment Assembly Compression

GFAz is a C++/CUDA library and command-line tool for compressing and
decompressing Graphical Fragment Assembly (GFA) files.

In our current benchmarks, GFAz reaches up to 20x higher compression ratio than
Gzip and 13x higher compression ratio than Zstd. At 16 CPU threads it reaches
up to 1.4 GB/s compression and 5.4 GB/s decompression; the experimental GPU
backend reaches up to 4.8 GB/s compression and 9.4 GB/s decompression.

It has two execution backends:

- CPU
- GPU (experimental, CUDA build required)

Both backends produce and consume the same `.gfaz` container format. The backend
changes how transforms are computed, not the on-disk format.

## Performance

Measured on an AMD Ryzen Threadripper PRO 9955WX (16 cores), 512 GB DDR5-6400
memory, a Samsung SSD 990 PRO NVMe drive, and an NVIDIA RTX Pro 6000. GFAz CPU
runs use 16 threads. Compression throughput is end-to-end CLI wall time
(parsing, compression, serialization) over the input GFA size; decompression
writes to `/dev/null`.

- Compression ratio: up to 84x on whole-genome HPRC graphs, up to 20x higher
  than Gzip and 13x higher than Zstd.
- CPU throughput: up to 1.4 GB/s compression and 5.4 GB/s decompression.
- GPU backend (experimental): up to 4.8 GB/s compression and 9.4 GB/s
  decompression.

Per-dataset results and the comparison against Gzip, Zstd, sqz, and GBZ are
reported in the paper (see [Citation](#citation)).

## Compute Engine Performance

Beyond compression, `gfaz` runs pangenome analyses **directly on the compressed
`.gfaz` container**, with no decompression back to GFA. Measured at 16 threads
against tools that read the uncompressed GFA (`vg deconstruct`, Panacus,
`odgi`), `gfaz` runs up to 614x faster with up to 27x lower peak memory where
both complete, and it finishes whole-genome analyses on the HPRC v2.x graphs
(about 400 GB of GFA) from a container under 5 GB, where `vg` and `odgi` cannot
load the input on a 512 GB node.

Outputs match the baselines: `growth` reproduces Panacus's curve exactly at
every point, `pav` matrices are identical to `odgi`'s, and `deconstruct`
reproduces more than 99.99% of `vg`'s sites at the same position with the same
reference allele. Per-analysis timings and memory are reported in the paper.

## What It Does

- Compresses GFA text into a shared `CompressedData` / `.gfaz` representation.
- Supports CPU and GPU compression against the same file format.
- Supports cross-backend decompression:
  CPU-compressed files can be decompressed with the GPU path, and GPU-compressed
  files can be decompressed with the CPU path.
- Exposes both a CLI (`gfaz`) and Python bindings (`gfa_compression`).
- Supports path and walk extraction from `.gfaz` without full round-trip
  conversion.
- Supports appending path-only or walk-only haplotypes to an existing `.gfaz`
  file using the stored rulebook.
- Runs pangenome analyses — `deconstruct` (GFA→VCF), `growth`, and `pav` —
  directly on the compressed `.gfaz`, without decompressing back to GFA
  (see [Compute Engine Performance](#compute-engine-performance)).

## Current Model

- Shared container: CPU and GPU workflows both serialize to the same `.gfaz`
  file format.
- CPU decompression default: streaming direct-writer mode, which reduces peak
  memory usage.
- CPU in-memory decompression is still available through `decompress_gfa(...)`
  and `gfaz decompress --legacy`.
- GPU backend is still experimental.
- Segment names are reconstructed canonically during decompression as dense
  1-based numeric IDs.

## Build

Initialize the environment first:

```bash
conda activate gfa
git submodule update --init --recursive
```

CPU-only build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
```

CPU + GPU build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON -DCUDA_PATH=/usr/local/cuda-12.8
cmake --build build -j"$(nproc)"
```

The CLI binary is:

```bash
build/bin/gfaz
```

## CLI

Compression:

```bash
# CPU compression
build/bin/gfaz compress example.gfa

# GPU compression; still writes the same .gfaz container
build/bin/gfaz compress --gpu example.gfa
```

Decompression:

```bash
# CPU default: streaming direct-writer path
build/bin/gfaz decompress example.gfa.gfaz

# CPU legacy: full in-memory graph reconstruction
build/bin/gfaz decompress --legacy example.gfa.gfaz

# GPU default: rolling-output GPU path
build/bin/gfaz decompress --gpu example.gfa.gfaz

# GPU legacy: whole-graph GPU decompression path
build/bin/gfaz decompress --gpu --gpu-legacy example.gfa.gfaz
```

Extraction and append workflows:

```bash
# Extract P-lines by path name
build/bin/gfaz extract-path example.gfa.gfaz chr1

# Extract a W-line by full identifier tuple
build/bin/gfaz extract-walk example.gfa.gfaz sample 0 seq1 0 1000

# Append path-only or walk-only haplotypes
build/bin/gfaz add-haplotypes example.gfa.gfaz new_paths.gfa
```

Compute-engine workflows (run directly on `.gfaz`, no GFA materialization):

```bash
# Compute growth curves directly from compressed paths/walks.
# -G/--group-by selects the grouping: path (default), sample-hap-seq, sample-hap,
# or sample (the last three mirror Panacus's default, -H, and -S).
build/bin/gfaz growth -i example.gfa.gfaz -j 8 -G sample-hap-seq

# Compute PAV ratios over BED ranges directly from compressed paths/walks
build/bin/gfaz pav -i example.gfa.gfaz -b ranges.bed -S -M -t 8

# Derive a VCF relative to a reference path directly from compressed traversals.
# The default matches `vg deconstruct` (one record per top-level snarl).
build/bin/gfaz deconstruct -i example.gfa.gfaz -r chr1 -S -t 16 > example.vcf
```

`growth` computes expected node accumulation curves from path/walk group
coverage (Panacus-equivalent). `pav` computes presence/absence ratios for BED
intervals by building node-to-group membership from compressed traversals
(odgi-compatible node semantics; supports `-S`/`-H` grouping, `-M` matrix output,
and `-B` thresholded binary output). `deconstruct` emits a VCF
of variant sites relative to a chosen reference path, with per-sample phased
genotypes (see [DECONSTRUCT_WORKFLOW.md](docs/workflows/DECONSTRUCT_WORKFLOW.md)). All three
operate on `.gfaz` without materializing the original GFA.

`deconstruct` has three site-finding modes. **The default emits one record per
top-level snarl** via a global biconnected decomposition, matching
`vg deconstruct`'s default granularity — producing output identical to `vg` is
the goal of this workflow. On full human chromosomes it reproduces vg's calls at
more than 99.99% position concordance while running an order of magnitude faster
with less memory (see [Compute Engine Performance](#compute-engine-performance)).
The two legacy modes — `--snarl` (leaf-superbubble
superset) and `--linear` (the flat reference-anchor heuristic) — are deprecated
and will be removed in a future release.

Notes:

- In CPU-only builds, `--gpu` falls back to CPU with a warning.
- CPU decompression defaults to streaming direct-writer mode.
- GPU decompression defaults to rolling traversal expansion.
- `extract-path`, `extract-walk`, and `add-haplotypes` all operate on the shared
  `.gfaz` representation.

## Python

Basic CPU workflow:

```python
import gfa_compression as gfac

graph = gfac.parse("example.gfa")
compressed = gfac.compress_file("example.gfa", rounds=8, threshold=2, delta_round=1)
gfac.serialize(compressed, "example.gfaz")

data = gfac.deserialize("example.gfaz")
roundtrip_graph = gfac.decompress_data(data)
gfac.write_gfa(roundtrip_graph, "example.roundtrip.gfa")
```

`delta_round=0` is supported on the CPU path and disables delta encoding.
The default remains `1`.

Lower-memory CPU write path:

```python
import gfa_compression as gfac

data = gfac.deserialize("example.gfaz")
gfac.write_gfa_from_compressed_data(data, "example.streamed.gfa")
```

GPU workflow:

```python
import gfa_compression as gfac

if gfac.has_gpu_backend():
    graph = gfac.parse("example.gfa")
    gpu_graph = gfac.convert_to_gpu_layout(graph)
    compressed = gfac.compress_gpu_graph(gpu_graph, 8)
    gfac.serialize(compressed, "example_gpu.gfaz")
```

Useful Python entry points:

- `parse(...)` / `parse_gfa(...)`
- `compress_file(...)`
- `decompress_data(...)`
- `serialize(...)`
- `deserialize(...)`
- `write_gfa(...)`
- `write_gfa_from_compressed_data(...)`
- `extract_path_line(...)` / `extract_path_lines(...)`
- `extract_walk_line(...)`
- `extract_walk_line_by_name(...)`
- `extract_walk_lines(...)`
- `extract_walk_lines_by_name(...)`
- `add_haplotypes(...)`

CUDA builds also expose:

- `has_gpu_backend()`
- `convert_to_gpu_layout(...)`
- `convert_from_gpu_layout(...)`
- `compress_gfa_gpu(...)`
- `compress_gpu_graph(...)`
- `decompress_to_gpu_layout(...)`

## Internal Data Model

The current in-memory CPU graph groups record families as follows:

- `segments` (`SegmentData`) for S-line state
- `paths_data` (`PathData`) for P-line state
- `walks` (`WalkData`) for W-line state
- `links` (`LinkData`) for L-line state
- `jumps` (`JumpData`) for J-line state
- `containments` (`ContainmentData`) for C-line state

The serialized `.gfaz` format remains shared across CPU and GPU backends.

## Validation

The whole suite (CLI regressions + golden-file concordance + the binding-based
round-trip matrix) runs from a single entry point, reported as PASS / SKIP / FAIL:

```bash
conda activate gfa
python3 tests/run_all.py            # PASS / SKIP / FAIL summary
ctest --test-dir build              # same, via the `gfaz_tests` CTest entry
```

A suite that cannot run (e.g. the compiled bindings are not importable, or an
external-tool golden is missing) exits with SKIP rather than FAIL. The GPU paths
are skipped unless a CUDA build is present. See [tests/README.md](tests/README.md)
for the full layout. To run just the binding-based round-trip checks:

```bash
python3 tests/regression/test_compression_regression.py example.gfa
```

## Documentation

Full docs live under [`docs/`](docs/README.md) (see that index for the complete
list). Highlights:

- [docs/BUILD_GUIDE.md](docs/BUILD_GUIDE.md): build instructions and CMake options
- [docs/WORKFLOW.md](docs/WORKFLOW.md): internal workflow and serialization reference
- [docs/workflows/GROWTH_WORKFLOW.md](docs/workflows/GROWTH_WORKFLOW.md): growth
  workflow and comparison with Panacus
- [docs/workflows/PAV_WORKFLOW.md](docs/workflows/PAV_WORKFLOW.md): PAV workflow
  and comparison with odgi
- [docs/workflows/DECONSTRUCT_WORKFLOW.md](docs/workflows/DECONSTRUCT_WORKFLOW.md):
  GFA→VCF deconstruct workflow, algorithm, and limitations
- [docs/design/](docs/design/): forward-looking design & roadmap notes

## Limitations

- GPU backend requires a CUDA-enabled build and runtime environment.
- GPU backend is still experimental.
- Decompression reconstructs canonical dense numeric segment IDs rather than the
  original segment-name strings.

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
