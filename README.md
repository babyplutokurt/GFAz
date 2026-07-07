# GFAz: State-of-the-Art Graphical Fragment Assembly Compression

GFAz is a C++/CUDA library and command-line tool for compressing and
decompressing Graphical Fragment Assembly (GFA) files.

In our current benchmarks, it reaches up to 20x higher compression ratio than
Gzip and 15x higher compression ratio than Zstd, with GB/s-level throughput.

It has two execution backends:

- CPU
- GPU (experimental, CUDA build required)

Both backends produce and consume the same `.gfaz` container format. The backend
changes how transforms are computed, not the on-disk format.

## Performance

| Dataset | Metrics | Gzip | Zstd | sqz | sqz+bgzip | GBZ | gfaz(CPU) | gfaz(GPU) |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| chr1. | Ratio | 5.59 | 7.54 | 3.09 | 18.0 | 9.52 | **35.4** | **31.7** |
| | Co. | 46.2 | 2178 | 3.95 | 3.97 | 12.1 | **385** | **2754** |
| | De. | 359 | 1618 | 21.6 | 21.4 | 284 | **1658** | **8124** |
| chr6. | Ratio | 5.04 | 6.99 | 5.51 | 20.8 | 19.2 | **35.4** | **28.18** |
| | Co. | 41.0 | 1712 | 3.56 | 3.56 | 10.7 | **348** | **3791** |
| | De. | 348 | 1515 | 20.1 | 20.4 | 281 | **1758** | **7230** |
| E.coli | Ratio | 4.69 | 5.67 | 1.26 | 7.46 | 5.58 | **18.3** | **16.7** |
| | Co. | 33.3 | 1356 | 4.57 | 4.53 | 20.2 | **190** | **678** |
| | De. | 310 | 1258 | 34.0 | 32.2 | 197 | **491** | **1430** |
| HPRCv1.1 | Ratio | 4.02 | 5.32 | - | - | 14.0 | **22.4** | **20.4** |
| | Co. | 36.4 | 1657 | - | - | 84.5 | **231** | **4843** |
| | De. | 319 | 1234 | - | - | 650 | **1058** | **9435** |
| HPRCv2.0 | Ratio | 4.19 | 6.49 | - | - | 66.8 | **83.9** | **76.4** |
| | Co. | 49.1 | 1514 | - | - | 130 | **367** | **-** |
| | De. | 342 | 1240 | - | - | 648 | **1652** | **-** |
| HPRCv2.1 | Ratio | 4.19 | 6.43 | - | - | 64.2 | **82.8** | **74.2** |
| | Co. | 48.9 | 1540 | - | - | 136 | **348** | **-** |
| | De. | 343 | 1241 | - | - | 652 | **1559** | **-** |

`Ratio` indicates compression ratio, `Co.` indicates compression speed/time, and
`De.` indicates decompression speed/time. Bold values indicate the best result
in each row. System configuration: AMD Ryzen Threadripper PRO 9955WX
(16 cores), NVIDIA RTX Pro 6000, and 512 GB DDR5-6400 memory.

## Compute Engine Performance

Beyond compression, `gfaz` runs pangenome analyses **directly on the compressed
`.gfaz` container** — with no decompression back to GFA. Each analysis reproduces
the reference tool's output while running one to three orders of magnitude faster
and using far less memory. Measured at 16 threads (wall-clock time / peak RSS);
baselines read the uncompressed GFA, `gfaz` reads the `.gfaz`.

| Analysis (vs. baseline) | Graph | Baseline | gfaz | Speedup | Mem. saving |
|:---|:---|---:|---:|---:|---:|
| `deconstruct` vs. `vg deconstruct` | chr1 | 805 s / 30.3 GB | 11.3 s / 8.3 GB | **71×** | **3.7×** |
| | HGSVC3 (80 GB GFA) | 132.9 min / 397 GB | 8.7 min / 51.9 GB | **15×** | **7.6×** |
| `growth` vs. Panacus | chr1 | 18.3 s / 6.16 GB | 0.74 s / 0.66 GB | **25×** | **9.3×** |
| | HPRC v2.0 (358 GB GFA) | 245 min / 327 GB | 39.8 s / 12.9 GB | **369×** | **25×** |
| `pav` vs. `odgi pav` | chr1 | 3499 s / 31.9 GB | 11.7 s / 9.5 GB | **299×** | **3.4×** |
| | chr6 | 4759 s / 19.4 GB | 7.8 s / 6.4 GB | **613×** | **3.0×** |

Outputs match the baselines: `deconstruct` VCF record counts agree with `vg` to
within ~0.2% (~99.99% position concordance), `growth` reproduces Panacus's growth
curve exactly at every point, and `pav` matrices are structurally identical to
`odgi`'s. On the largest whole-genome graphs (HPRC v2.0/v2.1, 358–369 GB GFA)
`vg` and `odgi` cannot load the input within memory, while `gfaz` runs from a
~4.5 GB container; for `deconstruct` and `growth`, peak RSS stays below the size
of the uncompressed GFA. System configuration as above.

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
**99.99% position concordance** and within **±0.13%** record count, while running
**17–24× faster** with **1.4–1.7× less memory** (chr1: 1,593,899 vs vg 1,593,956
records, 28.5 s vs 696 s). The two legacy modes — `--snarl` (leaf-superbubble
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

## License

MIT
