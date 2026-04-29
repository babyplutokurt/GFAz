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

Temporary downstream workflows:

```bash
# Compute growth curves directly from compressed paths/walks
build/bin/gfaz growth -i example.gfa.gfaz -j 8

# Compute PAV ratios over BED ranges directly from compressed paths/walks
build/bin/gfaz pav -i example.gfa.gfaz -b ranges.bed -S -M -t 8
```

`growth` computes expected node accumulation curves from path/walk group
coverage. `pav` computes presence/absence ratios for BED intervals by building
node-to-group membership from compressed traversals. Both operate on `.gfaz`
without materializing the original GFA.

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

Typical checks:

```bash
conda activate gfa
python tests/cpu/test_roundtrip.py example.gfa
python tests/cpu/test_streaming_roundtrip.py example.gfa
python tests/gpu/test_roundtrip.py example.gfa
python tests/regression/test_compression_regression.py example.gfa
build/bin/gfaz compress example.gfa
```

## Documentation

- [BUILD_GUIDE.md](BUILD_GUIDE.md): build instructions and CMake options
- [workflow.md](workflow.md): internal workflow and serialization reference
- [GROWTH_WORKFLOW.md](GROWTH_WORKFLOW.md): growth workflow and comparison with
  Panacus
- [PAV_WORKFLOW.md](PAV_WORKFLOW.md): PAV workflow and comparison with odgi

## Limitations

- GPU backend requires a CUDA-enabled build and runtime environment.
- GPU backend is still experimental.
- Decompression reconstructs canonical dense numeric segment IDs rather than the
  original segment-name strings.

## License

MIT
