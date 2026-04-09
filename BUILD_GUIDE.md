# Build Guide - GFA Compression

## Overview

This project has two build modes:

- CPU-only (default)
- CPU + GPU (`ENABLE_CUDA=ON`)

Both modes build the Python module (`gfa_compression`) and the CLI (`gfaz`).

## Build Flags (Current)

| Flag | Default | Effect |
|---|---:|---|
| `ENABLE_CUDA` | `OFF` | Enables CUDA backend, GPU workflows, and GPU Python APIs. |
| `ENABLE_PROFILING` | `OFF` | Links `gfaz` with gperftools profiler. |
| `CUDA_PATH` | empty | Optional CUDA toolkit root used to locate `nvcc`. |

Related standard CMake options you may set:

- `CMAKE_BUILD_TYPE` (e.g. `Release`, `Debug`)
- `CMAKE_CUDA_ARCHITECTURES` (GPU builds only)

## Optional Dependencies

- OpenMP: optional. If found, CPU k-mer collection is parallelized.
- CUDA Toolkit: required only when `ENABLE_CUDA=ON`.

## Environment Setup

Example Conda setup:

```bash
conda create -n gfa python=3.11
conda activate gfa
conda install -c conda-forge pybind11 numpy
```

Additional GPU-related packages used in some environments:

```bash
conda install -c conda-forge cudatoolkit
```

## Build Instructions

```bash
git submodule update --init --recursive
```

### CPU-only (default)

```bash
cd build
cmake ..
cmake --build . -j$(nproc)
```

CMake should report:

- `CUDA disabled - building CPU-only version`

### CPU + GPU

```bash
cd build
cmake -DENABLE_CUDA=ON ..
cmake --build . -j$(nproc)
```

If CUDA is not in default location, provide `CUDA_PATH`:

```bash
cmake -DENABLE_CUDA=ON -DCUDA_PATH=/usr/local/cuda-12.8 ..
```

## Outputs

- Python extension module: `gfa_compression` (in build tree)
- CLI executable: `build/bin/gfaz`

## Backend Responsibilities

### CPU backend

- Parse GFA text into `GfaGraph`
- CPU grammar compression/decompression workflow
- CPU serialization/deserialization (`.gfaz`)
- Baseline path for all builds

### GPU backend (`ENABLE_CUDA=ON`)

- GPU-friendly graph layout conversion (`GfaGraph <-> GfaGraph_gpu`)
- GPU compression/decompression workflows over the shared `.gfaz` format

## File Format Versions

Current binary format in code:

- Shared CPU/GPU format
  - Magic: `GFAZ`
  - Version: `5`

## Python API Capabilities by Build Mode

### Available in all builds (stable root APIs)

- `parse()`, `parse_gfa()`
- `compress()`, `decompress()`
- `serialize()`, `deserialize()`
- `write_gfa()`
- `verify_round_trip()` (alias `verify_roundtrip`)
- `has_gpu_backend()`

### CUDA builds only (stable GPU root APIs)

- `compress_gfa_gpu()`
- `compress_gpu_graph()`
- `decompress_to_gpu_layout()`
- `verify_gpu_round_trip()`
- `serialize_gpu()`, `deserialize_gpu()`
- `convert_to_gpu_layout()`, `convert_from_gpu_layout()`

### Experimental GPU helpers (CUDA builds only)

- `gfa_compression.experimental.gpu.*`

Backward-compatible top-level aliases for several legacy GPU helpers are still exported in CUDA builds.

### Runtime check

```python
import gfa_compression as gfac
print(gfac.has_gpu_backend())
```

## CLI Usage

### Compress

```bash
gfaz compress [OPTIONS] <input.gfa> [output.gfaz]
```

Options:

- `-r, --rounds <N>` default `8`
- `-d, --delta <N>` default `1` (CPU backend)
- `-t, --threshold <N>` default `2` (CPU backend)
- `-j, --threads <N>` default `0` (auto)
- `-g, --gpu` use GPU backend if available

### Decompress

```bash
gfaz decompress [OPTIONS] <input.gfaz> [output.gfa]
```

Options:

- `-j, --threads <N>` default `0` (auto, CPU backend)
- `-g, --gpu` use GPU backend if available

### CLI behavior notes

- CPU-only build + `--gpu`: prints warning and falls back to CPU backend.
- CPU `threads=0` uses auto policy:
  - `GFAZ_NUM_THREADS` if set
  - else `OMP_NUM_THREADS` if set
  - else `min(8, logical_cpus/2)`
- The same CPU thread policy is applied consistently across:
  - parser parallel sections (P/W line parsing)
  - CPU grammar compression/decompression OpenMP regions
- In CUDA build, GPU mode ignores CPU-only tuning knobs:
  - compress: `--delta`, `--threshold`, `--threads`
  - decompress: `--threads`
- Default output names:
  - CPU compress: `<input>.gfaz`
  - GPU compress: `<input>.gfaz`
  - Decompress without output strips `.gfaz` when present.

### Examples

```bash
# CPU compress/decompress
gfaz compress input.gfa
gfaz decompress input.gfa.gfaz

# GPU compress/decompress (CUDA build)
gfaz compress --gpu input.gfa
gfaz decompress --gpu input.gfa.gfaz
```

## Known Limitations

- CPU and GPU backends share the same `.gfaz` container; choose `--gpu` only to select the GPU decompression implementation.
- GPU backend currently uses a different tuning surface than CPU backend; some CLI knobs are intentionally ignored in GPU mode.

## Troubleshooting

### CUDA not found

```bash
which nvcc
```

If needed:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### CPU-only build accidentally requested GPU path

Reconfigure explicitly:

```bash
cmake .. -DENABLE_CUDA=OFF
```

### Profiling build fails

`ENABLE_PROFILING=ON` requires `libgoogle-perftools-dev` (or equivalent package providing `libprofiler`).
