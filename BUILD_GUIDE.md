# Build Guide - GFAz

## Overview

GFAz supports two backend configurations:

- CPU-only
- CPU + GPU (`ENABLE_CUDA=ON`)

Both backends use the same shared `.gfaz` container format. GPU support adds GPU
implementations for compression and decompression, but it does not introduce a
second file format.

The build can also independently enable or disable:

- Python bindings (`gfa_compression`)
- CLI executable (`gfaz`)
- system Zstd vs vendored Zstd
- optional CLI profiling support

## Current CMake Options

| Option | Default | Effect |
|---|---:|---|
| `ENABLE_CUDA` | `OFF` | Enable CUDA backend and GPU code paths. |
| `ENABLE_PROFILING` | `OFF` | Link `gfaz` with gperftools profiler support. |
| `GFAZ_USE_SYSTEM_ZSTD` | `ON` | Use system-installed Zstd instead of the vendored copy in `extern/zstd`. |
| `BUILD_PYTHON_BINDINGS` | `ON` | Build the `gfa_compression` Python extension module. |
| `BUILD_CLI` | `ON` | Build the `gfaz` CLI executable. |
| `CUDA_PATH` | empty | Optional CUDA toolkit root used to locate `nvcc`. |

Related standard CMake settings:

- `CMAKE_BUILD_TYPE`
- `CMAKE_CUDA_ARCHITECTURES`
- `CMAKE_INSTALL_PREFIX`

Notes:

- When `ENABLE_CUDA=ON`, the build requires CMake `3.23.1` or newer.
- If CUDA is enabled and `CMAKE_CUDA_ARCHITECTURES` is not set explicitly, the
  project uses `native`.

## Dependencies

Required for the default build:

- CMake >= 3.15
- C++17 compiler
- Python 3 development headers if `BUILD_PYTHON_BINDINGS=ON`
- pybind11 submodule
- Zstd

Optional:

- OpenMP
- CUDA toolkit for `ENABLE_CUDA=ON`
- gperftools for `ENABLE_PROFILING=ON`

## Environment Setup

Typical project environment:

```bash
conda activate gfa
git submodule update --init --recursive
```

If you need to create the environment first:

```bash
conda create -n gfa python=3.11
conda activate gfa
conda install -c conda-forge pybind11 numpy
```

## Common Builds

### Default Build

Builds:

- core library
- Python bindings
- CLI

Command:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
```

Expected configure behavior:

- CUDA disabled
- system Zstd preferred
- Python bindings enabled
- CLI enabled

### CPU-only Build Without Python Bindings

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_PYTHON_BINDINGS=OFF
cmake --build build -j"$(nproc)"
```

### CPU-only Build Without CLI

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_CLI=OFF
cmake --build build -j"$(nproc)"
```

### CPU + GPU Build

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_CUDA=ON
cmake --build build -j"$(nproc)"
```

If CUDA is not in the default search path, provide `CUDA_PATH`:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_CUDA=ON \
  -DCUDA_PATH=/usr/local/cuda-12.8
cmake --build build -j"$(nproc)"
```

### Build With Vendored Zstd

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGFAZ_USE_SYSTEM_ZSTD=OFF
cmake --build build -j"$(nproc)"
```

### Build With CLI Profiling

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_PROFILING=ON
cmake --build build -j"$(nproc)"
```

This only affects the `gfaz` executable.

## Outputs

### Core library

- `gfa_compression_core` static library

### Python bindings

If `BUILD_PYTHON_BINDINGS=ON`:

- `gfa_compression` Python extension module in the build tree

### CLI

If `BUILD_CLI=ON`:

- `build/bin/gfaz`

## Dependency Resolution Behavior

### Zstd

If `GFAZ_USE_SYSTEM_ZSTD=ON`:

- CMake first tries `find_package(ZSTD)`
- if that does not provide `ZSTD::ZSTD`, it falls back to `pkg-config`

If `GFAZ_USE_SYSTEM_ZSTD=OFF`:

- the vendored source under `extern/zstd` is built as `zstd_static`

### OpenMP

OpenMP is optional.

- If found, CPU OpenMP regions are enabled.
- If not found, the project still builds, but CPU parallel regions fall back to
  single-threaded execution.

### CUDA

If `ENABLE_CUDA=ON`:

- CUDA language is enabled
- `find_package(CUDAToolkit REQUIRED)` is used
- `extern/cuCollections` is added to the build

## What Each Build Contains

### CPU build

- GFA parser and writer
- CPU compression workflow
- CPU decompression workflow
- extraction workflows
- add-haplotypes workflow
- shared serializer/deserializer

### CUDA build

Includes everything from the CPU build, plus:

- `GfaGraph <-> GfaGraph_gpu` conversion
- GPU compression pipeline
- GPU decompression pipeline
- GPU direct-writer path
- GPU serializer compatibility aliases over the shared `.gfaz` format

## Python API Availability by Build Mode

### Available in all builds

- `parse(...)` / `parse_gfa(...)`
- `compress_file(...)`
- `decompress_data(...)`
- `serialize(...)`
- `deserialize(...)`
- `write_gfa(...)`
- `write_gfa_from_compressed_data(...)`
- extraction helpers
- `add_haplotypes(...)`
- `verify_round_trip(...)`
- `has_gpu_backend()`

### CUDA builds only

- `convert_to_gpu_layout(...)`
- `convert_from_gpu_layout(...)`
- `compress_gfa_gpu(...)`
- `compress_gpu_graph(...)`
- `decompress_to_gpu_layout(...)`
- `serialize_gpu(...)`
- `deserialize_gpu(...)`
- GPU verification helpers
- `gfa_compression.experimental.gpu.*`

Runtime check:

```python
import gfa_compression as gfac
print(gfac.has_gpu_backend())
```

## CLI Notes

The CLI is built only when `BUILD_CLI=ON`.

Key behavior:

- CPU-only build + `--gpu`: warns and falls back to CPU.
- CPU and GPU backends read and write the same `.gfaz` format.
- CPU decompression defaults to streaming direct-writer mode.
- GPU decompression defaults to rolling traversal expansion.
- CPU `threads=0` uses auto policy:
  `GFAZ_NUM_THREADS` -> `OMP_NUM_THREADS` -> `min(8, logical_cpus/2)`.
- Some CPU tuning knobs are ignored in GPU mode by design.

## Format Version

Current shared binary format:

- magic: `GFAZ`
- version: `5`

## Troubleshooting

### CUDA not found

Check:

```bash
which nvcc
```

If needed:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Or configure with:

```bash
cmake -S . -B build -DENABLE_CUDA=ON -DCUDA_PATH=/usr/local/cuda-12.8
```

### Python build fails

If `BUILD_PYTHON_BINDINGS=ON`, CMake requires:

- Python interpreter
- Python development headers/libraries

Disable bindings if you only need the core library or CLI:

```bash
cmake -S . -B build -DBUILD_PYTHON_BINDINGS=OFF
```

### System Zstd not found

Either install system Zstd development packages, or switch to vendored Zstd:

```bash
cmake -S . -B build -DGFAZ_USE_SYSTEM_ZSTD=OFF
```

### Profiling build fails

`ENABLE_PROFILING=ON` requires a system `profiler` library from gperftools.

On Debian/Ubuntu:

```bash
sudo apt install libgoogle-perftools-dev
```

### CPU-only build accidentally configured after a CUDA build

Reconfigure explicitly:

```bash
cmake -S . -B build -DENABLE_CUDA=OFF
```
