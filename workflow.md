# Workflow Reference

This document describes the current end-to-end data flow for CPU and GPU paths.

See also [backend_schema_map.md](/home/kurty/Release/gfa_compression/backend_schema_map.md) for a field-by-field mapping of `GfaGraph`, `CompressedData`, and `GfaGraph_gpu`.

## High-Level Pipelines

### CPU pipeline

1. Parse GFA text to `GfaGraph`.
2. Compress with CPU workflow (`compress_gfa(...)`) to `CompressedData`.
3. Serialize CPU payload (`serialize_compressed_data(...)`) to `.gfaz`.
4. Deserialize CPU payload (`deserialize_compressed_data(...)`).
5. Decompress (`decompress_gfa(...)`) back to `GfaGraph`.
6. Write GFA text (`write_gfa(...)`) if needed.

### GPU pipeline (`ENABLE_CUDA=ON`)

1. Parse GFA text to `GfaGraph`.
2. Convert to GPU layout (`convert_to_gpu_layout(...)`) -> `GfaGraph_gpu`.
3. Compress with GPU workflow (`compress_gpu_graph(...)` / `compress_gfa_gpu(...)`) to `CompressedData`.
4. Serialize GPU payload (`serialize_compressed_data_gpu(...)`) to `.gfaz_gpu`.
5. Deserialize GPU payload (`deserialize_compressed_data_gpu(...)`).
6. Decompress to GPU layout (`decompress_to_gpu_layout(...)`) -> `GfaGraph_gpu`.
7. Convert back (`convert_from_gpu_layout(...)`) -> `GfaGraph`.

## CPU Compression Workflow Details

Main implementation: `src/compression_workflow.cpp`

1. Parse input graph and flatten path/walk containers.
2. Apply delta transform (`delta_round` times) and track max absolute IDs.
3. Run 2-mer rule generation and iterative remap for `num_rounds`.
4. Record per-layer rule ranges.
5. Delta-encode rule arrays and ZSTD-compress blocks.
6. Flatten and compress paths, names, overlaps, walks, segments, links, optional fields, and J/C records.
7. Return `CompressedData`.

Notes:

- CPU compression parameters: rounds, frequency threshold, delta rounds, thread count.
- OpenMP is used when available.
- `num_threads=0` uses auto policy (`GFAZ_NUM_THREADS` -> `OMP_NUM_THREADS` -> `min(8, logical_cpus/2)`), and this same policy applies to parser parallel sections and CPU workflow OpenMP regions.

## CPU Decompression Workflow Details

Main implementation: `src/decompression_workflow.cpp`

1. Decompress and delta-decode rule arrays.
2. Reconstruct encoded path/walk sequences.
3. Expand grammar rules into original node IDs.
4. Inverse delta transform (`delta_round` times).
5. Rebuild metadata (names, overlaps, walks, segments, links, optional fields, J/C records).
6. Return reconstructed `GfaGraph`.

## GPU Compression/Decompression Workflow Details

Main implementations:

- `src/gpu/compression_workflow_gpu.cu`
- `src/gpu/decompression_workflow_gpu.cu`

Behavior:

- Uses GPU-oriented flattened structures and kernels for path/rule processing.
- Uses GPU-oriented flattened structures and kernels for path/rule processing, then stores the final payload in the shared `CompressedData` Zstd-based format.
- High-level public Python APIs:
  - `compress_gfa_gpu(...)`
  - `compress_gpu_graph(...)`
  - `decompress_to_gpu_layout(...)`

## Serialization Contracts

CPU serialization (`include/serialization.hpp`, `src/serialization.cpp`):

- Magic: `GFAZ`
- Version: `5`
- Type: `CompressedData`

GPU serialization (`include/gpu/serialization_gpu.hpp`, `src/gpu/serialization_gpu.cpp`):

- Magic: `GPUG`
- Version: `1`
- Type: `CompressedData`

Important:

- CPU and GPU serializers currently share the same `CompressedData` container and serializer implementation.
- Deserializers validate magic/version and throw on mismatch.

## CLI Workflow

CLI entrypoint: `src/gfaz_cli.cpp`

### `gfaz compress`

- CPU default path: parse -> `compress_gfa` -> `serialize_compressed_data`
- GPU path (`--gpu`, CUDA build): `compress_gfa_gpu` -> `serialize_compressed_data_gpu`
- CPU-only build with `--gpu`: warns and falls back to CPU path.

### `gfaz decompress`

- CPU default path: `deserialize_compressed_data` -> `decompress_gfa` -> `write_gfa`
- GPU path (`--gpu`, CUDA build): `deserialize_compressed_data_gpu` -> `decompress_to_gpu_layout` -> `convert_from_gpu_layout` -> `write_gfa`
- CPU-only build with `--gpu`: warns and falls back to CPU path.

## Python Binding Surface (Current)

Bindings file: `src/bindings.cpp`

Stable root APIs (all builds):

- parse/serialize/compress/decompress and write helpers
- `has_gpu_backend()`

Stable GPU root APIs (CUDA builds only):

- GPU graph conversion
- GPU compress/decompress and GPU serialization helpers

Experimental GPU APIs (CUDA builds only):

- `gfa_compression.experimental.gpu.*`

Legacy GPU alias names are retained for compatibility in CUDA builds.
