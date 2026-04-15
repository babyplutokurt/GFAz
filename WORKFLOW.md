# Workflow Reference

This document describes the current end-to-end data flow for CPU and GPU paths. The key architectural point is that both backends now emit and consume the same `CompressedData` representation and the same `.gfaz` file format.

## High-Level Pipelines

### CPU pipeline

1. Parse GFA text to `GfaGraph`.
2. Compress with CPU workflow (`compress_gfa(...)`) to `CompressedData`.
3. Serialize `CompressedData` (`serialize_compressed_data(...)`) to `.gfaz`.
4. Deserialize CPU payload (`deserialize_compressed_data(...)`).
5. Default CPU CLI decompression streams directly from `CompressedData` to GFA text (`write_gfa_from_compressed_data(...)`) to reduce peak memory usage.
6. In-memory CPU decompression (`decompress_gfa(...)` -> `GfaGraph` -> `write_gfa(...)`) remains available for APIs and legacy CLI mode when a full graph object is needed.

### GPU pipeline (`ENABLE_CUDA=ON`)

1. Parse GFA text to `GfaGraph`.
2. Convert to GPU layout (`convert_to_gpu_layout(...)`) -> `GfaGraph_gpu`.
3. Compress with GPU workflow (`compress_gpu_graph(...)` / `compress_gfa_gpu(...)`) to `CompressedData`.
4. Serialize `CompressedData` to `.gfaz`.
5. Deserialize `CompressedData`.
6. Decompress to GPU layout (`decompress_to_gpu_layout(...)`) -> `GfaGraph_gpu`.
7. Optionally convert back (`convert_from_gpu_layout(...)`) -> `GfaGraph` for CPU-side consumers.

The serializer entry points exposed to GPU callers are compatibility aliases over the shared serializer. There is no separate GPU container format.

## CPU Compression Workflow Details

Main implementation: `src/workflows/compression_workflow.cpp`

1. Parse input GFA into `GfaGraph`, whose segment and path state is grouped as `SegmentData` and `PathData`.
2. Flatten path and walk traversals for compression-oriented processing.
3. Apply delta transform (`delta_round` times) and track max absolute IDs.
4. Run 2-mer rule generation and iterative remap for `num_rounds`.
5. Record per-layer rule ranges.
6. Delta-encode rule arrays and ZSTD-compress blocks.
7. Flatten and compress paths, names, overlaps, walks, segments, links, optional fields, and J/C records.
8. Return `CompressedData`.

Notes:

- CPU compression parameters: rounds, frequency threshold, delta rounds, thread count.
- OpenMP is used when available.
- `num_threads=0` uses auto policy (`GFAZ_NUM_THREADS` -> `OMP_NUM_THREADS` -> `min(8, logical_cpus/2)`), and this same policy applies to parser parallel sections and CPU workflow OpenMP regions.

## CPU Decompression Workflow Details

Main implementation: `src/workflows/decompression_workflow.cpp`

The CPU backend has two decompression modes:

- Default CLI mode: streaming direct-writer (`write_gfa_from_compressed_data(...)`), which writes GFA output from `CompressedData` without materializing a full `GfaGraph`. This is the default because it lowers peak memory usage.
- In-memory mode: `decompress_gfa(...)`, which reconstructs a full `GfaGraph` for APIs and `gfaz decompress --legacy`.

`decompress_gfa(...)` performs these steps:

1. Decompress and delta-decode rule arrays.
2. Reconstruct encoded path/walk sequences.
3. Expand grammar rules into original node IDs.
4. Inverse delta transform (`delta_round` times).
5. Rebuild metadata (names, overlaps, walks, grouped segment/path data, links, optional fields, J/C records).
6. Return reconstructed `GfaGraph`.

## GPU Compression/Decompression Workflow Details

Main implementations:

- `src/gpu/compression/compression_workflow_gpu.cu`
- `src/gpu/decompression/decompression_workflow_gpu.cu`

Behavior:

- Uses GPU-oriented flattened structures and kernels for path/rule processing, then stores the final payload in the shared `CompressedData` Zstd-based format.
- GPU compression accelerates the transform pipeline; it does not define a different entropy layer or file format.
- GPU decompression reads the same serialized blocks as CPU decompression and reconstructs `GfaGraph_gpu` from the shared container.
- High-level public Python APIs:
  - `compress_gfa_gpu(...)`
  - `compress_gpu_graph(...)`
  - `decompress_to_gpu_layout(...)`

## Serialization Contracts

Shared serialization (`include/codec/serialization.hpp`, `src/codec/serialization.cpp`):

- Magic: `GFAZ`
- Version: `5`
- Type: `CompressedData`

GPU serialization (`include/gpu/core/serialization_gpu.hpp`, `src/gpu/core/serialization_gpu.cpp`):

- Alias over the shared serializer
- Magic: `GFAZ`
- Version: `5`
- Type: `CompressedData`

Important:

- CPU and GPU serializers share the same `CompressedData` container and serializer implementation.
- Deserializers validate magic/version and throw on mismatch.
- CPU-compressed `.gfaz` files can be decompressed through the GPU path.
- GPU-compressed `.gfaz` files can be decompressed through the CPU path.

## CLI Workflow

CLI entrypoint: `src/cli/gfaz_cli.cpp`

### `gfaz compress`

- CPU default path: parse -> `compress_gfa` -> `serialize_compressed_data`
- GPU path (`--gpu`, CUDA build): parse/convert -> `compress_gfa_gpu` -> shared serializer
- CPU-only build with `--gpu`: warns and falls back to CPU path.

### `gfaz decompress`

- CPU default path: `deserialize_compressed_data` -> `write_gfa_from_compressed_data` (streaming direct-writer, lower memory)
- CPU legacy path (`--legacy`): `deserialize_compressed_data` -> `decompress_gfa` -> `write_gfa`
- GPU default path (`--gpu`, CUDA build): shared deserialize -> `write_gfa_from_compressed_data_gpu` (rolling-output direct writer)
- GPU legacy path (`--gpu --gpu-legacy`, CUDA build): shared deserialize -> `decompress_to_gpu_layout` -> `convert_from_gpu_layout` -> `write_gfa`
- CPU-only build with `--gpu`: warns and falls back to CPU path.

Because the container is shared, the backend chosen at decompression time is independent of the backend that created the file.

## Python Binding Surface (Current)

Bindings file: `src/python/bindings.cpp`

Stable root APIs (all builds):

- parse/serialize/compress/decompress and write helpers
- `has_gpu_backend()`

Stable GPU root APIs (CUDA builds only):

- GPU graph conversion
- GPU compress/decompress and GPU serialization helpers

Experimental GPU APIs (CUDA builds only):

- `gfa_compression.experimental.gpu.*`

Legacy GPU alias names are retained for compatibility in CUDA builds.

## Semantic Notes

- `GfaGraph` now groups S-line state in `segments` (`SegmentData`) and P-line state in `paths_data` (`PathData`), while retaining `walks`, `links`, `jumps`, and `containments` as separate record groups.
- Segment names are reconstructed canonically during decompression as dense 1-based numeric IDs.
- Round-trip verification is therefore based on graph semantics rather than original segment-name strings.
