# Rolling GPU Decompression Direct-Writer Plan

## Goal

Refactor GPU decompression so that large traversal payloads are:

- decompressed on GPU in rolling chunks
- copied into pinned host buffers asynchronously
- consumed by a CPU writer thread directly into the output GFA

The target is to avoid the current end-to-end path:

1. decompress full traversal into host memory
2. materialize full `GfaGraph_gpu`
3. convert to full host `GfaGraph`
4. write the file afterward

Instead, the new path should overlap GPU traversal decompression with CPU file
formatting and disk output.

## Current Problem

The current GPU CLI decompression path in
[`src/gfaz_cli.cpp`](/home/kurty/Release/gfa_compression/src/gfaz_cli.cpp#L516)
does:

1. `deserialize_compressed_data_gpu(...)`
2. `decompress_to_gpu_layout(...)`
3. `convert_from_gpu_layout(...)`
4. `write_gfa(...)`

This means:

- traversal decompression finishes before writing starts
- traversal chunks are copied into one large host vector
- paths and walks are rebuilt into nested host vectors
- disk I/O does not overlap GPU work

The current rolling traversal decompressor in
[`src/gpu/codec_gpu.cu`](/home/kurty/Release/gfa_compression/src/gpu/codec_gpu.cu#L2657)
still:

- allocates the full host output vector with `h_result_data.resize(output_size)`
- decodes each chunk
- synchronously copies each chunk into the final host vector

That is chunked decompression, but not streaming decompression-to-writer.

## Target Architecture

The new GPU decompression path should be a direct-writer pipeline:

1. Deserialize compressed GPU payload.
2. Decode metadata blocks needed for `H`, `S`, `L`, `J`, and `C` lines.
3. Precompute rolling traversal chunk schedule.
4. Launch rolling GPU decompression chunk-by-chunk.
5. Copy each decoded traversal chunk into a pinned host buffer using
   `cudaMemcpyAsync`.
6. Hand completed buffers to a CPU writer thread.
7. Writer thread formats `P` and `W` lines directly from flat chunk data and
   writes them to disk.
8. Recycle the host buffer and continue.

This should overlap:

- GPU chunk expansion
- device-to-host transfer
- CPU line formatting
- file output

## Non-Goals

- Replacing the existing `decompress_to_gpu_layout(...)` API.
- Removing the in-memory GPU decompression path used by tests or bindings.
- Changing the compressed file format.
- Changing rule expansion semantics.

## High-Level Design

### Keep on GPU

- encoded traversal
- decoded rule arrays
- rule size and offset arrays
- per-chunk decode workspace
- per-chunk expanded traversal output before copy

### Keep on Host

- decoded metadata needed to write non-traversal lines
- pinned traversal staging buffers
- writer queue
- output file stream

### Avoid Materializing

The new direct-writer path should avoid creating:

- full host `FlattenedPaths::data`
- full `GfaGraph_gpu`
- full host `GfaGraph`
- nested `graph.paths`
- nested `graph.walks.walks`

## New Public Entry Point

Add a GPU-specific writer entry point parallel to the CPU direct writer:

- [`include/gfa_writer.hpp`](/home/kurty/Release/gfa_compression/include/gfa_writer.hpp)
  - `void write_gfa_from_compressed_data_gpu(const gpu_compression::CompressedData_gpu &data, const std::string &output_path);`

Use this from the GPU branch in
[`src/gfaz_cli.cpp`](/home/kurty/Release/gfa_compression/src/gfaz_cli.cpp#L516)
instead of:

- `decompress_to_gpu_layout(...)`
- `convert_from_gpu_layout(...)`
- `write_gfa(...)`

## Refactor Shape

### 1. Split traversal decompression from graph materialization

Today:

- [`decompress_paths_gpu(...)`](/home/kurty/Release/gfa_compression/src/gpu/decompression_workflow_gpu.cu#L284)
  returns a fully materialized `FlattenedPaths`

Target:

- keep `decompress_paths_gpu(...)` for compatibility
- add a rolling traversal streaming API used by the writer path

Recommended new API shape:

- `build_rolling_decode_schedule(...)`
- `decode_traversal_chunk_to_device(...)`
- `copy_decoded_chunk_to_pinned_host_async(...)`

These should be internal GPU decompression helpers.

### 2. Add a traversal chunk schedule

Each scheduled chunk should contain:

- `segment_begin`
- `segment_end`
- `encoded_begin`
- `encoded_end`
- `expanded_begin`
- `expanded_end`

This information is already derived inside
[`rolling_expand_and_inverse_delta_decode(...)`](/home/kurty/Release/gfa_compression/src/gpu/codec_gpu.cu#L2657).

Refactor that function so schedule creation is exposed separately from host
vector assembly.

### 3. Use pinned host buffers

Use 2 or 3 reusable pinned host buffers.

Each buffer should contain:

- `int32_t *host_nodes`
- `size_t node_capacity`
- `size_t node_count`
- `std::vector<uint32_t> lengths`
- `uint32_t segment_begin`
- `uint32_t segment_end`
- CUDA event for copy completion

Pinned memory is required to make `cudaMemcpyAsync` useful.

### 4. Add producer-consumer pipeline

Producer responsibilities:

- decode traversal chunk on GPU
- launch async D2H copy into pinned host buffer
- record completion event
- enqueue buffer for writer thread

Consumer responsibilities:

- wait for buffer completion event
- format `P` and `W` lines from flat chunk data
- write formatted bytes to output stream
- return buffer to free queue

## Metadata Handling

Decode non-traversal metadata once up front on host:

- header
- node names
- path names
- path overlaps
- walk sample IDs
- walk hap indices
- walk seq IDs
- walk seq starts
- walk seq ends
- segment sequences
- links
- jumps
- containments
- optional fields

Write file in standard order:

1. `H`
2. `S`
3. `L`
4. `P`
5. `W`
6. `J`
7. `C`

`P` and `W` lines are the only part that should be streaming.

## Writer Refactor

Refactor [`src/gfa_writer.cpp`](/home/kurty/Release/gfa_compression/src/gfa_writer.cpp)
to extract formatting helpers that do not require nested graph vectors.

Add helpers like:

- `format_path_line_from_flat(...)`
- `format_walk_line_from_flat(...)`

Inputs should be:

- node slice pointer or span
- slice length
- node-name lookup
- per-path or per-walk metadata

The direct GPU writer should call these helpers as it consumes traversal
buffers.

## Execution Model

Recommended buffer flow:

1. acquire free buffer
2. decode traversal chunk into device workspace
3. `cudaMemcpyAsync` into pinned host buffer
4. record completion event
5. push buffer to ready queue
6. writer thread waits for event
7. writer formats chunk lines into a large CPU string buffer
8. `ofstream.write(...)`
9. recycle buffer

Recommended streams:

- one compute stream for chunk decode
- one copy stream for D2H transfer

Recommended synchronization:

- CUDA events between compute and copy
- condition variable or lock-free queue between producer and consumer

## Avoidable Costs To Remove

The new path should remove these costs from CLI GPU decompression:

- full `FlattenedPaths` materialization
- full `GfaGraph_gpu` traversal ownership
- full `convert_from_gpu_layout(...)`
- nested path/walk vector reconstruction
- post-decompression `write_gfa(...)` on a full host graph

## Suggested Internal Types

### Rolling decode schedule

```cpp
struct DecodeTraversalChunk {
  uint32_t segment_begin;
  uint32_t segment_end;
  int64_t encoded_begin;
  int64_t encoded_end;
  int64_t expanded_begin;
  int64_t expanded_end;
};
```

### Host pinned buffer

```cpp
struct PinnedTraversalBuffer {
  int32_t *host_nodes = nullptr;
  size_t node_capacity = 0;
  size_t node_count = 0;
  uint32_t segment_begin = 0;
  uint32_t segment_end = 0;
  std::vector<uint32_t> lengths;
  cudaEvent_t ready = nullptr;
};
```

These names are suggestions, not required API.

## Implementation Phases

### Phase 1: Isolate rolling decode schedule

Refactor
[`rolling_expand_and_inverse_delta_decode(...)`](/home/kurty/Release/gfa_compression/src/gpu/codec_gpu.cu#L2657)
into:

- schedule construction
- per-chunk decode helper
- existing full-host-output wrapper

Goal:

- preserve current behavior
- make chunk scheduling reusable

### Phase 2: Add pinned-buffer chunk export

Add a helper that:

- decodes one scheduled chunk into device workspace
- asynchronously copies the decoded chunk into pinned host memory

Goal:

- replace synchronous `thrust::copy(...)` of chunk output

### Phase 3: Build direct GPU writer

Add `write_gfa_from_compressed_data_gpu(...)` that:

- decodes metadata
- writes `H/S/L`
- streams `P/W`
- writes `J/C`

Goal:

- produce correct output without materializing full graph objects

### Phase 4: Wire CLI to direct writer

Change the GPU decompression branch in
[`src/gfaz_cli.cpp`](/home/kurty/Release/gfa_compression/src/gfaz_cli.cpp#L516)
to use:

- `deserialize_compressed_data_gpu(...)`
- `write_gfa_from_compressed_data_gpu(...)`

Goal:

- make CLI benefit from the new streaming path immediately

### Phase 5: Add timing instrumentation

Log separate timings for:

- nvComp decompress
- rule decode
- chunk GPU expansion
- D2H copy
- writer formatting
- file write
- queue wait time

Goal:

- verify that disk writing is actually hiding GPU work

## Validation

### Correctness

Compare output of:

- old GPU CLI decompression path
- new GPU direct-writer path

Checks:

- byte-for-byte GFA output if ordering is identical
- otherwise parse both and compare reconstructed graphs

### Performance

Benchmark:

- old GPU CLI path
- new direct-writer path

Measure:

- total wall time
- throughput in MB/s
- GPU idle time
- writer idle time

Success condition:

- end-to-end runtime approaches writer time when disk is slower than GPU decode
- rolling decompression no longer regresses badly versus old path for large
  inputs

## Main Risks

- Writer formatting may become the new bottleneck if it emits too many small
  writes.
- Poor buffer sizing may cause GPU stalls or writer starvation.
- If pinned buffers are too small, the pipeline will thrash on scheduling
  overhead.
- If metadata decoding remains serialized and large, it may still dominate
  startup cost.

## Recommended Defaults

- start with 2 pinned buffers
- add a third if the writer often blocks the producer
- batch formatted output in a large CPU string buffer before `ofstream.write`
- keep chunk boundaries traversal-aligned

## Summary

The required fix is architectural, not a small kernel tweak.

The current rolling GPU decompression path is still:

- decode all traversal data
- materialize host structures
- write later

The target path should be:

- decode chunk on GPU
- async copy into pinned host buffer
- writer thread consumes and writes immediately
- overlap GPU work with disk output

That is the correct design if disk output is slower than traversal
decompression.
