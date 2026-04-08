# Rolling GPU Decompression Direct-Writer Plan

## Status

Current state: groundwork refactors are in place, but the direct-writer path is
not implemented yet.

Completed:

- extracted rolling traversal chunk schedule construction into
  `build_rolling_decode_schedule(...)`
- added explicit schedule types:
  - `gpu_codec::RollingDecodeChunk`
  - `gpu_codec::RollingDecodeSchedule`
- moved rolling decompression toward a prepare/execute/copy structure under
  `path_decompression_gpu_rolling.{hpp,cu}`
- added reusable rolling decode state:
  - `gpu_decompression::RollingPathDecodeContext`
- added reusable rolling chunk helpers:
  - `prepare_rolling_path_decode(...)`
  - `decode_rolling_path_chunk_to_device(...)`
  - `prepare_rolling_path_host_buffer(...)`
  - `copy_rolling_path_chunk_to_host_buffer(...)`
- introduced a caller-owned rolling host chunk buffer:
  - `gpu_decompression::RollingPathHostBuffer`
- introduced a pinned rolling host staging buffer:
  - `gpu_decompression::RollingPathPinnedHostBuffer`
- added pinned-buffer helper APIs:
  - `ensure_rolling_path_pinned_host_buffer_capacity(...)`
  - `release_rolling_path_pinned_host_buffer(...)`
  - `copy_rolling_path_chunk_to_pinned_host_async(...)`
  - `wait_for_rolling_path_pinned_host_buffer(...)`
- added a rolling streaming entry point with a bounded pinned-buffer queue and
  consumer callback:
  - `stream_decompress_paths_gpu_rolling(...)`
- kept the existing `decompress_paths_gpu_rolling(...)` behavior intact by
  rebuilding it on top of the new context/chunk APIs

Not done yet:

- GPU direct-writer CLI entry point
- streaming `P`/`W` formatting from flat traversal chunks

Immediate next step:

- connect the rolling streaming pipeline to actual `P`/`W` line formatting and
  a GPU direct-writer entry point

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

After the recent GPU decompression refactor, ownership is cleaner but the CLI
still ends in full host materialization:

- [`src/gpu/decompression_workflow_gpu.cu`](/home/kurty/Release/gfa_compression/src/gpu/decompression_workflow_gpu.cu)
  owns top-level metadata decode and legacy-vs-rolling dispatch
- [`src/gpu/path_decompression_gpu_legacy.cu`](/home/kurty/Release/gfa_compression/src/gpu/path_decompression_gpu_legacy.cu)
  owns full-device traversal expansion
- [`src/gpu/path_decompression_gpu_rolling.cu`](/home/kurty/Release/gfa_compression/src/gpu/path_decompression_gpu_rolling.cu)
  owns rolling traversal decompression scheduling
- [`src/gpu/path_expand_gpu.cu`](/home/kurty/Release/gfa_compression/src/gpu/path_expand_gpu.cu)
  owns rule expansion helpers and
  `rolling_expand_and_inverse_delta_decode(...)`

The current rolling traversal decompressor in
[`src/gpu/path_expand_gpu.cu`](/home/kurty/Release/gfa_compression/src/gpu/path_expand_gpu.cu)
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

### 1. Split traversal decompression from host-output assembly

This part is now mostly done for the rolling traversal path:

- `decompression_workflow_gpu.cu` now only orchestrates metadata decode and
  dispatch
- `path_decompression_gpu_rolling.cu` is now the rolling entry point
- `path_expand_gpu.cu` contains the reusable rolling expansion helper
- `path_expand_gpu.cu` now exposes `build_rolling_decode_schedule(...)`
- `path_decompression_gpu_rolling.cu` now owns:
  - rolling decode preparation
  - per-chunk device execution
  - per-chunk host copy for the compatibility path

Target:

- keep `decompress_paths_gpu(...)` for compatibility
- add a rolling traversal streaming API used by the writer path

Recommended new API shape:

- `build_rolling_decode_schedule(...)`
- `decode_traversal_chunk_to_device(...)`
- `copy_decoded_chunk_to_pinned_host_async(...)`

These should be internal GPU decompression helpers layered under
`path_decompression_gpu_rolling.cu`, not new top-level workflow logic.

Current code status:

- `build_rolling_decode_schedule(...)`: done
- `decode_traversal_chunk_to_device(...)`: effectively done as
  `decode_rolling_path_chunk_to_device(...)`
- caller-owned host chunk export: done via
  `prepare_rolling_path_host_buffer(...)` and
  `copy_rolling_path_chunk_to_host_buffer(...)`
- `copy_decoded_chunk_to_pinned_host_async(...)`: done as
  `copy_rolling_path_chunk_to_pinned_host_async(...)`
- bounded producer/consumer pipeline: done inside
  `stream_decompress_paths_gpu_rolling(...)`

### 2. Add a traversal chunk schedule

Each scheduled chunk should contain:

- `segment_begin`
- `segment_end`
- `encoded_begin`
- `encoded_end`
- `expanded_begin`
- `expanded_end`

This information used to be derived only inside
[`rolling_expand_and_inverse_delta_decode(...)`](/home/kurty/Release/gfa_compression/src/gpu/path_expand_gpu.cu).

Refactor that function so schedule creation is exposed separately from:

- host output vector sizing
- per-chunk synchronous `thrust::copy(...)`
- the compatibility wrapper that still returns full `FlattenedPaths`

Status: done.

The chunk schedule is now an explicit reusable object built before the rolling
loop. The compatibility path still uses synchronous `thrust::copy(...)`, but it
does so by consuming the precomputed schedule instead of recomputing chunk
boundaries inline.

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

Status: partially done.

The rolling path now supports caller-owned host chunk buffers carrying
per-chunk node counts and traversal lengths, and it now has pinned staging
buffer helpers with async copy and CUDA event completion. The compatibility
path still uses ordinary host memory and synchronous `thrust::copy(...)`.

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

Status: done for the rolling decompression module.

The rolling path now has a bounded pinned-buffer queue, a writer-consumer
thread, CUDA event handoff, and a streaming callback entry point. What remains
is connecting that callback to actual GFA `P/W` formatting and the CLI writer
path.

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

This separation is now easier than before because traversal decompression and
metadata decode are no longer interleaved in one source file:

- traversal decode path lives under `path_decompression_gpu_{legacy,rolling}.cu`
- metadata decode remains in `decompression_workflow_gpu.cu`

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

Status: not started.

The current writer still expects whole-graph or CPU direct-writer style inputs.
No GPU chunk-streaming writer entry point exists yet.

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

Current progress against that list:

- full `FlattenedPaths` materialization: not removed yet
- full `GfaGraph_gpu` traversal ownership: not removed yet
- full `convert_from_gpu_layout(...)`: not removed yet
- nested path/walk vector reconstruction: not removed yet
- post-decompression `write_gfa(...)` on a full host graph: not removed yet

What has been removed already is duplicated chunk-planning logic inside the
rolling decompressor. The hot path is now structured so these higher-level
removals can be done without another large refactor first.

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

## Current Module Boundaries

The direct-writer work should be built on the current refactored GPU
decompression layout, not the older pre-refactor ownership:

- [`src/gpu/decompression_workflow_gpu.cu`](/home/kurty/Release/gfa_compression/src/gpu/decompression_workflow_gpu.cu)
  - deserialize/decode metadata blocks
  - resolve debug and options
  - dispatch path decompression mode
- [`src/gpu/path_decompression_gpu_legacy.cu`](/home/kurty/Release/gfa_compression/src/gpu/path_decompression_gpu_legacy.cu)
  - legacy whole-device path expansion and inverse delta decode
- [`src/gpu/path_decompression_gpu_rolling.cu`](/home/kurty/Release/gfa_compression/src/gpu/path_decompression_gpu_rolling.cu)
  - rolling path decompression entry point
  - current place to add a streaming-oriented rolling API
- [`src/gpu/path_expand_gpu.cu`](/home/kurty/Release/gfa_compression/src/gpu/path_expand_gpu.cu)
  - rule expansion primitives
  - `expand_and_inverse_decode_chunk_device(...)`
  - `rolling_expand_and_inverse_delta_decode(...)`

For the direct writer, the main architectural change should happen under the
rolling path modules, while keeping `decompress_to_gpu_layout(...)` intact for
tests and Python bindings.

## Implementation Phases

### Phase 0: Refactor prerequisites

This is now complete:

- rolling decompression is extracted from `decompression_workflow_gpu.cu`
- legacy decompression is extracted from `decompression_workflow_gpu.cu`
- rolling expansion helpers are extracted from `codec_gpu.cu`

That means the direct-writer work can target the rolling decompression path
without further untangling the legacy path first.

### Phase 1: Isolate rolling decode schedule

Refactor
[`rolling_expand_and_inverse_delta_decode(...)`](/home/kurty/Release/gfa_compression/src/gpu/path_expand_gpu.cu)
into:

- schedule construction
- per-chunk decode helper
- existing full-host-output wrapper

Goal:

- preserve current behavior
- make chunk scheduling reusable
- keep `decompress_paths_gpu_rolling(...)` as a thin compatibility wrapper

### Phase 2: Add pinned-buffer chunk export

Add a helper that:

- decodes one scheduled chunk into device workspace
- asynchronously copies the decoded chunk into pinned host memory

Goal:

- replace synchronous `thrust::copy(...)` of chunk output
- keep the existing host-vector path available for tests and bindings

Recommended ownership:

- `path_expand_gpu.cu`: device decode into chunk workspace
- new rolling direct-writer helper near
  [`src/gpu/path_decompression_gpu_rolling.cu`](/home/kurty/Release/gfa_compression/src/gpu/path_decompression_gpu_rolling.cu):
  pinned-host export, events, and scheduling
- avoid pushing writer-thread concerns into `path_expand_gpu.cu`

Progress:

- done: decoded chunks can be exported into caller-owned host buffers without
  depending on one monolithic output vector API
- done: pinned staging buffers and async D2H completion events are now
  available under the rolling decompression module
- done: a producer/consumer streaming pipeline now exists under the rolling
  decompression module
- next: connect the streaming callback to GFA `P/W` formatting and the GPU
  direct-writer entry point

### Phase 3: Build direct GPU writer

Add `write_gfa_from_compressed_data_gpu(...)` that:

- decodes metadata
- writes `H/S/L`
- streams `P/W`
- writes `J/C`

Goal:

- produce correct output without materializing full graph objects

Recommended call shape:

- deserialize compressed GPU data
- decode non-traversal metadata once
- stream rolling traversal chunks to the writer
- never build full host `FlattenedPaths::data`

### Phase 4: Wire CLI to direct writer

Change the GPU decompression branch in
[`src/gfaz_cli.cpp`](/home/kurty/Release/gfa_compression/src/gfaz_cli.cpp#L516)
to use:

- `deserialize_compressed_data_gpu(...)`
- `write_gfa_from_compressed_data_gpu(...)`

Goal:

- make CLI benefit from the new streaming path immediately
- preserve the current in-memory decompression path behind existing APIs

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

After the refactor, the codebase is in a better position to do this cleanly:

- workflow dispatch is isolated
- rolling decompression is isolated
- legacy decompression is isolated
- expansion helpers are isolated

But the current rolling GPU decompression path is still:

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
