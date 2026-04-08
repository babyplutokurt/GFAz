# Rolling GPU Decompression Plan

## Status

Done:

- extracted rolling decode scheduling into `build_rolling_decode_schedule(...)`
- added rolling decode state and chunk helpers in
  `path_decompression_gpu_rolling.{hpp,cu}`
- added caller-owned host chunk buffers
- added pinned host buffers, async D2H copy helpers, and CUDA events
- added `stream_decompress_paths_gpu_rolling(...)`
- added `write_gfa_from_compressed_data_gpu(...)`
- switched GPU CLI decompression to the direct-writer path
- fixed streamed-writer correctness by preventing reuse of the shared device
  workspace before the prior D2H copy completes
- split GPU tests into:
  - `tests/gpu/test_gpu_legacy_roundtrip.py`
  - `tests/gpu/test_gpu_streaming_roundtrip.py`

Current behavior:

- CLI rolling decompression no longer shows a meaningful regression vs legacy
- in-memory rolling decompression (`decompress_to_gpu_layout(...)`) is still
  much slower than legacy
- profiling shows the in-memory regression is dominated by chunked D2H copies,
  not decode kernels

## Current Findings

- rolling CLI path is now effectively on par with legacy for end-to-end GFA
  writing
- rolling in-memory decompression is still slow because it materializes the full
  traversal in host memory
- debug timings show the main cost is `chunk device->host copies`, not:
  - nvcomp inflate
  - rule decode
  - rolling decode kernels
  - schedule construction

## Completed Architecture

- CLI GPU decompression path:
  - deserialize compressed GPU payload
  - decode metadata
  - stream `P/W` chunks through rolling decompression
  - write `H/S/L/P/W/J/C` directly to output GFA

- compatibility in-memory path remains:
  - `decompress_to_gpu_layout(...)`
  - `decompress_paths_gpu_rolling(...)`

## Remaining Work

### 1. Optimize in-memory rolling decompression

Target:

- reduce the cost of `decompress_to_gpu_layout(...)` for large graphs

Next steps:

- replace pageable host chunk copies with pinned destination memory for the
  compatibility path
- use `cudaHostRegister(...)` or equivalent pinned output ownership for
  `out_data`
- replace per-chunk `thrust::copy(...)` with `cudaMemcpyAsync(...)`
- measure again with the existing decompression timing breakdown

### 2. Restore safe overlap in the streaming path

Current correctness fix serializes workspace reuse.

Next steps:

- give each in-flight host buffer its own device chunk workspace
- remove the forced copy-stream synchronization between chunks
- allow decode and D2H copy to overlap safely again

### 3. Add CLI-side timing breakdown

Add timing around:

- metadata decode
- traversal streaming
- line formatting
- file output

Goal:

- confirm whether output formatting or disk writing is the dominant remaining
  CLI cost

### 4. Keep regression coverage aligned with behavior

Maintain:

- legacy in-memory GPU round-trip test for `GfaGraph_gpu` comparison
- rolling streaming GPU round-trip test for direct-writer correctness

Future additions:

- optional perf-focused scripts for:
  - legacy in-memory decompression
  - rolling in-memory decompression
  - rolling CLI streaming decompression

## Success Criteria

- CLI rolling path stays correct and remains at least comparable to legacy
- in-memory rolling decompression moves closer to legacy by removing D2H copy
  overhead
- streaming path regains overlap without reintroducing corruption
