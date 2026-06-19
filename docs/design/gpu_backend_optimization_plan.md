# GPU Backend Optimization Plan

## Goal

Improve GPU backend correctness, memory behavior, and end-to-end performance, with priority on decompression first. The current GPU path has too many host/device transfers, too much temporary allocation churn, and several places where GPU execution does not yet deliver a meaningful advantage over the CPU backend.

This plan is intended to be updated as work lands.

## Current State

- GPU rule expansion correctness issue was addressed by replacing the old fixed-stack rule materialization path with host-side bottom-up rule expansion followed by upload to device.
- The GPU backend is still functionally correct on the current regression suite.
- The GPU backend still has architectural inefficiencies:
  - repeated H2D/D2H transfers
  - frequent `thrust::device_vector` / `std::vector` staging
  - poor workspace reuse
  - large temporary allocations
  - host-heavy preprocessing in decompression

## Guiding Principles

- Correctness before speed.
- Measure before optimizing.
- Prefer reusable workspaces over per-call allocation.
- Prefer streaming/chunked execution over full materialization when possible.
- Avoid host/device round-trips unless they are clearly required.
- Keep the `.gfaz` container and CPU/GPU compatibility unchanged.

## Phase 1: Instrumentation and Baseline

### Objective

Establish a stage-by-stage performance and memory baseline for GPU decompression so later changes can be evaluated quantitatively.

### Tasks

- [ ] Add end-to-end timing breakdown for GPU decompression.
- [ ] Record byte counts for each H2D and D2H transfer.
- [ ] Record host-side preprocessing time for:
  - rule decode
  - traversal block decode
  - metadata decode
- [ ] Record GPU kernel time for:
  - rule/traversal expansion
  - inverse delta
  - rolling chunk decode
- [ ] Record temporary allocation sizes and counts where feasible.
- [ ] Record peak host RSS and best-effort GPU memory snapshots during major stages.

### Candidate Files

- [`src/gpu/decompression/traversal_decode_gpu.cu`](/home/kurty/Release/gfa_compression/src/gpu/decompression/traversal_decode_gpu.cu)
- [`src/gpu/decompression/decompression_primitives_gpu.cu`](/home/kurty/Release/gfa_compression/src/gpu/decompression/decompression_primitives_gpu.cu)
- [`src/gpu/decompression/path_expand_gpu.cu`](/home/kurty/Release/gfa_compression/src/gpu/decompression/path_expand_gpu.cu)
- [`src/gpu/io/gfa_writer_gpu.cu`](/home/kurty/Release/gfa_compression/src/gpu/io/gfa_writer_gpu.cu)

### Deliverables

- A debug/profiling output mode that reports stage times and transfer sizes.
- A baseline measurement on `example.gfa` and at least one larger real dataset.

### Exit Criteria

- We can identify the dominant costs in GPU decompression without guesswork.

## Phase 2: Reusable GPU Decompression Workspace

### Objective

Reduce allocation churn and improve memory locality by reusing buffers across GPU decompression operations.

### Tasks

- [ ] Introduce a `GpuDecompressionWorkspace` object.
- [ ] Reuse buffers for:
  - encoded traversal data
  - final lengths
  - rule sizes
  - rule offsets
  - expanded rule table
  - output offsets
  - rolling chunk scratch/output buffers
- [ ] Change helper APIs so they accept workspace storage instead of always allocating fresh vectors.
- [ ] Ensure buffers grow only when required and are otherwise reused.

### Candidate Files

- [`include/gpu/decompression/decompression_primitives_gpu.hpp`](/home/kurty/Release/gfa_compression/include/gpu/decompression/decompression_primitives_gpu.hpp)
- [`src/gpu/decompression/decompression_primitives_gpu.cu`](/home/kurty/Release/gfa_compression/src/gpu/decompression/decompression_primitives_gpu.cu)
- [`src/gpu/decompression/traversal_decode_gpu.cu`](/home/kurty/Release/gfa_compression/src/gpu/decompression/traversal_decode_gpu.cu)
- [`src/gpu/decompression/path_expand_gpu.cu`](/home/kurty/Release/gfa_compression/src/gpu/decompression/path_expand_gpu.cu)

### Deliverables

- Shared workspace object threaded through the main GPU decompression paths.
- Reduced allocation count in profiling logs.

### Exit Criteria

- Repeated decompression calls no longer allocate most major buffers from scratch.

## Phase 3: Reduce Host/Device Transfers

### Objective

Eliminate unnecessary staging and keep decompression data resident on device for longer.

### Tasks

- [ ] Audit all H2D/D2H boundaries in GPU decompression.
- [ ] Remove redundant copies between `std::vector` and `thrust::device_vector`.
- [ ] Keep prepared rule data on device once uploaded.
- [ ] Keep traversal decode fully device-resident until final output or direct-write emission.
- [ ] Minimize host-side materialization in the rolling path.

### Specific Questions

- Can encoded traversal payload be uploaded once and then decoded entirely on device?
- Can the direct-writer path emit chunks without full host graph materialization?
- Can prepared rulebook state be reused across multiple decode operations on the same file?

### Deliverables

- Reduced total transfer volume.
- Reduced number of copy steps in the GPU decompression path.

### Exit Criteria

- Profiling shows a measurable drop in H2D/D2H time and bytes moved.

## Phase 4: Make Rolling Direct-Writer the Fast Path

### Objective

Optimize the GPU backend around the path that avoids full host-side traversal materialization.

### Tasks

- [ ] Treat rolling direct-writer as the primary GPU decompression fast path.
- [ ] Audit and reduce intermediate buffering in the direct-writer pipeline.
- [ ] Reuse rolling chunk buffers between chunks.
- [ ] Verify that chunk boundaries do not force unnecessary host staging.
- [ ] Add focused benchmarks comparing:
  - CPU direct writer
  - GPU rolling materialized
  - GPU rolling direct writer

### Candidate Files

- [`src/gpu/io/gfa_writer_gpu.cu`](/home/kurty/Release/gfa_compression/src/gpu/io/gfa_writer_gpu.cu)
- [`src/gpu/decompression/traversal_decode_gpu.cu`](/home/kurty/Release/gfa_compression/src/gpu/decompression/traversal_decode_gpu.cu)
- [`src/gpu/decompression/path_decompression_gpu_rolling.cu`](/home/kurty/Release/gfa_compression/src/gpu/decompression/path_decompression_gpu_rolling.cu)

### Deliverables

- A benchmark-backed GPU direct-writer path that outperforms or meaningfully competes with CPU direct writer on intended workloads.

### Exit Criteria

- GPU direct-writer is the preferred decompression mode for large inputs.

## Phase 5: Revisit Device-Side Rule Preparation

### Objective

Replace the temporary host-side bottom-up rule preparation with a GPU-native implementation, but only after transfer and allocation issues are under control.

### Rationale

The current host-side rule expansion fixed a correctness bug and simplified the pipeline, but it is not the final architecture. A fully GPU-native rule preparation stage should be designed around bottom-up layered expansion, not a bounded DFS stack.

### Tasks

- [ ] Design a device-side bottom-up rule preparation algorithm using `layer_rule_ranges`.
- [ ] Compute per-rule final size layer by layer.
- [ ] Compute offsets from those sizes.
- [ ] Materialize expanded rule spans layer by layer.
- [ ] Validate orientation handling and reversed-copy correctness.
- [ ] Compare against the current host-side rule preparation path.

### Non-Goals

- Do not restore the old fixed-size stack-based GPU rule expansion.
- Do not accept silent truncation or silent fallback behavior.

### Deliverables

- A GPU-native rule preparation path with explicit correctness guarantees.

### Exit Criteria

- Device-side rule preparation is correct, measurably faster or lower-overhead, and does not reintroduce bounded-stack correctness risks.

## Phase 6: GPU Compression Follow-Up

### Objective

Apply the same measurement and workspace discipline to the GPU compression backend once decompression is under control.

### Tasks

- [ ] Add stage-level timing and transfer accounting for GPU compression.
- [ ] Reuse compression-side workspaces and staging buffers.
- [ ] Reduce host/device staging in rule generation/remap paths.
- [ ] Reevaluate metadata compression flow for avoidable host work.

## Benchmarks and Validation

For each phase, run at minimum:

- [ ] `python tests/regression/test_example_regression.py example.gfa`
- [ ] a larger GPU-capable decompression benchmark
- [ ] before/after timing comparison for:
  - total wall time
  - GPU kernel time
  - H2D bytes and time
  - D2H bytes and time
  - peak GPU memory
  - peak host memory

When a performance change is claimed, record:

- dataset name
- command used
- GPU model
- CUDA version
- build configuration

## Risks

- Host-side optimizations may hide deeper device-side bottlenecks.
- Aggressive buffer reuse can introduce lifetime and aliasing bugs.
- Device-side rule preparation may increase complexity substantially if layered assumptions are not enforced clearly.
- Improvements on small fixtures may not reflect behavior on large production graphs.

## Immediate Next Step

Implement Phase 1 first:

- add profiling counters and timing breakdowns for GPU decompression
- identify the top 2 to 3 transfer/allocation bottlenecks
- use that data to decide Phase 2 scope precisely
