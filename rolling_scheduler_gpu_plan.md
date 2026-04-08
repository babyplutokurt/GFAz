# GPU Rolling Scheduler Refactor Plan

## Goal

Refactor the GPU rolling scheduler so that only the per-chunk traversal payload
rolls through host memory. All round-global compression state should stay on the
GPU:

- per-chunk 2-mer histograms
- merged round histogram
- filtered round rules
- rule table
- round-global `rules_used`
- rule compaction metadata
- rule reorder metadata
- accumulated rules across rounds

The objective is to remove the current host-side histogram merge and most of the
per-chunk device-to-host control traffic, while preserving the rolling behavior
for traversal data that does not fit entirely on device.

## Current Bottlenecks

The current rolling path in
[`src/gpu/compression_workflow_gpu.cu`](/home/kurty/Release/gfa_compression/src/gpu/compression_workflow_gpu.cu)
is slow because it is host-driven in the hot path:

- Each chunk histogram is copied back to host and merged with
  `std::unordered_map`.
- Each chunk is uploaded to the GPU more than once per round.
- Each chunk result is copied back to host immediately after rule application.
- `rules_used` is copied back per chunk and OR-ed on the CPU.
- Rule compaction and rule sorting/remapping are performed on host-side data.
- There is no overlap of chunk upload, kernel work, and download.

In contrast, the full-device path keeps the traversal and all round-global state
on the GPU across rounds.

## Target Architecture

The rolling scheduler should become a host-scheduled, GPU-owned round pipeline.

For each compression round:

1. Host iterates chunks only to move traversal payload in and out.
2. GPU computes per-chunk 2-mer histograms.
3. GPU appends per-chunk histogram outputs into round-global device buffers.
4. GPU merges the full round histogram with a device-side sort/reduce pass.
5. GPU filters keys with `count >= 2` to produce `d_round_rules`.
6. GPU builds the rule lookup table directly from `d_round_rules`.
7. Host iterates chunks again for rule application only.
8. GPU accumulates one round-global `d_rules_used`.
9. GPU compacts and sorts rules, and builds remap metadata.
10. Host iterates chunks again only if a streamed remap pass is required.
11. Host stores the transformed chunk payload back into the rolling host buffers.

This keeps only the traversal payload and updated chunk lengths rolling through
host memory.

## Non-Goals

- Replacing the rolling scheduler with a fully device-resident path.
- Reworking the compression format.
- Introducing a persistent GPU-resident global counting hash table as the first
  step.

## Recommended Merge Strategy

Do not start with a GPU counting hash table.

Use the existing sort/reduce pattern already present in the codebase:

1. Compute per-chunk `(key, count)` pairs on the GPU.
2. Append them into round-global device buffers.
3. Run one device-side `sort_by_key`.
4. Run one device-side `reduce_by_key`.
5. Filter merged counts by threshold.

This approach is simpler, deterministic, and reuses the existing primitives in
`codec_gpu.cu`.

## Refactor Scope

### Keep on GPU for the whole round

- `d_round_hist_keys`
- `d_round_hist_counts`
- merged histogram outputs
- `d_round_rules`
- rule table
- round-global `d_rules_used`
- compaction prefix sums / new-index maps
- reorder maps
- `d_all_rules`

### Continue rolling through host

- chunk traversal payload
- chunk lengths after rule application
- host-side storage for `current_data`
- host-side storage for `current_lengths`

## Proposed New GPU Helpers

Add new interfaces in
[`include/gpu/codec_gpu.cuh`](/home/kurty/Release/gfa_compression/include/gpu/codec_gpu.cuh)
and implementations in
[`src/gpu/codec_gpu.cu`](/home/kurty/Release/gfa_compression/src/gpu/codec_gpu.cu).

### 1. Round histogram append helper

`append_counted_2mers_segmented_device_vec(...)`

Purpose:

- Compute one chunk's segmented histogram.
- Append the chunk's `d_unique_keys` and `d_counts` into round-global device
  buffers at a caller-managed offset.

Alternative:

- Keep `count_2mers_segmented_device_vec(...)` unchanged and do the append logic
  directly in the rolling workflow. This may be simpler for the first pass.

### 2. Round histogram merge helper

`merge_counted_2mers_device_vec(...)`

Purpose:

- Merge concatenated per-chunk histogram outputs entirely on GPU.

Expected implementation:

- `thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_counts.begin())`
- `thrust::reduce_by_key(...)`

Inputs:

- concatenated `d_all_keys`
- concatenated `d_all_counts`

Outputs:

- merged `d_unique_keys`
- merged `d_total_counts`

### 3. Rule filter helper

`filter_rules_by_count_device_vec(...)`

Purpose:

- Filter merged histogram results by threshold, normally `count >= 2`.

Inputs:

- `d_unique_keys`
- `d_total_counts`

Output:

- `d_round_rules`

### 4. Device OR-reduction for rule usage

`or_reduce_rules_used_device_vec(...)`

Purpose:

- OR a chunk-local `d_rules_used_chunk` into one round-global
  `d_rules_used_round`.

Expected implementation:

- a simple kernel or `thrust::transform` with logical OR

### 5. Rule compaction map builder

`build_rule_compaction_map_device_vec(...)`

Purpose:

- Build the exclusive-scan map for compacted rule IDs from
  `d_rules_used_round`.

Outputs:

- `d_new_indices`
- compacted rule count

### 6. Rule reorder map builder

`build_rule_reorder_map_device_vec(...)`

Purpose:

- Build `old_index -> new_index` mapping after sorting compacted rules.

Outputs:

- `d_reorder_map`

### 7. Chunk remap helper

`remap_chunk_rule_ids_device_vec(...)`

Purpose:

- Apply compaction/reorder remap to a chunk's encoded traversal on GPU before
  copying it back to host.

This is needed if remapping is no longer done over the full host traversal.

## Existing Code To Reuse

The current code already contains most of the required building blocks.

### Histogram generation

[`count_2mers_segmented_device_vec`](/home/kurty/Release/gfa_compression/src/gpu/codec_gpu.cu#L626)

### Device rule table construction

[`create_rule_table_gpu_from_device`](/home/kurty/Release/gfa_compression/src/gpu/codec_gpu.cu#L749)

### Device rule compaction/remap logic

[`compact_rules_and_remap_device_vec`](/home/kurty/Release/gfa_compression/src/gpu/codec_gpu.cu#L1488)

### Device rule sorting/remap logic

[`sort_rules_and_remap_device_vec`](/home/kurty/Release/gfa_compression/src/gpu/codec_gpu.cu#L1635)

### Rolling scheduler entry point

[`run_path_compression_gpu_rolling`](/home/kurty/Release/gfa_compression/src/gpu/compression_workflow_gpu.cu#L622)

## Detailed Workflow Changes

## Phase 1: Remove host histogram merge

Current behavior:

- For each chunk, histogram keys and counts are copied to host.
- Host merges them into `std::unordered_map<uint64_t, uint64_t>`.

Planned change:

- For each chunk, keep histogram output on device.
- Append chunk outputs into round-global device buffers.
- After all chunks are processed, perform one device-side merge.

Expected impact:

- Removes the largest current control-path overhead.
- Eliminates repeated D2H copies for chunk histograms.
- Eliminates CPU hash-map merge cost.

## Phase 2: Build round rules entirely on GPU

Current behavior:

- Round rules are materialized on the host after the host histogram merge.
- Rule table is built with `create_rule_table_gpu(...)`, which copies rules to
  device.

Planned change:

- Produce `d_round_rules` on device after the merged histogram filter.
- Build the rule table with `create_rule_table_gpu_from_device(...)`.

Expected impact:

- Removes another host round-trip for rule materialization.

## Phase 3: Keep `rules_used` round-global on GPU

Current behavior:

- Each chunk creates `d_rules_used_dev`.
- That bitmap is copied back to host and OR-ed into a host vector.

Planned change:

- Allocate one round-global `d_rules_used_round`.
- For each chunk, OR the chunk-local bitmap directly into the round-global
  device bitmap.
- Copy it to host only if required for debugging.

Expected impact:

- Removes per-chunk `rules_used` D2H traffic.

## Phase 4: Keep compaction and sorting metadata on GPU

Current behavior:

- Rule compaction and sorting in the rolling path are performed over host-side
  `current_data` and host-side rule vectors.

Planned change:

- Compact `d_round_rules` on device using `d_rules_used_round`.
- Sort compacted rules on device.
- Build the compaction and reorder maps on device.

Expected impact:

- Removes more host control work.
- Aligns rolling behavior more closely with the full-device path.

## Phase 5: Add a streamed chunk remap pass if needed

Current behavior:

- Host-side `current_data` is remapped after compaction and sort.

Planned change:

- If compaction/sort changes rule IDs after rule application, run one streamed
  GPU remap pass over chunks before copying them back to host for the next
  round.

Tradeoff:

- This adds one more chunk pass per round.
- It is still preferable to remapping the entire traversal on the CPU.

## Implementation Order

Recommended order of work:

1. Device-side round histogram merge.
2. Device-side threshold filtering into `d_round_rules`.
3. Build the rule table from device rules.
4. Device-side round-global `rules_used` accumulation.
5. Device-side compaction and sorting metadata.
6. Optional streamed chunk remap pass.
7. Optional overlap/streaming improvements.

This ordering gives the best performance return early while keeping the
refactor incremental.

## Pseudocode For New Rolling Round

```cpp
for each round:
  rebuild chunks from current_lengths

  clear d_round_hist_keys
  clear d_round_hist_counts

  // Pass 1: histogram only
  for each chunk:
    upload chunk data + chunk lengths
    build chunk offsets and boundary masks
    count segmented 2-mers on GPU
    append chunk (keys, counts) into round-global device buffers

  merge round-global histogram on GPU
  filter merged counts by threshold to produce d_round_rules
  if d_round_rules.empty():
    break

  build rule table from d_round_rules on GPU
  allocate d_rules_used_round = 0

  clear next_data
  clear next_lengths

  // Pass 2: rule application
  for each chunk:
    upload chunk data + chunk lengths
    build chunk offsets
    apply rules on GPU
    OR chunk rules_used into d_rules_used_round
    copy transformed chunk payload back to host next_data
    copy transformed chunk lengths back to host next_lengths

  compact rules on GPU using d_rules_used_round
  sort compacted rules on GPU
  build remap metadata on GPU

  // Optional Pass 3: streamed remap
  if remap is required after compaction/sort:
    for each chunk in next_data:
      upload chunk
      remap rule IDs on GPU
      copy remapped chunk back to host

  append compacted/sorted rules into d_all_rules
  current_data = next_data
  current_lengths = next_lengths
```

## Risks

### Device memory growth for round histogram buffers

Concatenating per-chunk histogram outputs for a full round may consume
substantial GPU memory when the number of unique chunk-local 2-mers is large.

Mitigations:

- reserve using a heuristic based on chunk node count
- fall back to staged merge if the round-global buffers become too large
- optionally merge every N chunks instead of only once at the end of the round

### Additional remap pass

If remapping after compaction/sort cannot be fused into the apply pass, an extra
streamed remap pass will be required.

This still keeps the expensive logic on GPU, but it adds another traversal over
chunk payload.

### Synchronization overhead

If implemented with many small synchronous `thrust` operations, the code may
still underperform.

Mitigations:

- reuse buffers
- minimize reallocations
- avoid unnecessary host synchronization
- move to explicit streams later if needed

## Validation Plan

### Correctness

- Compare compressed output from the rolling path against the existing rolling
  implementation on the same input.
- Verify decompression round-trip for the resulting `.gfaz_gpu`.
- Compare `layer_ranges`, total rule count, and reconstructed traversal output.
- Test both path-only and mixed path/walk inputs.

### Performance

Measure and report separately:

- chunk histogram time
- round histogram merge time
- rule table build time
- chunk rule-application time
- chunk remap time, if present
- final nvComp compression time

Compare against:

- current rolling scheduler
- full-device path on inputs that fit on device

## Success Criteria

The refactor is successful if:

- the rolling scheduler no longer copies per-chunk histograms to host
- the rolling scheduler no longer merges chunk histograms on the CPU
- round rules are created directly on device
- `rules_used` is accumulated on device for the full round
- host orchestration is limited to chunk scheduling and rolling traversal
  storage
- rolling-path performance improves substantially without changing output
  semantics
