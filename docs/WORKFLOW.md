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

Main implementation: `src/compress/compression_workflow.cpp`

1. Parse input GFA into `GfaGraph`, whose segment and path state is grouped as `SegmentData` and `PathData`.
2. Flatten path and walk traversals for compression-oriented processing.
3. Apply delta transform (`delta_round` times) and track max absolute IDs.
4. Run 2-mer rule generation and iterative remap for `num_rounds`.
5. Record per-layer rule ranges.
6. Delta-encode rule arrays and ZSTD-compress blocks.
7. Flatten and compress paths, names, overlaps, walks, segments, links, optional fields, and J/C records.
8. Return `CompressedData`.

Notes:

- CPU compression parameters: rounds, frequency threshold, delta rounds, thread count, and an optional traversal memory budget (below).
- OpenMP is used when available.
- `num_threads=0` uses auto policy (`GFAZ_NUM_THREADS` -> `OMP_NUM_THREADS` -> `min(8, logical_cpus/2)`), and this same policy applies to parser parallel sections and CPU workflow OpenMP regions.

### Parsing

Main implementation: `src/compress/io/gfa_parser.cpp`. The input is
memory-mapped and parsed in parallel; which route runs depends on the file:

- Direct two-pass parser: segment names are `1..N` in file order and optional
  fields are fixed `i`/`f`/`A` columns.
- General parser, parallel S/L lines: everything else whose S and L lines can
  be parsed in independent blocks. Segment IDs are still assigned in file
  order. When every name is a canonical decimal number (the HPRC and HGSVC3
  Minigraph-Cactus graphs, whose numbers do not start at 1 in file order),
  names resolve through a `number -> id` array instead of a hash map. Other
  names (`s1`, `s2`, ... in Minigraph graphs) are copied into one arena and
  resolved through a hash map built after the parallel pass. Either way,
  segment sequences are written straight into the flattened store. Per-block
  optional-field columns are concatenated in row order, so the result is the
  one the serial parser builds.
- General parser, serial S/L lines: inputs the block parsers hand back
  (repeated or empty segment names, sparse or non-canonical numbers, links that name
  unknown segments, a tag repeated within a row, a tag whose type changes).

All routes produce the same `GfaGraph`, so the `.gfaz` does not depend on the
route or the thread count (`tests/regression/test_parallel_parse_equivalence.py`).

### Rule generation

Main implementation: `src/compress/grammar/rule_generator.cpp`. A round's rules
are the canonical 2-mers that occur at least twice, in sorted order. Two scans
find them:

- Per-thread hash tables drop repeats inside a thread before the sort. Used
  in the early rounds, where most 2-mers repeat. Each table
  (`include/compress/grammar/kmer_seen_table.hpp`) costs one probe per 2-mer
  and keeps its "seen twice" mark inside the key, which needs node IDs below
  2^30; larger IDs fall back to two `robin_hood` sets per thread.
- Later rounds, where most 2-mers are distinct, write every 2-mer to the sort
  buffer and skip the sets. The switch uses the previous round's key count.

From the second round on, both scans and the encoder only look at 2-mers that
contain a rule created by the previous round. A 2-mer of two older symbols was
already adjacent before that round and was not replaced, so it occurred once
and still does. In the last rounds, where few symbols are new, this leaves
only a small share of the 2-mers to sort and look up.

Keys are sorted with a bucketed radix sort
(`src/compress/grammar/key_sort.cpp`: sampled splitters, then a byte-wise radix
sort per bucket). Both scans give the same rules
(`tests/regression/test_rule_scan_equivalence.py`).

### Test and debugging switches

These environment variables pin a code path. They do not change the output and
exist for the equivalence tests and for A/B timing:

- `GFAZ_SERIAL_PARSE=1`: use the serial S/L line parsers in the general parser.
- `GFAZ_RULE_SCAN=sets|sets-legacy|all`: always use the hash-table scan, its
  two-set fallback, or always sort every 2-mer.
- `GFAZ_NEW_SYMBOL_FILTER=0`: look at every 2-mer in every round instead of
  only those that contain a rule from the previous round.
- `GFAZ_FUSED_SCAN=0|1`: never, or in every round, collect a round's 2-mers
  while the previous round is encoded (default: only in rounds with spilled
  traversals).
- `GFAZ_SPILL_CODEC=raw|varint`: store every spilled traversal as plain `int32`
  or as varints instead of choosing per record.
- `GFAZ_SPILL_DROP_CACHE=1`: evict spill-file pages from the page cache as they
  are written and read.

### Lower-memory compression (`--memory-budget`)

By default every path/walk traversal is held in memory (4 bytes per step) until
the grammar rounds finish. `gfaz compress --memory-budget <SIZE>` (API:
`compress_gfa_with_options` with `CompressOptions::traversal_memory_budget_bytes`)
caps that:

- The parser hands each traversal to a `TraversalSpill`
  (`include/compress/grammar/traversal_spill.hpp`) through the `TraversalSink`
  interface instead of storing it in `GfaGraph`. The spill delta-transforms it
  and keeps it in memory while the budget allows; the overflow is written,
  varint-encoded, to an unlinked temp file in `--tmp-dir`.
- During a grammar round with spilled traversals, the rule scan and the encode
  pass load those traversals one at a time from the temp file (dynamic
  scheduling, one per worker), so resident traversal data is bounded by the
  budget plus `threads x largest traversal`. Each encode pass rewrites the
  still-spilled traversals into a second temp file and moves back into memory
  whatever now fits. After a round the steps are rule IDs, which are large
  numbers but few distinct ones in the early rounds, so rewritten records
  code each step as its rank in a per-thread table of frequent symbols (1 or
  2 bytes per step) and fall back to varints or plain `int32` when that does
  not pay. The encode pass also collects the next round's 2-mers from each
  traversal it has just encoded, so that round's rule scan does not read the
  temp file again. On HPRC v2.1 the table coding cuts the bytes written to
  the temp file from 211 GB to 82 GB. The rule-ID remap for spilled traversals is applied when
  they are next loaded rather than in a separate pass.
- Traversals roughly halve per round in the early rounds, so after a few rounds
  everything is in memory and the remaining rounds run the normal in-memory
  code. Whatever is still spilled after the last round is loaded before entropy
  coding.

Asking for a budget also switches two things that trade a little time for
memory:

- Segments and links are compressed and released right after parsing, before
  the grammar rounds, instead of being held through them and compressed
  alongside every other record group at the end.
- On glibc the malloc mmap threshold is fixed at 4 MiB, so the per-thread
  buffers used for streaming go back to the OS when freed rather than staying
  parked in each worker thread's arena.

The rules, their IDs, and the encoded traversals are the same as in the
in-memory run, so the `.gfaz` is byte-identical for any budget
(`tests/regression/test_low_memory_compress.py`). The budget itself only
covers traversal data: optional fields, rule-generation scratch, entropy-coding
buffers, and segments/links while they are being parsed are not counted
against it. Put `--tmp-dir` on fast local storage
(default: `$TMPDIR`, else the output directory); the temp file needs roughly
one byte per traversal step at its peak.

Independently of the budget, the encoder returns each traversal's unused
capacity after every round, so resident traversal memory follows the encoded
size instead of staying at the parsed size until the last round.

### Measured CPU throughput and memory

16 threads, input in the page cache, AMD Ryzen Threadripper PRO 9955WX, 512 GB
RAM, spill file and decompressed output on a Samsung 990 PRO NVMe drive.
Throughput is the GFA size over end-to-end CLI wall time. Medians of 3 runs,
single runs for HPRC v2.x. Reproduce with
`scripts/benchmark/compress_decompress.py`.

Compression, default and with `--memory-budget 2G`:

| Graph | GFA size | Time | Throughput | Peak RSS | Time (2G) | Throughput (2G) | Peak RSS (2G) |
|---|---|---|---|---|---|---|---|
| E. coli | 0.12 GB | 0.31 s | 0.38 GB/s | 0.24 GB | 0.32 s | 0.37 GB/s | 0.19 GB |
| HGSVC3-sv | 3.51 GB | 5.12 s | 0.69 GB/s | 5.92 GB | 5.16 s | 0.68 GB/s | 5.88 GB |
| HPRC chr6 | 4.06 GB | 1.88 s | 2.16 GB/s | 2.00 GB | 2.11 s | 1.93 GB/s | 2.37 GB |
| HPRC chr1 | 7.58 GB | 3.40 s | 2.23 GB/s | 4.00 GB | 3.87 s | 1.96 GB/s | 4.25 GB |
| HPRC v1.1 | 50.8 GB | 26.2 s | 1.94 GB/s | 29.9 GB | 31.5 s | 1.61 GB/s | 11.6 GB |
| HGSVC3 | 80.1 GB | 38.3 s | 2.09 GB/s | 44.1 GB | 44.7 s | 1.79 GB/s | 15.5 GB |
| HPRC v2.0 | 384.3 GB | 114.5 s | 3.36 GB/s | 156.8 GB | 135.2 s | 2.84 GB/s | 28.6 GB |
| HPRC v2.1 | 395.7 GB | 122.2 s | 3.24 GB/s | 161.4 GB | 125.0 s | 3.17 GB/s | 29.7 GB |

Decompression (default streaming mode), to `/dev/null` and to a file:

| Graph | Time (`/dev/null`) | Throughput | Peak RSS | Time (file) | Throughput | Peak RSS |
|---|---|---|---|---|---|---|
| E. coli | 0.06 s | 1.97 GB/s | 0.15 GB | 0.08 s | 1.48 GB/s | 0.17 GB |
| HGSVC3-sv | 2.66 s | 1.32 GB/s | 4.10 GB | 3.47 s | 1.01 GB/s | 4.11 GB |
| HPRC chr6 | 0.74 s | 5.49 GB/s | 1.21 GB | 1.54 s | 2.64 GB/s | 1.62 GB |
| HPRC chr1 | 1.89 s | 4.01 GB/s | 2.62 GB | 3.02 s | 2.51 GB/s | 2.99 GB |
| HPRC v1.1 | 11.0 s | 4.63 GB/s | 12.9 GB | 18.5 s | 2.74 GB/s | 13.4 GB |
| HGSVC3 | 16.4 s | 4.90 GB/s | 16.1 GB | 27.0 s | 2.97 GB/s | 17.5 GB |
| HPRC v2.0 | 45.4 s | 8.47 GB/s | 22.7 GB | 203.9 s | 1.88 GB/s | 24.2 GB |
| HPRC v2.1 | 49.8 s | 7.95 GB/s | 25.0 GB | 179.6 s | 2.20 GB/s | 27.4 GB |

Notes:

- The budget pays off on whole-genome graphs. On graphs whose traversals
  already fit in a few GB (chr1, chr6) it does not lower peak RSS, and on
  HGSVC3-sv, which has no paths or walks, it has nothing to spill.
- On the HPRC v2.x graphs the default run's memory plus the input exceed the
  node's RAM, so part of the input leaves the page cache during the run.
- Writing the ~390 GB HPRC v2.x outputs to a file is limited by the drive;
  repeated runs of the same binary ranged from 170 s to 280 s.
- These numbers are with transparent huge pages available on request (the
  kernel's `madvise` setting, or `always`). GFAz asks for huge pages on its
  large randomly accessed tables; with them unavailable, or with
  `GFAZ_NO_HUGE_PAGES=1`, compression is slower (HPRC v1.1: about 31 s
  instead of 27 s). Output bytes are the same either way.

## CPU Decompression Workflow Details

Main implementation: `src/compress/decompression_workflow.cpp`

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

- `src/compress/gpu/compression/compression_workflow_gpu.cu`
- `src/compress/gpu/decompression/decompression_workflow_gpu.cu`

Behavior:

- Uses GPU-oriented flattened structures and kernels for path/rule processing, then stores the final payload in the shared `CompressedData` Zstd-based format.
- GPU compression accelerates the transform pipeline; it does not define a different entropy layer or file format.
- GPU decompression reads the same serialized blocks as CPU decompression and reconstructs `GfaGraph_gpu` from the shared container.
- High-level public Python APIs:
  - `compress_gfa_gpu(...)`
  - `compress_gpu_graph(...)`
  - `decompress_to_gpu_layout(...)`

## Serialization Contracts

Shared serialization (`include/core/codec/serialization.hpp`, `src/core/codec/serialization.cpp`):

- Magic: `GFAZ`
- Version: `5`
- Type: `CompressedData`

GPU serialization (`include/compress/gpu/core/serialization_gpu.hpp`, `src/compress/gpu/core/serialization_gpu.cpp`):

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

- CPU default path: parse -> `compress_gfa_with_options` -> `serialize_compressed_data` (`--memory-budget`/`--tmp-dir` select the lower-memory mode)
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
- Segments are identified by dense 1-based IDs in S-line order. Their names are kept in the container (`SegmentNames`, `include/core/model/segment_names.hpp`) unless they are exactly `1..N`, and every writer prints the original name.
- Records are regrouped by type on decompression, so round-trip verification compares line sets rather than line order.
