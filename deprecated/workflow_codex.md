# GFA Compression Round-Trip Workflow (CPU, Codex Notes)

This document describes the **actual current implementation** of the CPU workflow in this repo, based on:
- `src/compression_workflow.cpp`
- `src/decompression_workflow.cpp`
- `src/codec.cpp`
- `src/rule_generator.cpp`
- `src/path_encoder.cpp`
- `src/serialization.cpp`
- `src/bindings.cpp`
- `gfa_compression_api.py`

GPU workflow is intentionally excluded.

## 1. What “round-trip” means in this project

There are two round-trip paths:

1. **In-memory round-trip**
   - `GFA file -> CompressedData -> GfaGraph`

2. **Persisted round-trip**
   - `GFA file -> CompressedData -> .gfaz file -> CompressedData -> GfaGraph`

Python entry points:
- `gfa_compression.compress(...)`
- `gfa_compression.decompress(...)`
- `gfa_compression.serialize(...)`
- `gfa_compression.deserialize(...)`
- `gfa_compression.verify_round_trip(...)`
- `gfa_compression.write_gfa(...)`

Wrapper class (`gfa_compression_api.py`): `GFACompressor`.

## 2. Core data model used by compression

Compression stores a `CompressedData` object (see `include/compression_workflow.hpp`) containing:

- Header
  - `header_line`
- Grammar rules
  - `layer_rule_ranges`
  - `rules_first_zstd`, `rules_second_zstd`
  - `delta_round`
- Paths
  - `sequence_lengths` (encoded lengths)
  - `original_path_lengths` (pre-transform lengths, used for exact decode allocation)
  - `paths_zstd`
  - names/overlaps blocks
- Walks
  - `walk_lengths` (encoded)
  - `original_walk_lengths` (pre-transform)
  - `walks_zstd` and metadata blocks
- Segments
  - `segment_sequences_zstd`, `segment_seq_lengths_zstd`
  - optional fields
- Links
  - IDs/orientations/overlap blocks
  - optional fields
- J/C lines
  - dedicated column blocks

## 3. Compression pipeline (CPU)

Function: `compress_gfa(...)` in `src/compression_workflow.cpp`.

### Step A: Parse input GFA
- `GfaParser::parse()` builds `GfaGraph`.
- `NodeId` is signed int32:
  - positive = forward
  - negative = reverse

### Step B: Snapshot original traversal lengths
Before any transform:
- store `original_path_lengths`
- store `original_walk_lengths`

These are used later to validate exact expansion length during decompression.

### Step C: Delta transform paths and walks (`delta_round` times)
- `Codec::delta_transform_and_max_abs(graph.paths)`
- `Codec::delta_transform_and_max_abs(graph.walks.walks)`

This is reverse-difference encoding per traversal.

Also tracks max absolute post-delta value to avoid rule-ID collision.

### Step D: Select rule-ID start
- Initial `next_id` is `graph.node_id_to_name.size()`.
- If delta generated larger absolute values, `next_id = max_abs + 1`.

### Step E: Multi-round 2-mer grammar compression
Function: `run_grammar_compression(...)`.

Per round:
1. Generate repeated 2-mers across **paths + walks**:
   - `RuleGenerator::generate_rules_2mer_combined(...)`
2. Apply rules greedily over each traversal:
   - `PathEncoder::encode_paths_2mer(...)` for paths then walks
3. Compact/sort/remap used rules:
   - keep only used rules
   - sort by packed 2-mer value (helps ZSTD)
   - remap rule IDs in traversals to dense sorted IDs
4. Append sorted rules to master rulebook

Exit conditions:
- no generated rules, or
- generated rules but none used

### Step F: Build layer metadata
- Adds one `LayerRuleRange`:
  - `k=2`, `start_id=layer_start`, `end_id=next_id`

### Step G: Materialize rule arrays + delta encode
- `process_rules(...)` splits packed 2-mer rules into:
  - `first[]`, `second[]`
- Delta encode both arrays:
  - `Codec::delta_encode_int32(first)`
  - `Codec::delta_encode_int32(second)`
- ZSTD compress to:
  - `rules_first_zstd`, `rules_second_zstd`

### Step H: Compress paths + metadata
- Flatten all encoded paths to one int32 array
- Save `sequence_lengths`
- Concatenate path names and overlaps with length vectors
- ZSTD compress each payload block

### Step I: Compress walks + metadata
If walks exist:
- flatten encoded walk nodes + `walk_lengths`
- compress walk metadata:
  - sample IDs, hap indices, seq IDs, seq starts/ends

### Step J: Compress segments
- Concatenate segment sequences (index 1..N)
- save segment lengths
- ZSTD compress
- compress optional field columns by type

### Step K: Compress links
- IDs: `compress_delta_varint_uint32`
- orientations: `compress_orientations` (bit-pack then ZSTD)
- overlap nums/ops: ZSTD
- compress link optional fields

### Step L: Compress J/C lines
- J lines: IDs/orientations + distance/rest strings
- C lines: IDs/orientations + positions + overlap/rest strings

### Step M: Return `CompressedData`
This is the in-memory compressed artifact used by both `decompress(...)` and `serialize(...)`.

## 4. Decompression pipeline (CPU)

Function: `decompress_gfa(const CompressedData&, GfaGraph&, ...)`.

### Step A: Decompress rule blocks and path block
- ZSTD decompress:
  - rules first/second arrays
  - flattened encoded paths

### Step B: Decode rules
- `Codec::delta_decode_int32(...)` for both arrays
- `min_rule_id = data.min_rule_id()`
- `num_rules = rules_first.size()`

### Step C: Reconstruct encoded path vectors
- `reconstruct_sequences(flattened_paths, sequence_lengths, graph.paths)`

### Step D: Expand grammar recursively
- `expand_sequences(...)`:
  - if node ID in rule range, recursively expand via `expand_rule(...)`
  - reverse orientation expansion is handled by swap+sign logic
  - reserves exact output length when `original_path_lengths` is present
  - logs mismatch if expanded length differs from original

### Step E: Inverse delta transform on paths
Apply `delta_round` times:
- `Codec::inverse_delta_transform(graph.paths)`

### Step F: Decompress path names and overlaps
Reconstruct string vectors from concatenated data + length vectors.

### Step G: Decompress walks (same path mechanics)
- `decompress_expand_sequences(...)` does:
  - ZSTD decode -> reconstruct sequences -> expand rules -> inverse delta
- then decode walk metadata blocks

### Step H: Release rules from memory
`rules_first/second` are cleared after path+walk expansion.

### Step I: Decompress segments and links
- segments:
  - decode concatenated sequence + lengths
  - rebuild `node_id_to_name`, `node_sequences`, `node_name_to_id`
- links:
  - decode IDs (delta+varint)
  - decode orientations (bit unpack)
  - decode overlap arrays

### Step J: Decompress optional fields + J/C lines
Rehydrate optional columns and jump/containment data.

### Step K: Return reconstructed `GfaGraph`

## 5. Serialization round-trip (`.gfaz`)

Functions:
- `serialize_compressed_data(...)`
- `deserialize_compressed_data(...)`

### File format markers
From `include/serialization.hpp`:
- magic: `GFAZ_MAGIC = 0x5A414647`
- version: `GFAZ_VERSION = 5`

Version 5 includes original path/walk lengths for exact allocation during decode.

### Stored order (important)
Serialize writes in fixed section order:
1. magic + version
2. header
3. rule ranges + path lengths + rule/path blocks + delta_round
4. path names/overlaps
5. segments + segment optional fields
6. links + link optional fields
7. jumps
8. containments
9. walks

Deserialize reads in exactly the same order and validates magic/version.

## 6. End-to-end API workflows

### In-memory round-trip (Python)
```python
import gfa_compression as gfac

compressed = gfac.compress(
    "input.gfa",
    num_rounds=8,
    freq_threshold=2,
    delta_round=1,
    num_threads=0,
)

graph_out = gfac.decompress(compressed, num_threads=0)
graph_in = gfac.parse("input.gfa")
ok = gfac.verify_round_trip(graph_in, graph_out)
```

### Persisted round-trip (Python)
```python
import gfa_compression as gfac

compressed = gfac.compress("input.gfa", 8, 2, 1, 0)
gfac.serialize(compressed, "input.gfa.gfaz")

loaded = gfac.deserialize("input.gfa.gfaz")
graph_out = gfac.decompress(loaded, num_threads=0)

gfac.write_gfa(graph_out, "input.gfa.decompressed")
```

### Wrapper class flow
`gfa_compression_api.GFACompressor` wraps the same sequence:
- `compress()` -> stores `self.compressed_data`
- `decompress()` -> returns `GfaGraph`
- `save()` / `load()` -> `.gfaz`
- `verify()` -> parse original and call `verify_round_trip`
- `write_gfa()` -> emit decompressed graph text

## 7. Threading and performance notes

- Parallelism is OpenMP-based (`num_threads=0` means default max threads).
- Major parallel regions:
  - rule generation
  - path/walk encoding
  - remap loops
  - many independent block compress/decompress sections
- Compression and decompression print timing breakdowns to stdout.

## 8. Behavior details and caveats (current implementation)

These matter for users expecting strict text-level GFA identity.

1. Link CIGAR overlap truncation
- L overlap currently stored as one numeric + one op char.
- Complex CIGAR strings are truncated.
- Documented in `KNOWN_ISSUES.md`.

2. P-line separator normalization
- Parser/writer uses `,` separator for path node lists.
- `;` (GFA v1.2 jump-separated paths) is not preserved.
- Documented in `KNOWN_ISSUES.md`.

3. Segment name reconstruction on decode
- Decompression rebuilds segment names as `"1"`, `"2"`, ...
- Original non-numeric segment labels are not stored in `CompressedData`.
- Graph topology/sequences can still round-trip, but segment label text fidelity is not guaranteed.

4. `freq_threshold > 2` effectively not supported in current fast rule generator
- Code warns and falls back behaviorally to repeated-2mer logic.

5. Output ordering is normalized by writer
- Writer emits lines in fixed block order: `H`, `S`, `L`, `P`, `W`, `J`, `C`.
- Original interleaving/order of lines is not preserved.

## 9. Parameter summary

- `num_rounds`: max grammar rounds
- `freq_threshold`: nominal 2-mer frequency threshold (see caveat above)
- `delta_round`: number of traversal delta-transform passes before grammar, and inverse passes during decode
- `num_threads`: OpenMP thread override (`0` = runtime default)

Environment variable:
- `GFA_COMPRESSION_ZSTD_LEVEL` (1..22, default 9)

## 10. Practical checklist for users

1. Compress with desired rounds/delta.
2. Decompress and run `verify_round_trip` against parsed original.
3. If you need file persistence, serialize to `.gfaz` and reload.
4. If textual GFA identity is required, account for caveats in Section 8.
