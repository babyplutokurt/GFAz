# Backend Schema Map

This document gives a schema-level map of the four main data structures in
GFAz and shows how data moves between them:

- `GfaGraph` -> parsed, CPU-oriented in-memory graph
- `CompressedData` -> CPU compressed representation (`.gfaz`)
- `GfaGraph_gpu` -> GPU-oriented flattened graph
- `CompressedData_gpu` -> GPU compressed representation (`.gfaz_gpu`)

It is intended as a change-planning reference: when adding a new feature or
field, use this map to identify every stage that must be updated.

## Pipeline Overview

### CPU backend

1. Parse GFA text into `GfaGraph`.
2. Delta-transform `paths` and `walks.walks`.
3. Run iterative 2-mer grammar compression over paths and walks.
4. Store grammar rules plus compressed metadata in `CompressedData`.
5. Serialize `CompressedData` to `.gfaz`.

### GPU backend

1. Parse GFA text into `GfaGraph`.
2. Convert `GfaGraph` into flattened `GfaGraph_gpu`.
3. Delta-transform and grammar-compress the concatenated path/walk stream on
   GPU.
4. Store nvComp-compressed blocks plus metadata in `CompressedData_gpu`.
5. Serialize `CompressedData_gpu` to `.gfaz_gpu`.

## Structure Map

### `GfaGraph`

Defined in [include/gfa_parser.hpp](/home/kurty/Release/gfa_compression/include/gfa_parser.hpp).

This is the shared parser output used by both backends.

| Field | Meaning | Notes |
| --- | --- | --- |
| `header_line` | Raw `H` line | Stored as a single string |
| `node_name_to_id` | Segment name -> numeric ID | 1-based IDs |
| `node_id_to_name` | Numeric ID -> segment name | Index 0 is placeholder |
| `node_sequences` | Segment sequences | Index 0 is placeholder |
| `segment_optional_fields` | S-line optional fields | Columnar by tag/type |
| `path_names` | `P` line names | One per path |
| `paths` | `P` traversal node IDs | `NodeId` sign encodes orientation |
| `path_overlaps` | `P` overlap strings | One per path |
| `walks.walks` | `W` traversal node IDs | Parallel to walk metadata |
| `walks.sample_ids` | `W` sample IDs | One per walk |
| `walks.hap_indices` | `W` haplotype index | One per walk |
| `walks.seq_ids` | `W` sequence IDs | One per walk |
| `walks.seq_starts` | `W` start coordinate | One per walk |
| `walks.seq_ends` | `W` end coordinate | One per walk |
| `links.*` | `L` columns | Already columnar |
| `link_optional_fields` | L-line optional fields | Columnar by tag/type |
| `jumps.*` | `J` columns | Structured columnar |
| `containments.*` | `C` columns | Structured columnar |

### `CompressedData`

Defined in [include/compression_workflow.hpp](/home/kurty/Release/gfa_compression/include/compression_workflow.hpp).

This is the CPU compressed representation written to `.gfaz`.

| Field | Source | Encoding / storage |
| --- | --- | --- |
| `header_line` | `GfaGraph.header_line` | Stored as raw string |
| `layer_rule_ranges` | CPU grammar rounds | Rule ID ranges per round |
| `rules_first_zstd` | Generated grammar rules | `int32` array, delta-encoded, then Zstd |
| `rules_second_zstd` | Generated grammar rules | `int32` array, delta-encoded, then Zstd |
| `delta_round` | CPU compress config | Scalar |
| `sequence_lengths` | Encoded `paths` lengths | After delta + grammar |
| `original_path_lengths` | Original `paths` lengths | Before delta + grammar |
| `paths_zstd` | Flattened encoded `paths` | Zstd-compressed `int32` array |
| `names_zstd` | `path_names` | Concatenated string, Zstd |
| `name_lengths_zstd` | `path_names` lengths | Zstd-compressed `uint32` array |
| `overlaps_zstd` | `path_overlaps` | Concatenated string, Zstd |
| `overlap_lengths_zstd` | `path_overlaps` lengths | Zstd-compressed `uint32` array |
| `segment_sequences_zstd` | `node_sequences[1..]` | Concatenated string, Zstd |
| `segment_seq_lengths_zstd` | Segment sequence lengths | Zstd-compressed `uint32` array |
| `segment_optional_fields_zstd` | `segment_optional_fields` | Per-column compressed blocks |
| `num_links` | `links` count | Scalar |
| `link_from_ids_zstd` | `links.from_ids` | Delta-varint compressed |
| `link_to_ids_zstd` | `links.to_ids` | Delta-varint compressed |
| `link_from_orients_zstd` | `links.from_orients` | Bit-packed then compressed |
| `link_to_orients_zstd` | `links.to_orients` | Bit-packed then compressed |
| `link_overlap_nums_zstd` | `links.overlap_nums` | Zstd-compressed `uint32` array |
| `link_overlap_ops_zstd` | `links.overlap_ops` | Zstd-compressed `char` array |
| `link_optional_fields_zstd` | `link_optional_fields` | Per-column compressed blocks |
| `num_jumps` | `jumps` count | Scalar |
| `jump_*_zstd` | `jumps.*` | Columnar compressed storage |
| `num_containments` | `containments` count | Scalar |
| `containment_*_zstd` | `containments.*` | Columnar compressed storage |
| `walk_lengths` | Encoded `walks.walks` lengths | After delta + grammar |
| `original_walk_lengths` | Original `walks.walks` lengths | Before delta + grammar |
| `walks_zstd` | Flattened encoded walks | Zstd-compressed `int32` array |
| `walk_sample_ids_zstd` | `walks.sample_ids` | Concatenated string, Zstd |
| `walk_sample_id_lengths_zstd` | `walks.sample_ids` lengths | Zstd-compressed `uint32` array |
| `walk_hap_indices_zstd` | `walks.hap_indices` | Zstd-compressed `uint32` array |
| `walk_seq_ids_zstd` | `walks.seq_ids` | Concatenated string, Zstd |
| `walk_seq_id_lengths_zstd` | `walks.seq_ids` lengths | Zstd-compressed `uint32` array |
| `walk_seq_starts_zstd` | `walks.seq_starts` | Varint-compressed `int64` |
| `walk_seq_ends_zstd` | `walks.seq_ends` | Varint-compressed `int64` |

### `GfaGraph_gpu`

Defined in [include/gpu/gfa_graph_gpu.hpp](/home/kurty/Release/gfa_compression/include/gpu/gfa_graph_gpu.hpp).

This is the GPU-friendly normalized layout. Most nested string/vector
structures are flattened into contiguous buffers plus length arrays.

| Field | Source in `GfaGraph` | Normalization |
| --- | --- | --- |
| `header_line` | `header_line` | Direct copy |
| `num_segments` | Segment count | Scalar metadata |
| `num_paths` | `paths.size()` | Scalar metadata |
| `num_walks` | `walks.walks.size()` | Scalar metadata |
| `num_links` | `links.from_ids.size()` | Scalar metadata |
| `node_names` | `node_id_to_name` | Flattened strings |
| `node_sequences` | `node_sequences` | Flattened strings |
| `paths.data` | `paths` then `walks.walks` | Concatenated node stream |
| `paths.lengths` | Path lengths then walk lengths | One entry per traversal |
| `path_names` | `path_names` | Flattened strings |
| `path_overlaps` | `path_overlaps` | Flattened strings |
| `walk_sample_ids` | `walks.sample_ids` | Flattened strings |
| `walk_hap_indices` | `walks.hap_indices` | Direct copy |
| `walk_seq_ids` | `walks.seq_ids` | Flattened strings |
| `walk_seq_starts` | `walks.seq_starts` | Direct copy |
| `walk_seq_ends` | `walks.seq_ends` | Direct copy |
| `link_*` | `links.*` | Direct copy |
| `segment_optional_fields` | `segment_optional_fields` | Converted to GPU column type |
| `link_optional_fields` | `link_optional_fields` | Converted to GPU column type |
| `jump_*` | `jumps.*` | Direct copy or flattened strings |
| `containment_*` | `containments.*` | Direct copy or flattened strings |

### `CompressedData_gpu`

Defined in [include/gpu/compression_workflow_gpu.hpp](/home/kurty/Release/gfa_compression/include/gpu/compression_workflow_gpu.hpp).

This is the GPU compressed representation written to `.gfaz_gpu`.

| Field | Source | Encoding / storage |
| --- | --- | --- |
| `encoded_path_zstd_nvcomp` | `GfaGraph_gpu.paths.data` | Delta-encoded, grammar-replaced, nvComp Zstd |
| `rules_first_zstd_nvcomp` | Generated GPU grammar rules | Split, delta-encoded, nvComp Zstd |
| `rules_second_zstd_nvcomp` | Generated GPU grammar rules | Split, delta-encoded, nvComp Zstd |
| `layer_ranges` | GPU grammar rounds | `start_id` + `count` per round |
| `num_paths` | `GfaGraph_gpu.num_paths` | Scalar |
| `num_walks` | `GfaGraph_gpu.num_walks` | Scalar |
| `path_lengths_zstd_nvcomp` | `GfaGraph_gpu.paths.lengths` | nvComp-compressed `uint32` array |
| `names_zstd_nvcomp` | `path_names.data` | nvComp-compressed bytes |
| `name_lengths_zstd_nvcomp` | `path_names.lengths` | nvComp-compressed `uint32` array |
| `overlaps_zstd_nvcomp` | `path_overlaps.data` | nvComp-compressed bytes |
| `overlap_lengths_zstd_nvcomp` | `path_overlaps.lengths` | nvComp-compressed `uint32` array |
| `walk_sample_ids_zstd_nvcomp` | `walk_sample_ids.data` | nvComp-compressed bytes |
| `walk_sample_id_lengths_zstd_nvcomp` | `walk_sample_ids.lengths` | nvComp-compressed `uint32` array |
| `walk_hap_indices_zstd_nvcomp` | `walk_hap_indices` | nvComp-compressed `uint32` array |
| `walk_seq_ids_zstd_nvcomp` | `walk_seq_ids.data` | nvComp-compressed bytes |
| `walk_seq_id_lengths_zstd_nvcomp` | `walk_seq_ids.lengths` | nvComp-compressed `uint32` array |
| `walk_seq_starts_zstd_nvcomp` | `walk_seq_starts` | Varint bytes then nvComp |
| `walk_seq_ends_zstd_nvcomp` | `walk_seq_ends` | Varint bytes then nvComp |
| `header_line` | `header_line` | Raw string |
| `segment_sequences_zstd_nvcomp` | `node_sequences.data` | nvComp-compressed bytes |
| `segment_seq_lengths_zstd_nvcomp` | `node_sequences.lengths` | nvComp-compressed `uint32` array |
| `node_names_zstd_nvcomp` | `node_names.data` | nvComp-compressed bytes |
| `node_name_lengths_zstd_nvcomp` | `node_names.lengths` | nvComp-compressed `uint32` array |
| `segment_optional_fields_zstd_nvcomp` | GPU segment optional columns | Per-column nvComp blocks |
| `link_from_ids_zstd_nvcomp` | `link_from_ids` | Delta-varint then nvComp |
| `link_to_ids_zstd_nvcomp` | `link_to_ids` | Delta-varint then nvComp |
| `link_from_orients_zstd_nvcomp` | `link_from_orients` | Bit-packed then nvComp |
| `link_to_orients_zstd_nvcomp` | `link_to_orients` | Bit-packed then nvComp |
| `link_overlap_nums_zstd_nvcomp` | `link_overlap_nums` | nvComp-compressed `uint32` array |
| `link_overlap_ops_zstd_nvcomp` | `link_overlap_ops` | nvComp-compressed bytes |
| `num_links` | `link_from_ids.size()` | Scalar |
| `link_optional_fields_zstd_nvcomp` | GPU link optional columns | Per-column nvComp blocks |
| `num_jumps_stored` | Jump count | Scalar |
| `jump_*_zstd_nvcomp` | `jump_*` | Columnar nvComp-compressed storage |
| `num_containments_stored` | Containment count | Scalar |
| `containment_*_zstd_nvcomp` | `containment_*` | Columnar nvComp-compressed storage |

## Field-by-Field Mapping

This section answers: "where does a given logical GFA field end up in each
backend?"

### Header

| Logical field | CPU path | GPU path |
| --- | --- | --- |
| `H` line | `GfaGraph.header_line -> CompressedData.header_line` | `GfaGraph.header_line -> GfaGraph_gpu.header_line -> CompressedData_gpu.header_line` |

### Segment identities and sequences

| Logical field | CPU path | GPU path |
| --- | --- | --- |
| Segment names | Parsed into `node_name_to_id` and `node_id_to_name`, but not serialized into CPU `.gfaz` | `GfaGraph.node_id_to_name -> GfaGraph_gpu.node_names -> node_names_zstd_nvcomp + node_name_lengths_zstd_nvcomp` |
| Segment sequences | `node_sequences[1..] -> segment_sequences_zstd + segment_seq_lengths_zstd` | `node_sequences -> GfaGraph_gpu.node_sequences -> segment_sequences_zstd_nvcomp + segment_seq_lengths_zstd_nvcomp` |
| Segment optional fields | `segment_optional_fields -> segment_optional_fields_zstd` | `segment_optional_fields -> GfaGraph_gpu.segment_optional_fields -> segment_optional_fields_zstd_nvcomp` |

Important:

- CPU decompression intentionally reconstructs dense numeric segment names
  rather than preserving original names.
- GPU round-trip preserves original segment names.

### Paths (`P` lines)

| Logical field | CPU path | GPU path |
| --- | --- | --- |
| Path traversals | `paths -> delta transform -> grammar compression -> flatten -> paths_zstd + sequence_lengths + original_path_lengths` | `paths -> GfaGraph_gpu.paths (paths first) -> combined path/walk GPU grammar compression -> encoded_path_zstd_nvcomp + path_lengths_zstd_nvcomp + num_paths` |
| Path names | `path_names -> names_zstd + name_lengths_zstd` | `path_names -> GfaGraph_gpu.path_names -> names_zstd_nvcomp + name_lengths_zstd_nvcomp` |
| Path overlaps | `path_overlaps -> overlaps_zstd + overlap_lengths_zstd` | `path_overlaps -> GfaGraph_gpu.path_overlaps -> overlaps_zstd_nvcomp + overlap_lengths_zstd_nvcomp` |

### Walks (`W` lines)

| Logical field | CPU path | GPU path |
| --- | --- | --- |
| Walk traversals | `walks.walks -> delta transform -> grammar compression -> flatten -> walks_zstd + walk_lengths + original_walk_lengths` | `walks.walks -> appended after paths in GfaGraph_gpu.paths -> encoded_path_zstd_nvcomp + path_lengths_zstd_nvcomp + num_walks` |
| `sample_id` | `walks.sample_ids -> walk_sample_ids_zstd + walk_sample_id_lengths_zstd` | `walk_sample_ids -> walk_sample_ids_zstd_nvcomp + walk_sample_id_lengths_zstd_nvcomp` |
| `hap_index` | `walks.hap_indices -> walk_hap_indices_zstd` | `walk_hap_indices -> walk_hap_indices_zstd_nvcomp` |
| `seq_id` | `walks.seq_ids -> walk_seq_ids_zstd + walk_seq_id_lengths_zstd` | `walk_seq_ids -> walk_seq_ids_zstd_nvcomp + walk_seq_id_lengths_zstd_nvcomp` |
| `seq_start` | `walks.seq_starts -> walk_seq_starts_zstd` | `walk_seq_starts -> walk_seq_starts_zstd_nvcomp` |
| `seq_end` | `walks.seq_ends -> walk_seq_ends_zstd` | `walk_seq_ends -> walk_seq_ends_zstd_nvcomp` |

Important:

- CPU compresses paths and walks through the same grammar machinery, but stores
  them in separate compressed blocks afterward.
- GPU concatenates paths and walks into one traversal stream before grammar
  compression, then uses `num_paths`, `num_walks`, and `path_lengths` to split
  them back apart.

### Links (`L` lines)

| Logical field | CPU path | GPU path |
| --- | --- | --- |
| `from_ids` | `link_from_ids_zstd` | `link_from_ids_zstd_nvcomp` |
| `to_ids` | `link_to_ids_zstd` | `link_to_ids_zstd_nvcomp` |
| `from_orients` | `link_from_orients_zstd` | `link_from_orients_zstd_nvcomp` |
| `to_orients` | `link_to_orients_zstd` | `link_to_orients_zstd_nvcomp` |
| overlap numbers | `link_overlap_nums_zstd` | `link_overlap_nums_zstd_nvcomp` |
| overlap ops | `link_overlap_ops_zstd` | `link_overlap_ops_zstd_nvcomp` |
| optional fields | `link_optional_fields_zstd` | `link_optional_fields_zstd_nvcomp` |

### Jumps (`J` lines)

| Logical field | CPU path | GPU path |
| --- | --- | --- |
| jump IDs/orients | `jump_*_zstd` | `jump_*_zstd_nvcomp` |
| distance strings | `jump_distances_zstd + jump_distance_lengths_zstd` | `jump_distances_zstd_nvcomp + jump_distance_lengths_zstd_nvcomp` |
| rest fields | `jump_rest_fields_zstd + jump_rest_lengths_zstd` | `jump_rest_fields_zstd_nvcomp + jump_rest_lengths_zstd_nvcomp` |

### Containments (`C` lines)

| Logical field | CPU path | GPU path |
| --- | --- | --- |
| IDs/orients/positions | `containment_*_zstd` | `containment_*_zstd_nvcomp` |
| overlap strings | `containment_overlaps_zstd + containment_overlap_lengths_zstd` | `containment_overlaps_zstd_nvcomp + containment_overlap_lengths_zstd_nvcomp` |
| rest fields | `containment_rest_fields_zstd + containment_rest_lengths_zstd` | `containment_rest_fields_zstd_nvcomp + containment_rest_lengths_zstd_nvcomp` |

### Grammar rulebook

| Logical concept | CPU path | GPU path |
| --- | --- | --- |
| Rule ID ranges | `layer_rule_ranges` with `k`, `start_id`, `end_id`, `flattened_offset`, `element_count` | `layer_ranges` with `start_id`, `count` |
| Rule payload | `rules_first_zstd` + `rules_second_zstd` | `rules_first_zstd_nvcomp` + `rules_second_zstd_nvcomp` |
| Input traversal domain | `paths` and `walks.walks`, processed as separate nested vectors | One concatenated `GfaGraph_gpu.paths.data` stream |

## Transformation Notes

### CPU traversal transform

For `paths` and `walks.walks`, the CPU backend applies:

1. Delta transform for `delta_round` passes
2. 2-mer rule generation across paths and walks
3. Rule substitution into the traversals
4. Rule compaction, sorting, and ID remapping
5. Flattening into compressed integer arrays

This is why CPU stores:

- encoded lengths (`sequence_lengths`, `walk_lengths`)
- original lengths (`original_path_lengths`, `original_walk_lengths`)
- separate encoded traversal blocks (`paths_zstd`, `walks_zstd`)

### GPU traversal transform

For `GfaGraph_gpu.paths.data`, the GPU backend applies:

1. Copy flattened path/walk stream to device
2. Delta encode the combined stream
3. Find repeated 2-mers on device
4. Apply rule substitution on device
5. Compact unused rules and remap IDs
6. Sort rules and remap again
7. Store one compressed traversal block plus one compressed lengths block

This is why GPU stores:

- one combined encoded traversal block: `encoded_path_zstd_nvcomp`
- one lengths block: `path_lengths_zstd_nvcomp`
- split metadata counts: `num_paths`, `num_walks`

## Serialization Contracts

| Backend | Type | Magic | Version |
| --- | --- | --- | --- |
| CPU | `CompressedData` | `GFAZ` | `5` |
| GPU | `CompressedData_gpu` | `GPUG` | `1` |

Definitions:

- [include/serialization.hpp](/home/kurty/Release/gfa_compression/include/serialization.hpp)
- [include/gpu/serialization_gpu.hpp](/home/kurty/Release/gfa_compression/include/gpu/serialization_gpu.hpp)

The CPU and GPU serialized formats are distinct and are not interchangeable.

## Change Checklist

When adding a new logical field or record type, check all applicable layers:

1. Parser model in `GfaGraph`
2. CPU compression into `CompressedData`
3. CPU serialization and deserialization
4. CPU decompression and GFA writer
5. `GfaGraph -> GfaGraph_gpu` conversion
6. GPU compression into `CompressedData_gpu`
7. GPU serialization and deserialization
8. GPU decompression
9. `GfaGraph_gpu -> GfaGraph` conversion if full-fidelity host round-trip is needed
10. Python bindings and regression tests
