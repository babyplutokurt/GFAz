# Python bindings

The extension module is `gfa_compression`. It is built with the project by
default (`-DBUILD_PYTHON_BINDINGS=ON`); see [BUILD_GUIDE.md](BUILD_GUIDE.md).

## Compress and decompress

```python
import gfa_compression as gfac

compressed = gfac.compress_file("example.gfa", rounds=8, threshold=2, delta_round=1)
gfac.serialize(compressed, "example.gfaz")

data = gfac.deserialize("example.gfaz")
gfac.write_gfa_from_compressed_data(data, "example.roundtrip.gfa")
```

`write_gfa_from_compressed_data` streams the GFA out and is the lower-memory
way to decompress. To get the whole graph in memory instead:

```python
graph = gfac.decompress_data(data)
gfac.write_gfa(graph, "example.roundtrip.gfa")
```

`delta_round=0` disables delta encoding on the CPU path. The default is `1`.

## Read paths in place

`PathReader` reads positions inside haplotypes without decompressing. Node ids
come back as numpy arrays, negative for reverse orientation.

```python
import gfa_compression as gfac

reader = gfac.PathReader("example.gfaz")       # index_path="example.gfaz.pidx" to reuse `gfaz index`
ref = reader.haplotypes_on("CHM13#0#chr6")[0]
first, nodes, starts = reader.steps_in_bases(ref, 30_000_000, 30_100_000)
visits = reader.locate(int(abs(nodes[0])))     # arrays: haplotype, step, start, reverse
name = reader.node_name(int(abs(nodes[0])))    # the segment's name in the GFA; node_id(name) goes back
```

## GPU (CUDA builds)

```python
import gfa_compression as gfac

if gfac.has_gpu_backend():
    graph = gfac.parse("example.gfa")
    gpu_graph = gfac.convert_to_gpu_layout(graph)
    compressed = gfac.compress_gpu_graph(gpu_graph, 8)
    gfac.serialize(compressed, "example_gpu.gfaz")
```

## Entry points

- `parse(...)` / `parse_gfa(...)`
- `compress_file(...)`
- `decompress_data(...)`
- `serialize(...)`
- `deserialize(...)`
- `write_gfa(...)`
- `write_gfa_from_compressed_data(...)`
- `PathReader(...)`: `steps`, `steps_in_bases`, `locate`, `node_lengths`
- `extract_path_line(...)` / `extract_path_lines(...)`
- `extract_walk_line(...)`
- `extract_walk_line_by_name(...)`
- `extract_walk_lines(...)`
- `extract_walk_lines_by_name(...)`
- `add_haplotypes(...)`

CUDA builds also expose:

- `has_gpu_backend()`
- `convert_to_gpu_layout(...)`
- `convert_from_gpu_layout(...)`
- `compress_gfa_gpu(...)`
- `compress_gpu_graph(...)`
- `decompress_to_gpu_layout(...)`

## In-memory graph

`parse` and `decompress_data` return a graph that groups GFA records by line
type:

- `segments` (`SegmentData`) for S-lines
- `paths_data` (`PathData`) for P-lines
- `walks` (`WalkData`) for W-lines
- `links` (`LinkData`) for L-lines
- `jumps` (`JumpData`) for J-lines
- `containments` (`ContainmentData`) for C-lines
