# GFAz: State-of-the-Art Graphical Fragment Assembly Compression

A high-performance C++/CUDA library and command-line tool for compressing and decompressing Graphical Fragment Assembly (GFA) files via grammar-based compression. It features both a CPU-only path as well as a GPU-accelerated backend using CUDA and nvComp.

## Features

- **High Performance**: Achieves up to 20X higher compression ratio compared to Gzip and 15X compared to Zstd, with GB/s-level throughput.
- **Dual Backends**: Run on CPU (with OpenMP parallelism) or GPU (CUDA + nvComp).
- **Python Extension**: Fully featured Python API (`gfa_compression`).
- **Command-Line Interface**: Easy to use `gfaz` CLI for quick compression/decompression.
- **Efficient Binary Formats**: Dedicated, magic-number-versioned binary file formats for CPU (`.gfaz`) and GPU (`.gfaz_gpu`).

## Quick Start (CLI)

First, install the prerequisites using conda:
```bash
conda create -n gfa python=3.11
conda activate gfa
conda install -c conda-forge pybind11 numpy
conda install -c conda-forge nvcomp
pip install nvidia-libnvcomp-cu12
```

Then, clone the repository and initialize submodules:
```bash
git submodule update --init --recursive
```

### Building the Project

**CPU-only Build (Default):**
```bash
mkdir build && cd build
cmake ..
cmake --build . -j$(nproc)
```

**CPU + GPU Build:**
```bash
mkdir build && cd build
cmake -DENABLE_CUDA=ON -DCUDA_PATH=/usr/local/cuda-12.8 ..
cmake --build . -j$(nproc)
```

Once built, the CLI tool `gfaz` can be found in `build/bin/gfaz`.

### Compress a GFA file

```bash
# CPU mode (creates example.gfa.gfaz)
gfaz compress example.gfa

# GPU mode (creates example.gfa.gfaz_gpu)
gfaz compress --gpu example.gfa
```

### Decompress a compressed file

```bash
# CPU mode (creates example.gfa)
gfaz decompress example.gfa.gfaz

# GPU mode
gfaz decompress --gpu example.gfa.gfaz_gpu
```

## Python API Usage

The library can also be used directly from Python via the `gfa_compression` module:

```python
import gfa_compression as gfac

# Check backend availability
has_gpu = gfac.has_gpu_backend()
print(f"GPU Backend Available: {has_gpu}")

# CPU compression workflow
graph = gfac.parse("example.gfa")
compressed_data = gfac.compress(graph, num_rounds=8, num_threads=0)
gfac.serialize(compressed_data, "example_output.gfaz")

# CPU decompression workflow
deserialized_data = gfac.deserialize("example_output.gfaz")
decompressed_graph = gfac.decompress(deserialized_data)
gfac.write_gfa(decompressed_graph, "example_output.gfa")

# GPU compression workflow
if has_gpu:
    gpu_graph = gfac.convert_to_gpu_layout(graph)
    gpu_compressed = gfac.compress_gpu_graph(gpu_graph)
    gfac.serialize_gpu(gpu_compressed, "example_output.gfaz_gpu")
```

## Documentation

- **[Build Guide](BUILD_GUIDE.md)**: Full instructions on how to build the project, including optional dependencies and CMake flags (`ENABLE_CUDA`, `ENABLE_PROFILING`).
- **[Workflow Reference](workflow.md)**: An overview of the internal architecture, compression pipelines, and serialization contracts.

## Known Limitations

- The CPU (`.gfaz`) and GPU (`.gfaz_gpu`) binary formats are fundamentally distinct and not interchangeable. 
- You must specify `--gpu` when decompressing a `.gfaz_gpu` file via the CLI. It does not auto-detect the backend via file magic.
