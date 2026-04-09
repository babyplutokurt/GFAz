# GFAz: State-of-the-Art Graphical Fragment Assembly Compression

A high-performance C++/CUDA library and command-line tool for compressing and decompressing Graphical Fragment Assembly (GFA) files via grammar-based compression. It features both a CPU-only path as well as a GPU-accelerated backend using CUDA.

## Features

- **High Performance**: Achieves up to 20X higher compression ratio compared to Gzip and 15X compared to Zstd, with GB/s-level throughput.
- **Dual Backends**: Run on CPU (with OpenMP parallelism) or GPU (CUDA).
- **Python Extension**: Fully featured Python API (`gfa_compression`).
- **Command-Line Interface**: Easy to use `gfaz` CLI for quick compression/decompression.
- **Efficient Binary Formats**: Dedicated, magic-number-versioned binary file formats for CPU (`.gfaz`) and GPU (`.gfaz_gpu`).

## Performance

| Dataset | Metrics | Gzip | Zstd | sqz | sqz+bgzip | sqz+Zstd | GBZ | gfaz(CPU) | gfaz(GPU) |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| chr1. | Ratio | 5.59 | 7.54 | 3.09 | 18.0 | 16.7 | 9.52 | **35.4** | **31.7** |
| | Co. | 46.2 | 2178 | 3.95 | 3.97 | 4.00 | 12.1 | **385** | **2754** |
| | De. | 359 | 1618 | 21.6 | 21.4 | 21.2 | 284 | **1658** | **8124** |
| chr6. | Ratio | 5.04 | 6.99 | 5.51 | 20.8 | 19.2 | 19.2 | **35.4** | **28.18** |
| | Co. | 41.0 | 1712 | 3.56 | 3.56 | 3.59 | 10.7 | **348** | **3791** |
| | De. | 348 | 1515 | 20.1 | 20.4 | 20.2 | 281 | **1758** | **7230** |
| E.coli | Ratio | 4.69 | 5.67 | 1.26 | 7.46 | 6.79 | 5.58 | **18.3** | **16.7** |
| | Co. | 33.3 | 1356 | 4.57 | 4.53 | 4.62 | 20.2 | **190** | **678** |
| | De. | 310 | 1258 | 34.0 | 32.2 | 34.5 | 197 | **491** | **1430** |
| HPRCv1.1 | Ratio | 4.02 | 5.32 | - | - | - | 14.0 | **22.4** | **20.4** |
| | Co. | 36.4 | 1657 | - | - | - | 84.5 | **231** | **4843** |
| | De. | 319 | 1234 | - | - | - | 650 | **1058** | **9435** |
| HPRCv2.0 | Ratio | 4.19 | 6.49 | - | - | - | 66.8 | **83.9** | **76.4** |
| | Co. | 49.1 | 1514 | - | - | - | 130 | **367** | **-** |
| | De. | 342 | 1240 | - | - | - | 648 | **1652** | **-** |
| HPRCv2.1 | Ratio | 4.19 | 6.43 | - | - | - | 64.2 | **82.8** | **74.2** |
| | Co. | 48.9 | 1540 | - | - | - | 136 | **348** | **-** |
| | De. | 343 | 1241 | - | - | - | 652 | **1559** | **-** |

> **Note**: `Ratio` indicates compression ratio, `Co.` indicates compression speed/time, and `De.` indicates decompression speed/time. Bold values indicate the best performance. The system configuration used: AMD Ryzen Threadripper PRO 9955WX (16 cores), an NVIDIA RTX Pro 6000 GPU, and 512 GB of DDR5 6400 MHz memory.

## Quick Start (CLI)

First, install the prerequisites using conda:
```bash
conda create -n gfa python=3.11
conda activate gfa
conda install -c conda-forge pybind11 numpy
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

- **[Build Guide](BUILD_GUIDE.md)**: Full instructions on how to build the project, including CMake flags (`ENABLE_CUDA`, `ENABLE_PROFILING`).
- **[Workflow Reference](workflow.md)**: An overview of the internal architecture, compression pipelines, and serialization contracts.

## Known Limitations

- CPU and GPU backends now share the same Zstd-based compressed representation internally; the difference is in how the data transforms are computed.

## License

This project is licensed under the MIT License.
