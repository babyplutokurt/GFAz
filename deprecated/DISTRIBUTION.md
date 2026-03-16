# Distribution Guide

## Quick Answer

**YES!** `cmake ..` builds a **pure CPU binary** with **zero CUDA dependencies**.

---

## Two Build Modes

### 1. CPU-Only Build (Default)
```bash
cmake ..
make -j$(nproc)
```

**Output:**
- `bin/gfaz` - Pure CPU CLI (987 KB)
- `gfa_compression.so` - Pure CPU Python module (1.7 MB)

**Dependencies:** Only standard C++ libraries (libstdc++, libgomp, libc)

**Works on:** ANY Linux system (no GPU or CUDA required)

---

### 2. GPU Build
```bash
cmake -DENABLE_CUDA=ON ..
make -j$(nproc)
```

**Output:**
- `bin/gfaz` - GPU-accelerated CLI (1.2 MB)
- `gfa_compression.so` - GPU-accelerated Python module (1.8 MB)

**Dependencies:** CUDA runtime (`libcudart.so.12`)

**Works on:** Linux systems with NVIDIA GPU + CUDA Toolkit installed

---

## Building Both Versions for Distribution

Use the provided build script:

```bash
./build_release.sh
```

This creates `release/` directory with:
- `gfaz` - CPU-only CLI
- `gfaz_gpu` - GPU CLI
- `gfa_compression_cpu.so` - CPU Python module
- `gfa_compression_gpu.so` - GPU Python module

---

## CUDA Architecture Auto-Detection

**Current behavior:** CMake auto-detects the GPU on the build machine.

### Your RTX Pro 6000 Ada
When you build with `cmake -DENABLE_CUDA=ON ..`, CMake detects:
- Compute capability: **sm_89** (Ada Lovelace architecture)
- Binary optimized specifically for your GPU
- **Will work on:** RTX 4060/4070/4080/4090, RTX 6000 Ada, and newer

### For Other Users

Each user should **build from source** for optimal performance:

| GPU | Build Command | Auto-Detected Arch | Binary Works On |
|-----|--------------|-------------------|-----------------|
| GTX 1080 | `cmake -DENABLE_CUDA=ON ..` | sm_61 | Pascal GPUs |
| V100 | `cmake -DENABLE_CUDA=ON ..` | sm_70 | Volta GPUs |
| RTX 3090 | `cmake -DENABLE_CUDA=ON ..` | sm_86 | Ampere GPUs |
| RTX 6000 Ada | `cmake -DENABLE_CUDA=ON ..` | sm_89 | Ada GPUs |

**Advantage:** Each user gets a binary optimized for their specific GPU.

**Disadvantage:** Binary compiled on one GPU type may not work on different GPU types.

---

## If You Want to Distribute Pre-Built GPU Binaries

If you need ONE binary to work on multiple GPU types, specify architectures:

```bash
cmake -DENABLE_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES="60;70;75;80;86;89" \
      ..
```

**Trade-offs:**
- ✅ Works on Pascal (GTX 1060+), Volta (V100), Turing (RTX 20xx), Ampere (RTX 30xx), Ada (RTX 40xx)
- ❌ Larger binary size (~6x larger)
- ❌ Longer compile time (~6x longer)
- ❌ Slightly slower runtime (JIT overhead)

**Recommendation:** Let users build from source instead!

---

## Recommended Distribution Strategy

### Option 1: Source Distribution (Recommended)
Provide source code + build instructions:
```bash
# CPU-only (works everywhere)
cmake ..
make -j$(nproc)

# GPU (auto-detects user's GPU for optimal performance)
cmake -DENABLE_CUDA=ON ..
make -j$(nproc)
```

**Users get:** Binary optimized for their exact GPU model.

---

### Option 2: Binary Distribution
Ship 2 binaries:

1. **`gfaz`** (CPU-only)
   - Works on all Linux systems
   - No dependencies beyond glibc/libstdc++

2. **`gfaz_gpu`** (GPU-accelerated)
   - Requires CUDA Toolkit installed
   - User must ensure their GPU architecture matches
   - Or build with multiple architectures (large binary)

**Users choose:** Download `gfaz` if no GPU, `gfaz_gpu` if they have compatible GPU.

---

## Summary

### Your Questions Answered:

**Q: Does `cmake ..` build CPU-only binary?**
✅ **YES** - Pure CPU, zero CUDA dependencies (987 KB)

**Q: Will users with different GPUs be able to build?**
✅ **YES** - CMake auto-detects their GPU and compiles for it

**Q: Should we call GPU version `gfaz_gpu`?**
✅ **YES** - Good naming convention:
- `gfaz` = CPU-only (maximum compatibility)
- `gfaz_gpu` = GPU-accelerated (requires CUDA)

**Q: Do we need to specify CUDA architectures in CMakeLists.txt?**
❌ **NO** - Auto-detection works perfectly for source distribution
✅ **YES** - Only if distributing pre-built binaries for multiple GPU types

---

## Testing Binary Dependencies

### CPU binary (should have NO CUDA):
```bash
ldd release/gfaz | grep cuda
# (empty output = no CUDA dependency ✓)
```

### GPU binary (should have CUDA):
```bash
ldd release/gfaz_gpu | grep cuda
# libcudart.so.12 => ... (CUDA dependency present ✓)
```

---

## File Size Reference

| Binary | CPU-only | GPU |
|--------|----------|-----|
| CLI | 987 KB | 1.2 MB |
| Python module | 1.7 MB | 1.8 MB |

The ~200KB difference is due to CUDA device code and runtime overhead.
