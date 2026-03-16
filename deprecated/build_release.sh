#!/bin/bash
# Build script for creating CPU and GPU release binaries

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
RELEASE_DIR="$SCRIPT_DIR/release"

echo "=========================================="
echo "GFA Compression - Release Build Script"
echo "=========================================="

# Clean and create directories
rm -rf "$BUILD_DIR" "$RELEASE_DIR"
mkdir -p "$BUILD_DIR" "$RELEASE_DIR"

# ========== Build 1: CPU-only version ==========
echo ""
echo "[1/2] Building CPU-only version..."
echo "--------------------------------------"
cd "$BUILD_DIR"
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Copy CPU binaries to release directory
if [ -f "bin/gfaz" ]; then
    cp bin/gfaz "$RELEASE_DIR/gfaz"
    echo "✓ CPU CLI: release/gfaz ($(du -h "$RELEASE_DIR/gfaz" | cut -f1))"
else
    echo "✗ Failed to build CPU CLI"
    exit 1
fi

if [ -f "gfa_compression.so" ]; then
    cp gfa_compression.so "$RELEASE_DIR/gfa_compression_cpu.so"
    echo "✓ CPU Python module: release/gfa_compression_cpu.so ($(du -h "$RELEASE_DIR/gfa_compression_cpu.so" | cut -f1))"
else
    echo "✗ Failed to build CPU Python module"
    exit 1
fi

# ========== Build 2: GPU version ==========
echo ""
echo "[2/2] Building GPU version..."
echo "--------------------------------------"
cd "$SCRIPT_DIR"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON
make -j$(nproc)

# Copy GPU binaries to release directory
if [ -f "bin/gfaz" ]; then
    cp bin/gfaz "$RELEASE_DIR/gfaz_gpu"
    echo "✓ GPU CLI: release/gfaz_gpu ($(du -h "$RELEASE_DIR/gfaz_gpu" | cut -f1))"
else
    echo "✗ Failed to build GPU CLI"
    exit 1
fi

if [ -f "gfa_compression.so" ]; then
    cp gfa_compression.so "$RELEASE_DIR/gfa_compression_gpu.so"
    echo "✓ GPU Python module: release/gfa_compression_gpu.so ($(du -h "$RELEASE_DIR/gfa_compression_gpu.so" | cut -f1))"
else
    echo "✗ Failed to build GPU Python module"
    exit 1
fi

# ========== Verify dependencies ==========
echo ""
echo "=========================================="
echo "Dependency Verification"
echo "=========================================="

echo ""
echo "CPU binaries (should have NO CUDA):"
echo "  gfaz:"
ldd "$RELEASE_DIR/gfaz" | grep cuda && echo "    ✗ ERROR: Found CUDA dependency!" || echo "    ✓ No CUDA dependencies"
echo "  gfa_compression_cpu.so:"
ldd "$RELEASE_DIR/gfa_compression_cpu.so" | grep cuda && echo "    ✗ ERROR: Found CUDA dependency!" || echo "    ✓ No CUDA dependencies"

echo ""
echo "GPU binaries (should have CUDA):"
echo "  gfaz_gpu:"
ldd "$RELEASE_DIR/gfaz_gpu" | grep cuda && echo "    ✓ Has CUDA dependencies" || echo "    ✗ ERROR: Missing CUDA dependency!"
echo "  gfa_compression_gpu.so:"
ldd "$RELEASE_DIR/gfa_compression_gpu.so" | grep cuda && echo "    ✓ Has CUDA dependencies" || echo "    ✗ ERROR: Missing CUDA dependency!"

# ========== Summary ==========
echo ""
echo "=========================================="
echo "Build Complete!"
echo "=========================================="
echo ""
echo "Release binaries in: $RELEASE_DIR"
ls -lh "$RELEASE_DIR"
echo ""
echo "Distribution instructions:"
echo "  - gfaz              → CPU-only CLI (works everywhere)"
echo "  - gfaz_gpu          → GPU-accelerated CLI (requires CUDA)"
echo "  - gfa_compression_cpu.so  → CPU-only Python module"
echo "  - gfa_compression_gpu.so  → GPU-accelerated Python module"
echo ""
