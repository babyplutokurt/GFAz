#!/usr/bin/env python3
"""
Test script to verify GPU compression workflow:
1. Parse GFA file
2. Convert to GPU layout
3. Full path compression on GPU (new GPU-resident API)
"""

import sys
import time
sys.path.insert(0, 'build')

import gfa_compression as gfac


def print_header(title):
    """Print formatted section header"""
    print(f"\n{title}")
    print("=" * 70)


def delta_encode_cpu(values):
    """Return delta-encoded copy of values (CPU-side helper for verification)."""
    if not values:
        return []
    out = [values[0]]
    for i in range(1, len(values)):
        out.append(values[i] - values[i - 1])
    return out


def test_gpu_path_compression(gfa_file):
    """Test the new GPU-resident path compression API"""
    
    print_header(f"GPU Path Compression Test: {gfa_file}")
    
    # ========================================================================
    # Step 1: Parse GFA file
    # ========================================================================
    print("\n[Step 1] Parsing GFA file...")
    graph_original = gfac.parse(gfa_file)
    print(f"  ✓ Parsed {len(graph_original.paths)} paths, "
          f"{len(graph_original.node_sequences)} segments")
    
    # ========================================================================
    # Step 2: Convert to GPU layout
    # ========================================================================
    print("\n[Step 2] Converting to GPU layout...")
    gpu_graph = gfac.convert_to_gpu_layout(graph_original)
    
    total_nodes = gpu_graph.paths.total_nodes()
    print(f"  ✓ GPU layout created:")
    print(f"    - Segments: {gpu_graph.num_segments:,}")
    print(f"    - Paths: {gpu_graph.num_paths:,}")
    print(f"    - Total path nodes: {total_nodes:,}")
    print(f"    - Links: {gpu_graph.num_links:,}")
    
    # ========================================================================
    # Step 3: Capture original delta-encoded path for verification
    # ========================================================================
    print("\n[Step 3] Computing reference delta-encoded path on CPU...")
    original_delta_path = delta_encode_cpu(list(gpu_graph.paths.data))
    print(f"  ✓ Reference path computed ({len(original_delta_path):,} elements)")
    if len(original_delta_path) < 500:
        print(f"  Raw Delta Path: {original_delta_path}")
    
    # ========================================================================
    # Step 4: Run NEW GPU compression (full GFA entry point)
    # ========================================================================
    print("\n[Step 4] Running GPU compression (compress_gfa_gpu)...")
    
    t_start = time.time()
    compressed_data = gfac.compress_gfa_gpu(gfa_file, 8)
    t_end = time.time()
    
    # Decompress encoded_path and path_lengths for verification
    encoded_path = gfac.decompress_encoded_path(compressed_data.encoded_path_zstd_nvcomp)
    path_lengths = gfac.decompress_path_lengths(compressed_data.path_lengths_zstd_nvcomp)
    
    print(f"  ✓ Compression complete in {t_end - t_start:.3f}s")
    print(f"    - Encoded path length: {len(encoded_path):,}")
    print(f"    - Total rules created: {compressed_data.total_rules():,}")
    print(f"    - Number of rounds: {compressed_data.num_rounds()}")
    print(f"    - Min rule ID: {compressed_data.min_rule_id()}")
    
    # Print per-round statistics
    print(f"\n  Per-round breakdown:")
    for i, rng in enumerate(compressed_data.layer_ranges):
        print(f"    Round {i+1}: {rng.count:,} rules (start_id={rng.start_id})")
    
    # Compression statistics
    reduction = total_nodes - len(encoded_path)
    ratio = (1 - len(encoded_path) / total_nodes) * 100
    print(f"\n  Compression statistics:")
    print(f"    - Original nodes:   {total_nodes:,}")
    print(f"    - Compressed nodes: {len(encoded_path):,}")
    print(f"    - Reduction:        {reduction:,} nodes ({ratio:.1f}%)")
    
    # ========================================================================
    # Step 5: Build rulebook and verify round-trip
    # ========================================================================
    print("\n[Step 5] Building rulebook from CompressedData_gpu...")
    
    t_build_start = time.time()
    rulebook = gfac.build_rulebook(compressed_data)
    min_rule_id = compressed_data.min_rule_id()
    t_build_end = time.time()
    
    print(f"  ✓ Rulebook built in {t_build_end - t_build_start:.4f}s")
    print(f"    - Rulebook size: {len(rulebook):,}")
    print(f"    - Min rule ID: {min_rule_id:,}")
    
    # ========================================================================
    # Step 6: CPU Reconstruction verification
    # ========================================================================
    print("\n[Step 6] Verifying with CPU stack-based reconstruction...")
    
    t_recon_start = time.time()
    reconstructed_path = gfac.cpu_reconstruct_path(
        encoded_path,  # Use decompressed encoded_path
        rulebook, 
        min_rule_id
    )
    t_recon_end = time.time()
    
    print(f"  ✓ Reconstruction complete in {t_recon_end - t_recon_start:.3f}s")
    print(f"    - Reconstructed length: {len(reconstructed_path):,}")
    
    # Compare lengths
    if len(reconstructed_path) != len(original_delta_path):
        print(f"  ✗ Length mismatch! Original: {len(original_delta_path):,}, "
              f"Reconstructed: {len(reconstructed_path):,}")
        return False
    
    # Compare content
    if reconstructed_path == original_delta_path:
        print("  ✓ VERIFICATION SUCCESSFUL: Reconstructed path matches original exactly!")
    else:
        print("  ✗ Content mismatch found!")
        # Find first mismatch
        for i, (orig, recon) in enumerate(zip(original_delta_path, reconstructed_path)):
            if orig != recon:
                print(f"    First mismatch at index {i}: original={orig}, reconstructed={recon}")
                # Show context
                start = max(0, i - 5)
                end = min(len(original_delta_path), i + 10)
                print(f"    Original context [{start}:{end}]: {original_delta_path[start:end]}")
                print(f"    Reconstructed context [{start}:{end}]: {reconstructed_path[start:end]}")
                break
        return False
    
    # ========================================================================
    # Step 7: GPU Decompression Test
    # ========================================================================
    print("\n[Step 7] Testing GPU decompression (path expansion on GPU)...")
    
    t_decomp_start = time.time()
    decompressed_paths = gfac.decompress_paths_gpu(compressed_data)
    t_decomp_end = time.time()
    
    decomp_time = t_decomp_end - t_decomp_start
    print(f"  ✓ GPU decompression complete in {decomp_time:.3f}s")
    print(f"    - Decompressed length: {len(decompressed_paths.data):,}")
    
    # Verify decompression matches original (before delta encoding)
    original_path = list(gpu_graph.paths.data)
    decompressed = list(decompressed_paths.data)
    
    if len(decompressed) != len(original_path):
        print(f"  ✗ Length mismatch! Original: {len(original_path):,}, Decompressed: {len(decompressed):,}")
        return False
    
    if decompressed == original_path:
        print("  ✓ GPU DECOMPRESSION VERIFIED: Matches original path exactly!")
    else:
        print("  ✗ Content mismatch in GPU decompression!")
        # Find first mismatch
        for i, (orig, decomp) in enumerate(zip(original_path, decompressed)):
            if orig != decomp:
                print(f"    First mismatch at index {i}: original={orig}, decompressed={decomp}")
                start = max(0, i - 5)
                end = min(len(original_path), i + 10)
                print(f"    Original context [{start}:{end}]: {original_path[start:end]}")
                print(f"    Decompressed context [{start}:{end}]: {decompressed[start:end]}")
                break
        return False
    
    # Compare timing: CPU reconstruction vs GPU decompression
    speedup = (t_recon_end - t_recon_start) / decomp_time if decomp_time > 0 else 0
    print(f"\n  Performance comparison:")
    print(f"    - CPU reconstruction: {t_recon_end - t_recon_start:.3f}s")
    print(f"    - GPU decompression:  {decomp_time:.3f}s")
    print(f"    - Speedup:            {speedup:.1f}x")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print_header("✓ GPU COMPRESSION + DECOMPRESSION TEST PASSED!")
    print(f"  Total path nodes:    {total_nodes:,}")
    print(f"  Compressed to:       {len(encoded_path):,} ({ratio:.1f}% reduction)")
    print(f"  Total rules:         {compressed_data.total_rules():,}")
    print(f"  Compression rounds:  {compressed_data.num_rounds()}")
    print(f"  Compression time:    {t_end - t_start:.3f}s")
    print(f"  GPU Decomp time:     {decomp_time:.3f}s")
    print(f"  CPU Recon time:      {t_recon_end - t_recon_start:.3f}s")
    print(f"  GPU vs CPU speedup:  {speedup:.1f}x")
    print("=" * 70)
    
    return True


def test_old_api_comparison(gfa_file):
    """Compare new API with old API for consistency"""
    
    print_header(f"API Comparison Test: {gfa_file}")
    
    # Parse and convert
    graph = gfac.parse(gfa_file)
    gpu_graph_new = gfac.convert_to_gpu_layout(graph)
    gpu_graph_old = gfac.convert_to_gpu_layout(graph)
    
    original_delta = delta_encode_cpu(list(gpu_graph_new.paths.data))
    
    # Run new API
    print("\n[New API] Running compress_gfa_gpu...")
    compressed_data = gfac.compress_gfa_gpu(gfa_file, 8)
    
    # Decompress encoded_path for comparison
    encoded_path_new = gfac.decompress_encoded_path(compressed_data.encoded_path_zstd_nvcomp)
    
    # Run old API
    print("[Old API] Running gpu_run_compression_2mer_device...")
    start_id = max(abs(v) for v in original_delta) + 1 if original_delta else 1
    next_id, rulebook_old = gfac.gpu_run_compression_2mer_device(
        gpu_graph_old.paths, start_id, 8
    )
    
    # Compare results
    print("\n[Comparison]")
    print(f"  New API - encoded path length: {len(encoded_path_new):,}")
    print(f"  Old API - encoded path length: {len(gpu_graph_old.paths.data):,}")
    print(f"  New API - total rules: {compressed_data.total_rules():,}")
    print(f"  Old API - total rules: {len(rulebook_old):,}")
    
    # Verify both can reconstruct
    rulebook_new = gfac.build_rulebook(compressed_data)
    min_id_new = compressed_data.min_rule_id()
    
    recon_new = gfac.cpu_reconstruct_path(encoded_path_new, rulebook_new, min_id_new)
    recon_old = gfac.cpu_reconstruct_path(list(gpu_graph_old.paths.data), rulebook_old, start_id)
    
    if recon_new == original_delta and recon_old == original_delta:
        print("  ✓ Both APIs produce correct reconstructions!")
        return True
    else:
        print("  ✗ Reconstruction mismatch between APIs")
        return False


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python test_gpu_conversion.py <gfa_file> [--compare]")
        print("\nTests the complete GPU path compression workflow:")
        print("  1. Parse GFA → GPU layout conversion")
        print("  2. GPU-resident path compression (CompressedData_gpu)")
        print("  3. Round-trip verification")
        print("\nOptions:")
        print("  --compare  Also run comparison with old API")
        sys.exit(1)
    
    gfa_file = sys.argv[1]
    compare = "--compare" in sys.argv
    
    success = test_gpu_path_compression(gfa_file)
    
    if compare and success:
        success = test_old_api_comparison(gfa_file)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
