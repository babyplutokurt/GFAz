#!/usr/bin/env python3
"""
GPU Round-trip Test: GfaGraph → GfaGraph_gpu → CompressedData_gpu → GfaGraph_gpu

This test verifies that the full compression/decompression pipeline
preserves all data correctly.
"""

import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests._bootstrap import add_build_to_syspath
add_build_to_syspath()

import gfa_compression as gfac


def print_header(title):
    """Print formatted section header"""
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}")


def print_step(step_num, description):
    """Print step header"""
    print(f"\n[Step {step_num}] {description}")


def calculate_compressed_size(compressed):
    """Calculate total size of CompressedData_gpu in bytes"""
    total = 0
    
    # Path data
    total += len(compressed.encoded_path_zstd_nvcomp.payload)
    total += len(compressed.path_lengths_zstd_nvcomp.payload)
    total += len(compressed.rules_first_zstd_nvcomp.payload)
    total += len(compressed.rules_second_zstd_nvcomp.payload)
    
    # Path metadata
    total += len(compressed.names_zstd_nvcomp.payload)
    total += len(compressed.name_lengths_zstd_nvcomp.payload)
    total += len(compressed.overlaps_zstd_nvcomp.payload)
    total += len(compressed.overlap_lengths_zstd_nvcomp.payload)
    
    # Segment data
    total += len(compressed.segment_sequences_zstd_nvcomp.payload)
    total += len(compressed.segment_seq_lengths_zstd_nvcomp.payload)
    
    # Link data
    total += len(compressed.link_from_ids_zstd_nvcomp.payload)
    total += len(compressed.link_to_ids_zstd_nvcomp.payload)
    total += len(compressed.link_from_orients_zstd_nvcomp.payload)
    total += len(compressed.link_to_orients_zstd_nvcomp.payload)
    total += len(compressed.link_overlap_nums_zstd_nvcomp.payload)
    total += len(compressed.link_overlap_ops_zstd_nvcomp.payload)
    
    # Optional fields
    for col in compressed.segment_optional_fields_zstd_nvcomp:
        if col.int_values_zstd_nvcomp.payload:
            total += len(col.int_values_zstd_nvcomp.payload)
        if col.float_values_zstd_nvcomp.payload:
            total += len(col.float_values_zstd_nvcomp.payload)
        if col.char_values_zstd_nvcomp.payload:
            total += len(col.char_values_zstd_nvcomp.payload)
        if col.strings_zstd_nvcomp.payload:
            total += len(col.strings_zstd_nvcomp.payload)
        if col.string_lengths_zstd_nvcomp.payload:
            total += len(col.string_lengths_zstd_nvcomp.payload)
    
    # Walk metadata
    if compressed.num_walks > 0:
        for attr in ['walk_sample_ids_zstd_nvcomp', 'walk_hap_indices_zstd_nvcomp',
                      'walk_seq_ids_zstd_nvcomp', 'walk_seq_starts_zstd_nvcomp',
                      'walk_seq_ends_zstd_nvcomp']:
            block = getattr(compressed, attr, None)
            if block and block.payload:
                total += len(block.payload)
    
    # Jump data (structured columnar)
    if compressed.num_jumps_stored > 0:
        for attr in ['jump_from_ids_zstd_nvcomp', 'jump_to_ids_zstd_nvcomp',
                      'jump_from_orients_zstd_nvcomp', 'jump_to_orients_zstd_nvcomp',
                      'jump_distances_zstd_nvcomp', 'jump_distance_lengths_zstd_nvcomp',
                      'jump_rest_fields_zstd_nvcomp', 'jump_rest_lengths_zstd_nvcomp']:
            block = getattr(compressed, attr, None)
            if block and block.payload:
                total += len(block.payload)
    
    # Containment data (structured columnar)
    if compressed.num_containments_stored > 0:
        for attr in ['containment_container_ids_zstd_nvcomp', 'containment_contained_ids_zstd_nvcomp',
                      'containment_container_orients_zstd_nvcomp', 'containment_contained_orients_zstd_nvcomp',
                      'containment_positions_zstd_nvcomp', 'containment_overlaps_zstd_nvcomp',
                      'containment_overlap_lengths_zstd_nvcomp', 'containment_rest_fields_zstd_nvcomp',
                      'containment_rest_lengths_zstd_nvcomp']:
            block = getattr(compressed, attr, None)
            if block and block.payload:
                total += len(block.payload)
    # Node names (for full-fidelity round-trip)
    for attr in ['node_names_zstd_nvcomp', 'node_name_lengths_zstd_nvcomp']:
        block = getattr(compressed, attr, None)
        if block and block.payload:
            total += len(block.payload)
    
    # Header line (stored as raw string)
    if hasattr(compressed, 'header_line') and compressed.header_line:
        total += len(compressed.header_line)
    
    return total





def test_gpu_roundtrip(gfa_file):
    """Test full GPU compression/decompression round-trip"""
    
    print_header(f"GPU Round-Trip Test: {gfa_file}")
    
    # ========================================================================
    # Step 1: Parse GFA file
    # ========================================================================
    print_step(1, "Parsing GFA file...")
    t_parse_start = time.time()
    graph = gfac.parse(gfa_file)
    t_parse_end = time.time()
    print(f"  ✓ Parsed in {t_parse_end - t_parse_start:.3f}s")
    print(f"    - Segments: {len(graph.node_sequences):,}")
    print(f"    - Paths: {len(graph.paths):,}")
    
    # ========================================================================
    # Step 2: Convert to GPU layout
    # ========================================================================
    print_step(2, "Converting to GfaGraph_gpu...")
    t_convert_start = time.time()
    gpu_graph_original = gfac.convert_to_gpu_layout(graph)
    t_convert_end = time.time()
    print(f"  ✓ Converted in {t_convert_end - t_convert_start:.3f}s")
    print(f"    - num_segments: {gpu_graph_original.num_segments:,}")
    print(f"    - num_paths: {gpu_graph_original.num_paths:,}")
    print(f"    - num_walks: {gpu_graph_original.num_walks:,}")
    print(f"    - num_links: {gpu_graph_original.num_links:,}")
    print(f"    - paths.total_nodes: {gpu_graph_original.paths.total_nodes():,}")
    
    # Get original file size
    original_file_size = os.path.getsize(gfa_file)
    
    # ========================================================================
    # Step 3: Compress to CompressedData_gpu (using pre-converted gpu_graph)
    # ========================================================================
    print_step(3, "Compressing GfaGraph_gpu → CompressedData_gpu...")
    t_compress_start = time.time()
    compressed = gfac.compress_gpu_graph(gpu_graph_original, 8)
    t_compress_end = time.time()
    
    # Calculate compression stats
    compressed_size = calculate_compressed_size(compressed)
    compression_ratio = original_file_size / compressed_size if compressed_size > 0 else 0
    
    print(f"  ✓ Compressed in {t_compress_end - t_compress_start:.3f}s")
    print(f"    - Encoded path: {len(gfac.decompress_encoded_path(compressed.encoded_path_zstd_nvcomp)):,} elements")
    print(f"    - Total rules: {compressed.total_rules():,}")
    print(f"    - Compression rounds: {compressed.num_rounds()}")
    print(f"    - Original size: {original_file_size / (1024*1024):.2f} MB")
    print(f"    - Compressed size: {compressed_size / (1024*1024):.2f} MB")
    print(f"    - Compression ratio: {compression_ratio:.2f}x")
    
    # ========================================================================
    # Step 4: Serialize to temp file, reload, then decompress
    # ========================================================================
    print_step(4, "Serialize compressed GPU data to temp file...")
    tmp_handle = tempfile.NamedTemporaryFile(
        mode="wb", suffix=".gfaz_gpu", prefix="gfa_gpu_roundtrip_", delete=False
    )
    tmp_path = tmp_handle.name
    tmp_handle.close()
    print(f"  Temp file: {tmp_path}")

    t_decompress_start = None
    t_decompress_end = None
    t_serialize_start = None
    t_serialize_end = None

    try:
        t_serialize_start = time.time()
        gfac.serialize_gpu(compressed, tmp_path)
        t_serialize_end = time.time()
        print("  ✓ Serialized")

        print_step(5, "Load temp file and decompress to GfaGraph_gpu...")
        loaded_compressed = gfac.deserialize_gpu(tmp_path)

        t_decompress_start = time.time()
        gpu_graph_decompressed = gfac.decompress_to_gpu_layout(loaded_compressed)
        t_decompress_end = time.time()
        print(f"  ✓ Decompressed in {t_decompress_end - t_decompress_start:.3f}s")
        print(f"    - num_segments: {gpu_graph_decompressed.num_segments:,}")
        print(f"    - num_paths: {gpu_graph_decompressed.num_paths:,}")
        print(f"    - num_walks: {gpu_graph_decompressed.num_walks:,}")
        print(f"    - num_links: {gpu_graph_decompressed.num_links:,}")

        # ====================================================================
        # Step 6: Compare original and decompressed GfaGraph_gpu
        # ====================================================================
        print_step(6, "Verifying round-trip (C++ verification)...")
        success = gfac.verify_gpu_round_trip(gpu_graph_original, gpu_graph_decompressed)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # ========================================================================
    # Step 7: Report results
    # ========================================================================
    print_step(7, "Results")
    
    if success:
        print_header("✅ GPU ROUND-TRIP TEST PASSED!")
        print(f"  GfaGraph → GfaGraph_gpu → CompressedData_gpu → GfaGraph_gpu")
        print(f"  All {gpu_graph_original.num_segments:,} segments verified")
        print(f"  All {gpu_graph_original.num_paths:,} paths verified ({gpu_graph_original.paths.total_nodes():,} nodes)")
        print(f"  All {gpu_graph_original.num_links:,} links verified")
        print()
        print(f"  Compression:")
        print(f"    - Original:    {original_file_size / (1024*1024):.2f} MB")
        print(f"    - Compressed:  {compressed_size / (1024*1024):.2f} MB")
        print(f"    - Ratio:       {compression_ratio:.2f}x")
        print()
        print(f"  Timings:")
        print(f"    - Parse:       {t_parse_end - t_parse_start:.3f}s")
        print(f"    - Convert:     {t_convert_end - t_convert_start:.3f}s")
        print(f"    - Compress:    {t_compress_end - t_compress_start:.3f}s")
        print(f"    - Serialize:   {t_serialize_end - t_serialize_start:.3f}s")
        print(f"    - Decompress:  {t_decompress_end - t_decompress_start:.3f}s")
        print(
            f"    - Total:       "
            f"{(t_parse_end - t_parse_start) + (t_convert_end - t_convert_start) + (t_compress_end - t_compress_start) + (t_serialize_end - t_serialize_start) + (t_decompress_end - t_decompress_start):.3f}s"
        )
        print("=" * 70)
        return True
    else:
        print_header("❌ GPU ROUND-TRIP TEST FAILED!")
        print("  See error messages above for details.")
        print("=" * 70)
        return False


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python tests/gpu/test_roundtrip.py <gfa_file>")
        print()
        print("Tests the full GPU round-trip:")
        print("  GfaGraph → GfaGraph_gpu → CompressedData_gpu → GfaGraph_gpu")
        print()
        print("Verifies that decompressed data matches original exactly.")
        sys.exit(1)
    
    gfa_file = sys.argv[1]
    success = test_gpu_roundtrip(gfa_file)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
