#!/usr/bin/env python3
"""
GPU legacy round-trip test:
1) Parse GFA to original GfaGraph
2) Convert original graph to GfaGraph_gpu
3) GPU compress using the legacy full-device path
4) Serialize and reload temporary .gfaz_gpu
5) GPU decompress using the legacy full-device path
6) Verify original vs decompressed GfaGraph_gpu
"""
import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests._bootstrap import add_build_to_syspath
add_build_to_syspath()

import gfa_compression as gfac


def calculate_compressed_size(compressed):
    total = 0

    total += len(compressed.encoded_path_zstd_nvcomp.payload)
    total += len(compressed.path_lengths_zstd_nvcomp.payload)
    total += len(compressed.rules_first_zstd_nvcomp.payload)
    total += len(compressed.rules_second_zstd_nvcomp.payload)

    total += len(compressed.names_zstd_nvcomp.payload)
    total += len(compressed.name_lengths_zstd_nvcomp.payload)
    total += len(compressed.overlaps_zstd_nvcomp.payload)
    total += len(compressed.overlap_lengths_zstd_nvcomp.payload)

    total += len(compressed.segment_sequences_zstd_nvcomp.payload)
    total += len(compressed.segment_seq_lengths_zstd_nvcomp.payload)

    total += len(compressed.link_from_ids_zstd_nvcomp.payload)
    total += len(compressed.link_to_ids_zstd_nvcomp.payload)
    total += len(compressed.link_from_orients_zstd_nvcomp.payload)
    total += len(compressed.link_to_orients_zstd_nvcomp.payload)
    total += len(compressed.link_overlap_nums_zstd_nvcomp.payload)
    total += len(compressed.link_overlap_ops_zstd_nvcomp.payload)

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

    if compressed.num_walks > 0:
        for attr in [
            "walk_sample_ids_zstd_nvcomp",
            "walk_hap_indices_zstd_nvcomp",
            "walk_seq_ids_zstd_nvcomp",
            "walk_seq_starts_zstd_nvcomp",
            "walk_seq_ends_zstd_nvcomp",
        ]:
            block = getattr(compressed, attr, None)
            if block and block.payload:
                total += len(block.payload)

    if compressed.num_jumps_stored > 0:
        for attr in [
            "jump_from_ids_zstd_nvcomp",
            "jump_to_ids_zstd_nvcomp",
            "jump_from_orients_zstd_nvcomp",
            "jump_to_orients_zstd_nvcomp",
            "jump_distances_zstd_nvcomp",
            "jump_distance_lengths_zstd_nvcomp",
            "jump_rest_fields_zstd_nvcomp",
            "jump_rest_lengths_zstd_nvcomp",
        ]:
            block = getattr(compressed, attr, None)
            if block and block.payload:
                total += len(block.payload)

    if compressed.num_containments_stored > 0:
        for attr in [
            "containment_container_ids_zstd_nvcomp",
            "containment_contained_ids_zstd_nvcomp",
            "containment_container_orients_zstd_nvcomp",
            "containment_contained_orients_zstd_nvcomp",
            "containment_positions_zstd_nvcomp",
            "containment_overlaps_zstd_nvcomp",
            "containment_overlap_lengths_zstd_nvcomp",
            "containment_rest_fields_zstd_nvcomp",
            "containment_rest_lengths_zstd_nvcomp",
        ]:
            block = getattr(compressed, attr, None)
            if block and block.payload:
                total += len(block.payload)

    for attr in ["node_names_zstd_nvcomp", "node_name_lengths_zstd_nvcomp"]:
        block = getattr(compressed, attr, None)
        if block and block.payload:
            total += len(block.payload)

    if hasattr(compressed, "header_line") and compressed.header_line:
        total += len(compressed.header_line)

    return total


def parse_args():
    parser = argparse.ArgumentParser(
        description="GPU legacy round-trip test for GfaGraph_gpu"
    )
    parser.add_argument("gfa_file", help="Input GFA file")
    parser.add_argument(
        "--rounds",
        type=int,
        default=8,
        help="Grammar compression rounds (default: 8)",
    )
    parser.add_argument(
        "--debug-decompression",
        action="store_true",
        help="Enable verbose GPU decompression timing output",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=== GPU Legacy Round-Trip Test ===")
    print(f"Input:      {args.gfa_file}")
    print(f"Rounds:     {args.rounds}")
    print("Mode:       legacy compression + legacy decompression")

    if args.debug_decompression:
        gfac.set_gpu_decompression_debug(True)

    original_file_size = os.path.getsize(args.gfa_file)

    print("\n[1] Parse original GFA")
    t_parse_start = time.perf_counter()
    original_graph = gfac.parse(args.gfa_file)
    t_parse_end = time.perf_counter()
    print(
        f"  Parsed: {len(original_graph.paths)} paths, "
        f"{len(original_graph.walks.walks)} walks, "
        f"{len(original_graph.node_sequences) - 1} segments"
    )
    print(f"  Parse time: {t_parse_end - t_parse_start:.3f}s")

    print("\n[2] Convert original graph to GfaGraph_gpu")
    t_convert_start = time.perf_counter()
    gpu_graph_original = gfac.convert_to_gpu_layout(original_graph)
    t_convert_end = time.perf_counter()
    print(
        f"  Converted: {gpu_graph_original.num_paths:,} paths, "
        f"{gpu_graph_original.num_walks:,} walks, "
        f"{gpu_graph_original.num_segments:,} segments"
    )
    print(f"  Convert time: {t_convert_end - t_convert_start:.3f}s")

    print("\n[3] Compress with GPU legacy path and save to temporary .gfaz_gpu")
    t_compress_start = time.perf_counter()
    comp_opts = gfac.GpuCompressionOptions()
    comp_opts.force_full_device_legacy = True
    compressed = gfac.compress_gpu_graph(gpu_graph_original, args.rounds, comp_opts)
    t_compress_end = time.perf_counter()

    tmp_handle = tempfile.NamedTemporaryFile(
        mode="wb", suffix=".gfaz_gpu", prefix="gfa_gpu_legacy_", delete=False
    )
    tmp_gfaz = tmp_handle.name
    tmp_handle.close()

    try:
        t_serialize_start = time.perf_counter()
        gfac.serialize_gpu(compressed, tmp_gfaz)
        t_serialize_end = time.perf_counter()
        compressed_size = os.path.getsize(tmp_gfaz)
        compression_ratio = (
            original_file_size / compressed_size if compressed_size > 0 else 0.0
        )
        print(f"  Saved temporary file: {tmp_gfaz}")

        print("\n[4] Load temporary .gfaz_gpu and legacy-decompress")
        loaded = gfac.deserialize_gpu(tmp_gfaz)
        decomp_opts = gfac.GpuDecompressionOptions()
        decomp_opts.use_legacy_full_decompression = True

        t_decompress_start = time.perf_counter()
        gpu_graph_decompressed = gfac.decompress_to_gpu_layout(loaded, decomp_opts)
        t_decompress_end = time.perf_counter()

        print("\n[5] Verify original vs decompressed GfaGraph_gpu")
        ok = gfac.verify_gpu_round_trip(gpu_graph_original, gpu_graph_decompressed)
        if not ok:
            print("❌ GPU legacy round-trip verification FAILED")
            return 1
    finally:
        if os.path.exists(tmp_gfaz):
            os.remove(tmp_gfaz)

    print("✅ PASS GPU legacy round-trip verification PASSED")
    print("\nResults")
    print(f"  Original size:   {original_file_size / (1024 * 1024):.2f} MB")
    print(f"  Compressed size: {compressed_size / (1024 * 1024):.2f} MB")
    print(f"  Ratio:           {compression_ratio:.2f}x")
    print(
        f"  Payload size:    "
        f"{calculate_compressed_size(compressed) / (1024 * 1024):.2f} MB"
    )
    print(f"  Parse:           {t_parse_end - t_parse_start:.3f}s")
    print(f"  Convert:         {t_convert_end - t_convert_start:.3f}s")
    print(f"  Compress:        {t_compress_end - t_compress_start:.3f}s")
    print(f"  Serialize:       {t_serialize_end - t_serialize_start:.3f}s")
    print(f"  Decompress:      {t_decompress_end - t_decompress_start:.3f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
