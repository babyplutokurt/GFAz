#!/usr/bin/env python3
"""
GPU rolling direct-writer round-trip test:
1) Parse GFA to original GfaGraph
2) Convert original graph to GfaGraph_gpu
3) GPU compress using the rolling scheduler
4) Serialize to a temporary .gfaz file
5) Decompress through the CLI rolling direct-writer path to a temporary GFA
6) Parse the streamed GFA output
7) Verify original vs reparsed streamed output
"""
import argparse
import os
import subprocess
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
    for attr in [
        "paths_zstd",
        "walks_zstd",
        "rules_first_zstd",
        "rules_second_zstd",
        "names_zstd",
        "name_lengths_zstd",
        "overlaps_zstd",
        "overlap_lengths_zstd",
        "segment_sequences_zstd",
        "segment_seq_lengths_zstd",
        "link_from_ids_zstd",
        "link_to_ids_zstd",
        "link_from_orients_zstd",
        "link_to_orients_zstd",
        "link_overlap_nums_zstd",
        "link_overlap_ops_zstd",
        "walk_sample_ids_zstd",
        "walk_sample_id_lengths_zstd",
        "walk_hap_indices_zstd",
        "walk_seq_ids_zstd",
        "walk_seq_id_lengths_zstd",
        "walk_seq_starts_zstd",
        "walk_seq_ends_zstd",
        "jump_from_ids_zstd",
        "jump_to_ids_zstd",
        "jump_from_orients_zstd",
        "jump_to_orients_zstd",
        "jump_distances_zstd",
        "jump_distance_lengths_zstd",
        "jump_rest_fields_zstd",
        "jump_rest_lengths_zstd",
        "containment_container_ids_zstd",
        "containment_contained_ids_zstd",
        "containment_container_orients_zstd",
        "containment_contained_orients_zstd",
        "containment_positions_zstd",
        "containment_overlaps_zstd",
        "containment_overlap_lengths_zstd",
        "containment_rest_fields_zstd",
        "containment_rest_lengths_zstd",
    ]:
        block = getattr(compressed, attr, None)
        if block and block.payload:
            total += len(block.payload)

    for col in compressed.segment_optional_fields_zstd:
        for attr in [
            "int_values_zstd",
            "float_values_zstd",
            "char_values_zstd",
            "strings_zstd",
            "string_lengths_zstd",
            "b_subtypes_zstd",
            "b_lengths_zstd",
            "b_concat_bytes_zstd",
        ]:
            block = getattr(col, attr, None)
            if block and block.payload:
                total += len(block.payload)

    for col in compressed.link_optional_fields_zstd:
        for attr in [
            "int_values_zstd",
            "float_values_zstd",
            "char_values_zstd",
            "strings_zstd",
            "string_lengths_zstd",
            "b_subtypes_zstd",
            "b_lengths_zstd",
            "b_concat_bytes_zstd",
        ]:
            block = getattr(col, attr, None)
            if block and block.payload:
                total += len(block.payload)

    if hasattr(compressed, "header_line") and compressed.header_line:
        total += len(compressed.header_line)

    return total


def find_gfaz_binary():
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / "build" / "bin" / "gfaz"
    if not candidate.exists():
        raise FileNotFoundError(f"gfaz binary not found: {candidate}")
    return candidate


def parse_args():
    parser = argparse.ArgumentParser(
        description="GPU rolling direct-writer round-trip test through CLI writer"
    )
    parser.add_argument("gfa_file", help="Input GFA file")
    parser.add_argument(
        "--rounds",
        type=int,
        default=8,
        help="Grammar compression rounds (default: 8)",
    )
    parser.add_argument(
        "--chunk-GB",
        dest="chunk_gb",
        type=float,
        default=4096 / (1024 * 1024 * 1024),
        help="Compression rolling chunk size in GiB (e.g. 0.5 for 512MiB)",
    )
    parser.add_argument(
        "--traversals-per-chunk",
        type=int,
        default=16,
        help="Rolling GPU decompression traversals per chunk (default: 16)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Pass --stats to gfaz decompress for detailed timing output",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    gfaz_bin = find_gfaz_binary()

    print("=== GPU Rolling Direct-Writer Round-Trip Test ===")
    print(f"Input:      {args.gfa_file}")
    print(f"Rounds:     {args.rounds}")
    print("Mode:       rolling compression + rolling direct writer")
    print(f"Chunk GiB:  {args.chunk_gb}")
    print(f"Traversals: {args.traversals_per_chunk}")
    print(f"CLI:        {gfaz_bin}")

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

    print("\n[3] Compress with GPU rolling scheduler and save to temporary .gfaz")
    t_compress_start = time.perf_counter()
    comp_opts = gfac.GpuCompressionOptions()
    comp_opts.force_rolling_scheduler = True
    if args.chunk_gb is not None:
        comp_opts.rolling_chunk_bytes = int(args.chunk_gb * 1024 * 1024 * 1024)
    compressed = gfac.compress_gpu_graph(gpu_graph_original, args.rounds, comp_opts)
    t_compress_end = time.perf_counter()

    tmp_gfaz = tempfile.NamedTemporaryFile(
        mode="wb", suffix=".gfaz", prefix="gfa_gpu_stream_", delete=False
    )
    tmp_out = tempfile.NamedTemporaryFile(
        mode="w", suffix=".gfa", prefix="gfa_gpu_stream_", delete=False
    )
    tmp_gfaz_path = tmp_gfaz.name
    tmp_out_path = tmp_out.name
    tmp_gfaz.close()
    tmp_out.close()

    try:
        t_serialize_start = time.perf_counter()
        gfac.serialize_gpu(compressed, tmp_gfaz_path)
        t_serialize_end = time.perf_counter()
        compressed_size = os.path.getsize(tmp_gfaz_path)
        compression_ratio = (
            original_file_size / compressed_size if compressed_size > 0 else 0.0
        )
        print(f"  Saved temporary file: {tmp_gfaz_path}")

        print("\n[4] Decompress through gfaz rolling direct writer")
        cmd = [
            str(gfaz_bin),
            "decompress",
            "--gpu",
            "--gpu-traversals",
            str(args.traversals_per_chunk),
        ]
        if args.stats:
            cmd.append("--stats")
        cmd.extend([tmp_gfaz_path, tmp_out_path])

        t_write_start = time.perf_counter()
        subprocess.run(cmd, check=True)
        t_write_end = time.perf_counter()
        print(f"  Streamed temporary GFA: {tmp_out_path}")

        print("\n[5] Parse streamed GFA output")
        t_reparse_start = time.perf_counter()
        streamed_graph = gfac.parse(tmp_out_path)
        t_reparse_end = time.perf_counter()

        print("\n[6] Verify original vs streamed-output GfaGraph")
        ok = gfac.verify_round_trip(original_graph, streamed_graph)
        if not ok:
            print("❌ GPU rolling direct-writer round-trip verification FAILED")
            return 1
    finally:
        if os.path.exists(tmp_gfaz_path):
            os.remove(tmp_gfaz_path)
        if os.path.exists(tmp_out_path):
            os.remove(tmp_out_path)

    print("✅ PASS GPU rolling direct-writer round-trip verification PASSED")
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
    print(f"  Write GFA:       {t_write_end - t_write_start:.3f}s")
    print(f"  Re-parse:        {t_reparse_end - t_reparse_start:.3f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
