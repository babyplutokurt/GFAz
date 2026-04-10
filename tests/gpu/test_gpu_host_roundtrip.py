#!/usr/bin/env python3
"""
GPU rolling host-graph round-trip test:
1) Parse GFA to original GfaGraph
2) Convert original graph to GfaGraph_gpu
3) GPU compress using the rolling scheduler
4) Serialize and reload temporary .gfaz
5) GPU decompress using rolling traversal expansion to host-side GfaGraph
6) Verify original vs reconstructed host GfaGraph
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="GPU rolling host-graph round-trip test for GfaGraph"
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
        type=float,
        default=4096 / (1024 * 1024 * 1024),
        help="Compression rolling chunk size in GiB (e.g. 0.5 for 512MiB)",
    )
    parser.add_argument(
        "--traversals-per-chunk",
        type=int,
        default=16,
        help="Rolling decompression traversals per chunk (default: 16)",
    )
    parser.add_argument(
        "--debug-decompression",
        action="store_true",
        help="Enable verbose GPU decompression timing output",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=== GPU Rolling Host-Graph Round-Trip Test ===")
    print(f"Input:      {args.gfa_file}")
    print(f"Rounds:     {args.rounds}")
    print("Mode:       rolling compression + rolling host-graph reconstruction")
    print(f"Chunk GiB:  {args.chunk_GB}")
    print(f"Traversals: {args.traversals_per_chunk}")

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

    print("\n[3] Compress with GPU rolling scheduler and save to temporary .gfaz")
    t_compress_start = time.perf_counter()
    comp_opts = gfac.GpuCompressionOptions()
    comp_opts.force_rolling_scheduler = True
    if args.chunk_GB is not None:
        comp_opts.rolling_chunk_bytes = int(args.chunk_GB * 1024 * 1024 * 1024)
    compressed = gfac.compress_gpu_graph(gpu_graph_original, args.rounds, comp_opts)
    t_compress_end = time.perf_counter()

    tmp_handle = tempfile.NamedTemporaryFile(
        mode="wb", suffix=".gfaz", prefix="gfa_gpu_host_", delete=False
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

        print("\n[4] Load temporary .gfaz and reconstruct host GfaGraph")
        loaded = gfac.deserialize_gpu(tmp_gfaz)
        decomp_opts = gfac.GpuDecompressionOptions()
        decomp_opts.traversals_per_chunk = args.traversals_per_chunk

        t_decompress_start = time.perf_counter()
        host_graph = gfac.decompress_to_host_graph_gpu(loaded, decomp_opts)
        t_decompress_end = time.perf_counter()

        print("\n[5] Verify original vs reconstructed host GfaGraph")
        ok = gfac.verify_round_trip(original_graph, host_graph)
        if not ok:
            print("❌ GPU rolling host-graph round-trip verification FAILED")
            return 1
    finally:
        if os.path.exists(tmp_gfaz):
            os.remove(tmp_gfaz)

    print("✅ PASS GPU rolling host-graph round-trip verification PASSED")
    print("\nResults")
    print(f"  Original size:   {original_file_size / (1024 * 1024):.2f} MB")
    print(f"  Compressed size: {compressed_size / (1024 * 1024):.2f} MB")
    print(f"  Ratio:           {compression_ratio:.2f}x")
    print(f"  Parse:           {t_parse_end - t_parse_start:.3f}s")
    print(f"  Convert:         {t_convert_end - t_convert_start:.3f}s")
    print(f"  Compress:        {t_compress_end - t_compress_start:.3f}s")
    print(f"  Serialize:       {t_serialize_end - t_serialize_start:.3f}s")
    print(f"  Decompress:      {t_decompress_end - t_decompress_start:.3f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
