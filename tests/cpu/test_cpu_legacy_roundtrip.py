#!/usr/bin/env python3
"""
CPU legacy materialized round-trip test:
1) Parse GFA to original GfaGraph
2) Compress and serialize to a temporary .gfaz file
3) Deserialize and decompress through the CPU legacy materialized path
4) Verify original vs decompressed GfaGraph
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
import gfa_compression as gfa_lib


def parse_args():
    parser = argparse.ArgumentParser(
        description="CPU legacy materialized round-trip test for GFA compression"
    )
    parser.add_argument("gfa_file", help="Input GFA file")
    parser.add_argument(
        "--rounds",
        type=int,
        default=int(os.environ.get("GFA_COMPRESSION_ROUNDS", "8")),
        help="Grammar compression rounds (default: 8)",
    )
    parser.add_argument(
        "--delta-rounds",
        type=int,
        default=int(os.environ.get("GFA_COMPRESSION_DELTA_ROUNDS", "1")),
        help="Delta encoding rounds (default: 1)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=int(os.environ.get("GFA_COMPRESSION_FREQ_THRESHOLD", "2")),
        help="2-mer frequency threshold (default: 2)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="Threads for compress/decompress (0 = all available)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=== CPU Legacy Materialized Round-Trip Test ===")
    print(f"Input:      {args.gfa_file}")
    print(f"Rounds:     {args.rounds}")
    print(f"Delta:      {args.delta_rounds}")
    print(f"Threshold:  {args.threshold}")
    print(f"Threads:    {args.threads if args.threads > 0 else 'all available'}")
    print("Mode:       CPU legacy materialized")
    original_file_size = os.path.getsize(args.gfa_file)

    print("\n[1] Parse original GFA")
    t_parse_start = time.perf_counter()
    original_graph = gfa_lib.parse(args.gfa_file)
    t_parse_end = time.perf_counter()
    print(
        f"  Parsed: {len(original_graph.paths_data.traversals)} paths, "
        f"{len(original_graph.walks.walks)} walks, "
        f"{len(original_graph.segments.node_sequences) - 1} segments"
    )
    print(f"  Parse time: {t_parse_end - t_parse_start:.3f}s")

    print("\n[2] Compress and save to temporary .gfaz")
    t_compress_start = time.perf_counter()
    compressed = gfa_lib.compress(
        args.gfa_file,
        num_rounds=args.rounds,
        freq_threshold=args.threshold,
        delta_round=args.delta_rounds,
        num_threads=args.threads,
    )
    t_compress_end = time.perf_counter()

    tmp_handle = tempfile.NamedTemporaryFile(
        mode="wb", suffix=".gfaz", prefix="gfa_cpu_roundtrip_", delete=False
    )
    tmp_gfaz = tmp_handle.name
    tmp_handle.close()

    try:
        t_serialize_start = time.perf_counter()
        gfa_lib.serialize(compressed, tmp_gfaz)
        t_serialize_end = time.perf_counter()
        compressed_size = os.path.getsize(tmp_gfaz)
        compression_ratio = (
            original_file_size / compressed_size if compressed_size > 0 else 0.0
        )
        print(f"  Saved temporary file: {tmp_gfaz}")

        print("\n[3] Load temporary .gfaz and legacy-decompress")
        t_decompress_start = time.perf_counter()
        loaded = gfa_lib.deserialize(tmp_gfaz)
        decompressed_graph = gfa_lib.decompress(loaded, num_threads=args.threads)
        t_decompress_end = time.perf_counter()

        print("\n[4] Verify original vs decompressed GfaGraph")
        ok = gfa_lib.verify_round_trip(original_graph, decompressed_graph)
        if not ok:
            print("❌ CPU legacy materialized round-trip verification FAILED")
            return 1
    finally:
        if os.path.exists(tmp_gfaz):
            os.remove(tmp_gfaz)

    print("✅ PASS CPU legacy materialized round-trip verification PASSED")
    print("\nResults")
    print(f"  Original size:   {original_file_size / (1024 * 1024):.2f} MB")
    print(f"  Compressed size: {compressed_size / (1024 * 1024):.2f} MB")
    print(f"  Ratio:           {compression_ratio:.2f}x")
    print(f"  Parse:           {t_parse_end - t_parse_start:.3f}s")
    print(f"  Compress:        {t_compress_end - t_compress_start:.3f}s")
    print(f"  Serialize:       {t_serialize_end - t_serialize_start:.3f}s")
    print(f"  Decompress:      {t_decompress_end - t_decompress_start:.3f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
