#!/usr/bin/env python3
"""
CPU direct-writer round-trip test:
1) Parse GFA to original GfaGraph
2) Compress and serialize to a temporary .gfaz file
3) Deserialize and write GFA through the CPU direct-writer path
4) Parse the streamed GFA output
5) Verify original vs reparsed streamed output
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
        description="CPU direct-writer round-trip test for direct GFA writing"
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
        help="Threads for compress/write (0 = all available)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=== CPU Direct-Writer Round-Trip Test ===")
    print(f"Input:      {args.gfa_file}")
    print(f"Rounds:     {args.rounds}")
    print(f"Delta:      {args.delta_rounds}")
    print(f"Threshold:  {args.threshold}")
    print(f"Threads:    {args.threads if args.threads > 0 else 'all available'}")
    print("Mode:       CPU direct writer")
    original_file_size = os.path.getsize(args.gfa_file)

    print("\n[1] Parse original GFA")
    t_parse_start = time.perf_counter()
    original_graph = gfa_lib.parse(args.gfa_file)
    t_parse_end = time.perf_counter()
    print(
        f"  Parsed: {len(original_graph.paths)} paths, "
        f"{len(original_graph.walks.walks)} walks, "
        f"{len(original_graph.node_sequences) - 1} segments"
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

    tmp_gfaz = tempfile.NamedTemporaryFile(
        mode="wb", suffix=".gfaz", prefix="gfa_cpu_stream_", delete=False
    )
    tmp_out = tempfile.NamedTemporaryFile(
        mode="w", suffix=".gfa", prefix="gfa_cpu_stream_", delete=False
    )
    tmp_gfaz_path = tmp_gfaz.name
    tmp_out_path = tmp_out.name
    tmp_gfaz.close()
    tmp_out.close()

    try:
        t_serialize_start = time.perf_counter()
        gfa_lib.serialize(compressed, tmp_gfaz_path)
        t_serialize_end = time.perf_counter()
        compressed_size = os.path.getsize(tmp_gfaz_path)
        compression_ratio = (
            original_file_size / compressed_size if compressed_size > 0 else 0.0
        )
        print(f"  Saved temporary file: {tmp_gfaz_path}")

        print("\n[3] Load temporary .gfaz and write GFA directly")
        t_write_start = time.perf_counter()
        loaded = gfa_lib.deserialize(tmp_gfaz_path)
        gfa_lib.write_gfa_from_compressed_data(
            loaded, tmp_out_path, num_threads=args.threads
        )
        t_write_end = time.perf_counter()
        print(f"  Streamed temporary GFA: {tmp_out_path}")

        print("\n[4] Parse streamed GFA output")
        t_reparse_start = time.perf_counter()
        streamed_graph = gfa_lib.parse(tmp_out_path)
        t_reparse_end = time.perf_counter()

        print("\n[5] Verify original vs streamed-output GfaGraph")
        ok = gfa_lib.verify_round_trip(original_graph, streamed_graph)
        if not ok:
            print("❌ FAIL CPU direct-writer round-trip verification FAILED")
            return 1
    finally:
        if os.path.exists(tmp_gfaz_path):
            os.remove(tmp_gfaz_path)
        if os.path.exists(tmp_out_path):
            os.remove(tmp_out_path)

    print("✅ PASS CPU direct-writer round-trip verification PASSED")
    print("\nResults")
    print(f"  Original size:   {original_file_size / (1024 * 1024):.2f} MB")
    print(f"  Compressed size: {compressed_size / (1024 * 1024):.2f} MB")
    print(f"  Ratio:           {compression_ratio:.2f}x")
    print(f"  Parse:           {t_parse_end - t_parse_start:.3f}s")
    print(f"  Compress:        {t_compress_end - t_compress_start:.3f}s")
    print(f"  Serialize:       {t_serialize_end - t_serialize_start:.3f}s")
    print(f"  Write GFA:       {t_write_end - t_write_start:.3f}s")
    print(f"  Re-parse:        {t_reparse_end - t_reparse_start:.3f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
