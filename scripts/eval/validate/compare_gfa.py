#!/usr/bin/env python3
"""
Compare two GFA files semantically using the project parser/bindings.

This validator is intended for GFAz compression/decompression round-trip checks.
It validates graph content rather than raw line-for-line textual equality.

Important:
    GFAz currently reconstructs segment names canonically as dense 1-based
    numeric IDs during decompression. Because of that, raw `S` lines may differ
    even when the graph is semantically equivalent. This script is therefore the
    preferred validator over raw text diffs such as compare.sh.

Usage:
    python scripts/eval/validate/compare_gfa.py <original.gfa> <decompressed.gfa>
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Add the build directory to the Python path.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "build"))
import gfa_compression as gfa_lib


def graph_summary(graph) -> str:
    return (f"segments={len(graph.node_sequences) - 1}, "
            f"paths={len(graph.paths)}, "
            f"walks={len(graph.walks.walks)}")


def segment_name_stats(graph) -> tuple[int, int]:
    total = max(0, len(graph.node_id_to_name) - 1)
    numeric_dense = 0
    for idx in range(1, len(graph.node_id_to_name)):
        if graph.node_id_to_name[idx] == str(idx):
            numeric_dense += 1
    return numeric_dense, total


def compare_gfa_files(file1: str, file2: str) -> bool:
    """
    Compare two GFA files using the C++ library's semantic verifier.

    Returns True if graphs are equivalent, False otherwise.
    """
    if not os.path.exists(file1):
        print(f"Error: File not found: {file1}")
        return False
    
    if not os.path.exists(file2):
        print(f"Error: File not found: {file2}")
        return False
    
    file1_size = os.path.getsize(file1)
    file2_size = os.path.getsize(file2)
    
    print(f"=== GFA Comparison ===")
    print(f"File 1: {file1} ({file1_size:,} bytes)")
    print(f"File 2: {file2} ({file2_size:,} bytes)")
    
    # Parse first file
    print(f"\nParsing {os.path.basename(file1)}...")
    start = time.perf_counter()
    graph1 = gfa_lib.parse(file1)
    t1 = time.perf_counter() - start
    print(f"  Time: {t1:.2f}s")
    
    # Parse second file
    print(f"\nParsing {os.path.basename(file2)}...")
    start = time.perf_counter()
    graph2 = gfa_lib.parse(file2)
    t2 = time.perf_counter() - start
    print(f"  Time: {t2:.2f}s")
    
    # Print statistics
    print(f"\n=== Graph Statistics ===")
    print(f"Graph 1: {graph_summary(graph1)}")
    print(f"Graph 2: {graph_summary(graph2)}")

    graph1_numeric_dense, graph1_total = segment_name_stats(graph1)
    graph2_numeric_dense, graph2_total = segment_name_stats(graph2)

    print(f"\n=== Segment Name Notes ===")
    print(f"Graph 1 numeric dense IDs: {graph1_numeric_dense}/{graph1_total}")
    print(f"Graph 2 numeric dense IDs: {graph2_numeric_dense}/{graph2_total}")
    if graph2_total > 0 and graph2_numeric_dense == graph2_total:
        print("Note: Graph 2 uses canonical dense numeric segment names.")
    print("Note: validation is semantic; raw S-line text may still differ.")

    # Verify using C++ library
    print(f"\n=== Verifying ===")
    start = time.perf_counter()
    success = gfa_lib.verify_round_trip(graph1, graph2)
    t_verify = time.perf_counter() - start
    print(f"  Time: {t_verify:.2f}s")
    
    print(f"\n{'='*50}")
    if success:
        print("✅ MATCH: Graphs are semantically equivalent")
        if graph1.node_id_to_name != graph2.node_id_to_name:
            print("Note: segment names differ, which is expected when the")
            print("      decompressed graph uses canonical numeric IDs.")
    else:
        print("❌ MISMATCH: Graphs differ semantically")
        print("Hint: if compare.sh reports an S-line mismatch, check whether")
        print("      the difference is only segment names rather than sequences.")
    print(f"{'='*50}")
    
    return success


def main():
    parser = argparse.ArgumentParser(
        description="Compare two GFA files for lossless compression validation"
    )
    parser.add_argument("file1", help="First GFA file (original)")
    parser.add_argument("file2", help="Second GFA file (decompressed)")
    
    args = parser.parse_args()
    
    try:
        success = compare_gfa_files(args.file1, args.file2)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
