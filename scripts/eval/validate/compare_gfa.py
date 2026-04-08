#!/usr/bin/env python3
"""
Compare two GFA files to verify they represent the same graph.
Used to validate lossless compression by comparing original and decompressed GFA files.

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


def compare_gfa_files(file1: str, file2: str) -> bool:
    """
    Compare two GFA files using the C++ library's verify function.
    
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
    
    # Print statistics (use available attributes)
    print(f"\n=== Graph Statistics ===")
    print(f"Graph 1: {len(graph1.node_sequences)-1} segments, "
          f"{len(graph1.paths)} paths, "
          f"{len(graph1.walks.walks)} walks")
    print(f"Graph 2: {len(graph2.node_sequences)-1} segments, "
          f"{len(graph2.paths)} paths, "
          f"{len(graph2.walks.walks)} walks")
    
    # Verify using C++ library (handles all fields including links)
    print(f"\n=== Verifying ===")
    start = time.perf_counter()
    success = gfa_lib.verify_round_trip(graph1, graph2)
    t_verify = time.perf_counter() - start
    print(f"  Time: {t_verify:.2f}s")
    
    print(f"\n{'='*50}")
    if success:
        print("✅ MATCH: Graphs are equivalent")
    else:
        print("❌ MISMATCH: Graphs differ")
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
