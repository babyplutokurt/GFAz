#!/usr/bin/env python3
"""
Regression suite for example.gfa.

This keeps one curated fixture and validates the major invariants we care about:
CPU round-trip, CPU streaming path, GPU round-trip, CPU/GPU parity, and CLI
stats behavior for both backends.
"""

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.regression.regression_utils import (
    CLI_PATH,
    add_repo_and_build_to_syspath,
    assert_stats_output,
    calculate_gpu_compressed_size,
    ensure_cli_exists,
    file_size,
    format_size_mb,
    graph_summary,
    has_gpu_bindings,
    is_gpu_runtime_unavailable,
    require_success,
    run_command,
)

add_repo_and_build_to_syspath()
import gfa_compression as gfa_lib


def parse_args():
  parser = argparse.ArgumentParser(
      description="Regression suite for example.gfa"
  )
  parser.add_argument(
      "gfa_file",
      nargs="?",
      default="example.gfa",
      help="Fixture to test (default: example.gfa)",
  )
  parser.add_argument(
      "--gfaz",
      default=str(CLI_PATH),
      help="Path to gfaz CLI binary",
  )
  parser.add_argument(
      "--rounds",
      type=int,
      default=int(os.environ.get("GFA_COMPRESSION_ROUNDS", "8")),
      help="Compression rounds (default: 8)",
  )
  parser.add_argument(
      "--delta-rounds",
      type=int,
      default=int(os.environ.get("GFA_COMPRESSION_DELTA_ROUNDS", "1")),
      help="Delta rounds for CPU compression (default: 1)",
  )
  parser.add_argument(
      "--threshold",
      type=int,
      default=int(os.environ.get("GFA_COMPRESSION_FREQ_THRESHOLD", "2")),
      help="2-mer frequency threshold for CPU compression (default: 2)",
  )
  parser.add_argument(
      "--threads",
      type=int,
      default=0,
      help="Threads for CPU compression/decompression (0 = auto)",
  )
  parser.add_argument(
      "--skip-gpu",
      action="store_true",
      help="Skip GPU-specific regressions",
  )
  return parser.parse_args()


def print_header(title: str):
  print(f"\n{'=' * 72}")
  print(title)
  print(f"{'=' * 72}")


def print_result(name: str, elapsed: float, details: str = ""):
  print(f"PASS {name} ({elapsed:.3f}s)")
  if details:
    print(f"  {details}")


def test_cpu_legacy(args, original_graph):
  start = time.perf_counter()
  compressed = gfa_lib.compress(
      args.gfa_file,
      num_rounds=args.rounds,
      freq_threshold=args.threshold,
      delta_round=args.delta_rounds,
      num_threads=args.threads,
  )

  handle = tempfile.NamedTemporaryFile(
      mode="wb", suffix=".gfaz", prefix="reg_cpu_legacy_", delete=False
  )
  tmp_path = Path(handle.name)
  handle.close()

  try:
    gfa_lib.serialize(compressed, str(tmp_path))
    loaded = gfa_lib.deserialize(str(tmp_path))
    decompressed = gfa_lib.decompress(loaded, num_threads=args.threads)
    if not gfa_lib.verify_round_trip(original_graph, decompressed):
      raise AssertionError("CPU legacy round-trip verification failed")
  finally:
    if tmp_path.exists():
      tmp_path.unlink()

  elapsed = time.perf_counter() - start
  print_result("cpu_legacy_roundtrip", elapsed, graph_summary(original_graph))


def test_cpu_streaming(args, original_graph):
  start = time.perf_counter()
  compressed = gfa_lib.compress(
      args.gfa_file,
      num_rounds=args.rounds,
      freq_threshold=args.threshold,
      delta_round=args.delta_rounds,
      num_threads=args.threads,
  )

  gfaz_handle = tempfile.NamedTemporaryFile(
      mode="wb", suffix=".gfaz", prefix="reg_cpu_stream_", delete=False
  )
  out_handle = tempfile.NamedTemporaryFile(
      mode="w", suffix=".gfa", prefix="reg_cpu_stream_", delete=False
  )
  gfaz_path = Path(gfaz_handle.name)
  out_path = Path(out_handle.name)
  gfaz_handle.close()
  out_handle.close()

  try:
    gfa_lib.serialize(compressed, str(gfaz_path))
    loaded = gfa_lib.deserialize(str(gfaz_path))
    gfa_lib.write_gfa_from_compressed_data(
        loaded, str(out_path), num_threads=args.threads
    )
    streamed = gfa_lib.parse(str(out_path))
    if not gfa_lib.verify_round_trip(original_graph, streamed):
      raise AssertionError("CPU streaming round-trip verification failed")
  finally:
    if gfaz_path.exists():
      gfaz_path.unlink()
    if out_path.exists():
      out_path.unlink()

  elapsed = time.perf_counter() - start
  print_result("cpu_streaming_roundtrip", elapsed)


def test_gpu_roundtrip(args, original_graph):
  if args.skip_gpu or not has_gpu_bindings(gfa_lib):
    print("SKIP gpu_roundtrip (GPU bindings unavailable or --skip-gpu set)")
    return

  start = time.perf_counter()
  try:
    gpu_graph = gfa_lib.convert_to_gpu_layout(original_graph)
    compressed = gfa_lib.compress_gpu_graph(gpu_graph, args.rounds)
  except Exception as exc:
    if is_gpu_runtime_unavailable(exc):
      print(f"SKIP gpu_roundtrip ({exc})")
      return
    raise

  handle = tempfile.NamedTemporaryFile(
      mode="wb", suffix=".gfaz_gpu", prefix="reg_gpu_", delete=False
  )
  tmp_path = Path(handle.name)
  handle.close()

  try:
    try:
      gfa_lib.serialize_gpu(compressed, str(tmp_path))
      loaded = gfa_lib.deserialize_gpu(str(tmp_path))
      decompressed_gpu = gfa_lib.decompress_to_gpu_layout(loaded)
      if not gfa_lib.verify_gpu_round_trip(gpu_graph, decompressed_gpu):
        raise AssertionError("GPU round-trip verification failed")
      compressed_size = calculate_gpu_compressed_size(compressed)
    except Exception as exc:
      if is_gpu_runtime_unavailable(exc):
        print(f"SKIP gpu_roundtrip ({exc})")
        return
      raise
  finally:
    if tmp_path.exists():
      tmp_path.unlink()

  elapsed = time.perf_counter() - start
  ratio = file_size(Path(args.gfa_file)) / compressed_size if compressed_size else 0.0
  print_result(
      "gpu_roundtrip",
      elapsed,
      f"compressed={format_size_mb(compressed_size)}, ratio={ratio:.2f}x",
  )


def test_backend_parity(args, original_graph):
  if args.skip_gpu or not has_gpu_bindings(gfa_lib):
    print("SKIP backend_parity (GPU bindings unavailable or --skip-gpu set)")
    return

  start = time.perf_counter()
  cpu_compressed = gfa_lib.compress(
      args.gfa_file,
      num_rounds=args.rounds,
      freq_threshold=args.threshold,
      delta_round=args.delta_rounds,
      num_threads=args.threads,
  )
  cpu_graph = gfa_lib.decompress(cpu_compressed, num_threads=args.threads)

  try:
    gpu_graph = gfa_lib.convert_to_gpu_layout(original_graph)
    gpu_compressed = gfa_lib.compress_gpu_graph(gpu_graph, args.rounds)
    gpu_roundtrip = gfa_lib.decompress_to_gpu_layout(gpu_compressed)
    gpu_host_graph = gfa_lib.convert_from_gpu_layout(gpu_roundtrip)
  except Exception as exc:
    if is_gpu_runtime_unavailable(exc):
      print(f"SKIP backend_parity ({exc})")
      return
    raise

  if not gfa_lib.verify_round_trip(original_graph, cpu_graph):
    raise AssertionError("CPU graph diverged from original before parity check")
  if not gfa_lib.verify_round_trip(original_graph, gpu_host_graph):
    raise AssertionError("GPU graph diverged from original before parity check")
  if not gfa_lib.verify_round_trip(cpu_graph, gpu_host_graph):
    raise AssertionError("CPU and GPU decompressed graphs do not match")

  elapsed = time.perf_counter() - start
  print_result("backend_parity_cpu_vs_gpu", elapsed)


def test_cli_cpu_stats(args, original_graph, cli_path: Path):
  start = time.perf_counter()
  out_handle = tempfile.NamedTemporaryFile(
      mode="wb", suffix=".gfaz", prefix="reg_cli_cpu_", delete=False
  )
  gfaz_path = Path(out_handle.name)
  out_handle.close()
  gfa_path = gfaz_path.with_suffix(".roundtrip.gfa")

  try:
    result = run_command(
        [
            str(cli_path),
            "compress",
            "--stats",
            args.gfa_file,
            str(gfaz_path),
        ]
    )
    require_success(result, "CLI CPU compress --stats")
    assert_stats_output(result.stdout, expect_ratio=True)

    result = run_command(
        [
            str(cli_path),
            "decompress",
            "--stats",
            str(gfaz_path),
            str(gfa_path),
        ]
    )
    require_success(result, "CLI CPU decompress --stats")
    assert_stats_output(result.stdout, expect_ratio=False)

    roundtrip_graph = gfa_lib.parse(str(gfa_path))
    if not gfa_lib.verify_round_trip(original_graph, roundtrip_graph):
      raise AssertionError("CLI CPU stats round-trip verification failed")
  finally:
    if gfaz_path.exists():
      gfaz_path.unlink()
    if gfa_path.exists():
      gfa_path.unlink()

  elapsed = time.perf_counter() - start
  print_result("cli_cpu_stats", elapsed)


def test_cli_gpu_stats(args, original_graph, cli_path: Path):
  if args.skip_gpu or not has_gpu_bindings(gfa_lib):
    print("SKIP cli_gpu_stats (GPU bindings unavailable or --skip-gpu set)")
    return

  start = time.perf_counter()
  out_handle = tempfile.NamedTemporaryFile(
      mode="wb", suffix=".gfaz_gpu", prefix="reg_cli_gpu_", delete=False
  )
  gfaz_path = Path(out_handle.name)
  out_handle.close()
  gfa_path = gfaz_path.with_suffix(".roundtrip.gfa")

  try:
    result = run_command(
        [
            str(cli_path),
            "compress",
            "--gpu",
            "--stats",
            args.gfa_file,
            str(gfaz_path),
        ]
    )
    try:
      require_success(result, "CLI GPU compress --stats")
    except AssertionError as exc:
      if is_gpu_runtime_unavailable(exc):
        print(f"SKIP cli_gpu_stats ({exc})")
        return
      raise
    assert_stats_output(result.stdout, expect_ratio=True)

    result = run_command(
        [
            str(cli_path),
            "decompress",
            "--gpu",
            "--stats",
            str(gfaz_path),
            str(gfa_path),
        ]
    )
    try:
      require_success(result, "CLI GPU decompress --stats")
    except AssertionError as exc:
      if is_gpu_runtime_unavailable(exc):
        print(f"SKIP cli_gpu_stats ({exc})")
        return
      raise
    assert_stats_output(result.stdout, expect_ratio=False)

    roundtrip_graph = gfa_lib.parse(str(gfa_path))
    if not gfa_lib.verify_round_trip(original_graph, roundtrip_graph):
      raise AssertionError("CLI GPU stats round-trip verification failed")
  finally:
    if gfaz_path.exists():
      gfaz_path.unlink()
    if gfa_path.exists():
      gfa_path.unlink()

  elapsed = time.perf_counter() - start
  print_result("cli_gpu_stats", elapsed)


def main():
  args = parse_args()
  cli_path = Path(args.gfaz)
  ensure_cli_exists(cli_path)

  print_header("Example Regression Suite")
  print(f"Fixture: {args.gfa_file}")
  print(f"CLI:     {cli_path}")
  print(f"Rounds:  {args.rounds}")
  print(f"Delta:   {args.delta_rounds}")
  print(f"Threads: {args.threads if args.threads > 0 else 'auto'}")

  original_graph = gfa_lib.parse(args.gfa_file)
  print(f"Graph:   {graph_summary(original_graph)}")

  started = time.perf_counter()
  test_cpu_legacy(args, original_graph)
  test_cpu_streaming(args, original_graph)
  test_gpu_roundtrip(args, original_graph)
  test_backend_parity(args, original_graph)
  test_cli_cpu_stats(args, original_graph, cli_path)
  test_cli_gpu_stats(args, original_graph, cli_path)
  total = time.perf_counter() - started

  print_header("Regression Summary")
  print(f"PASS all requested regressions in {total:.3f}s")
  return 0


if __name__ == "__main__":
  sys.exit(main())
