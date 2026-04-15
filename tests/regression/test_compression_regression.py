#!/usr/bin/env python3
"""
Compression regression suite.

This suite explicitly covers each CPU/GPU compression and decompression path
that matters for backend compatibility:
  - CPU legacy materialized round-trip
  - CPU direct-writer round-trip
  - GPU legacy materialized round-trip
  - GPU rolling materialized round-trip
  - GPU rolling host-materialized round-trip
  - GPU rolling direct-writer round-trip
  - Cross-backend container compatibility
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
    ensure_cli_exists,
    graph_summary,
    has_gpu_bindings,
    is_gpu_runtime_unavailable,
    require_success,
    run_command,
)

add_repo_and_build_to_syspath()
import gfa_compression as gfa_lib


DEFAULT_GPU_ROLLING_CHUNK_BYTES = 4096
DEFAULT_GPU_TRAVERSALS_PER_CHUNK = 16


def parse_args():
  parser = argparse.ArgumentParser(
      description="Explicit compression regression suite across CPU/GPU paths"
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
  parser.add_argument(
      "--gpu-rolling-chunk-bytes",
      type=int,
      default=int(
          os.environ.get(
              "GFA_GPU_ROLLING_CHUNK_BYTES",
              str(DEFAULT_GPU_ROLLING_CHUNK_BYTES),
          )
      ),
      help=(
          "GPU rolling compression chunk size in bytes. "
          "Defaults small to force chunking on regression fixtures."
      ),
  )
  parser.add_argument(
      "--gpu-traversals-per-chunk",
      type=int,
      default=int(
          os.environ.get(
              "GFA_GPU_TRAVERSALS_PER_CHUNK",
              str(DEFAULT_GPU_TRAVERSALS_PER_CHUNK),
          )
      ),
      help=(
          "GPU rolling decompression traversals per chunk. "
          "Defaults small to force chunking on regression fixtures."
      ),
  )
  return parser.parse_args()


def print_header(title: str):
  print(f"\n{'=' * 72}")
  print(title)
  print(f"{'=' * 72}")


def print_result(name: str, elapsed: float, details: str = ""):
  print(f"✅ PASS {name} ({elapsed:.3f}s)")
  if details:
    print(f"  {details}")


def cpu_compress(args):
  return gfa_lib.compress(
      args.gfa_file,
      num_rounds=args.rounds,
      freq_threshold=args.threshold,
      delta_round=args.delta_rounds,
      num_threads=args.threads,
  )


def gpu_compress_options_legacy():
  options = gfa_lib.GpuCompressionOptions()
  options.force_full_device_legacy = True
  return options


def gpu_compress_options_rolling(args):
  options = gfa_lib.GpuCompressionOptions()
  options.force_rolling_scheduler = True
  options.rolling_input_chunk_bytes = args.gpu_rolling_chunk_bytes
  return options


def gpu_decompress_options_legacy():
  options = gfa_lib.GpuDecompressionOptions()
  options.use_legacy_full_decompression = True
  return options


def gpu_decompress_options_rolling(args):
  options = gfa_lib.GpuDecompressionOptions()
  options.traversals_per_chunk = args.gpu_traversals_per_chunk
  return options


def with_temp_gfaz(prefix: str):
  handle = tempfile.NamedTemporaryFile(
      mode="wb", suffix=".gfaz", prefix=prefix, delete=False
  )
  path = Path(handle.name)
  handle.close()
  return path


def with_temp_gfa(prefix: str):
  handle = tempfile.NamedTemporaryFile(
      mode="w", suffix=".gfa", prefix=prefix, delete=False
  )
  path = Path(handle.name)
  handle.close()
  return path


def serialize_cpu_roundtrip(compressed, prefix: str):
  gfaz_path = with_temp_gfaz(prefix)
  try:
    gfa_lib.serialize(compressed, str(gfaz_path))
    return gfa_lib.deserialize(str(gfaz_path))
  finally:
    if gfaz_path.exists():
      gfaz_path.unlink()


def serialize_gpu_roundtrip(compressed, prefix: str):
  gfaz_path = with_temp_gfaz(prefix)
  try:
    gfa_lib.serialize_gpu(compressed, str(gfaz_path))
    return gfa_lib.deserialize_gpu(str(gfaz_path))
  finally:
    if gfaz_path.exists():
      gfaz_path.unlink()


def assert_host_graph_matches(original_graph, actual_graph, case_name: str):
  if not gfa_lib.verify_round_trip(original_graph, actual_graph):
    raise AssertionError(f"❌ {case_name} verification failed")


def assert_gpu_graph_matches(original_gpu_graph, actual_gpu_graph, case_name: str):
  if not gfa_lib.verify_gpu_round_trip(original_gpu_graph, actual_gpu_graph):
    raise AssertionError(f"❌ {case_name} verification failed")


def test_cpu_legacy_roundtrip(args, original_graph):
  start = time.perf_counter()
  compressed = cpu_compress(args)
  loaded = serialize_cpu_roundtrip(compressed, "reg_cpu_legacy_")
  decompressed = gfa_lib.decompress(loaded, num_threads=args.threads)
  assert_host_graph_matches(
      original_graph, decompressed, "CPU legacy round-trip"
  )
  elapsed = time.perf_counter() - start
  print_result("cpu_legacy_roundtrip", elapsed)


def test_cpu_direct_writer_roundtrip(args, original_graph):
  start = time.perf_counter()
  compressed = cpu_compress(args)
  gfaz_path = with_temp_gfaz("reg_cpu_direct_")
  out_path = with_temp_gfa("reg_cpu_direct_")

  try:
    gfa_lib.serialize(compressed, str(gfaz_path))
    loaded = gfa_lib.deserialize(str(gfaz_path))
    gfa_lib.write_gfa_from_compressed_data(
        loaded, str(out_path), num_threads=args.threads
    )
    reparsed = gfa_lib.parse(str(out_path))
    assert_host_graph_matches(
        original_graph, reparsed, "CPU direct-writer round-trip"
    )
  finally:
    if gfaz_path.exists():
      gfaz_path.unlink()
    if out_path.exists():
      out_path.unlink()

  elapsed = time.perf_counter() - start
  print_result("cpu_direct_writer_roundtrip", elapsed)


def test_gpu_legacy_roundtrip(args, original_graph):
  if args.skip_gpu or not has_gpu_bindings(gfa_lib):
    print("SKIP gpu_legacy_roundtrip (GPU bindings unavailable or --skip-gpu set)")
    return

  start = time.perf_counter()
  try:
    original_gpu_graph = gfa_lib.convert_to_gpu_layout(original_graph)
    compressed = gfa_lib.compress_gpu_graph(
        original_gpu_graph, args.rounds, gpu_compress_options_legacy()
    )
    loaded = serialize_gpu_roundtrip(compressed, "reg_gpu_legacy_")
    decompressed_gpu = gfa_lib.decompress_to_gpu_layout(
        loaded, gpu_decompress_options_legacy()
    )
    assert_gpu_graph_matches(
        original_gpu_graph, decompressed_gpu, "GPU legacy round-trip"
    )
  except Exception as exc:
    if is_gpu_runtime_unavailable(exc):
      print(f"SKIP gpu_legacy_roundtrip ({exc})")
      return
    raise

  elapsed = time.perf_counter() - start
  print_result("gpu_legacy_roundtrip", elapsed)


def test_gpu_rolling_device_graph_roundtrip(args, original_graph):
  if args.skip_gpu or not has_gpu_bindings(gfa_lib):
    print("SKIP gpu_rolling_device_graph_roundtrip (GPU bindings unavailable or --skip-gpu set)")
    return

  start = time.perf_counter()
  try:
    original_gpu_graph = gfa_lib.convert_to_gpu_layout(original_graph)
    compressed = gfa_lib.compress_gpu_graph(
        original_gpu_graph, args.rounds, gpu_compress_options_rolling(args)
    )
    loaded = serialize_gpu_roundtrip(compressed, "reg_gpu_rolling_")
    decompressed_gpu = gfa_lib.decompress_to_gpu_layout(
        loaded, gpu_decompress_options_rolling(args)
    )
    assert_gpu_graph_matches(
        original_gpu_graph, decompressed_gpu, "GPU rolling device-graph round-trip"
    )
  except Exception as exc:
    if is_gpu_runtime_unavailable(exc):
      print(f"SKIP gpu_rolling_device_graph_roundtrip ({exc})")
      return
    raise

  elapsed = time.perf_counter() - start
  print_result(
      "gpu_rolling_device_graph_roundtrip",
      elapsed,
      (
          f"chunk_bytes={args.gpu_rolling_chunk_bytes}, "
          f"traversals_per_chunk={args.gpu_traversals_per_chunk}"
      ),
  )


def test_gpu_rolling_host_graph_roundtrip(args, original_graph):
  if args.skip_gpu or not has_gpu_bindings(gfa_lib):
    print("SKIP gpu_rolling_host_graph_roundtrip (GPU bindings unavailable or --skip-gpu set)")
    return

  start = time.perf_counter()
  try:
    original_gpu_graph = gfa_lib.convert_to_gpu_layout(original_graph)
    compressed = gfa_lib.compress_gpu_graph(
        original_gpu_graph, args.rounds, gpu_compress_options_rolling(args)
    )
    loaded = serialize_gpu_roundtrip(compressed, "reg_gpu_host_")
    host_graph = gfa_lib.decompress_to_host_graph_gpu(
        loaded, gpu_decompress_options_rolling(args)
    )
    assert_host_graph_matches(
        original_graph,
        host_graph,
        "GPU rolling host-graph round-trip",
    )
  except Exception as exc:
    if is_gpu_runtime_unavailable(exc):
      print(f"SKIP gpu_rolling_host_graph_roundtrip ({exc})")
      return
    raise

  elapsed = time.perf_counter() - start
  print_result("gpu_rolling_host_graph_roundtrip", elapsed)


def test_gpu_direct_writer_roundtrip(args, original_graph, cli_path: Path):
  if args.skip_gpu or not has_gpu_bindings(gfa_lib):
    print("SKIP gpu_direct_writer_roundtrip (GPU bindings unavailable or --skip-gpu set)")
    return

  start = time.perf_counter()
  gfaz_path = with_temp_gfaz("reg_gpu_direct_")
  out_path = with_temp_gfa("reg_gpu_direct_")

  try:
    try:
      original_gpu_graph = gfa_lib.convert_to_gpu_layout(original_graph)
      compressed = gfa_lib.compress_gpu_graph(
          original_gpu_graph, args.rounds, gpu_compress_options_rolling(args)
      )
      gfa_lib.serialize_gpu(compressed, str(gfaz_path))
    except Exception as exc:
      if is_gpu_runtime_unavailable(exc):
        print(f"SKIP gpu_direct_writer_roundtrip ({exc})")
        return
      raise

    result = run_command(
        [
            str(cli_path),
            "decompress",
            "--gpu",
            str(gfaz_path),
            str(out_path),
        ]
    )
    try:
      require_success(result, "GPU direct-writer CLI decompression")
    except AssertionError as exc:
      if is_gpu_runtime_unavailable(exc):
        print(f"SKIP gpu_direct_writer_roundtrip ({exc})")
        return
      raise

    reparsed = gfa_lib.parse(str(out_path))
    assert_host_graph_matches(
        original_graph, reparsed, "GPU rolling direct-writer round-trip"
    )
  finally:
    if gfaz_path.exists():
      gfaz_path.unlink()
    if out_path.exists():
      out_path.unlink()

  elapsed = time.perf_counter() - start
  print_result("gpu_direct_writer_roundtrip", elapsed)


def test_cpu_container_to_gpu_rolling(args, original_graph):
  if args.skip_gpu or not has_gpu_bindings(gfa_lib):
    print("SKIP cpu_container_to_gpu_rolling (GPU bindings unavailable or --skip-gpu set)")
    return

  start = time.perf_counter()
  try:
    cpu_compressed = cpu_compress(args)
    gfaz_path = with_temp_gfaz("reg_cpu_to_gpu_roll_")
    try:
      gfa_lib.serialize(cpu_compressed, str(gfaz_path))
      loaded = gfa_lib.deserialize_gpu(str(gfaz_path))
      host_graph = gfa_lib.decompress_to_host_graph_gpu(
          loaded, gpu_decompress_options_rolling(args)
      )
      assert_host_graph_matches(
          original_graph,
          host_graph,
          "CPU container -> GPU rolling decompression",
      )
    finally:
      if gfaz_path.exists():
        gfaz_path.unlink()
  except Exception as exc:
    if is_gpu_runtime_unavailable(exc):
      print(f"SKIP cpu_container_to_gpu_rolling ({exc})")
      return
    raise

  elapsed = time.perf_counter() - start
  print_result("cpu_container_to_gpu_rolling", elapsed)


def test_cpu_container_to_gpu_legacy(args, original_graph):
  if args.skip_gpu or not has_gpu_bindings(gfa_lib):
    print("SKIP cpu_container_to_gpu_legacy (GPU bindings unavailable or --skip-gpu set)")
    return

  start = time.perf_counter()
  try:
    cpu_compressed = cpu_compress(args)
    gfaz_path = with_temp_gfaz("reg_cpu_to_gpu_legacy_")
    try:
      gfa_lib.serialize(cpu_compressed, str(gfaz_path))
      loaded = gfa_lib.deserialize_gpu(str(gfaz_path))
      host_graph = gfa_lib.decompress_to_host_graph_gpu(
          loaded, gpu_decompress_options_legacy()
      )
      assert_host_graph_matches(
          original_graph,
          host_graph,
          "CPU container -> GPU legacy decompression",
      )
    finally:
      if gfaz_path.exists():
        gfaz_path.unlink()
  except Exception as exc:
    if is_gpu_runtime_unavailable(exc):
      print(f"SKIP cpu_container_to_gpu_legacy ({exc})")
      return
    raise

  elapsed = time.perf_counter() - start
  print_result("cpu_container_to_gpu_legacy", elapsed)


def test_gpu_container_to_cpu(args, original_graph):
  if args.skip_gpu or not has_gpu_bindings(gfa_lib):
    print("SKIP gpu_container_to_cpu (GPU bindings unavailable or --skip-gpu set)")
    return

  start = time.perf_counter()
  try:
    original_gpu_graph = gfa_lib.convert_to_gpu_layout(original_graph)
    compressed = gfa_lib.compress_gpu_graph(
        original_gpu_graph, args.rounds, gpu_compress_options_rolling(args)
    )
    gfaz_path = with_temp_gfaz("reg_gpu_to_cpu_")
    try:
      gfa_lib.serialize_gpu(compressed, str(gfaz_path))
      loaded = gfa_lib.deserialize(str(gfaz_path))
      host_graph = gfa_lib.decompress(loaded, num_threads=args.threads)
      assert_host_graph_matches(
          original_graph,
          host_graph,
          "GPU container -> CPU decompression",
      )
    finally:
      if gfaz_path.exists():
        gfaz_path.unlink()
  except Exception as exc:
    if is_gpu_runtime_unavailable(exc):
      print(f"SKIP gpu_container_to_cpu ({exc})")
      return
    raise

  elapsed = time.perf_counter() - start
  print_result("gpu_container_to_cpu", elapsed)


def main():
  args = parse_args()
  cli_path = Path(args.gfaz)
  ensure_cli_exists(cli_path)

  print_header("Example Regression Suite")
  print(f"Fixture:               {args.gfa_file}")
  print(f"CLI:                   {cli_path}")
  print(f"Rounds:                {args.rounds}")
  print(f"Delta:                 {args.delta_rounds}")
  print(f"Threads:               {args.threads if args.threads > 0 else 'auto'}")
  print(f"GPU rolling bytes:     {args.gpu_rolling_chunk_bytes}")
  print(f"GPU traversals/chunk:  {args.gpu_traversals_per_chunk}")

  original_graph = gfa_lib.parse(args.gfa_file)
  print(f"Graph:                 {graph_summary(original_graph)}")

  started = time.perf_counter()
  test_cpu_legacy_roundtrip(args, original_graph)
  test_cpu_direct_writer_roundtrip(args, original_graph)
  test_gpu_legacy_roundtrip(args, original_graph)
  test_gpu_rolling_device_graph_roundtrip(args, original_graph)
  test_gpu_rolling_host_graph_roundtrip(args, original_graph)
  test_gpu_direct_writer_roundtrip(args, original_graph, cli_path)
  test_cpu_container_to_gpu_rolling(args, original_graph)
  test_cpu_container_to_gpu_legacy(args, original_graph)
  test_gpu_container_to_cpu(args, original_graph)
  total = time.perf_counter() - started

  print_header("Regression Summary")
  print(f"✅ PASS all requested regressions in {total:.3f}s")
  return 0


if __name__ == "__main__":
  sys.exit(main())
