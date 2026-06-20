import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = REPO_ROOT / "build"
CLI_PATH = BUILD_DIR / "bin" / "gfaz"

# Exit code the runner (tests/run_all.py) interprets as "skipped, not failed".
# Matches the automake convention so CTest/CI can treat it specially too.
SKIP_EXIT_CODE = 77


class SkipTest(Exception):
  """Raised by a test to signal it cannot run (missing tool, dataset, or
  golden file). The runner reports SKIP rather than FAIL."""


def run_main(main_fn):
  """Entry-point wrapper: run a test's main(), translating SkipTest into the
  runner's SKIP exit code (77) instead of a hard failure."""
  try:
    main_fn()
  except SkipTest as exc:
    print(f"⏭️  SKIP: {exc}")
    sys.exit(SKIP_EXIT_CODE)


def add_repo_and_build_to_syspath():
  sys.path.insert(0, str(REPO_ROOT))
  sys.path.insert(0, str(BUILD_DIR))


def run_command(cmd, cwd=None):
  return subprocess.run(
      cmd,
      cwd=str(cwd or REPO_ROOT),
      text=True,
      capture_output=True,
      check=False,
  )


def require_success(result, step_name: str):
  if result.returncode == 0:
    return
  message = [
      f"{step_name} failed with exit code {result.returncode}",
      "STDOUT:",
      result.stdout.strip() or "<empty>",
      "STDERR:",
      result.stderr.strip() or "<empty>",
  ]
  raise AssertionError("\n".join(message))


def has_gpu_bindings(gfa_lib) -> bool:
  return all(
      hasattr(gfa_lib, attr)
      for attr in [
          "convert_to_gpu_layout",
          "convert_from_gpu_layout",
          "compress_gpu_graph",
          "decompress_to_gpu_layout",
          "verify_gpu_round_trip",
      ]
  )


def is_gpu_runtime_unavailable(exc: Exception) -> bool:
  text = str(exc)
  markers = [
      "cudaError",
      "CUDA",
      "std::bad_alloc",
      "operation not supported on this OS",
  ]
  return any(marker in text for marker in markers)


def graph_summary(graph) -> str:
  parts = [
      f"segments={len(graph.segments.node_sequences) - 1}",
      f"paths={len(graph.paths_data.traversals)}",
      f"walks={len(graph.walks.walks)}",
  ]
  if hasattr(graph, "links") and hasattr(graph.links, "from_nodes"):
    parts.append(f"links={len(graph.links.from_nodes)}")
  return ", ".join(parts)


def ensure_cli_exists(cli_path: Path):
  if cli_path.exists():
    return
  raise FileNotFoundError(
      f"CLI binary not found at {cli_path}. Build the project before running regressions."
  )


def file_size(path: Path) -> int:
  return os.path.getsize(path)
