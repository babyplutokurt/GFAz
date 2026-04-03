import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = REPO_ROOT / "build"
CLI_PATH = BUILD_DIR / "bin" / "gfaz"


def add_repo_and_build_to_syspath():
  sys.path.insert(0, str(REPO_ROOT))
  sys.path.insert(0, str(BUILD_DIR))


def format_size_mb(num_bytes: int) -> str:
  return f"{num_bytes / (1024 * 1024):.2f} MB"


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


def assert_stats_output(output: str, expect_ratio: bool):
  required = ["Time:", "Input:", "Output:"]
  if expect_ratio:
    required.append("Ratio:")
  missing = [token for token in required if token not in output]
  if missing:
    raise AssertionError(
        f"Missing stats fields {missing} in CLI output:\n{output.strip()}"
    )


def calculate_gpu_compressed_size(compressed):
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
    for attr in [
        "int_values_zstd_nvcomp",
        "float_values_zstd_nvcomp",
        "char_values_zstd_nvcomp",
        "strings_zstd_nvcomp",
        "string_lengths_zstd_nvcomp",
        "b_subtypes_zstd_nvcomp",
        "b_lengths_zstd_nvcomp",
        "b_concat_bytes_zstd_nvcomp",
    ]:
      block = getattr(col, attr, None)
      if block and block.payload:
        total += len(block.payload)

  for attr in [
      "walk_sample_ids_zstd_nvcomp",
      "walk_sample_id_lengths_zstd_nvcomp",
      "walk_hap_indices_zstd_nvcomp",
      "walk_seq_ids_zstd_nvcomp",
      "walk_seq_id_lengths_zstd_nvcomp",
      "walk_seq_starts_zstd_nvcomp",
      "walk_seq_ends_zstd_nvcomp",
      "jump_from_ids_zstd_nvcomp",
      "jump_to_ids_zstd_nvcomp",
      "jump_from_orients_zstd_nvcomp",
      "jump_to_orients_zstd_nvcomp",
      "jump_distances_zstd_nvcomp",
      "jump_distance_lengths_zstd_nvcomp",
      "jump_rest_fields_zstd_nvcomp",
      "jump_rest_lengths_zstd_nvcomp",
      "containment_container_ids_zstd_nvcomp",
      "containment_contained_ids_zstd_nvcomp",
      "containment_container_orients_zstd_nvcomp",
      "containment_contained_orients_zstd_nvcomp",
      "containment_positions_zstd_nvcomp",
      "containment_overlaps_zstd_nvcomp",
      "containment_overlap_lengths_zstd_nvcomp",
      "containment_rest_fields_zstd_nvcomp",
      "containment_rest_lengths_zstd_nvcomp",
      "node_names_zstd_nvcomp",
      "node_name_lengths_zstd_nvcomp",
  ]:
    block = getattr(compressed, attr, None)
    if block and block.payload:
      total += len(block.payload)

  if hasattr(compressed, "header_line") and compressed.header_line:
    total += len(compressed.header_line)

  return total


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
      "nvcomp",
      "std::bad_alloc",
      "operation not supported on this OS",
  ]
  return any(marker in text for marker in markers)


def graph_summary(graph) -> str:
  parts = [
      f"segments={len(graph.node_sequences) - 1}",
      f"paths={len(graph.paths)}",
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
