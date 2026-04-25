#!/usr/bin/env python3
"""
CLI command regression suite for extract-path, extract-walk, and add-haplotypes.
"""

import argparse
import sys
import tempfile
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.regression.regression_utils import (
    CLI_PATH,
    ensure_cli_exists,
    require_success,
    run_command,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures"
DEFAULT_INPUT_GFA = FIXTURE_DIR / "cli_fixture.gfa"
DEFAULT_APPEND_PATHS = FIXTURE_DIR / "append_paths.gfa"
DEFAULT_APPEND_WALKS = FIXTURE_DIR / "append_walks.gfa"
DEFAULT_APPEND_MIXED = FIXTURE_DIR / "append_mixed.gfa"
DEFAULT_PAV_BED = FIXTURE_DIR / "pav_fixture.bed"
DEFAULT_PAV_PATH_ONLY_BED = FIXTURE_DIR / "pav_path_only_fixture.bed"


def parse_args():
  parser = argparse.ArgumentParser(
      description="CLI regression suite for extract and add-haplotypes commands"
  )
  parser.add_argument(
      "--gfaz",
      default=str(CLI_PATH),
      help="Path to gfaz CLI binary",
  )
  parser.add_argument(
      "--fixture",
      default=str(DEFAULT_INPUT_GFA),
      help="Base GFA fixture with paths and walks",
  )
  parser.add_argument(
      "--append-paths",
      default=str(DEFAULT_APPEND_PATHS),
      help="Path-only append fixture",
  )
  parser.add_argument(
      "--append-walks",
      default=str(DEFAULT_APPEND_WALKS),
      help="Walk-only append fixture",
  )
  parser.add_argument(
      "--append-mixed",
      default=str(DEFAULT_APPEND_MIXED),
      help="Mixed P/W append fixture expected to fail",
  )
  parser.add_argument(
      "--pav-bed",
      default=str(DEFAULT_PAV_BED),
      help="BED fixture for pav command",
  )
  parser.add_argument(
      "--pav-path-only-bed",
      default=str(DEFAULT_PAV_PATH_ONLY_BED),
      help="BED fixture that exercises the pav node-to-groups plan",
  )
  return parser.parse_args()


def temp_file(suffix: str, prefix: str) -> Path:
  handle = tempfile.NamedTemporaryFile(
      mode="w", suffix=suffix, prefix=prefix, delete=False
  )
  path = Path(handle.name)
  handle.close()
  return path


def compress_fixture(cli_path: Path, fixture: Path) -> Path:
  gfaz_path = temp_file(".gfaz", "gfaz_cli_cmds_")
  result = run_command(
      [str(cli_path), "compress", str(fixture), str(gfaz_path)]
  )
  require_success(result, "compress fixture")
  return gfaz_path


def assert_stdout(result, expected: str, step_name: str):
  actual = result.stdout
  if actual != expected:
    raise AssertionError(
        f"{step_name} produced unexpected stdout.\n"
        f"Expected:\n{expected}\n"
        f"Actual:\n{actual}\n"
        f"STDERR:\n{result.stderr}"
    )


def test_extract_path(cli_path: Path, gfaz_path: Path):
  result = run_command(
      [str(cli_path), "extract-path", str(gfaz_path), "pathB", "pathA"]
  )
  require_success(result, "extract-path")
  expected = (
      "P\tpathB\t1+,3-,4+\t*\n"
      "P\tpathA\t1+,2+,3+\t2M,1M\n"
  )
  assert_stdout(result, expected, "extract-path")


def test_extract_walk(cli_path: Path, gfaz_path: Path):
  result = run_command(
      [
          str(cli_path),
          "extract-walk",
          str(gfaz_path),
          "sampleB",
          "1",
          "chr2",
          "*",
          "*",
      ]
  )
  require_success(result, "extract-walk")
  expected = "W\tsampleB\t1\tchr2\t*\t*\t>1<3>4\n"
  assert_stdout(result, expected, "extract-walk")


def test_add_haplotypes_paths(cli_path: Path, gfaz_path: Path, append_paths: Path):
  updated_path = temp_file(".gfaz", "gfaz_cli_paths_updated_")
  try:
    result = run_command(
        [
            str(cli_path),
            "add-haplotypes",
            str(gfaz_path),
            str(append_paths),
            str(updated_path),
        ]
    )
    require_success(result, "add-haplotypes paths")

    extract_result = run_command(
        [str(cli_path), "extract-path", str(updated_path), "pathA", "pathC"]
    )
    require_success(extract_result, "extract-path after path append")
    expected = (
        "P\tpathA\t1+,2+,3+\t2M,1M\n"
        "P\tpathC\t2+,4+,3-\t*\n"
    )
    assert_stdout(extract_result, expected, "extract-path after path append")
  finally:
    if updated_path.exists():
      updated_path.unlink()


def test_add_haplotypes_walks(cli_path: Path, gfaz_path: Path, append_walks: Path):
  updated_path = temp_file(".gfaz", "gfaz_cli_walks_updated_")
  try:
    result = run_command(
        [
            str(cli_path),
            "add-haplotypes",
            str(gfaz_path),
            str(append_walks),
            str(updated_path),
        ]
    )
    require_success(result, "add-haplotypes walks")

    extract_result = run_command(
        [
            str(cli_path),
            "extract-walk",
            str(updated_path),
            "sampleC",
            "0",
            "chr3",
            "*",
            "*",
        ]
    )
    require_success(extract_result, "extract-walk after walk append")
    expected = "W\tsampleC\t0\tchr3\t*\t*\t>2>4>1\n"
    assert_stdout(extract_result, expected, "extract-walk after walk append")
  finally:
    if updated_path.exists():
      updated_path.unlink()


def test_add_haplotypes_rejects_mixed(
    cli_path: Path, gfaz_path: Path, append_mixed: Path
):
  updated_path = temp_file(".gfaz", "gfaz_cli_mixed_updated_")
  try:
    result = run_command(
        [
            str(cli_path),
            "add-haplotypes",
            str(gfaz_path),
            str(append_mixed),
            str(updated_path),
        ]
    )
    if result.returncode == 0:
      raise AssertionError(
          "add-haplotypes unexpectedly accepted a mixed P/W append file"
      )
  finally:
    if updated_path.exists():
      updated_path.unlink()


def test_pav_matrix(cli_path: Path, gfaz_path: Path, pav_bed: Path):
  result = run_command(
      [
          str(cli_path),
          "pav",
          "-i",
          str(gfaz_path),
          "-b",
          str(pav_bed),
          "-M",
          "-t",
          "2",
      ]
  )
  require_success(result, "pav matrix")
  expected = (
      "chrom\tstart\tend\tname\tpathA\tpathB\tsampleA#0#chr1\tsampleB#1#chr2\n"
      "pathA\t0\t6\tpathA_all\t1\t0.83333\t1\t0.83333\n"
      "pathB\t0\t6\tpathB_all\t0.83333\t1\t0.83333\t1\n"
  )
  assert_stdout(result, expected, "pav matrix")


def test_pav_matrix_grouped_path_only_plan(
    cli_path: Path, gfaz_path: Path, pav_bed: Path
):
  result = run_command(
      [
          str(cli_path),
          "pav",
          "-i",
          str(gfaz_path),
          "-b",
          str(pav_bed),
          "-S",
          "-M",
          "-t",
          "2",
      ]
  )
  require_success(result, "pav grouped matrix")
  expected = (
      "chrom\tstart\tend\tname\tpathA\tpathB\tsampleA\tsampleB\n"
      "pathA\t0\t6\tpathA_all\t1\t0.83333\t1\t0.83333\n"
      "pathB\t0\t6\tpathB_all\t0.83333\t1\t0.83333\t1\n"
      "pathA\t0\t2\tpathA_first_node\t1\t1\t1\t1\n"
      "pathB\t4\t6\tpathB_tail\t0.75\t1\t0.75\t1\n"
  )
  assert_stdout(result, expected, "pav grouped matrix")


def main():
  args = parse_args()
  cli_path = Path(args.gfaz)
  fixture = Path(args.fixture)
  append_paths = Path(args.append_paths)
  append_walks = Path(args.append_walks)
  append_mixed = Path(args.append_mixed)
  pav_bed = Path(args.pav_bed)
  pav_path_only_bed = Path(args.pav_path_only_bed)

  ensure_cli_exists(cli_path)

  gfaz_path = compress_fixture(cli_path, fixture)
  try:
    test_extract_path(cli_path, gfaz_path)
    test_extract_walk(cli_path, gfaz_path)
    test_pav_matrix(cli_path, gfaz_path, pav_bed)
    test_pav_matrix_grouped_path_only_plan(
        cli_path, gfaz_path, pav_path_only_bed
    )
    test_add_haplotypes_paths(cli_path, gfaz_path, append_paths)
    test_add_haplotypes_walks(cli_path, gfaz_path, append_walks)
    test_add_haplotypes_rejects_mixed(cli_path, gfaz_path, append_mixed)
    print("✅ PASS cli_command_regressions")
  finally:
    if gfaz_path.exists():
      gfaz_path.unlink()


if __name__ == "__main__":
  main()
