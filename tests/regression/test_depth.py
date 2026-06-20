#!/usr/bin/env python3
"""
Regression suite for the `gfaz depth` subcommand (node coverage depth).

Fixture `similarity_fixture.gfa` (path-only; node lengths 1:4 2:2 3:3 4:1 5:2
6:4 7:3). Visits across the 5 paths:
  node 1: A#0,A#1,B#0,B#1            -> depth 4, distinct paths 4
  node 2: A#0,A#1(x2),B#1            -> depth 4, distinct paths 3
  node 3: A#0,A#1,B#0                -> depth 3, distinct paths 3
  nodes 4,5,6,7: 1 each              -> depth 1, distinct paths 1
Summary: node.count=7 graph.length=19 step.count=15 path.length=43
  mean.node.depth = 15/7 = 2.14286 ; mean.graph.depth = 43/19 = 2.26316
"""

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
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "similarity_fixture.gfa"


def temp_gfaz() -> Path:
  h = tempfile.NamedTemporaryFile(suffix=".gfaz", prefix="gfaz_depth_",
                                  delete=False)
  p = Path(h.name)
  h.close()
  return p


def compress(cli: Path) -> Path:
  out = temp_gfaz()
  require_success(run_command([str(cli), "compress", str(FIXTURE), str(out)]),
                  "compress depth fixture")
  return out


def lines(text: str):
  return [ln for ln in text.splitlines() if ln.strip() != ""]


def test_summary(cli: Path, gfaz: Path):
  r = run_command([str(cli), "depth", "-i", str(gfaz), "-t", "3"])
  require_success(r, "depth")
  expected = [
      "#node.count\tgraph.length\tstep.count\tpath.length\tmean.node.depth\t"
      "mean.graph.depth",
      "7\t19\t15\t43\t2.14286\t2.26316",
  ]
  if lines(r.stdout) != expected:
    raise AssertionError(
        "depth summary mismatch.\n--- expected ---\n" + "\n".join(expected) +
        "\n--- actual ---\n" + "\n".join(lines(r.stdout)))


def test_summary_flag_is_default(cli: Path, gfaz: Path):
  a = run_command([str(cli), "depth", "-i", str(gfaz)])
  b = run_command([str(cli), "depth", "-i", str(gfaz), "-S"])
  require_success(a, "depth")
  require_success(b, "depth -S")
  if a.stdout != b.stdout:
    raise AssertionError("depth and depth -S must produce identical output")


def test_per_node(cli: Path, gfaz: Path):
  r = run_command([str(cli), "depth", "-i", str(gfaz), "-d", "-t", "2"])
  require_success(r, "depth -d")
  expected = [
      "#node.id\tdepth\tdepth.uniq",
      "1\t4\t4",
      "2\t4\t3",
      "3\t3\t3",
      "4\t1\t1",
      "5\t1\t1",
      "6\t1\t1",
      "7\t1\t1",
  ]
  if lines(r.stdout) != expected:
    raise AssertionError(
        "depth -d mismatch.\n--- expected ---\n" + "\n".join(expected) +
        "\n--- actual ---\n" + "\n".join(lines(r.stdout)))


def test_determinism(cli: Path, gfaz: Path):
  """Per-node table is thread-count invariant."""
  a = run_command([str(cli), "depth", "-i", str(gfaz), "-d", "-t", "1"])
  b = run_command([str(cli), "depth", "-i", str(gfaz), "-d", "-t", "4"])
  require_success(a, "depth -d -t 1")
  require_success(b, "depth -d -t 4")
  if a.stdout != b.stdout:
    raise AssertionError("depth -d output differs between -t 1 and -t 4")


def main():
  cli = Path(CLI_PATH)
  ensure_cli_exists(cli)
  gfaz = compress(cli)
  try:
    test_summary(cli, gfaz)
    test_summary_flag_is_default(cli, gfaz)
    test_per_node(cli, gfaz)
    test_determinism(cli, gfaz)
    print("✅ PASS depth_regressions")
  finally:
    if gfaz.exists():
      gfaz.unlink()


if __name__ == "__main__":
  main()
