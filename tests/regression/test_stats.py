#!/usr/bin/env python3
"""
Regression suite for the `gfaz stats` subcommand (graph dimension summary).

Fixture `similarity_fixture.gfa` (path-only; node lengths 1:4 2:2 3:3 4:1 5:2
6:4 7:3; 7 L-lines; 5 P-lines). Hand-verifiable:
  length = 4+2+3+1+2+4+3 = 19 ; nodes = 7 ; edges = 7 ; paths = 5
  steps  = |A#0|3 + |A#1|4 + |B#0|3 + |B#1|3 + |C#0|2 = 15
  base content over the segments: A=8 C=3 G=7 T=1
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
  h = tempfile.NamedTemporaryFile(suffix=".gfaz", prefix="gfaz_stats_",
                                  delete=False)
  p = Path(h.name)
  h.close()
  return p


def compress(cli: Path) -> Path:
  out = temp_gfaz()
  require_success(run_command([str(cli), "compress", str(FIXTURE), str(out)]),
                  "compress stats fixture")
  return out


def lines(text: str):
  return [ln for ln in text.splitlines() if ln.strip() != ""]


def test_summary(cli: Path, gfaz: Path):
  r = run_command([str(cli), "stats", "-i", str(gfaz)])
  require_success(r, "stats")
  expected = [
      "#length\tnodes\tedges\tpaths\tsteps",
      "19\t7\t7\t5\t15",
  ]
  if lines(r.stdout) != expected:
    raise AssertionError(
        "stats summary mismatch.\n--- expected ---\n" + "\n".join(expected) +
        "\n--- actual ---\n" + "\n".join(lines(r.stdout)))


def test_summary_flag_is_default(cli: Path, gfaz: Path):
  """-S is the default; passing it explicitly changes nothing."""
  a = run_command([str(cli), "stats", "-i", str(gfaz)])
  b = run_command([str(cli), "stats", "-i", str(gfaz), "-S"])
  require_success(a, "stats")
  require_success(b, "stats -S")
  if a.stdout != b.stdout:
    raise AssertionError("stats and stats -S must produce identical output")


def test_base_content(cli: Path, gfaz: Path):
  r = run_command([str(cli), "stats", "-i", str(gfaz), "-b"])
  require_success(r, "stats -b")
  expected = ["A\t8", "C\t3", "G\t7", "T\t1"]
  if lines(r.stdout) != expected:
    raise AssertionError(
        "stats -b mismatch.\n--- expected ---\n" + "\n".join(expected) +
        "\n--- actual ---\n" + "\n".join(lines(r.stdout)))


def test_positional_input(cli: Path, gfaz: Path):
  """The .gfaz may be given positionally instead of with -i."""
  r = run_command([str(cli), "stats", str(gfaz)])
  require_success(r, "stats <positional>")
  if lines(r.stdout)[0] != "#length\tnodes\tedges\tpaths\tsteps":
    raise AssertionError("positional input form failed")


def main():
  cli = Path(CLI_PATH)
  ensure_cli_exists(cli)
  gfaz = compress(cli)
  try:
    test_summary(cli, gfaz)
    test_summary_flag_is_default(cli, gfaz)
    test_base_content(cli, gfaz)
    test_positional_input(cli, gfaz)
    print("✅ PASS stats_regressions")
  finally:
    if gfaz.exists():
      gfaz.unlink()


if __name__ == "__main__":
  main()
