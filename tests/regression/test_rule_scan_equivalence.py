#!/usr/bin/env python3
"""
Both ways of finding repeated 2-mers must give the same rules.

Each grammar round turns every 2-mer that occurs at least twice into a rule.
The generator either collects 2-mers through per-thread sets (which drop
repeats before the sort) or, once most 2-mers are distinct, writes all of them
to the sort buffer. It picks per round; `GFAZ_RULE_SCAN=sets|all` pins one.
`sets-legacy` pins the two-sets-per-thread fallback that the set route uses
when node IDs do not fit the single-probe table.
The .gfaz must be byte-identical for both pinned routes and the default.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.regression.regression_utils import (
    CLI_PATH,
    REPO_ROOT,
    ensure_cli_exists,
)
from tests.regression.test_low_memory_compress import (
    FIXTURE_DIR,
    FIXTURES,
    make_synthetic_graph,
)


def compress(cli: Path, src: Path, dst: Path, extra, scan):
  env = dict(os.environ)
  env.pop("GFAZ_RULE_SCAN", None)
  if scan:
    env["GFAZ_RULE_SCAN"] = scan
  result = subprocess.run(
      [str(cli), "compress", *extra, str(src), str(dst)],
      cwd=str(REPO_ROOT), text=True, capture_output=True, check=False, env=env)
  if result.returncode != 0:
    raise AssertionError(
        f"compress {src.name} (scan={scan}) failed:\n{result.stderr}")
  return dst.read_bytes()


def check(cli: Path, src: Path, workdir: Path):
  # A small budget makes the scans read traversals back from the spill file.
  budget = ["--memory-budget", "8K", "--tmp-dir", str(workdir)]
  for delta in ("0", "1", "2"):
    for threads in ("1", "4"):
      extra = ["-d", delta, "-j", threads]
      reference = compress(cli, src, workdir / "sets.gfaz", extra, "sets")
      for scan in ("sets-legacy", "all", None):
        for spill in ([], budget):
          got = compress(cli, src, workdir / "other.gfaz", extra + spill, scan)
          if got != reference:
            raise AssertionError(
                f"{src.name}: rule scan '{scan or 'default'}' changed the "
                f"output (delta={delta}, threads={threads}, "
                f"budget={'yes' if spill else 'no'})")


def main():
  cli = Path(CLI_PATH)
  ensure_cli_exists(cli)
  with tempfile.TemporaryDirectory() as dd:
    workdir = Path(dd)
    for fixture in FIXTURES:
      check(cli, FIXTURE_DIR / fixture, workdir)
    for numeric in (True, False):
      src = workdir / f"synthetic_{numeric}.gfa"
      make_synthetic_graph(src, numeric_names=numeric)
      check(cli, src, workdir)
  print("✅ PASS rule_scan_equivalence")


if __name__ == "__main__":
  main()
