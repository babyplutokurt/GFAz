#!/usr/bin/env python3
"""
Large-scale deconstruct concordance: `gfaz deconstruct` vs `vg deconstruct` on a
whole human chromosome.

This backs the headline claim that gfaz's default mode reproduces
`vg deconstruct` at ~99.99% position concordance. Unlike the hermetic toy-fixture
test (test_deconstruct_vs_vg.py), this one runs `vg` live on a multi-hundred-MB
GFA and is therefore OPT-IN and SKIPPED by default: it requires
GFAZ_LARGE_CONCORDANCE to be set, plus the chrY inputs and the vg binary.

Enable with, e.g.:
    GFAZ_LARGE_CONCORDANCE=1 python3 tests/concordance/test_deconstruct_vs_vg_large.py

Inputs (env-overridable):
    GFAZ_CHRY_GFAZ   compressed graph for gfaz   (default ./chrY.gfaz)
    GFAZ_CHRY_GFA    same graph as GFA for vg    (default the smoothed chrY GFA)
    GFAZ_CHRY_REF    reference path name         (default grch38#chrY)
    GFAZ_VG_BIN      vg binary                   (default /home/kurty/Release/vg/bin/vg)

Predicate (position level, matching the documented metric): the (CHROM, POS) set
overlap must be >= 0.99 and the record-count delta <= 0.5%. REF/ALT spelling and
GT are intentionally not required to match exactly at this scale.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.regression.regression_utils import (
    CLI_PATH,
    SkipTest,
    ensure_cli_exists,
    require_success,
    run_command,
    run_main,
)
from tests.concordance.concordance_utils import VG_BIN, tool_or_skip, vcf_pos_keys

REPO_ROOT = Path(__file__).resolve().parents[2]

CHRY_GFAZ = Path(os.environ.get("GFAZ_CHRY_GFAZ", str(REPO_ROOT / "chrY.gfaz")))
CHRY_GFA = Path(os.environ.get(
    "GFAZ_CHRY_GFA",
    "/home/kurty/data/chrY.pan.fa.a2fb268.4030258.bc221f9.smooth.gfa"))
CHRY_REF = os.environ.get("GFAZ_CHRY_REF", "grch38#chrY")

MIN_POS_OVERLAP = 0.99
MAX_COUNT_DELTA = 0.005


def main():
  if not os.environ.get("GFAZ_LARGE_CONCORDANCE"):
    raise SkipTest("set GFAZ_LARGE_CONCORDANCE=1 to run the chrY-scale "
                   "deconstruct concordance test")
  cli = Path(CLI_PATH)
  ensure_cli_exists(cli)
  vg = tool_or_skip(VG_BIN, "vg")
  if not CHRY_GFAZ.exists():
    raise SkipTest(f"chrY .gfaz not found at {CHRY_GFAZ} (set GFAZ_CHRY_GFAZ)")
  if not CHRY_GFA.exists():
    raise SkipTest(f"chrY .gfa not found at {CHRY_GFA} (set GFAZ_CHRY_GFA)")

  t0 = time.time()
  vg_proc = subprocess.run(
      [str(vg), "deconstruct", "-p", CHRY_REF, "-t", "8", str(CHRY_GFA)],
      capture_output=True, text=True)
  if vg_proc.returncode != 0:
    raise AssertionError(f"vg deconstruct failed:\n{vg_proc.stderr[:2000]}")
  vg_time = time.time() - t0

  gfaz_res = run_command(
      [str(cli), "deconstruct", "-i", str(CHRY_GFAZ), "-r", CHRY_REF, "-S",
       "-t", "8"])
  require_success(gfaz_res, "gfaz deconstruct chrY")

  vg_pos = vcf_pos_keys(vg_proc.stdout)
  gfaz_pos = vcf_pos_keys(gfaz_res.stdout)
  if not vg_pos:
    raise AssertionError("vg produced no records")
  shared = vg_pos & gfaz_pos
  overlap = len(shared) / max(len(vg_pos), len(gfaz_pos))
  count_delta = abs(len(vg_pos) - len(gfaz_pos)) / len(vg_pos)

  print(f"  vg records={len(vg_pos)} ({vg_time:.1f}s)  gfaz records={len(gfaz_pos)}")
  print(f"  POS overlap={overlap:.5f}  count delta={count_delta:.5f}")

  if overlap < MIN_POS_OVERLAP:
    raise AssertionError(
        f"POS overlap {overlap:.5f} < {MIN_POS_OVERLAP}")
  if count_delta > MAX_COUNT_DELTA:
    raise AssertionError(
        f"record-count delta {count_delta:.5f} > {MAX_COUNT_DELTA}")
  print("✅ PASS deconstruct_vs_vg_large")


if __name__ == "__main__":
  run_main(main)
