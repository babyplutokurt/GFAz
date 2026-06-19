#!/usr/bin/env python3
"""
Concordance: `gfaz growth` vs `panacus growth` (committed golden).

panacus reports the integer floor of the expected-distinct-nodes value that gfaz
prints as a float: panacus[k] == floor(gfaz[k]). Grouping modes are paired
gfaz -G sample-hap-seq <-> panacus (default), -G sample-hap <-> -H,
-G sample <-> -S. The golden holds panacus's parsed {k: int} curve; this test
reads the golden and runs only gfaz.
"""

import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.regression.regression_utils import (
    CLI_PATH,
    ensure_cli_exists,
    require_success,
    run_command,
    run_main,
)
from tests.concordance.concordance_utils import (
    load_golden,
    parse_growth_gfaz,
    parse_growth_panacus,
    strip_golden_comments,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
GFA = REPO_ROOT / "tests" / "fixtures" / "growth_fixture.gfa"

MODES = (
    ("sample-hap-seq", "growth_fixture.sample-hap-seq.golden"),
    ("sample-hap", "growth_fixture.sample-hap.golden"),
    ("sample", "growth_fixture.sample.golden"),
)


def compress(cli: Path) -> Path:
  handle = tempfile.NamedTemporaryFile(
      mode="w", suffix=".gfaz", prefix="gfaz_growthcon_", delete=False)
  out = Path(handle.name)
  handle.close()
  require_success(run_command([str(cli), "compress", str(GFA), str(out)]),
                  "compress growth fixture")
  return out


def check(cli: Path, gfaz: Path, mode: str, golden_name: str):
  result = run_command([str(cli), "growth", "-G", mode, str(gfaz)])
  require_success(result, f"gfaz growth -G {mode}")
  gfaz_curve = parse_growth_gfaz(result.stdout)
  golden = parse_growth_panacus(strip_golden_comments(load_golden(golden_name)))
  if set(gfaz_curve) != set(golden):
    raise AssertionError(
        f"{mode}: k set mismatch gfaz={sorted(gfaz_curve)} panacus={sorted(golden)}")
  for k in sorted(gfaz_curve):
    if math.floor(gfaz_curve[k]) != golden[k]:
      raise AssertionError(
          f"{mode}: floor(gfaz[{k}]={gfaz_curve[k]})={math.floor(gfaz_curve[k])}"
          f" != panacus {golden[k]}")


def main():
  cli = Path(CLI_PATH)
  ensure_cli_exists(cli)
  gfaz = compress(cli)
  try:
    for mode, golden_name in MODES:
      check(cli, gfaz, mode, golden_name)
    print("✅ PASS growth_vs_panacus")
  finally:
    if gfaz.exists():
      gfaz.unlink()


if __name__ == "__main__":
  run_main(main)
