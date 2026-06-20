#!/usr/bin/env python3
"""
Concordance: `gfaz stats` vs `odgi stats` (committed golden).

The graph-dimension summary (#length nodes edges paths steps) and the A/C/G/T
base content are pure aggregates, independent of node identity, so the two tools
agree byte-for-byte. Goldens hold odgi's output (regenerate with
scripts/gen_golden.py); this test never invokes odgi.

Correspondence: gfaz stats (default) == odgi stats -S ; gfaz stats -b == odgi
stats -b. odgi drops W-lines on `odgi build`, so the fixture is path-only.
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
    run_main,
)
from tests.concordance.concordance_utils import (
    load_golden,
    normalize_stats,
    strip_golden_comments,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
GFA = REPO_ROOT / "tests" / "fixtures" / "similarity_fixture.gfa"


def compress(cli: Path) -> Path:
  handle = tempfile.NamedTemporaryFile(
      mode="w", suffix=".gfaz", prefix="gfaz_statscon_", delete=False)
  out = Path(handle.name)
  handle.close()
  require_success(run_command([str(cli), "compress", str(GFA), str(out)]),
                  "compress stats fixture")
  return out


def check(cli: Path, gfaz: Path, extra, golden_name: str):
  result = run_command([str(cli), "stats", "-i", str(gfaz)] + extra)
  require_success(result, f"gfaz stats {' '.join(extra) or '(summary)'}")
  actual = normalize_stats(result.stdout)
  expected = strip_golden_comments(load_golden(golden_name))
  if actual != expected:
    raise AssertionError(
        f"stats concordance mismatch for {golden_name}.\n"
        f"--- expected (odgi golden) ---\n{expected}\n"
        f"--- actual (gfaz) ---\n{actual}")


def main():
  cli = Path(CLI_PATH)
  ensure_cli_exists(cli)
  gfaz = compress(cli)
  try:
    check(cli, gfaz, [], "stats_fixture.summarize.golden")
    check(cli, gfaz, ["-b"], "stats_fixture.base.golden")
    print("✅ PASS stats_vs_odgi")
  finally:
    if gfaz.exists():
      gfaz.unlink()


if __name__ == "__main__":
  run_main(main)
