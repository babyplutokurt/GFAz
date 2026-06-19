#!/usr/bin/env python3
"""
Concordance: `gfaz pav` vs `odgi pav` (committed golden).

gfaz pav is meant to reproduce odgi pav's node-coverage semantics. odgi drops
W-lines on `odgi build`, so the fixture is path-only. gfaz emits rows in BED
order and odgi in path-id order, so comparison is header-verbatim + sorted-body
(see normalize_pav). The golden holds odgi's normalized output; this test only
reads the golden and never invokes odgi.
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
    normalize_pav,
    strip_golden_comments,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures"
GFA = FIXTURE_DIR / "pav_pathonly_concordance.gfa"
BED = FIXTURE_DIR / "pav_pathonly_concordance.bed"


def compress(cli: Path) -> Path:
  handle = tempfile.NamedTemporaryFile(
      mode="w", suffix=".gfaz", prefix="gfaz_pavcon_", delete=False)
  out = Path(handle.name)
  handle.close()
  require_success(run_command([str(cli), "compress", str(GFA), str(out)]),
                  "compress pav fixture")
  return out


def check(cli: Path, gfaz: Path, extra, golden_name: str):
  result = run_command(
      [str(cli), "pav", "-i", str(gfaz), "-b", str(BED), "-t", "2"] + extra)
  require_success(result, f"gfaz pav {' '.join(extra) or '(long)'}")
  actual = normalize_pav(result.stdout)
  expected = strip_golden_comments(load_golden(golden_name))
  if actual != expected:
    raise AssertionError(
        f"pav concordance mismatch for {golden_name}.\n"
        f"--- expected (odgi golden) ---\n{expected}\n"
        f"--- actual (gfaz) ---\n{actual}")


def main():
  cli = Path(CLI_PATH)
  ensure_cli_exists(cli)
  gfaz = compress(cli)
  try:
    check(cli, gfaz, [], "pav_pathonly_concordance.long.golden")
    check(cli, gfaz, ["-M"], "pav_pathonly_concordance.matrix.golden")
    print("✅ PASS pav_vs_odgi")
  finally:
    if gfaz.exists():
      gfaz.unlink()


if __name__ == "__main__":
  run_main(main)
