#!/usr/bin/env python3
"""
Concordance: `gfaz similarity` vs `odgi similarity` (committed golden).

gfaz similarity reproduces odgi similarity's multiplicity-aware coverage
intersection. The output is in terms of group names + bp (no node ids), so the
two tools agree on values *exactly* (not just position-level), modulo row order:
gfaz emits pairs in (a,b) ascending order, odgi in hash-map order, so we compare
header-verbatim + sorted-body (see normalize_similarity). odgi drops W-lines on
`odgi build`, so the fixture is path-only. Goldens hold odgi's normalized output
(regenerate with scripts/gen_golden.py); this test never invokes odgi.

Grouping correspondence: gfaz -p (per path/walk) == odgi default (no -D);
gfaz -S (per sample) == odgi -D '#' -p 1.
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
    normalize_similarity,
    strip_golden_comments,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures"
GFA = FIXTURE_DIR / "similarity_fixture.gfa"


def compress(cli: Path) -> Path:
  handle = tempfile.NamedTemporaryFile(
      mode="w", suffix=".gfaz", prefix="gfaz_simcon_", delete=False)
  out = Path(handle.name)
  handle.close()
  require_success(run_command([str(cli), "compress", str(GFA), str(out)]),
                  "compress similarity fixture")
  return out


def check(cli: Path, gfaz: Path, extra, golden_name: str):
  result = run_command(
      [str(cli), "similarity", "-i", str(gfaz), "-t", "2"] + extra)
  require_success(result, f"gfaz similarity {' '.join(extra) or '(per-path)'}")
  actual = normalize_similarity(result.stdout)
  expected = strip_golden_comments(load_golden(golden_name))
  if actual != expected:
    raise AssertionError(
        f"similarity concordance mismatch for {golden_name}.\n"
        f"--- expected (odgi golden) ---\n{expected}\n"
        f"--- actual (gfaz) ---\n{actual}")


def main():
  cli = Path(CLI_PATH)
  ensure_cli_exists(cli)
  gfaz = compress(cli)
  try:
    check(cli, gfaz, ["-p"], "similarity_fixture.per_path.golden")
    check(cli, gfaz, ["-S"], "similarity_fixture.sample.golden")
    check(cli, gfaz, ["-S", "-d"], "similarity_fixture.sample.distances.golden")
    check(cli, gfaz, ["-S", "-a"], "similarity_fixture.sample.all.golden")
    print("✅ PASS similarity_vs_odgi")
  finally:
    if gfaz.exists():
      gfaz.unlink()


if __name__ == "__main__":
  run_main(main)
