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


def check_binary_consistency(cli: Path, gfaz: Path, threshold: float):
  """-B output must equal the -M float matrix thresholded at the same value.
  Self-contained (no odgi): guards the binary-emission path."""
  floats = run_command(
      [str(cli), "pav", "-i", str(gfaz), "-b", str(BED), "-M", "-t", "2"])
  require_success(floats, "gfaz pav -M")
  binary = run_command(
      [str(cli), "pav", "-i", str(gfaz), "-b", str(BED), "-M",
       "-B", str(threshold), "-t", "2"])
  require_success(binary, "gfaz pav -M -B")

  flines = floats.stdout.splitlines()
  blines = binary.stdout.splitlines()
  if flines[0] != blines[0] or len(flines) != len(blines):
    raise AssertionError("pav -B: header/row count differs from -M")
  for fl, bl in zip(flines[1:], blines[1:]):
    ff, bf = fl.split("\t"), bl.split("\t")
    # first 4 columns are chrom/start/end/name; the rest are values
    if ff[:4] != bf[:4]:
      raise AssertionError("pav -B: range columns differ from -M")
    for fv, bv in zip(ff[4:], bf[4:]):
      expected = "1" if float(fv) >= threshold else "0"
      if bv != expected:
        raise AssertionError(
            f"pav -B: value {bv} != threshold({fv}>={threshold})={expected}")


def main():
  cli = Path(CLI_PATH)
  ensure_cli_exists(cli)
  gfaz = compress(cli)
  try:
    check(cli, gfaz, [], "pav_pathonly_concordance.long.golden")
    check(cli, gfaz, ["-M"], "pav_pathonly_concordance.matrix.golden")
    check(cli, gfaz, ["-S"], "pav_pathonly_concordance.sample.long.golden")
    # NB: the -S *matrix* form is intentionally not compared to odgi: values
    # agree but the sample column order differs (odgi sorts sample names, gfaz
    # is ref-first/insertion order) -- a presentation choice, not a correctness
    # difference. The -S long form (above) validates the grouped values.
    check_binary_consistency(cli, gfaz, 0.95)
    print("✅ PASS pav_vs_odgi")
  finally:
    if gfaz.exists():
      gfaz.unlink()


if __name__ == "__main__":
  run_main(main)
