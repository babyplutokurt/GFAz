#!/usr/bin/env python3
"""
CPU compress -> decompress round-trip matrix, driven entirely through the `gfaz`
CLI (no Python bindings), so it runs under any interpreter and on builds without
the bindings module.

Covers, across several feature-rich fixtures:
  - both CPU decompress paths: default streaming direct-writer, and --legacy
    (CompressedData -> GfaGraph -> write_gfa);
  - a compression-parameter sweep over --delta (0..3), --rounds, --threshold,
    all of which affect the on-disk format. The --delta axis specifically guards
    the delta-domain id-space handling (see the segment-drop fix in
    compression_workflow.cpp::apply_delta_transform).

The binding-based structural round-trip suite
(tests/regression/test_compression_regression.py) is the deeper check when the
bindings are built; this CLI suite is the always-runnable floor.

Comparison: gfaz canonicalizes segment names to dense numeric IDs and reorders
lines by type, so equivalence is checked as a sorted, blank-stripped line set.
Fixtures therefore use dense numeric segment IDs (the canonical form), making the
round-trip exact modulo line order.
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
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures"

# Feature-rich fixtures with dense numeric segment IDs:
#   cli_fixture            P + W + L, optional LN fields, CIGAR overlap
#   deconstruct_fixture    11 segments, 5 P-lines (SNP/indel topology)
#   optional_fields        J + C lines, S optional fields i/f/A/Z/B
#   growth_fixture_walks   W-lines only (walk decode path)
#   roundtrip_orientations reverse-orientation P/W, cyclic/duplicate visits
#   roundtrip_btypes       all B-array subtypes (c/C/s/S/i/I/f), lowercase seq
FIXTURES = [
    "cli_fixture.gfa",
    "deconstruct_fixture.gfa",
    "compressor_optional_fields_fixture.gfa",
    "growth_fixture_walks.gfa",
    "roundtrip_orientations_fixture.gfa",
    "roundtrip_btypes_fixture.gfa",
]

# (delta, rounds, threshold) combinations. Defaults are (1, 8, 2).
PARAM_SETS = [
    (0, 8, 2),   # delta off
    (1, 8, 2),   # default
    (2, 8, 2),   # multi-round delta
    (3, 8, 2),   # deeper delta
    (1, 1, 2),   # minimal grammar
    (1, 8, 1),   # lower frequency threshold
]

DECOMPRESS_MODES = [
    ("default", []),         # streaming direct-writer
    ("legacy", ["--legacy"]),
]


def normal_form(path: Path):
  return sorted(ln for ln in path.read_text().splitlines() if ln.strip() != "")


def round_trip(cli: Path, fixture: Path, params, dec_flags):
  delta, rounds, threshold = params
  with tempfile.TemporaryDirectory() as d:
    gfaz = Path(d) / "c.gfaz"
    out = Path(d) / "c.gfa"
    require_success(
        run_command([str(cli), "compress",
                     "-d", str(delta), "-r", str(rounds), "-t", str(threshold),
                     str(fixture), str(gfaz)]),
        f"compress {fixture.name} d={delta} r={rounds} t={threshold}")
    require_success(
        run_command([str(cli), "decompress", *dec_flags, str(gfaz), str(out)]),
        f"decompress {fixture.name} {dec_flags or 'default'}")
    if normal_form(fixture) != normal_form(out):
      raise AssertionError(
          f"round-trip mismatch: {fixture.name} d={delta} r={rounds} "
          f"t={threshold} decompress={dec_flags or 'default'}")


def main():
  cli = Path(CLI_PATH)
  ensure_cli_exists(cli)
  n = 0
  for name in FIXTURES:
    fixture = FIXTURE_DIR / name
    for params in PARAM_SETS:
      for _mode, flags in DECOMPRESS_MODES:
        round_trip(cli, fixture, params, flags)
        n += 1
  print(f"✅ PASS cpu_roundtrip ({n} compress/decompress combinations)")


if __name__ == "__main__":
  main()
