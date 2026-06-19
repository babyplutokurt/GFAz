#!/usr/bin/env python3
"""
Degenerate inputs and error handling.

Two kinds of coverage:
  - degenerate-but-valid graphs round-trip cleanly (segments-only, walks-only,
    header-only);
  - clearly-invalid invocations fail with a non-zero exit instead of producing
    garbage (missing/malformed BED, unresolved pav/deconstruct reference).
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


def norm(path: Path):
  return sorted(ln for ln in path.read_text().splitlines() if ln.strip() != "")


def assert_failure(result, label: str):
  if result.returncode == 0:
    raise AssertionError(f"{label}: expected non-zero exit, got 0")


def round_trip_ok(cli: Path, gfa: Path, d: Path, label: str):
  gfaz = d / "x.gfaz"
  out = d / "x.gfa"
  require_success(run_command([str(cli), "compress", str(gfa), str(gfaz)]),
                  f"compress {label}")
  require_success(run_command([str(cli), "decompress", str(gfaz), str(out)]),
                  f"decompress {label}")
  if norm(gfa) != norm(out):
    raise AssertionError(f"{label}: round-trip mismatch")


def test_degenerate_round_trips(cli: Path):
  with tempfile.TemporaryDirectory() as dd:
    d = Path(dd)
    seg = d / "segonly.gfa"
    seg.write_text("H\tVN:Z:1.1\nS\t1\tAAAA\nS\t2\tCCCC\nL\t1\t+\t2\t+\t0M\n")
    round_trip_ok(cli, seg, d, "segments-only")

    walk = d / "walkonly.gfa"
    walk.write_text("H\tVN:Z:1.1\nS\t1\tAAAA\nS\t2\tCCCC\nW\ts1\t0\tc1\t0\t8\t>1>2\n")
    round_trip_ok(cli, walk, d, "walks-only")

    # Header-only: must at least compress+decompress without error.
    hdr = d / "hdr.gfa"
    hdr.write_text("H\tVN:Z:1.1\n")
    gfaz = d / "h.gfaz"
    require_success(run_command([str(cli), "compress", str(hdr), str(gfaz)]),
                    "compress header-only")
    require_success(run_command([str(cli), "decompress", str(gfaz), str(d / 'h.gfa')]),
                    "decompress header-only")


def test_error_paths(cli: Path):
  with tempfile.TemporaryDirectory() as dd:
    d = Path(dd)
    gfa = d / "g.gfa"
    gfa.write_text("H\tVN:Z:1.1\nS\t1\tAAAA\nS\t2\tCCCC\nP\tref\t1+,2+\t*\n")
    gfaz = d / "g.gfaz"
    require_success(run_command([str(cli), "compress", str(gfa), str(gfaz)]),
                    "compress")

    assert_failure(
        run_command([str(cli), "pav", "-i", str(gfaz), "-b",
                     str(d / "nope.bed")]),
        "pav missing BED file")

    bad_ref = d / "badref.bed"
    bad_ref.write_text("nosuchpath\t0\t5\tx\n")
    assert_failure(
        run_command([str(cli), "pav", "-i", str(gfaz), "-b", str(bad_ref)]),
        "pav unresolved BED reference")

    malformed = d / "malformed.bed"
    malformed.write_text("ref\tnotanumber\t5\tx\n")
    assert_failure(
        run_command([str(cli), "pav", "-i", str(gfaz), "-b", str(malformed)]),
        "pav malformed BED line")

    assert_failure(
        run_command([str(cli), "deconstruct", "-i", str(gfaz), "-r", "nosuch"]),
        "deconstruct unresolved reference")


def main():
  cli = Path(CLI_PATH)
  ensure_cli_exists(cli)
  test_degenerate_round_trips(cli)
  test_error_paths(cli)
  print("✅ PASS degenerate_inputs")


if __name__ == "__main__":
  main()
