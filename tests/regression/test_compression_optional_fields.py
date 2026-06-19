#!/usr/bin/env python3
"""
Round-trip coverage for GFA features that the other fixtures miss: J-lines,
C-lines, and S-line optional fields of every type (i / f / A / Z / B,i / B,f).

This is a pure-CLI round-trip (compress -> decompress) with no Python bindings,
so it runs even when the bindings are not built. gfaz reconstructs segment names
as dense numeric IDs and may reorder lines by type, so the fixture already uses
numeric IDs and the comparison is line-set based (sorted), not positional.

Compression uses default settings (delta encoding on). This also guards a
regression that previously dropped segments on small / monotonic graphs under
delta encoding: the rule-ID floor was seeded only from the (small) delta-domain
maximum, truncating the stored segment table below the real node count. The fix
floors it at the node count (compression_workflow.cpp::apply_delta_transform).
Large-graph default-path round-trip is covered by
tests/regression/test_compression_regression.py (bindings required).
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
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "compressor_optional_fields_fixture.gfa"


def line_set(path: Path):
  return sorted(
      ln for ln in path.read_text().splitlines() if ln.strip() != "")


def test_round_trip(cli: Path):
  with tempfile.TemporaryDirectory() as d:
    gfaz = Path(d) / "opt.gfaz"
    out = Path(d) / "opt.out.gfa"
    require_success(
        run_command([str(cli), "compress", str(FIXTURE), str(gfaz)]),
        "compress optional-fields fixture")
    require_success(
        run_command([str(cli), "decompress", str(gfaz), str(out)]),
        "decompress optional-fields fixture")

    want, got = line_set(FIXTURE), line_set(out)
    if want != got:
      missing = [ln for ln in want if ln not in got]
      extra = [ln for ln in got if ln not in want]
      raise AssertionError(
          "optional-fields round-trip mismatch.\n"
          f"missing from output:\n" + "\n".join(missing) + "\n"
          f"unexpected in output:\n" + "\n".join(extra))


def test_small_monotonic_graphs(cli: Path):
  # Direct guard for the delta-domain segment-drop regression: a single
  # monotonic path 1+,2+,...,n+ delta-encodes to all-1s, which previously
  # pushed the rule-ID floor below the node count and truncated the segment
  # table. Every segment (and its sequence) must survive under default settings.
  for n in (2, 3, 4, 6, 8, 12, 20, 50):
    with tempfile.TemporaryDirectory() as d:
      src = Path(d) / f"n{n}.gfa"
      gfaz = Path(d) / f"n{n}.gfaz"
      out = Path(d) / f"n{n}.out.gfa"
      lines = ["H\tVN:Z:1.1"]
      lines += [f"S\t{i}\tAC" for i in range(1, n + 1)]
      lines.append("P\tp\t" + ",".join(f"{i}+" for i in range(1, n + 1)) + "\t*")
      src.write_text("\n".join(lines) + "\n")
      require_success(run_command([str(cli), "compress", str(src), str(gfaz)]),
                      f"compress n={n}")
      require_success(run_command([str(cli), "decompress", str(gfaz), str(out)]),
                      f"decompress n={n}")
      out_segs = sum(1 for ln in out.read_text().splitlines()
                     if ln.startswith("S\t"))
      if out_segs != n:
        raise AssertionError(
            f"n={n}: expected {n} segments after round-trip, got {out_segs} "
            "(delta-domain segment-drop regression)")


def main():
  cli = Path(CLI_PATH)
  ensure_cli_exists(cli)
  test_round_trip(cli)
  test_small_monotonic_graphs(cli)
  print("✅ PASS compression_optional_fields")


if __name__ == "__main__":
  main()
