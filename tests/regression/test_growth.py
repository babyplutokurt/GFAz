#!/usr/bin/env python3
"""
Regression suite for the `gfaz growth` subcommand (pangenome growth curve,
the Panacus-equivalent count=node statistic).

Coverage:
  - A self-contained, hand-verified expected curve on `growth_fixture.gfa`
    (path-only) for every grouping mode. No external tools involved.
  - The same curve from the W-line variant `growth_fixture_walks.gfa`, which
    exercises gfaz's walk-decode path (the curve is identical because the node
    coverage is identical).

Concordance with the real `panacus` binary lives in
`tests/concordance/test_growth_vs_panacus.py`; this file only guards gfaz's own
output so it runs with no external dependencies.

Fixture topology (shared with deconstruct_fixture.gfa): 11 nodes, 5 haplotypes
  ref          1 2 3 4 5 6 7 8 9
  HG001#0#chr1 1 10 3 4 5 6 7 8 9
  HG001#1#chr1 1 2 3 4 5 6 7 8 9
  HG002#0#chr1 1 2 3 4 6 7 8 9      (drops node 5)
  HG002#1#chr1 1 2 3 4 5 6 7 11 8 9 (adds node 11)

Expected count=node growth curve E[distinct nodes | k random haplotypes],
verified against gfaz and (floored) against panacus:
  k=1 -> 9.0   k=2 -> 9.8   k=3 -> 10.2   k=4 -> 10.6   k=5 -> 11.0
"""

import sys
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
GROWTH_FIXTURE = FIXTURE_DIR / "growth_fixture.gfa"
GROWTH_WALKS_FIXTURE = FIXTURE_DIR / "growth_fixture_walks.gfa"

EXPECTED_CURVE = {1: 9.0, 2: 9.8, 3: 10.2, 4: 10.6, 5: 11.0}


def compress(cli: Path, fixture: Path) -> Path:
  import tempfile

  handle = tempfile.NamedTemporaryFile(
      mode="w", suffix=".gfaz", prefix="gfaz_growth_", delete=False
  )
  out = Path(handle.name)
  handle.close()
  require_success(
      run_command([str(cli), "compress", str(fixture), str(out)]),
      f"compress {fixture.name}",
  )
  return out


def parse_growth(stdout: str) -> dict:
  """Parse a gfaz growth table into {k: float}, skipping '#' comments and the
  'k\tgrowth' header row."""
  curve = {}
  in_body = False
  for line in stdout.splitlines():
    if not line or line.startswith("#"):
      continue
    if line.startswith("k\t") or line.split("\t")[0] == "k":
      in_body = True
      continue
    if not in_body:
      continue
    fields = line.split("\t")
    curve[int(fields[0])] = float(fields[1])
  return curve


def assert_curve(actual: dict, expected: dict, step: str):
  if set(actual) != set(expected):
    raise AssertionError(
        f"{step}: k set mismatch. expected {sorted(expected)}, got {sorted(actual)}"
    )
  for k in expected:
    if abs(actual[k] - expected[k]) > 1e-4:
      raise AssertionError(
          f"{step}: growth[{k}]={actual[k]} != expected {expected[k]}"
      )


def test_grouping_modes(cli: Path, gfaz: Path):
  # All four modes collapse to the same per-haplotype curve here because every
  # path is its own (sample, hap, seq) and no two paths share a (sample, hap)
  # except via distinct seq — except `sample` mode, which groups HG001/HG002.
  # We assert the per-line curve for path/sample-hap-seq/sample-hap (5 groups)
  # and a separate hand-checked curve for `sample` (3 groups).
  for mode in ("path", "sample-hap-seq", "sample-hap"):
    result = run_command([str(cli), "growth", "-G", mode, str(gfaz)])
    require_success(result, f"growth -G {mode}")
    assert_curve(parse_growth(result.stdout), EXPECTED_CURVE, f"growth -G {mode}")


def test_sample_mode(cli: Path, gfaz: Path):
  # `sample` groups into {ref, HG001, HG002} = 3 groups. A group covers a node
  # if any of its haplotypes does.
  #   ref:   {1..9}                  (9 nodes)
  #   HG001: {1,2,3,4,5,6,7,8,9,10}  (10 nodes; union of its two haplotypes)
  #   HG002: {1,2,3,4,5,6,7,8,9,11}  (10 nodes)
  # Hand-computed E[distinct | k of 3 groups]:
  #   k=1 -> (9+10+10)/3 = 9.6667
  #   k=2 -> ({ref,HG001}=10 + {ref,HG002}=10 + {HG001,HG002}=11)/3 = 10.3333
  #   k=3 -> 11 (all nodes 1..11)
  expected = {1: 29.0 / 3.0, 2: 31.0 / 3.0, 3: 11.0}
  result = run_command([str(cli), "growth", "-G", "sample", str(gfaz)])
  require_success(result, "growth -G sample")
  curve = parse_growth(result.stdout)
  if set(curve) != {1, 2, 3}:
    raise AssertionError(f"growth -G sample: expected k=1..3, got {sorted(curve)}")
  for k, want in expected.items():
    if abs(curve[k] - want) > 1e-3:
      raise AssertionError(
          f"growth -G sample: growth[{k}]={curve[k]} != expected {want}"
      )


def test_walks_variant(cli: Path, gfaz_walks: Path):
  # Same topology expressed as W-lines: per-line curve must be identical.
  result = run_command([str(cli), "growth", "-G", "path", str(gfaz_walks)])
  require_success(result, "growth walks -G path")
  assert_curve(parse_growth(result.stdout), EXPECTED_CURVE, "growth walks -G path")


def main():
  cli = Path(CLI_PATH)
  ensure_cli_exists(cli)

  gfaz = compress(cli, GROWTH_FIXTURE)
  gfaz_walks = compress(cli, GROWTH_WALKS_FIXTURE)
  try:
    test_grouping_modes(cli, gfaz)
    test_sample_mode(cli, gfaz)
    test_walks_variant(cli, gfaz_walks)
    print("✅ PASS growth_regressions")
  finally:
    for p in (gfaz, gfaz_walks):
      if p.exists():
        p.unlink()


if __name__ == "__main__":
  main()
