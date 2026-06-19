#!/usr/bin/env python3
"""
Locks the shared PanSN grouping contract used by the whole compute engine.

path_group_key / parse_pansn_path_name live in compute/traversal_query.cpp and
are used identically by growth, pav, and deconstruct, so every module groups
haplotypes the same way. This test pins the parser's behavior on the cases that
distinguish the grouping modes:
  - 2-field PanSN  "GRCh38#chr1"        -> (sample=GRCh38, hap=chr1)
  - 3-field PanSN  "HG002#1#chr1"       -> (sample, hap, seq)
  - no '#'         "chm13"              -> sample only
  - coord suffix   "chm13:0-100"        -> coords stripped, groups with "chm13"

We assert the number of haplotype groups N (= the largest k in the growth
curve) per mode, which is exactly what the grouping key determines. Concordance
with the real panacus binary is covered separately in
tests/concordance/test_growth_vs_panacus.py; this suite is hermetic.
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

GFA = (
    "H\tVN:Z:1.1\n"
    "S\t1\tA\n"
    "S\t2\tC\n"
    "S\t3\tG\n"
    "S\t4\tT\n"
    "P\tGRCh38#chr1\t1+,2+\t*\n"
    "P\tGRCh38#chr2\t1+,3+\t*\n"
    "P\tHG002#1#chr1\t1+,2+\t*\n"
    "P\tHG002#1#chr2\t1+,3+\t*\n"
    "P\tHG002#2#chr1\t1+,4+\t*\n"
    "P\tchm13\t2+,3+\t*\n"
    "P\tchm13:0-100\t2+,4+\t*\n"
)

# Expected number of haplotype groups per grouping mode:
#   path           : every P-line is its own group                       -> 7
#   sample-hap-seq : GRCh38#chr1, GRCh38#chr2, HG002#1#chr1,
#                    HG002#1#chr2, HG002#2#chr1, chm13 (x2 coalesce)      -> 6
#   sample-hap     : GRCh38#chr1, GRCh38#chr2, HG002#1, HG002#2, chm13#   -> 5
#   sample         : GRCh38, HG002, chm13                                 -> 3
EXPECTED_GROUPS = {
    "path": 7,
    "sample-hap-seq": 6,
    "sample-hap": 5,
    "sample": 3,
}


def num_groups(stdout: str) -> int:
  """The growth curve runs k=1..N, so N (the group count) is the max k."""
  ks = []
  in_body = False
  for line in stdout.splitlines():
    if not line or line.startswith("#"):
      continue
    if line.split("\t")[0] == "k":
      in_body = True
      continue
    if not in_body:
      continue
    ks.append(int(line.split("\t")[0]))
  if not ks:
    raise AssertionError("no growth curve rows parsed from output")
  return max(ks)


def test_grouping_counts(cli: Path):
  with tempfile.TemporaryDirectory() as dd:
    d = Path(dd)
    gfa = d / "pansn.gfa"
    gfa.write_text(GFA)
    gfaz = d / "pansn.gfaz"
    require_success(run_command([str(cli), "compress", str(gfa), str(gfaz)]),
                    "compress pansn fixture")

    for mode, expected in EXPECTED_GROUPS.items():
      result = run_command([str(cli), "growth", "-G", mode, str(gfaz)])
      require_success(result, f"growth -G {mode}")
      got = num_groups(result.stdout)
      if got != expected:
        raise AssertionError(
            f"-G {mode}: expected {expected} haplotype groups, got {got}")


def main():
  cli = Path(CLI_PATH)
  ensure_cli_exists(cli)
  test_grouping_counts(cli)
  print("✅ PASS pansn_grouping")


if __name__ == "__main__":
  main()
