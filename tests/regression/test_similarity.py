#!/usr/bin/env python3
"""
Regression suite for the `gfaz similarity` subcommand (all-vs-all group
similarity matrix).

Fixture `similarity_fixture.gfa` (path-only; node lengths 1:4 2:2 3:3 4:1 5:2
6:4 7:3) is hand-verifiable. Per-sample coverage (cov = node_length x visits,
multiplicity-aware, matching odgi):
  sampleA = A#0(1,2,3) + A#1(1,2,3,2) -> {1:8, 2:6, 3:6}            L=20
  sampleB = B#0(1,3,4) + B#1(1,2,5)   -> {1:8, 2:2, 3:3, 4:1, 5:2}  L=16
  sampleC = C#0(6,7)                  -> {6:4, 7:3}                  L=7
Intersection I(A,B)=min per node = 8+2+3 = 13; I(A,C)=I(B,C)=0.
  jaccard(A,B)  = 13/(20+16-13) = 0.565217
  cosine(A,B)   = 13/sqrt(20*16) = 0.726722
  dice(A,B)     = 26/36 = 0.722222 ; estimated.identity == dice always.
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
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "similarity_fixture.gfa"

SIM_HEADER = ("group.a\tgroup.b\tgroup.a.length\tgroup.b.length\tintersection\t"
              "jaccard.similarity\tcosine.similarity\tdice.similarity\t"
              "estimated.identity")
DIST_HEADER = ("group.a\tgroup.b\tgroup.a.length\tgroup.b.length\tintersection\t"
               "jaccard.distance\tcosine.distance\tdice.distance\t"
               "estimated.difference.rate\teuclidean.distance\t"
               "manhattan.distance")


def temp_gfaz() -> Path:
  h = tempfile.NamedTemporaryFile(suffix=".gfaz", prefix="gfaz_sim_",
                                  delete=False)
  p = Path(h.name)
  h.close()
  return p


def compress(cli: Path) -> Path:
  out = temp_gfaz()
  require_success(run_command([str(cli), "compress", str(FIXTURE), str(out)]),
                  "compress similarity fixture")
  return out


def lines(text: str):
  return [ln for ln in text.splitlines() if ln.strip() != ""]


def test_sample_values(cli: Path, gfaz: Path):
  """-S output is exact and in deterministic (a,b ascending) order."""
  r = run_command([str(cli), "similarity", "-i", str(gfaz), "-S", "-t", "3"])
  require_success(r, "similarity -S")
  expected = [
      SIM_HEADER,
      "sampleA\tsampleA\t20\t20\t20\t1.000000\t1.000000\t1.000000\t1.000000",
      "sampleA\tsampleB\t20\t16\t13\t0.565217\t0.726722\t0.722222\t0.722222",
      "sampleB\tsampleA\t16\t20\t13\t0.565217\t0.726722\t0.722222\t0.722222",
      "sampleB\tsampleB\t16\t16\t16\t1.000000\t1.000000\t1.000000\t1.000000",
      "sampleC\tsampleC\t7\t7\t7\t1.000000\t1.000000\t1.000000\t1.000000",
  ]
  if lines(r.stdout) != expected:
    raise AssertionError(
        "similarity -S mismatch.\n--- expected ---\n" + "\n".join(expected) +
        "\n--- actual ---\n" + "\n".join(lines(r.stdout)))


def test_distances_header_and_zero(cli: Path, gfaz: Path):
  r = run_command([str(cli), "similarity", "-i", str(gfaz), "-S", "-d", "-t",
                   "2"])
  require_success(r, "similarity -S -d")
  if lines(r.stdout)[0] != DIST_HEADER:
    raise AssertionError("similarity -d header mismatch:\n" + lines(r.stdout)[0])


def test_all_pairs_includes_zero(cli: Path, gfaz: Path):
  """-a emits zero-intersection pairs (sampleA vs disjoint sampleC); the sparse
  default omits them."""
  sparse = run_command([str(cli), "similarity", "-i", str(gfaz), "-S", "-t",
                        "2"])
  require_success(sparse, "similarity -S")
  allp = run_command([str(cli), "similarity", "-i", str(gfaz), "-S", "-a", "-t",
                      "2"])
  require_success(allp, "similarity -S -a")
  zero = ("sampleA\tsampleC\t20\t7\t0\t"
          "0.000000\t0.000000\t0.000000\t0.000000")
  body_sparse = lines(sparse.stdout)[1:]
  body_all = lines(allp.stdout)[1:]
  if any(ln.startswith("sampleA\tsampleC\t") for ln in body_sparse):
    raise AssertionError("sparse default must omit zero-intersection pairs")
  if zero not in body_all:
    raise AssertionError(
        "similarity -a must emit the zero-intersection sampleA/sampleC pair.\n"
        + "\n".join(body_all))
  # N=3 groups -> 9 ordered pairs with -a.
  if len(body_all) != 9:
    raise AssertionError(f"-a expected 9 pairs, got {len(body_all)}")


def test_per_path_grouping(cli: Path, gfaz: Path):
  """-p uses full path/walk names as groups (matches odgi's default)."""
  r = run_command([str(cli), "similarity", "-i", str(gfaz), "-p", "-t", "2"])
  require_success(r, "similarity -p")
  body = lines(r.stdout)[1:]
  names = {ln.split("\t")[0] for ln in body}
  for want in ("sampleA#0#chr1", "sampleC#0#chr2"):
    if want not in names:
      raise AssertionError(f"-p should group by full path name; missing {want}")


def test_determinism(cli: Path, gfaz: Path):
  """Output is thread-count invariant."""
  a = run_command([str(cli), "similarity", "-i", str(gfaz), "-S", "-t", "1"])
  b = run_command([str(cli), "similarity", "-i", str(gfaz), "-S", "-t", "4"])
  require_success(a, "similarity -t 1")
  require_success(b, "similarity -t 4")
  if a.stdout != b.stdout:
    raise AssertionError("similarity output differs between -t 1 and -t 4")


def test_grouping_exclusivity(cli: Path, gfaz: Path):
  r = run_command([str(cli), "similarity", "-i", str(gfaz), "-S", "-H"])
  if r.returncode == 0:
    raise AssertionError("similarity must reject combining -S and -H")


def main():
  cli = Path(CLI_PATH)
  ensure_cli_exists(cli)
  gfaz = compress(cli)
  try:
    test_sample_values(cli, gfaz)
    test_distances_header_and_zero(cli, gfaz)
    test_all_pairs_includes_zero(cli, gfaz)
    test_per_path_grouping(cli, gfaz)
    test_determinism(cli, gfaz)
    test_grouping_exclusivity(cli, gfaz)
    print("✅ PASS similarity_regressions")
  finally:
    if gfaz.exists():
      gfaz.unlink()


if __name__ == "__main__":
  main()
