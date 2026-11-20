#!/usr/bin/env python3
"""
Regression suite for `gfaz locate` and `gfaz index`.

`locate` is checked against a brute-force scan of the decompressed GFA on the
generated multi-block graph of test_extract_region.py, with its segments named
1..N, by other numbers, and by strings. The saved index must
give the same answers as an index built for the query, be picked up from
<input>.pidx, and be rejected for a different container.
"""

import shutil
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
from tests.regression.test_extract_region import (
    FIXTURES,
    SEGMENT_NAMINGS,
    Graph,
    make_synthetic,
)


def expected_locate(graph: Graph, nodes):
  rows = ["#node\tpath\tstrand\tstep\tstart\tend"]
  for wanted in nodes:
    for hap in graph.haps:
      pos = hap["offset"]
      for step, (node, orient) in enumerate(hap["steps"]):
        length = graph.lengths[node]
        if node == wanted:
          strand = "+" if orient in "+>" else "-"
          rows.append(f"{node}\t{hap['contig']}\t{strand}\t{step}\t{pos}\t"
                      f"{pos + length}")
        pos += length
  return rows


def check(cli: str, tmp: Path, name_of):
  """Run every check on the generated graph, its n-th segment named
  name_of(n)."""
  gfa = tmp / "synthetic.gfa"
  make_synthetic(gfa, name_of)
  gfaz = tmp / "synthetic.gfaz"
  full = tmp / "full.gfa"
  require_success(run_command([cli, "compress", str(gfa), str(gfaz)]),
                  "compress")
  require_success(run_command([cli, "decompress", str(gfaz), str(full)]),
                  "decompress")
  graph = Graph(full.read_text())

  nodes = [name_of(n) for n in (1, 17, 250, 1499, 2999, 3000, 250)]
  expected = expected_locate(graph, nodes)
  node_args = [arg for name in nodes for arg in ("-n", name)]

  def locate(extra, threads):
    result = run_command([cli, "locate", "-i", str(gfaz), "-t", str(threads)]
                         + node_args + extra)
    require_success(result, "locate")
    return result.stdout.splitlines()

  for threads in (1, 4):
    if locate([], threads) != expected:
      raise AssertionError(f"locate (threads={threads}) differs from the "
                           "brute-force scan")

  # A saved index, at several block sizes, answers the same.
  for block_steps in (1, 5, 4096):
    index = tmp / f"b{block_steps}.pidx"
    require_success(run_command([cli, "index", "-i", str(gfaz), "-o",
                                 str(index), "-b", str(block_steps)]),
                    "index")
    if locate(["-x", str(index)], 2) != expected:
      raise AssertionError(f"locate with a saved index (block_steps="
                           f"{block_steps}) differs")

  # <input>.pidx is picked up without -x, by locate and by extract-region.
  require_success(run_command([cli, "index", "-i", str(gfaz), "-b", "3"]),
                  "index default path")
  if not Path(str(gfaz) + ".pidx").exists():
    raise AssertionError("index did not write <input>.pidx")
  if locate([], 2) != expected:
    raise AssertionError("locate with the default index differs")
  region = ["extract-region", "-i", str(gfaz), "-r", "p0#1#chrS:2000-2400"]
  with_index = run_command([cli] + region)
  built = run_command([cli] + region + ["--index-block-steps", "4096"])
  require_success(with_index, "extract-region with index")
  require_success(built, "extract-region building index")
  if with_index.stdout != built.stdout:
    raise AssertionError("extract-region differs with a saved index")

  # An index of another container is refused.
  other = tmp / "other.gfaz"
  require_success(run_command([cli, "compress",
                               str(FIXTURES / "deconstruct_links_fixture.gfa"),
                               str(other)]), "compress other")
  shutil.copy(str(gfaz) + ".pidx", str(other) + ".pidx")
  stale = run_command([cli, "locate", "-i", str(other), "-n", "1"])
  if stale.returncode == 0 or "different container" not in stale.stderr:
    raise AssertionError("a stale index was not rejected: " + stale.stderr)

  bad = run_command([cli, "locate", "-i", str(gfaz), "-n", "999999"])
  if bad.returncode == 0 or "no segment named" not in bad.stderr:
    raise AssertionError("an unknown node was not rejected")


def main():
  ensure_cli_exists(CLI_PATH)
  with tempfile.TemporaryDirectory(prefix="gfaz_locate_") as tmp_name:
    for label, name_of in SEGMENT_NAMINGS.items():
      tmp = Path(tmp_name) / label
      tmp.mkdir()
      check(str(CLI_PATH), tmp, name_of)
  print("✅ locate/index regression checks passed")


if __name__ == "__main__":
  main()
