#!/usr/bin/env python3
"""
Regression suite for the Python `gfa_compression.PathReader`.

Every method is compared with a brute-force reading of the decompressed GFA of
the generated multi-block graph, at several index block sizes and with a saved
index. Needs the compiled bindings (run by tests/run_all.py with a matching
interpreter); SKIPs when they or numpy cannot be imported.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.regression.regression_utils import (
    BUILD_DIR,
    CLI_PATH,
    SkipTest,
    ensure_cli_exists,
    require_success,
    run_command,
    run_main,
)
from tests.regression.test_extract_region import Graph, make_synthetic


def signed(node, orient):
  return node if orient in "+>" else -node


def check_reader(reader, graph: Graph):
  haps = graph.haps
  if reader.num_haplotypes != len(haps) or reader.num_nodes != len(graph.lengths):
    raise AssertionError("haplotype or node count differs")
  if reader.contigs != [h["contig"] for h in haps]:
    raise AssertionError("contigs differ")
  if list(reader.contig_offsets) != [h["offset"] for h in haps]:
    raise AssertionError("contig offsets differ")
  lengths = reader.node_lengths()
  if [int(x) for x in lengths] != [graph.lengths[n] for n in sorted(graph.lengths)]:
    raise AssertionError("node lengths differ")

  for h, hap in enumerate(haps):
    expected = [signed(n, o) for n, o in hap["steps"]]
    starts, pos = [], 0
    for n, _ in hap["steps"]:
      starts.append(pos)
      pos += graph.lengths[n]
    if reader.step_count(h) != len(expected) or reader.length(h) != pos:
      raise AssertionError(f"haplotype {h}: step count or length differs")
    if reader.steps(h).tolist() != expected:
      raise AssertionError(f"haplotype {h}: steps differ")
    for a, b in ((0, 1), (5, 37), (len(expected) - 3, len(expected) + 10)):
      if reader.steps(h, a, b).tolist() != expected[a:b]:
        raise AssertionError(f"haplotype {h}: steps[{a}:{b}] differ")
    for begin, end in ((0, 1), (pos // 3, pos // 3 + 200), (pos - 5, pos + 50)):
      first, nodes, where = reader.steps_in_bases(h, begin, end)
      want = [i for i, s in enumerate(starts)
              if s < min(end, pos) and s + graph.lengths[hap["steps"][i][0]] > begin]
      if nodes.tolist() != [expected[i] for i in want] or \
         where.tolist() != [starts[i] for i in want] or \
         (want and first != want[0]):
        raise AssertionError(f"haplotype {h}: steps_in_bases({begin},{end}) differ")

  for node in (1, 250, 1499, 3000):
    want = [(h, i, signed(n, o) < 0)
            for h, hap in enumerate(haps)
            for i, (n, o) in enumerate(hap["steps"]) if n == node]
    got = reader.locate(node)
    rows = list(zip(got["haplotype"].tolist(), got["step"].tolist(),
                    got["reverse"].tolist()))
    if rows != want:
      raise AssertionError(f"locate({node}) differs")
  if reader.haplotypes_on(haps[-1]["contig"]) != [
      i for i, h in enumerate(haps) if h["contig"] == haps[-1]["contig"]]:
    raise AssertionError("haplotypes_on differs")


def main():
  ensure_cli_exists(CLI_PATH)
  sys.path.insert(0, str(BUILD_DIR))
  try:
    import numpy  # noqa: F401
    import gfa_compression
  except ImportError as exc:
    raise SkipTest(f"bindings or numpy not importable: {exc}")
  if not hasattr(gfa_compression, "PathReader"):
    raise SkipTest("bindings were built without PathReader")

  cli = str(CLI_PATH)
  with tempfile.TemporaryDirectory(prefix="gfaz_reader_") as tmp_name:
    tmp = Path(tmp_name)
    gfa = tmp / "synthetic.gfa"
    make_synthetic(gfa)
    gfaz, full = tmp / "synthetic.gfaz", tmp / "full.gfa"
    require_success(run_command([cli, "compress", str(gfa), str(gfaz)]), "compress")
    require_success(run_command([cli, "decompress", str(gfaz), str(full)]),
                    "decompress")
    graph = Graph(full.read_text())

    for block_steps in (1, 7, 4096):
      check_reader(gfa_compression.PathReader(str(gfaz), block_steps=block_steps),
                   graph)
    index = tmp / "saved.pidx"
    require_success(run_command([cli, "index", "-i", str(gfaz), "-o", str(index),
                                 "-b", "9"]), "index")
    check_reader(gfa_compression.PathReader(str(gfaz), index_path=str(index)),
                 graph)
  print("✅ PathReader regression checks passed")


if __name__ == "__main__":
  run_main(main)
