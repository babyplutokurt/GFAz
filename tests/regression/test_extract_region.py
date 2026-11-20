#!/usr/bin/env python3
"""
Regression suite for `gfaz extract-region`.

Every case is checked against a brute-force reference computed in Python from
the decompressed GFA: mark the nodes the reference visits inside the region,
find every visit of every haplotype to a marked node, join visits at most
--max-gap bases apart, and keep the joined stretches, the segments they visit
and the links between those segments.

Besides the small fixtures, a generated graph with long haplotypes, reverse
steps, repeats and offset walks exercises the multi-block index paths; each
case runs with several index block sizes and thread counts, which must not
change the output.
"""

import random
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
FIXTURES = REPO_ROOT / "tests" / "fixtures"


class Graph:
  """The parts of a GFA the reference computation needs."""

  def __init__(self, text: str):
    self.header = []
    self.segments = {}   # segment name -> line
    self.lengths = {}    # segment name -> bases
    self.ids = {}        # segment name -> 1-based number in S-line order
    self.links = []      # (from, to, line)
    self.haps = []       # dict(kind, contig, offset, steps, fields)
    for line in text.splitlines():
      f = line.split("\t")
      if f[0] == "H":
        self.header.append(line)
      elif f[0] == "S":
        self.segments[f[1]] = line
        self.lengths[f[1]] = len(f[2])
        self.ids[f[1]] = len(self.ids) + 1
      elif f[0] == "L":
        self.links.append((f[1], f[3], line))
      elif f[0] == "P":
        steps = [(s[:-1], s[-1]) for s in f[2].split(",")]
        self.haps.append(dict(kind="P", contig=f[1], offset=0, steps=steps,
                              fields=f))
      elif f[0] == "W":
        steps, i, walk = [], 0, f[6]
        while i < len(walk):
          j = i + 1
          while j < len(walk) and walk[j] not in "<>":
            j += 1
          steps.append((walk[i + 1:j], walk[i]))
          i = j
        offset = int(f[4]) if f[4] != "*" else 0
        self.haps.append(dict(kind="W", contig=f"{f[1]}#{f[2]}#{f[3]}",
                              offset=offset, steps=steps, fields=f))


def expected_region(graph: Graph, regions, max_gap: int):
  """Brute-force expected output lines of extract-region, as a set."""
  marked = set()
  for contig, begin, end in regions:
    for hap in graph.haps:
      if hap["contig"] != contig:
        continue
      pos = hap["offset"]
      for node, _ in hap["steps"]:
        length = graph.lengths[node]
        if pos < end and pos + length > begin:
          marked.add(node)
        pos += length

  lines, nodes = set(), set()
  for hap in graph.haps:
    pos, runs, cur = 0, [], None
    for i, (node, _) in enumerate(hap["steps"]):
      length = graph.lengths[node]
      if node in marked:
        if cur and pos <= cur[3] + max_gap:
          cur[1], cur[3] = i + 1, pos + length
        else:
          if cur:
            runs.append(cur)
          cur = [i, i + 1, pos, pos + length]
      pos += length
    if cur:
      runs.append(cur)
    total = pos
    for first, last, bp_begin, bp_end in runs:
      steps = hap["steps"][first:last]
      nodes.update(node for node, _ in steps)
      f = hap["fields"]
      whole = bp_begin == 0 and bp_end == total
      if hap["kind"] == "P":
        name = f[1] if whole else f"{f[1]}:{bp_begin}-{bp_end}"
        body = ",".join(f"{node}{orient}" for node, orient in steps)
        lines.add(f"P\t{name}\t{body}\t*")
      else:
        body = "".join(f"{orient}{node}" for node, orient in steps)
        lines.add(f"W\t{f[1]}\t{f[2]}\t{f[3]}\t{hap['offset'] + bp_begin}\t"
                  f"{hap['offset'] + bp_end}\t{body}")
  lines.update(graph.header)
  lines.update(graph.segments[n] for n in nodes)
  lines.update(line for a, b, line in graph.links if a in nodes and b in nodes)
  return lines


# Ways to name segment number n of a generated graph: as the number itself,
# as other numbers, as strings, and as numbers that turn into strings.
SEGMENT_NAMINGS = {
    "dense": str,
    "sparse": lambda n: str(1000 + 3 * n),
    "named": lambda n: f"s{n}x",
    "mixed": lambda n: f"{n + 5:04d}" if n < 100 else f"utg{n}",
    "zero_based": lambda n: str(n - 1),
}


def make_synthetic(path: Path, name_of=str):
  """A graph whose haplotypes span many index blocks. name_of gives the name
  of the n-th segment."""
  rng = random.Random(7)
  num_nodes = 3000
  out = ["H\tVN:Z:1.1"]
  lengths = {}
  for n in range(1, num_nodes + 1):
    sequence = "".join(rng.choice("ACGT") for _ in range(rng.randint(1, 9)))
    lengths[n] = len(sequence)
    out.append(f"S\t{name_of(n)}\t{sequence}")
  links = set()
  haps = []
  for h in range(10):
    steps, n, repeats = [], 1, 6
    while n <= num_nodes:
      roll = rng.random()
      if roll < 0.02 and n + 40 < num_nodes:      # inversion
        steps += [(m, "-") for m in range(n + 20, n - 1, -1)]
        n += 21
      elif roll < 0.03 and n > 200 and repeats:   # jump back: a repeat
        n -= rng.randint(50, 150)
        repeats -= 1
      elif roll < 0.10:                           # deletion
        n += rng.randint(2, 6)
      else:
        steps.append((n, "+"))
        n += 1
    haps.append(steps)
    for (a, ao), (b, bo) in zip(steps, steps[1:]):
      links.add((a, ao, b, bo))
  for a, ao, b, bo in sorted(links):
    out.append(f"L\t{name_of(a)}\t{ao}\t{name_of(b)}\t{bo}\t0M")
  for h, steps in enumerate(haps):
    if h < 4:
      body = ",".join(f"{name_of(n)}{o}" for n, o in steps)
      out.append(f"P\tp{h}#1#chrS\t{body}\t*")
    else:
      # Two walks on one contig, the second starting where the first ends.
      cut = len(steps) // 2
      first = sum(lengths[n] for n, _ in steps[:cut])
      total = sum(lengths[n] for n, _ in steps)
      for part, begin, end in ((steps[:cut], 0, first), (steps[cut:], first, total)):
        body = "".join(f"{'>' if o == '+' else '<'}{name_of(n)}"
                       for n, o in part)
        out.append(f"W\tw{h}\t1\tchrS\t{begin}\t{end}\t{body}")
  path.write_text("\n".join(out) + "\n")


def parse_region(text: str):
  contig, _, rng = text.rpartition(":")
  begin, end = rng.split("-")
  return contig, int(begin), int(end)


def check(cli: Path, gfa: Path, cases, tmp: Path):
  gfaz = tmp / (gfa.stem + ".gfaz")
  require_success(run_command([str(cli), "compress", str(gfa), str(gfaz)]),
                  f"compress {gfa.name}")
  full = tmp / (gfa.stem + ".full.gfa")
  require_success(run_command([str(cli), "decompress", str(gfaz), str(full)]),
                  f"decompress {gfa.name}")
  graph = Graph(full.read_text())

  for regions, max_gap in cases:
    expected = expected_region(graph, [parse_region(r) for r in regions], max_gap)
    for block_steps in (1, 7, 4096):
      for threads in (1, 4):
        cmd = [str(cli), "extract-region", "-i", str(gfaz), "-g", str(max_gap),
               "--index-block-steps", str(block_steps), "-t", str(threads)]
        for region in regions:
          cmd += ["-r", region]
        result = run_command(cmd)
        require_success(result, f"extract-region {regions}")
        got = result.stdout.splitlines()
        if len(got) != len(set(got)):
          raise AssertionError(f"{gfa.name} {regions}: duplicate output lines")
        if set(got) != expected:
          missing = sorted(expected - set(got))[:3]
          extra = sorted(set(got) - expected)[:3]
          raise AssertionError(
              f"{gfa.name} {regions} gap={max_gap} block={block_steps} "
              f"threads={threads}: output differs\nmissing: {missing}\n"
              f"extra: {extra}")


def test_errors(cli: Path, tmp: Path):
  gfaz = tmp / "err.gfaz"
  require_success(run_command([str(cli), "compress",
                               str(FIXTURES / "deconstruct_links_fixture.gfa"),
                               str(gfaz)]), "compress")
  for region, needle in (("nope:0-5", "no path or walk named"),
                         ("ref:5-5", "empty interval"),
                         ("ref:1000-2000", "does not overlap")):
    result = run_command([str(cli), "extract-region", "-i", str(gfaz),
                          "-r", region])
    if result.returncode == 0 or needle not in result.stderr:
      raise AssertionError(f"region {region}: expected error '{needle}', got "
                           f"rc={result.returncode} {result.stderr.strip()}")


def main():
  ensure_cli_exists(CLI_PATH)
  cli = CLI_PATH
  with tempfile.TemporaryDirectory(prefix="gfaz_region_") as tmp_name:
    tmp = Path(tmp_name)
    check(cli, FIXTURES / "deconstruct_links_fixture.gfa", [
        (["ref:0-6"], 100000),
        (["ref:5-11"], 0),
        (["ref:9-12", "HG002#1#chr1:20-24"], 2),
        (["ref:0-26"], 100000),
    ], tmp)
    check(cli, FIXTURES / "growth_fixture_walks.gfa", [
        (["ref#0#chr1:5-11"], 100000),
        (["HG002#1#chr1:18-25"], 0),
    ], tmp)
    for label, name_of in SEGMENT_NAMINGS.items():
      synthetic = tmp / f"synthetic_{label}.gfa"
      make_synthetic(synthetic, name_of)
      check(cli, synthetic, [
          (["p0#1#chrS:2000-2400"], 100000),
          (["p0#1#chrS:2000-2400"], 0),
          (["p1#1#chrS:100-9000"], 50),
          (["w5#1#chrS:6000-9000"], 300),
          (["w7#1#chrS:0-40", "p3#1#chrS:12000-12100"], 1000),
      ], tmp)
    test_errors(cli, tmp)
  print("✅ extract-region regression checks passed")


if __name__ == "__main__":
  main()
