#!/usr/bin/env python3
"""
Thread-count invariance for the compute-engine commands.

pav, growth and deconstruct all parallelize over traversals with OpenMP (atomics
/ reductions / per-thread accumulators). Their output must not depend on the
thread count. This suite runs each command at 1, 4 and 8 threads and asserts the
stdout is byte-identical (and non-empty, so the comparison is meaningful).

Inputs:
  - growth:      example.gfa (143 haplotypes -> real parallelism).
  - pav:         a synthetic path-only graph generated here (a `ref` path plus
                 many sample paths that each drop one node, so PAV values vary
                 and there are enough slices to exercise threading).
  - deconstruct: deconstruct_links_fixture.gfa (carries L-lines so the default
                 snarl mode emits records).
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
EXAMPLE_GFA = REPO_ROOT / "example.gfa"
DECON_FIXTURE = FIXTURE_DIR / "deconstruct_links_fixture.gfa"

THREAD_COUNTS = [1, 4, 8]


def compress(cli: Path, src: Path, dst: Path):
  require_success(run_command([str(cli), "compress", str(src), str(dst)]),
                  f"compress {src.name}")


def assert_invariant(cli: Path, build_argv, label: str):
  """build_argv(thread_count) -> argv (excluding the cli path)."""
  outputs = []
  for t in THREAD_COUNTS:
    result = run_command([str(cli), *build_argv(t)])
    require_success(result, f"{label} -t {t}")
    outputs.append(result.stdout)
  if not outputs[0].strip():
    raise AssertionError(f"{label}: empty output, comparison is meaningless")
  for t, out in zip(THREAD_COUNTS[1:], outputs[1:]):
    if out != outputs[0]:
      raise AssertionError(
          f"{label}: output differs between {THREAD_COUNTS[0]} and {t} threads")


def make_pav_graph(d: Path):
  """ref path over 100 nodes + 50 sample paths each dropping one distinct node;
  a BED with several windows over ref."""
  n = 100
  gfa = d / "pav_thread.gfa"
  bed = d / "pav_thread.bed"
  lines = ["H\tVN:Z:1.1"]
  lines += [f"S\t{i}\tACGT" for i in range(1, n + 1)]
  lines.append("P\tref\t" + ",".join(f"{i}+" for i in range(1, n + 1)) + "\t*")
  for s in range(1, 51):
    drop = s + 1  # drop node 2..51 for samples 1..50
    nodes = [i for i in range(1, n + 1) if i != drop]
    lines.append(f"P\tsample{s}#0#chr1\t" + ",".join(f"{i}+" for i in nodes) + "\t*")
  gfa.write_text("\n".join(lines) + "\n")
  bed.write_text("ref\t0\t400\tw1\nref\t100\t220\tw2\nref\t200\t360\tw3\n")
  return gfa, bed


def test_growth(cli: Path):
  if not EXAMPLE_GFA.exists():
    # example.gfa is the large fixture; fall back to a committed one.
    src = FIXTURE_DIR / "growth_fixture.gfa"
  else:
    src = EXAMPLE_GFA
  with tempfile.TemporaryDirectory() as dd:
    d = Path(dd)
    gfaz = d / "g.gfaz"
    compress(cli, src, gfaz)
    assert_invariant(
        cli,
        lambda t: ["growth", "-G", "sample-hap-seq", "-j", str(t), str(gfaz)],
        "growth")


def test_pav(cli: Path):
  with tempfile.TemporaryDirectory() as dd:
    d = Path(dd)
    gfa, bed = make_pav_graph(d)
    gfaz = d / "p.gfaz"
    compress(cli, gfa, gfaz)
    assert_invariant(
        cli,
        lambda t: ["pav", "-i", str(gfaz), "-b", str(bed), "-M", "-t", str(t)],
        "pav")


def test_deconstruct(cli: Path):
  with tempfile.TemporaryDirectory() as dd:
    d = Path(dd)
    gfaz = d / "dl.gfaz"
    compress(cli, DECON_FIXTURE, gfaz)
    assert_invariant(
        cli,
        lambda t: ["deconstruct", "-i", str(gfaz), "-r", "ref", "-S",
                   "-t", str(t)],
        "deconstruct")


def main():
  cli = Path(CLI_PATH)
  ensure_cli_exists(cli)
  test_growth(cli)
  test_pav(cli)
  test_deconstruct(cli)
  print("✅ PASS thread_determinism")


if __name__ == "__main__":
  main()
