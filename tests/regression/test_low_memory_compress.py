#!/usr/bin/env python3
"""
`gfaz compress --memory-budget` must not change the output.

With a budget, path/walk traversals that do not fit are kept in a temp file and
streamed through the grammar rounds, returning to memory as they shrink. The
.gfaz must be byte-identical to the fully in-memory run for every budget:

  - no budget          : reference
  - 1 byte             : every traversal spilled, through all rounds
  - a mid-size budget  : mixed in-memory / spilled, switching back part-way
  - a huge budget      : sink path active but nothing spilled

Checked across thread counts and delta settings, on committed fixtures and on a
synthetic graph with enough repeated structure for several grammar rounds.
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
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures"

FIXTURES = [
    "cli_fixture.gfa",
    "growth_fixture.gfa",
    "growth_fixture_walks.gfa",
    "append_mixed.gfa",
    "roundtrip_orientations_fixture.gfa",
    "compressor_optional_fields_fixture.gfa",
]


def make_synthetic_graph(path: Path, numeric_names: bool):
  """Haplotypes that are mosaics of a few founder traversals, as P and W
  lines, with some reverse-strand stretches. Shared stretches give the grammar
  several productive rounds; lengths vary so budgets split the set."""
  rng = random.Random(7)
  n = 3000
  name = (lambda i: str(i)) if numeric_names else (lambda i: f"s{i}")
  lines = ["H\tVN:Z:1.1"]
  lines += [f"S\t{name(i)}\t{'ACGT'[i % 4]}" for i in range(1, n + 1)]

  founders = []
  for _ in range(4):
    founder = [i for i in range(1, n + 1) if rng.random() > 0.1]
    founders.append(founder)

  def haplotype():
    steps = []
    position = 0
    length = rng.randint(50, 2500)
    while len(steps) < length:
      founder = rng.choice(founders)
      run = rng.randint(20, 400)
      chunk = founder[position % len(founder):][:run]
      if rng.random() < 0.15:
        steps.extend((node, "-") for node in reversed(chunk))
      else:
        steps.extend((node, "+") for node in chunk)
      position += run
    return steps[:length]

  for h in range(40):
    steps = haplotype()
    lines.append(f"P\tsample{h}#1#chr1\t" +
                 ",".join(f"{name(node)}{orient}" for node, orient in steps) +
                 "\t*")
  for h in range(40):
    steps = haplotype()
    walk = "".join((">" if orient == "+" else "<") + name(node)
                   for node, orient in steps)
    lines.append(f"W\twalk{h}\t1\tchr1\t0\t{len(steps)}\t{walk}")
  path.write_text("\n".join(lines) + "\n")


def compress(cli: Path, src: Path, dst: Path, extra):
  result = run_command([str(cli), "compress", *extra, str(src), str(dst)])
  require_success(result, f"compress {src.name} {' '.join(extra)}")
  return dst.read_bytes()


def check_input(cli: Path, src: Path, workdir: Path, label: str):
  tmp_dir = workdir / "spill"
  tmp_dir.mkdir(exist_ok=True)
  for delta in ("0", "1", "2"):
    for threads in ("1", "4"):
      common = ["-d", delta, "-j", threads]
      reference = compress(cli, src, workdir / "ref.gfaz", common)
      for budget in ("1", "6K", "40K", "1G"):
        got = compress(
            cli, src, workdir / "budget.gfaz",
            common + ["--memory-budget", budget, "--tmp-dir", str(tmp_dir)])
        if got != reference:
          raise AssertionError(
              f"{label}: --memory-budget {budget} changed the output "
              f"(delta={delta}, threads={threads})")
  leftovers = list(tmp_dir.iterdir())
  if leftovers:
    raise AssertionError(f"{label}: temp files left behind: {leftovers}")


def check_bad_tmp_dir(cli: Path, src: Path, workdir: Path):
  result = run_command([
      str(cli), "compress", "--memory-budget", "1", "--tmp-dir",
      str(workdir / "does-not-exist"), str(src), str(workdir / "bad.gfaz")
  ])
  if result.returncode == 0:
    raise AssertionError("compress succeeded with an unusable --tmp-dir")
  if "temp file" not in (result.stderr + result.stdout):
    raise AssertionError(
        "unusable --tmp-dir did not produce a clear error:\n" + result.stderr)


def check_bad_budget(cli: Path, src: Path, workdir: Path):
  for bad in ("0", "abc", "12Q"):
    result = run_command([
        str(cli), "compress", "--memory-budget", bad, str(src),
        str(workdir / "bad.gfaz")
    ])
    if result.returncode == 0:
      raise AssertionError(f"--memory-budget {bad} was accepted")


def main():
  cli = Path(CLI_PATH)
  ensure_cli_exists(cli)
  with tempfile.TemporaryDirectory() as dd:
    workdir = Path(dd)
    for fixture in FIXTURES:
      check_input(cli, FIXTURE_DIR / fixture, workdir, fixture)

    numeric = workdir / "synthetic_numeric.gfa"
    make_synthetic_graph(numeric, numeric_names=True)
    check_input(cli, numeric, workdir, "synthetic (numeric ids)")

    named = workdir / "synthetic_named.gfa"
    make_synthetic_graph(named, numeric_names=False)
    check_input(cli, named, workdir, "synthetic (named segments)")

    check_bad_tmp_dir(cli, numeric, workdir)
    check_bad_budget(cli, numeric, workdir)
  print("✅ PASS low_memory_compress")


if __name__ == "__main__":
  main()
