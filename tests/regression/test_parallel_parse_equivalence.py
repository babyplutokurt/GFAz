#!/usr/bin/env python3
"""
The chunked (parallel) S/L parsers must reproduce the serial line parsers.

Graphs that miss the direct two-pass parser (segment names that are not 1..N in
file order, irregular or string-typed optional fields) have their S and L
lines parsed in parallel blocks, with numeric segment names resolved through an
array instead of a hash map. `GFAZ_SERIAL_PARSE=1` forces the line-at-a-time
parsers; the .gfaz must be byte-identical with and without it.

Covers the shapes the chunked parsers accept (permuted numeric names, dense
names with irregular fields, named segments, numbers mixed with names) and the ones they must hand back
to the serial parsers (repeated names, non-canonical numbers, links or steps
that name unknown segments, sparse numbering, a tag repeated in one row).
"""

import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.regression.regression_utils import (
    CLI_PATH,
    REPO_ROOT,
    ensure_cli_exists,
)

SEGMENTS = 4000


def segment_tags(rng, i):
  """Irregular optional fields of every column type."""
  tags = []
  if rng.random() < 0.3:
    tags.append(f"SN:Z:chr{i % 5}#ctg")
    tags.append(f"SO:i:{i * 7}")
  if rng.random() < 0.2:
    tags.append(f"RC:f:{i / 8}")
  if rng.random() < 0.1:
    tags.append(f"XA:A:{'abc'[i % 3]}")
  if rng.random() < 0.05:
    tags.append(f"XB:B:i,{i},{i + 1},-3")
  return tags


def make_graph(path: Path, names, seed=11, link_tags=True, extra_lines=()):
  """`names[i]` is the name of the i-th segment in file order. S and L lines
  are interleaved in blocks, followed by paths and walks."""
  rng = random.Random(seed)
  n = len(names)
  lines = ["H\tVN:Z:1.1"]
  block = 500
  for start in range(0, n, block):
    for i in range(start, min(start + block, n)):
      fields = ["S", names[i], "ACGT"[i % 4] * (1 + i % 7)]
      lines.append("\t".join(fields + segment_tags(rng, i)))
    for i in range(start, min(start + block, n) - 1):
      fields = ["L", names[i], "+-"[i % 2], names[i + 1], "+", f"{i % 3}M"]
      if link_tags and rng.random() < 0.25:
        fields.append(f"EC:i:{i}")
      if link_tags and rng.random() < 0.1:
        fields.append(f"ID:Z:e{i}")
      lines.append("\t".join(fields))
  for h in range(12):
    start = rng.randrange(0, n - 600)
    steps = [(names[i], rng.random() < 0.9)
             for i in range(start, start + rng.randint(100, 600))
             if rng.random() > 0.1]
    lines.append(f"P\tsample{h}#1#chr1\t" +
                 ",".join(f"{node}{'+' if fwd else '-'}"
                          for node, fwd in steps) + "\t*")
    walk = "".join(("><"[not fwd]) + node for node, fwd in steps)
    lines.append(f"W\twalk{h}\t1\tchr1\t0\t{len(steps)}\t{walk}")
  lines.extend(extra_lines)
  path.write_text("\n".join(lines) + "\n")


def compress(cli: Path, src: Path, dst: Path, threads: str, serial: bool):
  env = dict(os.environ)
  env.pop("GFAZ_SERIAL_PARSE", None)
  if serial:
    env["GFAZ_SERIAL_PARSE"] = "1"
  result = subprocess.run(
      [str(cli), "compress", "-j", threads, str(src), str(dst)],
      cwd=str(REPO_ROOT), text=True, capture_output=True, check=False, env=env)
  if result.returncode != 0:
    raise AssertionError(
        f"compress {src.name} (serial={serial}) failed:\n{result.stderr}")
  return dst.read_bytes()


def check(cli: Path, src: Path, workdir: Path):
  reference = compress(cli, src, workdir / "serial.gfaz", "1", serial=True)
  for threads in ("1", "4", "16"):
    got = compress(cli, src, workdir / "chunked.gfaz", threads, serial=False)
    if got != reference:
      raise AssertionError(
          f"{src.name}: chunked parse changed the output at {threads} threads")


def main():
  cli = Path(CLI_PATH)
  ensure_cli_exists(cli)
  rng = random.Random(3)
  dense = [str(i) for i in range(1, SEGMENTS + 1)]
  # HPRC-style: numbers 1..N, emitted in a different order.
  rotated = dense[1500:] + dense[:1500]
  shuffled = dense[:]
  rng.shuffle(shuffled)
  offset = [str(i + 1000) for i in range(1, SEGMENTS + 1)]
  named = [f"s{i}" for i in range(1, SEGMENTS + 1)]

  repeated = rotated[:]
  repeated[2500] = repeated[10]
  leading_zero = rotated[:]
  leading_zero[3000] = "0" + leading_zero[3000]
  sparse = rotated[:]
  sparse[100] = "4000000000"

  # Named segments resolve through a name lookup; mixes and awkward names
  # must behave as in the serial parser.
  mixed = [name if i % 3 else f"s{name}" for i, name in enumerate(rotated)]
  named_repeated = named[:]
  named_repeated[2500] = named_repeated[10]
  named_leading_zero = named[:]
  named_leading_zero[3000] = "0042"
  dense_then_named = dense[:2000] + [f"n{i}" for i in range(2000, SEGMENTS)]

  cases = {
      "dense_irregular_fields": (dense, {}),
      "rotated": (rotated, {}),
      "shuffled": (shuffled, {}),
      "offset": (offset, {}),
      "named": (named, {}),
      "mixed_numbers_and_names": (mixed, {}),
      "dense_then_named": (dense_then_named, {}),
      "named_repeated": (named_repeated, {}),
      "named_leading_zero_number": (named_leading_zero, {
          "extra_lines": ["L\t42\t+\ts7\t+\t0M", "L\t0042\t+\ts8\t+\t0M"]
      }),
      "named_unknown_references": (named, {
          "extra_lines": [
              "L\ts17\t+\tnope\t+\t0M",
              "P\todd\ts17+,nope+,s20-\t*",
              "W\todd\t1\tchr1\t0\t3\t>s17<nope>s20",
          ]
      }),
      "no_link_tags": (rotated, {"link_tags": False}),
      "repeated_name": (repeated, {}),
      "leading_zero_name": (leading_zero, {}),
      "sparse_numbers": (sparse, {}),
      "unknown_link_endpoint": (rotated, {
          "extra_lines": ["L\t17\t+\t999999\t+\t0M", "L\t18\t+\t19\t-\t1M"]
      }),
      "noncanonical_references": (rotated, {
          "extra_lines": [
              "L\t017\t+\t18\t+\t0M",
              "P\todd\t17+,018+,19x+,20-,999999+\t*",
              "W\todd\t1\tchr1\t0\t4\t>17<018>19x>20>999999",
          ]
      }),
      "repeated_tag_in_row": (rotated, {
          "extra_lines": ["S\t900001\tAC\tSO:i:1\tSO:i:2", "S\t900002\tG"]
      }),
  }
  with tempfile.TemporaryDirectory() as dd:
    workdir = Path(dd)
    for label, (names, kwargs) in cases.items():
      src = workdir / f"{label}.gfa"
      make_graph(src, names, **kwargs)
      check(cli, src, workdir)
  print("✅ PASS parallel_parse_equivalence")


if __name__ == "__main__":
  main()
