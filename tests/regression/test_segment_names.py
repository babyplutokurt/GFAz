#!/usr/bin/env python3
"""
Segment names survive compression.

A container numbers segments 1..N in S-line order. When the GFA already names
them that way nothing is stored and the file is format version 5, as before;
any other naming adds a name column (version 6). This suite checks, for each
kind of naming:

  - compress -> decompress reproduces every line with its original names, for
    both decompression paths and for the serial and chunked parsers;
  - the serial and chunked parsers write identical containers;
  - the format version is 5 exactly when the names are 1..N;
  - extract-path, extract-walk, add-haplotypes and depth -d speak the
    original names.
"""

import hashlib
import os
import random
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from regression_utils import CLI_PATH, ensure_cli_exists, run_main  # noqa: E402

NUM_SEGMENTS = 1500


def naming_schemes():
  shuffled = list(range(1, NUM_SEGMENTS + 1))
  random.Random(7).shuffle(shuffled)
  # name -> (function of 1-based index, expected format version)
  return {
      "dense": (str, 5),
      "prefixed": (lambda i: f"s{i}", 6),
      "sparse_numbers": (lambda i: str(1000 + 3 * i), 6),
      "shuffled_numbers": (lambda i: str(shuffled[i - 1]), 6),
      "padded_with_suffix": (lambda i: f"utg{i:06d}l", 6),
      "padding_overflows": (lambda i: f"n{i + 9000:04d}", 6),
      "zero_padded": (lambda i: f"{i:05d}", 6),
      "zero_based": (lambda i: str(i - 1), 6),
      "mixed": (lambda i: str(i) if i % 7 else f"alt_{i}", 6),
      "unrelated": (lambda i: hashlib.md5(str(i).encode()).hexdigest()[:9], 6),
  }


def make_gfa(name_of, seed):
  rng = random.Random(seed)
  names = [name_of(i) for i in range(1, NUM_SEGMENTS + 1)]
  lines = ["H\tVN:Z:1.1"]
  for i, name in enumerate(names):
    sequence = "".join(rng.choice("ACGT") for _ in range(rng.randint(1, 9)))
    lines.append(f"S\t{name}\t{sequence}\tLN:i:{len(sequence)}")
  for i in range(NUM_SEGMENTS - 1):
    lines.append(f"L\t{names[i]}\t+\t{names[i + 1]}\t{'+-'[i % 2]}\t0M")
  lines.append(f"J\t{names[0]}\t+\t{names[5]}\t-\t*")
  lines.append(f"C\t{names[0]}\t+\t{names[1]}\t+\t0\t1M")
  for p in range(4):
    steps = [names[j] + rng.choice("+-")
             for j in range(0, NUM_SEGMENTS, rng.randint(1, 3))]
    lines.append(f"P\tpath{p}\t{','.join(steps)}\t*")
  for w in range(4):
    steps = "".join(rng.choice("<>") + names[j]
                    for j in range(0, NUM_SEGMENTS, rng.randint(1, 3)))
    lines.append(f"W\tsample{w}\t1\tchr1\t0\t100\t{steps}")
  return lines, names


def run(args, env=None):
  result = subprocess.run([str(CLI_PATH), *args], capture_output=True,
                          text=True, env=env)
  if result.returncode != 0:
    raise AssertionError(f"gfaz {' '.join(args)} failed:\n{result.stderr}")
  return result.stdout


def format_version(path: Path) -> int:
  return struct.unpack("<I", path.read_bytes()[4:8])[0]


def check_scheme(label, name_of, version, d: Path):
  lines, names = make_gfa(name_of, seed=len(label))
  gfa = d / f"{label}.gfa"
  gfa.write_text("\n".join(lines) + "\n")
  expected = sorted(lines)

  chunked = d / f"{label}.gfaz"
  serial = d / f"{label}.serial.gfaz"
  run(["compress", str(gfa), str(chunked)])
  run(["compress", str(gfa), str(serial)],
      env={**os.environ, "GFAZ_SERIAL_PARSE": "1"})
  if chunked.read_bytes() != serial.read_bytes():
    raise AssertionError(f"{label}: serial and chunked parsers disagree")
  if format_version(chunked) != version:
    raise AssertionError(f"{label}: format version "
                         f"{format_version(chunked)}, expected {version}")

  for flags in ([], ["--legacy"]):
    out = d / f"{label}.out.gfa"
    run(["decompress", *flags, str(chunked), str(out)])
    if sorted(out.read_text().splitlines()) != expected:
      raise AssertionError(
          f"{label}: round trip {flags or 'default'} changed the graph")

  path_line = next(ln for ln in lines if ln.startswith("P\tpath2\t"))
  if run(["extract-path", str(chunked), "path2"]).strip() != path_line:
    raise AssertionError(f"{label}: extract-path differs from the P-line")
  walk_line = next(ln for ln in lines if ln.startswith("W\tsample1\t"))
  walk = run(["extract-walk", str(chunked), "sample1", "1", "chr1", "0",
              "100"]).strip()
  if walk != walk_line:
    raise AssertionError(f"{label}: extract-walk differs from the W-line")

  depth_rows = run(["depth", "-i", str(chunked), "-d"]).splitlines()[1:]
  if [row.split("\t")[0] for row in depth_rows] != names:
    raise AssertionError(f"{label}: depth -d does not list the segment names")

  added = f"P\tadded\t{names[0]}+,{names[9]}-,{names[4]}+\t*"
  extra = d / f"{label}.add.gfa"
  extra.write_text(added + "\n")
  updated = d / f"{label}.added.gfaz"
  run(["add-haplotypes", str(chunked), str(extra), str(updated)])
  if run(["extract-path", str(updated), "added"]).strip() != added:
    raise AssertionError(f"{label}: appended path does not read back")
  if format_version(updated) != version:
    raise AssertionError(f"{label}: add-haplotypes changed the version")

  unknown = d / f"{label}.unknown.gfa"
  unknown.write_text("P\tbad\tno_such_segment+\t*\n")
  result = subprocess.run(
      [str(CLI_PATH), "add-haplotypes", str(chunked), str(unknown),
       str(d / "bad.gfaz")], capture_output=True, text=True)
  if result.returncode == 0:
    raise AssertionError(f"{label}: an unknown segment name was accepted")


def main():
  ensure_cli_exists(CLI_PATH)
  with tempfile.TemporaryDirectory() as dd:
    for label, (name_of, version) in naming_schemes().items():
      check_scheme(label, name_of, version, Path(dd))
  print("test_segment_names: all checks passed")


if __name__ == "__main__":
  run_main(main)
