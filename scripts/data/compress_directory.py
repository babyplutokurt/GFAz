#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Compress all .gfa files in a directory into .gfaz files."
  )
  parser.add_argument("input_dir", help="Directory containing input .gfa files")
  parser.add_argument(
      "--output-dir",
      help="Directory for output .gfaz files (default: input_dir)",
  )
  parser.add_argument(
      "--gfaz-bin",
      default="build/bin/gfaz",
      help="Path to the gfaz executable (default: build/bin/gfaz)",
  )
  parser.add_argument(
      "--threads",
      "-j",
      type=int,
      default=0,
      help="Threads to pass to gfaz compress (default: 0 = auto)",
  )
  parser.add_argument(
      "--recursive",
      action="store_true",
      help="Scan input_dir recursively for .gfa files",
  )
  parser.add_argument(
      "--overwrite",
      action="store_true",
      help="Overwrite existing .gfaz files instead of skipping them",
  )
  return parser.parse_args()


def find_gfa_files(input_dir: Path, recursive: bool) -> list[Path]:
  if recursive:
    return sorted(path for path in input_dir.rglob("*.gfa") if path.is_file())
  return sorted(path for path in input_dir.glob("*.gfa") if path.is_file())


def output_path_for(
    *,
    input_dir: Path,
    output_dir: Path,
    gfa_path: Path,
    recursive: bool,
) -> Path:
  if recursive:
    relative_parent = gfa_path.parent.relative_to(input_dir)
    target_dir = output_dir / relative_parent
  else:
    target_dir = output_dir
  return target_dir / f"{gfa_path.stem}.gfaz"


def main() -> int:
  args = parse_args()

  input_dir = Path(args.input_dir)
  output_dir = Path(args.output_dir) if args.output_dir else input_dir
  gfaz_bin = Path(args.gfaz_bin)

  if not input_dir.is_dir():
    print(f"Error: input_dir is not a directory: {input_dir}", file=sys.stderr)
    return 1
  if not gfaz_bin.exists():
    print(f"Error: gfaz executable not found: {gfaz_bin}", file=sys.stderr)
    return 1
  if args.threads < 0:
    print("Error: threads must be >= 0", file=sys.stderr)
    return 1

  gfa_files = find_gfa_files(input_dir, args.recursive)
  if not gfa_files:
    print(f"No .gfa files found in {input_dir}")
    return 0

  print(f"Found {len(gfa_files)} .gfa files")

  failures = 0
  skipped = 0

  for index, gfa_path in enumerate(gfa_files, start=1):
    output_path = output_path_for(
        input_dir=input_dir,
        output_dir=output_dir,
        gfa_path=gfa_path,
        recursive=args.recursive,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not args.overwrite:
      skipped += 1
      print(
          f"[{index}/{len(gfa_files)}] skip {gfa_path} -> {output_path} "
          "(already exists)"
      )
      continue

    cmd = [
        str(gfaz_bin),
        "compress",
        "-j",
        str(args.threads),
        str(gfa_path),
        str(output_path),
    ]
    print(f"[{index}/{len(gfa_files)}] {' '.join(cmd)}")

    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
      failures += 1
      print(f"  failed with exit code {proc.returncode}", file=sys.stderr)

  print(
      f"Completed: total={len(gfa_files)} skipped={skipped} failed={failures}"
  )
  return 1 if failures else 0


if __name__ == "__main__":
  raise SystemExit(main())
