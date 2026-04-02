#!/usr/bin/env python3

import argparse
from collections import deque
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description=(
          "Split a GFA into a trimmed GFA and a text file containing the "
          "last N P/W lines."
      )
  )
  parser.add_argument("input_gfa", help="Input GFA file")
  parser.add_argument(
      "count",
      type=int,
      help="Number of trailing P/W lines to extract",
  )
  parser.add_argument(
      "--trimmed-output",
      help="Output path for the trimmed GFA (default: <input>.trimmed.gfa)",
  )
  parser.add_argument(
      "--extracted-output",
      help="Output path for the extracted P/W text (default: <input>.tail.txt)",
  )
  return parser.parse_args()


def default_outputs(input_path: Path) -> tuple[Path, Path]:
  if input_path.suffix:
    stem = input_path.with_suffix("")
  else:
    stem = input_path
  return Path(f"{stem}.trimmed.gfa"), Path(f"{stem}.tail.txt")


def main() -> int:
  args = parse_args()
  input_path = Path(args.input_gfa)

  if not input_path.is_file():
    print(f"Error: input_gfa not found: {input_path}", file=sys.stderr)
    return 1
  if args.count < 0:
    print("Error: count must be non-negative", file=sys.stderr)
    return 1

  trimmed_output, extracted_output = default_outputs(input_path)
  if args.trimmed_output:
    trimmed_output = Path(args.trimmed_output)
  if args.extracted_output:
    extracted_output = Path(args.extracted_output)

  if trimmed_output == extracted_output:
    print(
        "Error: trimmed_output and extracted_output must be different files",
        file=sys.stderr,
    )
    return 1

  trimmed_output.parent.mkdir(parents=True, exist_ok=True)
  extracted_output.parent.mkdir(parents=True, exist_ok=True)

  pending_pw: deque[str] = deque()
  total_pw = 0

  try:
    with input_path.open("r", encoding="utf-8") as f_in, \
         trimmed_output.open("w", encoding="utf-8") as f_trim, \
         extracted_output.open("w", encoding="utf-8") as f_ext:
      for line in f_in:
        if line.startswith("P\t") or line.startswith("W\t"):
          pending_pw.append(line)
          total_pw += 1
          if len(pending_pw) > args.count:
            f_trim.write(pending_pw.popleft())
        else:
          f_trim.write(line)
      while pending_pw:
        f_ext.write(pending_pw.popleft())
  except OSError as exc:
    print(f"Error: failed while processing files: {exc}", file=sys.stderr)
    return 1

  extracted_count = min(args.count, total_pw)
  print(f"Input: {input_path}")
  print(f"Total P/W lines: {total_pw}")
  print(f"Extracted trailing P/W lines: {extracted_count}")
  print(f"Trimmed GFA: {trimmed_output}")
  print(f"Extracted text: {extracted_output}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
