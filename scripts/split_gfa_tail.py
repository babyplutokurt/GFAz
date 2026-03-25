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
        help="Number of trailing path/walk lines to extract",
    )
    parser.add_argument(
        "--trimmed-output",
        help="Output path for the trimmed GFA "
        "(default: <input>.trimmed.gfa)",
    )
    parser.add_argument(
        "--extracted-output",
        help="Output path for the extracted P/W text "
        "(default: <input>.tail.txt)",
    )
    return parser.parse_args()


def default_outputs(input_path: Path) -> tuple[Path, Path]:
    if input_path.suffix:
        stem = input_path.with_suffix("")
    else:
        stem = input_path
    trimmed = Path(str(stem) + ".trimmed.gfa")
    extracted = Path(str(stem) + ".tail.txt")
    return trimmed, extracted


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_gfa)

    if args.count < 0:
        print("Error: count must be non-negative", file=sys.stderr)
        return 1

    trimmed_output, extracted_output = default_outputs(input_path)
    if args.trimmed_output:
        trimmed_output = Path(args.trimmed_output)
    if args.extracted_output:
        extracted_output = Path(args.extracted_output)

    try:
        with open(input_path, "r") as f_in, \
             open(trimmed_output, "w") as f_trim, \
             open(extracted_output, "w") as f_ext:
            pending_pw = deque()
            total_pw = 0
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
        print(f"Error: failed to write outputs: {exc}", file=sys.stderr)
        return 1

    extract_count = min(args.count, total_pw)

    print(f"Input: {input_path}")
    print(f"Total P/W lines: {total_pw}")
    print(f"Extracted trailing P/W lines: {extract_count}")
    print(f"Trimmed GFA: {trimmed_output}")
    print(f"Extracted text: {extracted_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
