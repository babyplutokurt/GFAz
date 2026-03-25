#!/usr/bin/env python3

import argparse
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
        lines = input_path.read_text().splitlines(keepends=True)
    except OSError as exc:
        print(f"Error: failed to read {input_path}: {exc}", file=sys.stderr)
        return 1

    pw_indices = [
        i for i, line in enumerate(lines) if line.startswith("P\t") or line.startswith("W\t")
    ]
    extract_count = min(args.count, len(pw_indices))
    extracted_index_set = set(pw_indices[-extract_count:])

    trimmed_lines = []
    extracted_lines = []
    for i, line in enumerate(lines):
        if i in extracted_index_set:
            extracted_lines.append(line)
        else:
            trimmed_lines.append(line)

    try:
        trimmed_output.write_text("".join(trimmed_lines))
        extracted_output.write_text("".join(extracted_lines))
    except OSError as exc:
        print(f"Error: failed to write outputs: {exc}", file=sys.stderr)
        return 1

    print(f"Input: {input_path}")
    print(f"Total P/W lines: {len(pw_indices)}")
    print(f"Extracted trailing P/W lines: {extract_count}")
    print(f"Trimmed GFA: {trimmed_output}")
    print(f"Extracted text: {extracted_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
