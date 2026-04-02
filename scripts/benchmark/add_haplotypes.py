#!/usr/bin/env python3

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark 'gfaz add-haplotypes' across a directory of .gfaz files "
            "and matching tail files, recording elapsed time and peak memory."
        )
    )
    parser.add_argument(
        "input_dir",
        help=(
            "Directory containing input .gfaz files and matching tail files, "
            "for example '*.trimmed.gfaz' and '*.tail.txt'"
        ),
    )
    parser.add_argument(
        "--csv",
        default="add_haplotypes_benchmark.csv",
        help="Output CSV path (default: add_haplotypes_benchmark.csv)",
    )
    parser.add_argument(
        "--gfaz-bin",
        default="build/bin/gfaz",
        help="Path to the gfaz executable (default: build/bin/gfaz)",
    )
    parser.add_argument(
        "--time-bin",
        default="/usr/bin/time",
        help="Path to /usr/bin/time (default: /usr/bin/time)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="Threads to pass to gfaz add-haplotypes (default: 0 = auto)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of benchmark repeats per dataset (default: 1)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan the input directory recursively",
    )
    parser.add_argument(
        "--scratch-dir",
        default=None,
        help=(
            "Directory for temporary output files. Defaults to the input "
            "directory so large outputs stay on the same filesystem."
        ),
    )
    parser.add_argument(
        "--keep-output",
        action="store_true",
        help="Keep generated .updated.gfaz outputs instead of deleting them",
    )
    parser.add_argument(
        "--suffix",
        default=".updated.gfaz",
        help=(
            "Suffix for generated output files when --keep-output is used "
            "(default: .updated.gfaz)"
        ),
    )
    return parser.parse_args()


def find_files(root: Path, pattern: str, recursive: bool) -> list[Path]:
    if recursive:
        return sorted(root.rglob(pattern))
    return sorted(root.glob(pattern))


def parse_max_rss_kb(stderr_text: str) -> int | None:
    for line in stderr_text.splitlines():
        if "Maximum resident set size" in line:
            _, _, value = line.partition(":")
            value = value.strip()
            try:
                return int(value)
            except ValueError:
                return None
    return None


def parse_user_seconds(stderr_text: str) -> float | None:
    for line in stderr_text.splitlines():
        if "User time (seconds)" in line:
            _, _, value = line.partition(":")
            value = value.strip()
            try:
                return float(value)
            except ValueError:
                return None
    return None


def parse_system_seconds(stderr_text: str) -> float | None:
    for line in stderr_text.splitlines():
        if "System time (seconds)" in line:
            _, _, value = line.partition(":")
            value = value.strip()
            try:
                return float(value)
            except ValueError:
                return None
    return None


def dataset_name(gfaz_path: Path) -> str:
    name = gfaz_path.name
    if name.endswith(".trimmed.gfaz"):
        return name[: -len(".trimmed.gfaz")]
    if name.endswith(".gfaz"):
        return name[: -len(".gfaz")]
    return gfaz_path.stem


def tail_candidates(gfaz_path: Path) -> list[str]:
    name = gfaz_path.name
    candidates = []
    if name.endswith(".trimmed.gfaz"):
        candidates.append(name[: -len(".trimmed.gfaz")] + ".tail.txt")
    if name.endswith(".gfa.gfaz"):
        candidates.append(name[: -len(".gfa.gfaz")] + ".tail.txt")
    if name.endswith(".gfaz"):
        candidates.append(name[: -len(".gfaz")] + ".tail.txt")
    candidates.append(gfaz_path.stem + ".tail.txt")
    return candidates


def build_tail_index(tail_paths: list[Path]) -> dict[str, Path]:
    return {path.name: path for path in tail_paths}


def match_tail_file(gfaz_path: Path, tail_index: dict[str, Path]) -> Path | None:
    for candidate in tail_candidates(gfaz_path):
        if candidate in tail_index:
            return tail_index[candidate]
    return None


def output_path_for_keep(gfaz_path: Path, repeat: int, suffix: str) -> Path:
    dataset = dataset_name(gfaz_path)
    if repeat == 1:
        return gfaz_path.with_name(dataset + suffix)
    return gfaz_path.with_name(f"{dataset}.repeat{repeat}{suffix}")


def run_add_haplotypes_benchmark(
    *,
    time_bin: str,
    gfaz_bin: str,
    gfaz_path: Path,
    tail_path: Path,
    output_path: Path,
    threads: int,
) -> tuple[float, int | None, float | None, float | None, int, str, str]:
    cmd = [
        time_bin,
        "-v",
        gfaz_bin,
        "add-haplotypes",
        "-j",
        str(threads),
        str(gfaz_path),
        str(tail_path),
        str(output_path),
    ]
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - start
    combined_output = proc.stdout
    if proc.stderr:
        if combined_output:
            combined_output += "\n"
        combined_output += proc.stderr
    return (
        elapsed,
        parse_max_rss_kb(proc.stderr),
        parse_user_seconds(proc.stderr),
        parse_system_seconds(proc.stderr),
        proc.returncode,
        " ".join(cmd),
        combined_output.strip(),
    )


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    csv_path = Path(args.csv)
    gfaz_bin = Path(args.gfaz_bin)
    time_bin = Path(args.time_bin)
    scratch_root = Path(args.scratch_dir) if args.scratch_dir else input_dir

    if not input_dir.is_dir():
        print(f"Error: input_dir is not a directory: {input_dir}", file=sys.stderr)
        return 1
    if not gfaz_bin.exists():
        print(f"Error: gfaz executable not found: {gfaz_bin}", file=sys.stderr)
        return 1
    if not time_bin.exists():
        print(f"Error: time executable not found: {time_bin}", file=sys.stderr)
        return 1
    if args.repeats < 1:
        print("Error: repeats must be at least 1", file=sys.stderr)
        return 1
    if args.threads < 0:
        print("Error: threads must be >= 0", file=sys.stderr)
        return 1
    if not scratch_root.is_dir():
        print(
            f"Error: scratch_dir is not a directory: {scratch_root}",
            file=sys.stderr,
        )
        return 1

    gfaz_paths = find_files(input_dir, "*.gfaz", args.recursive)
    tail_paths = find_files(input_dir, "*.tail.txt", args.recursive)
    if not gfaz_paths:
        print(f"Error: no .gfaz files found in {input_dir}", file=sys.stderr)
        return 1
    if not tail_paths:
        print(f"Error: no .tail.txt files found in {input_dir}", file=sys.stderr)
        return 1

    tail_index = build_tail_index(tail_paths)
    rows: list[dict[str, object]] = []

    for gfaz_path in gfaz_paths:
        dataset = dataset_name(gfaz_path)
        tail_path = match_tail_file(gfaz_path, tail_index)

        if tail_path is None:
            rows.append(
                {
                    "dataset": dataset,
                    "gfaz_path": str(gfaz_path),
                    "tail_path": "",
                    "repeat": "",
                    "threads": args.threads,
                    "elapsed_seconds": "",
                    "max_rss_kb": "",
                    "user_seconds": "",
                    "system_seconds": "",
                    "output_path": "",
                    "output_size_bytes": "",
                    "status": "skipped_missing_tail_match",
                    "return_code": "",
                    "error": "",
                    "command_line": "",
                }
            )
            continue

        for repeat in range(1, args.repeats + 1):
            temp_dir = None
            return_code = None
            if args.keep_output:
                output_path = output_path_for_keep(gfaz_path, repeat, args.suffix)
                output_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                temp_dir = tempfile.TemporaryDirectory(
                    prefix=f"gfaz_add_haps_{dataset}_",
                    dir=str(scratch_root),
                )
                output_path = Path(temp_dir.name) / f"{dataset}.updated.gfaz"

            try:
                (
                    elapsed,
                    max_rss_kb,
                    user_seconds,
                    system_seconds,
                    return_code,
                    command_line,
                    combined_output,
                ) = run_add_haplotypes_benchmark(
                    time_bin=str(time_bin),
                    gfaz_bin=str(gfaz_bin),
                    gfaz_path=gfaz_path,
                    tail_path=tail_path,
                    output_path=output_path,
                    threads=args.threads,
                )
            except OSError as exc:
                rows.append(
                    {
                        "dataset": dataset,
                        "gfaz_path": str(gfaz_path),
                        "tail_path": str(tail_path),
                        "repeat": repeat,
                        "threads": args.threads,
                        "elapsed_seconds": "",
                        "max_rss_kb": "",
                        "user_seconds": "",
                        "system_seconds": "",
                        "output_path": str(output_path),
                        "output_size_bytes": "",
                        "status": "error",
                        "return_code": "",
                        "error": str(exc),
                        "command_line": "",
                    }
                )
            else:
                if return_code == 0 and output_path.exists():
                    output_size_bytes = output_path.stat().st_size
                    status = "ok"
                    error = ""
                else:
                    output_size_bytes = ""
                    status = "error"
                    error = combined_output

                rows.append(
                    {
                        "dataset": dataset,
                        "gfaz_path": str(gfaz_path),
                        "tail_path": str(tail_path),
                        "repeat": repeat,
                        "threads": args.threads,
                        "elapsed_seconds": f"{elapsed:.6f}",
                        "max_rss_kb": max_rss_kb if max_rss_kb is not None else "",
                        "user_seconds": (
                            f"{user_seconds:.2f}"
                            if user_seconds is not None
                            else ""
                        ),
                        "system_seconds": (
                            f"{system_seconds:.2f}"
                            if system_seconds is not None
                            else ""
                        ),
                        "output_path": str(output_path),
                        "output_size_bytes": output_size_bytes,
                        "status": status,
                        "return_code": return_code,
                        "error": error,
                        "command_line": command_line,
                    }
                )
            finally:
                if temp_dir is not None:
                    temp_dir.cleanup()
                elif return_code == 0 and output_path.exists():
                    pass
                elif output_path.exists():
                    try:
                        if output_path.is_dir():
                            shutil.rmtree(output_path)
                        else:
                            output_path.unlink()
                    except OSError:
                        pass

    fieldnames = [
        "dataset",
        "gfaz_path",
        "tail_path",
        "repeat",
        "threads",
        "elapsed_seconds",
        "max_rss_kb",
        "user_seconds",
        "system_seconds",
        "output_path",
        "output_size_bytes",
        "status",
        "return_code",
        "error",
        "command_line",
    ]

    try:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    except OSError as exc:
        print(f"Error: failed to write {csv_path}: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {len(rows)} rows to {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
