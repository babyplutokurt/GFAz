#!/usr/bin/env python3

import argparse
import csv
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark gfaz extract-path and extract-walk on a folder of "
            ".gfaz files using names collected from matching .gfa files."
        )
    )
    parser.add_argument("gfa_dir", help="Directory containing source .gfa files")
    parser.add_argument("gfaz_dir", help="Directory containing .gfaz files")
    parser.add_argument(
        "--csv",
        default="extract_benchmark.csv",
        help="Output CSV path (default: extract_benchmark.csv)",
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
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 4, 16],
        help="Batch sizes to benchmark (default: 1 4 16)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of random trials per file, command, and batch size "
        "(default: 1)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="Threads to pass to gfaz extract commands (default: 0 = auto)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for reproducible sampling (default: 1)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan input directories recursively",
    )
    return parser.parse_args()


def find_files(root: Path, pattern: str, recursive: bool) -> list[Path]:
    if recursive:
        return sorted(root.rglob(pattern))
    return sorted(root.glob(pattern))


def build_gfa_index(gfa_paths: list[Path]) -> dict[str, Path]:
    index = {}
    for path in gfa_paths:
        index[path.name] = path
    return index


def match_gfaz_to_gfa(gfaz_path: Path, gfa_index: dict[str, Path]) -> Path | None:
    candidates = []
    if gfaz_path.name.endswith(".gfaz"):
        without_suffix = gfaz_path.name[: -len(".gfaz")]
        candidates.append(without_suffix)
        if not without_suffix.endswith(".gfa"):
            candidates.append(without_suffix + ".gfa")
    candidates.append(gfaz_path.stem)
    if not gfaz_path.stem.endswith(".gfa"):
        candidates.append(gfaz_path.stem + ".gfa")

    for candidate in candidates:
        if candidate in gfa_index:
            return gfa_index[candidate]
    return None


def collect_names(gfa_path: Path) -> tuple[list[str], list[str], int]:
    path_names = []
    walk_keys = []
    total_pw_lines = 0

    try:
        with gfa_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("P\t"):
                    fields = line.rstrip("\n").split("\t")
                    if len(fields) > 1:
                        path_names.append(fields[1])
                        total_pw_lines += 1
                elif line.startswith("W\t"):
                    fields = line.rstrip("\n").split("\t")
                    if len(fields) >= 6:
                        walk_keys.append(
                            (
                                fields[1],
                                fields[2],
                                fields[3],
                                fields[4],
                                fields[5],
                            )
                        )
                        total_pw_lines += 1
    except OSError as exc:
        raise RuntimeError(f"failed to read {gfa_path}: {exc}") from exc

    return path_names, walk_keys, total_pw_lines


def unique_names_with_duplicate_count(names: list[str]) -> tuple[list[str], int]:
    counts = Counter(names)
    unique = [name for name in names if counts[name] == 1]
    duplicate_count = sum(1 for name in names if counts[name] > 1)
    return unique, duplicate_count


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


def run_extract_benchmark(
    time_bin: str,
    gfaz_bin: str,
    subcommand: str,
    gfaz_path: Path,
    query_args: list[str],
    threads: int,
) -> tuple[float, int | None, str]:
    cmd = [
        time_bin,
        "-v",
        gfaz_bin,
        subcommand,
        "-j",
        str(threads),
        str(gfaz_path),
        *query_args,
    ]
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip())
    return elapsed, parse_max_rss_kb(proc.stderr), " ".join(cmd)


def dataset_name(gfaz_path: Path) -> str:
    if gfaz_path.name.endswith(".gfaz"):
        return gfaz_path.name[: -len(".gfaz")]
    return gfaz_path.stem


def benchmark_name_set(
    rows: list[dict[str, object]],
    rng: random.Random,
    *,
    dataset: str,
    gfa_path: Path,
    gfaz_path: Path,
    subcommand: str,
    label: str,
    names: list[str],
    duplicate_count: int,
    batch_sizes: list[int],
    repeats: int,
    threads: int,
    gfaz_bin: str,
    time_bin: str,
) -> None:
    available = len(names)
    if available == 0:
        rows.append(
            {
                "dataset": dataset,
                "gfa_path": str(gfa_path),
                "gfaz_path": str(gfaz_path),
                "command": label,
                "batch_size": "",
                "repeat": "",
                "available_names": 0,
                "duplicate_name_count": duplicate_count,
                "sampled_names": "",
                "elapsed_seconds": "",
                "max_rss_kb": "",
                "status": f"skipped_no_{label}",
                "error": "",
                "command_line": "",
            }
        )
        return

    for batch_size in batch_sizes:
        if batch_size > available:
            rows.append(
                {
                    "dataset": dataset,
                    "gfa_path": str(gfa_path),
                    "gfaz_path": str(gfaz_path),
                    "command": label,
                    "batch_size": batch_size,
                    "repeat": "",
                    "available_names": available,
                    "duplicate_name_count": duplicate_count,
                    "sampled_names": "",
                    "elapsed_seconds": "",
                    "max_rss_kb": "",
                    "status": "skipped_insufficient_names",
                    "error": "",
                    "command_line": "",
                }
            )
            continue

        for repeat in range(1, repeats + 1):
            sampled_names = rng.sample(names, batch_size)
            try:
                elapsed, max_rss_kb, command_line = run_extract_benchmark(
                    time_bin=time_bin,
                    gfaz_bin=gfaz_bin,
                    subcommand=subcommand,
                    gfaz_path=gfaz_path,
                    query_args=sampled_names,
                    threads=threads,
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "gfa_path": str(gfa_path),
                        "gfaz_path": str(gfaz_path),
                        "command": label,
                        "batch_size": batch_size,
                        "repeat": repeat,
                        "available_names": available,
                        "duplicate_name_count": duplicate_count,
                        "sampled_names": ";".join(sampled_names),
                        "elapsed_seconds": f"{elapsed:.6f}",
                        "max_rss_kb": max_rss_kb if max_rss_kb is not None else "",
                        "status": "ok",
                        "error": "",
                        "command_line": command_line,
                    }
                )
            except RuntimeError as exc:
                rows.append(
                    {
                        "dataset": dataset,
                        "gfa_path": str(gfa_path),
                        "gfaz_path": str(gfaz_path),
                        "command": label,
                        "batch_size": batch_size,
                        "repeat": repeat,
                        "available_names": available,
                        "duplicate_name_count": duplicate_count,
                        "sampled_names": ";".join(sampled_names),
                        "elapsed_seconds": "",
                        "max_rss_kb": "",
                        "status": "error",
                        "error": str(exc),
                        "command_line": "",
                    }
                )


def format_walk_key(walk_key: tuple[str, str, str, str, str]) -> str:
    return "|".join(walk_key)


def benchmark_walk_set(
    rows: list[dict[str, object]],
    rng: random.Random,
    *,
    dataset: str,
    gfa_path: Path,
    gfaz_path: Path,
    walk_keys: list[tuple[str, str, str, str, str]],
    batch_sizes: list[int],
    repeats: int,
    threads: int,
    gfaz_bin: str,
    time_bin: str,
) -> None:
    available = len(walk_keys)
    if available == 0:
        rows.append(
            {
                "dataset": dataset,
                "gfa_path": str(gfa_path),
                "gfaz_path": str(gfaz_path),
                "command": "walk",
                "batch_size": "",
                "repeat": "",
                "available_names": 0,
                "duplicate_name_count": 0,
                "sampled_names": "",
                "elapsed_seconds": "",
                "max_rss_kb": "",
                "status": "skipped_no_walk",
                "error": "",
                "command_line": "",
            }
        )
        return

    for batch_size in batch_sizes:
        if batch_size > available:
            rows.append(
                {
                    "dataset": dataset,
                    "gfa_path": str(gfa_path),
                    "gfaz_path": str(gfaz_path),
                    "command": "walk",
                    "batch_size": batch_size,
                    "repeat": "",
                    "available_names": available,
                    "duplicate_name_count": 0,
                    "sampled_names": "",
                    "elapsed_seconds": "",
                    "max_rss_kb": "",
                    "status": "skipped_insufficient_names",
                    "error": "",
                    "command_line": "",
                }
            )
            continue

        for repeat in range(1, repeats + 1):
            sampled_walk_keys = rng.sample(walk_keys, batch_size)
            query_args = []
            for walk_key in sampled_walk_keys:
                query_args.extend(walk_key)

            try:
                elapsed, max_rss_kb, command_line = run_extract_benchmark(
                    time_bin=time_bin,
                    gfaz_bin=gfaz_bin,
                    subcommand="extract-walk",
                    gfaz_path=gfaz_path,
                    query_args=query_args,
                    threads=threads,
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "gfa_path": str(gfa_path),
                        "gfaz_path": str(gfaz_path),
                        "command": "walk",
                        "batch_size": batch_size,
                        "repeat": repeat,
                        "available_names": available,
                        "duplicate_name_count": 0,
                        "sampled_names": ";".join(
                            format_walk_key(walk_key)
                            for walk_key in sampled_walk_keys
                        ),
                        "elapsed_seconds": f"{elapsed:.6f}",
                        "max_rss_kb": max_rss_kb if max_rss_kb is not None else "",
                        "status": "ok",
                        "error": "",
                        "command_line": command_line,
                    }
                )
            except RuntimeError as exc:
                rows.append(
                    {
                        "dataset": dataset,
                        "gfa_path": str(gfa_path),
                        "gfaz_path": str(gfaz_path),
                        "command": "walk",
                        "batch_size": batch_size,
                        "repeat": repeat,
                        "available_names": available,
                        "duplicate_name_count": 0,
                        "sampled_names": ";".join(
                            format_walk_key(walk_key)
                            for walk_key in sampled_walk_keys
                        ),
                        "elapsed_seconds": "",
                        "max_rss_kb": "",
                        "status": "error",
                        "error": str(exc),
                        "command_line": "",
                    }
                )


def main() -> int:
    args = parse_args()
    gfa_dir = Path(args.gfa_dir)
    gfaz_dir = Path(args.gfaz_dir)
    gfaz_bin = Path(args.gfaz_bin)
    time_bin = Path(args.time_bin)
    csv_path = Path(args.csv)

    if not gfa_dir.is_dir():
        print(f"Error: gfa_dir is not a directory: {gfa_dir}", file=sys.stderr)
        return 1
    if not gfaz_dir.is_dir():
        print(f"Error: gfaz_dir is not a directory: {gfaz_dir}", file=sys.stderr)
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
    if any(batch_size < 1 for batch_size in args.batch_sizes):
        print("Error: batch sizes must all be positive", file=sys.stderr)
        return 1

    gfa_paths = find_files(gfa_dir, "*.gfa", args.recursive)
    gfaz_paths = find_files(gfaz_dir, "*.gfaz", args.recursive)
    if not gfa_paths:
        print(f"Error: no .gfa files found in {gfa_dir}", file=sys.stderr)
        return 1
    if not gfaz_paths:
        print(f"Error: no .gfaz files found in {gfaz_dir}", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    gfa_index = build_gfa_index(gfa_paths)
    rows = []

    for gfaz_path in gfaz_paths:
        matched_gfa = match_gfaz_to_gfa(gfaz_path, gfa_index)
        dataset = dataset_name(gfaz_path)

        if matched_gfa is None:
            rows.append(
                {
                    "dataset": dataset,
                    "gfa_path": "",
                    "gfaz_path": str(gfaz_path),
                    "command": "",
                    "batch_size": "",
                    "repeat": "",
                    "available_names": "",
                    "duplicate_name_count": "",
                    "sampled_names": "",
                    "elapsed_seconds": "",
                    "max_rss_kb": "",
                    "status": "skipped_missing_gfa_match",
                    "error": "",
                    "command_line": "",
                }
            )
            continue

        try:
            path_names, walk_keys, _ = collect_names(matched_gfa)
        except RuntimeError as exc:
            rows.append(
                {
                    "dataset": dataset,
                    "gfa_path": str(matched_gfa),
                    "gfaz_path": str(gfaz_path),
                    "command": "",
                    "batch_size": "",
                    "repeat": "",
                    "available_names": "",
                    "duplicate_name_count": "",
                    "sampled_names": "",
                    "elapsed_seconds": "",
                    "max_rss_kb": "",
                    "status": "error",
                    "error": str(exc),
                    "command_line": "",
                }
            )
            continue

        unique_path_names, path_duplicate_count = unique_names_with_duplicate_count(
            path_names
        )

        benchmark_name_set(
            rows,
            rng,
            dataset=dataset,
            gfa_path=matched_gfa,
            gfaz_path=gfaz_path,
            subcommand="extract-path",
            label="path",
            names=unique_path_names,
            duplicate_count=path_duplicate_count,
            batch_sizes=args.batch_sizes,
            repeats=args.repeats,
            threads=args.threads,
            gfaz_bin=str(gfaz_bin),
            time_bin=str(time_bin),
        )
        benchmark_walk_set(
            rows,
            rng,
            dataset=dataset,
            gfa_path=matched_gfa,
            gfaz_path=gfaz_path,
            walk_keys=walk_keys,
            batch_sizes=args.batch_sizes,
            repeats=args.repeats,
            threads=args.threads,
            gfaz_bin=str(gfaz_bin),
            time_bin=str(time_bin),
        )

    fieldnames = [
        "dataset",
        "gfa_path",
        "gfaz_path",
        "command",
        "batch_size",
        "repeat",
        "available_names",
        "duplicate_name_count",
        "sampled_names",
        "elapsed_seconds",
        "max_rss_kb",
        "status",
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
