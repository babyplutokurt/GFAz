#!/usr/bin/env python3

import argparse
import os
import subprocess
import tempfile
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark gfaz CPU decompression modes on a .gfaz file and "
            "compare wall time and peak RSS."
        )
    )
    parser.add_argument("gfaz_file", help="Input CPU .gfaz file")
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
        help="Threads to pass to gfaz decompress (default: 0 = auto)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of runs per mode (default: 1)",
    )
    return parser.parse_args()


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


def format_seconds(seconds: float) -> str:
    return f"{seconds:.3f}s"


def format_rss_kb(rss_kb: int | None) -> str:
    if rss_kb is None:
        return "n/a"
    return f"{rss_kb / 1024.0:.1f} MB"


def run_mode(
    *,
    time_bin: str,
    gfaz_bin: str,
    gfaz_file: str,
    threads: int,
    legacy: bool,
) -> tuple[float, int | None, str]:
    fd, output_path = tempfile.mkstemp(suffix=".gfa", prefix="gfaz_bench_")
    os.close(fd)

    cmd = [
        time_bin,
        "-v",
        gfaz_bin,
        "decompress",
        "-j",
        str(threads),
    ]
    if legacy:
        cmd.append("--legacy")
    cmd.extend([gfaz_file, output_path])

    try:
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
    finally:
        if os.path.exists(output_path):
            os.remove(output_path)


def summarize(values: list[float]) -> float:
    return sum(values) / len(values)


def summarize_int(values: list[int | None]) -> int | None:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return max(filtered)


def main() -> int:
    args = parse_args()

    gfaz_file = Path(args.gfaz_file)
    if not gfaz_file.exists():
        raise FileNotFoundError(f".gfaz file not found: {gfaz_file}")

    modes = [
        ("streaming", False),
        ("legacy", True),
    ]

    results: list[dict[str, object]] = []

    print("=== CPU Decompression Mode Benchmark ===")
    print(f"Input:    {gfaz_file}")
    print(f"Threads:  {args.threads}")
    print(f"Repeats:  {args.repeats}")
    print()

    for label, legacy in modes:
        times: list[float] = []
        rss_values: list[int | None] = []
        command_line = ""

        for repeat in range(args.repeats):
            elapsed, rss_kb, command_line = run_mode(
                time_bin=args.time_bin,
                gfaz_bin=args.gfaz_bin,
                gfaz_file=str(gfaz_file),
                threads=args.threads,
                legacy=legacy,
            )
            times.append(elapsed)
            rss_values.append(rss_kb)
            print(
                f"{label:10s} run {repeat + 1}: "
                f"time={format_seconds(elapsed):>8s}  "
                f"peak_rss={format_rss_kb(rss_kb):>10s}"
            )

        results.append(
            {
                "mode": label,
                "avg_time": summarize(times),
                "peak_rss_kb": summarize_int(rss_values),
                "command_line": command_line,
            }
        )

    print("\n--- Summary ---")
    print(f"{'Mode':10s} {'Avg Time':>12s} {'Peak RSS':>12s}")
    for result in results:
        print(
            f"{result['mode']:10s} "
            f"{format_seconds(result['avg_time']):>12s} "
            f"{format_rss_kb(result['peak_rss_kb']):>12s}"
        )

    streaming = next(result for result in results if result["mode"] == "streaming")
    legacy = next(result for result in results if result["mode"] == "legacy")

    time_ratio = (
        legacy["avg_time"] / streaming["avg_time"]
        if streaming["avg_time"] > 0
        else float("inf")
    )
    print("\n--- Comparison ---")
    print(f"legacy/streaming time ratio: {time_ratio:.3f}x")

    if streaming["peak_rss_kb"] is not None and legacy["peak_rss_kb"] is not None:
        rss_ratio = legacy["peak_rss_kb"] / streaming["peak_rss_kb"]
        print(f"legacy/streaming peak RSS ratio: {rss_ratio:.3f}x")

    print("\nCommands:")
    for result in results:
        print(f"{result['mode']:10s} {result['command_line']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
