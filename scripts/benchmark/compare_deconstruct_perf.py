#!/usr/bin/env python3
"""Benchmark `vg deconstruct` against `gfaz deconstruct`.

Takes a GFA file as input. `vg` reads the GFA directly; for `gfaz` the GFA is
compressed to a `.gfaz` container first (the compression step is timed and
reported separately, not counted in the vg-vs-gfaz deconstruct comparison).

Each command runs under `/usr/bin/time -v`, capturing elapsed wall time, peak
RSS, exit code, and -- for deconstruct runs -- the number of VCF records
emitted. Results are written as JSON + CSV and printed as a table with ratios.

gfaz deconstruct is benchmarked in `--snarl` mode by default (the topology-based
mode that mirrors `vg deconstruct`). Use --gfaz-modes to also time the linear
heuristic and the vg-compat (acyclic-reference) refinement.

Examples:
  # From a GFA (auto-compresses to a temp .gfaz):
  python scripts/benchmark/compare_deconstruct_perf.py \
      --gfa /home/kurty/data/chrY.pan...smooth.gfa --ref grch38#chrY

  # Reuse an existing .gfaz and time all three gfaz modes with 8 threads:
  python scripts/benchmark/compare_deconstruct_perf.py --preset chrY \
      --gfaz /home/kurty/data/chrY.gfaz --gfaz-modes all --threads 8

  # Auto-detect the reference (first P/W line) and keep the output VCFs:
  python scripts/benchmark/compare_deconstruct_perf.py --gfa graph.gfa --keep-output
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VG = Path("/home/kurty/Release/vg/bin/vg")
DEFAULT_GFAZ = REPO_ROOT / "build" / "bin" / "gfaz"
DEFAULT_DATA = Path("/home/kurty/data")
PRESETS = {
    "chrY": {
        "gfa": DEFAULT_DATA / "chrY.pan.fa.a2fb268.4030258.bc221f9.smooth.gfa",
        "gfaz": DEFAULT_DATA / "chrY.gfaz",
        "ref": "grch38#chrY",
    },
    "chr1": {
        "gfa": DEFAULT_DATA / "chr1.pan.fa.a2fb268.4030258.bc221f9.smooth.gfa",
        "gfaz": DEFAULT_DATA / "chr1.gfaz",
        "ref": "chm13#chr1",
    },
}

# gfaz deconstruct mode -> extra CLI flags (all are per-sample, -S).
GFAZ_MODES = {
    "snarl": ["-S", "--snarl"],
    "linear": ["-S"],
    "vg-compat": ["-S", "--vg-compat"],
}


@dataclass
class BenchResult:
    dataset: str
    tool: str
    command: str
    exit_code: int | None
    timed_out: bool
    wall_seconds: float
    time_elapsed: str | None
    user_seconds: float | None
    system_seconds: float | None
    max_rss_kb: int | None
    record_count: int | None
    stdout_path: str
    stdout_bytes: int | None
    stderr_path: str
    time_log_path: str


def first_reference_name(gfa: Path) -> str:
    """Return the first P-line path name, falling back to W-line sample#hap#seq."""
    with gfa.open("rt", errors="replace") as handle:
        for line in handle:
            if line.startswith("P\t"):
                parts = line.split("\t", 2)
                if len(parts) > 1 and parts[1]:
                    return parts[1]
            if line.startswith("W\t"):
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 5:
                    return f"{parts[1]}#{parts[2]}#{parts[3]}"
    raise RuntimeError(f"No P/W reference path found in {gfa}")


def preferred_reference_name(gfa: Path) -> str:
    """Prefer a common linear reference (grch38/chm13) if present, else first P/W."""
    preferred_first: str | None = None
    fallback: str | None = None
    with gfa.open("rt", errors="replace") as handle:
        for line in handle:
            name: str | None = None
            if line.startswith("P\t"):
                parts = line.split("\t", 2)
                name = parts[1] if len(parts) > 1 and parts[1] else None
            elif line.startswith("W\t"):
                parts = line.rstrip("\n").split("\t")
                name = f"{parts[1]}#{parts[2]}#{parts[3]}" if len(parts) >= 5 else None
            if name is None:
                continue
            if fallback is None:
                fallback = name
            if preferred_first is None and re.search(r"grch38|chm13", name, re.I):
                preferred_first = name
                break
    chosen = preferred_first or fallback
    if chosen is None:
        raise RuntimeError(f"No P/W reference path found in {gfa}")
    return chosen


def parse_time_v(stderr_text: str) -> dict[str, object]:
    out: dict[str, object] = {
        "time_elapsed": None,
        "user_seconds": None,
        "system_seconds": None,
        "max_rss_kb": None,
    }
    patterns = {
        "time_elapsed": r"Elapsed \(wall clock\) time .*: (.+)",
        "user_seconds": r"User time \(seconds\): ([0-9.]+)",
        "system_seconds": r"System time \(seconds\): ([0-9.]+)",
        "max_rss_kb": r"Maximum resident set size \(kbytes\): ([0-9]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, stderr_text)
        if not match:
            continue
        value = match.group(1).strip()
        if key in {"user_seconds", "system_seconds"}:
            out[key] = float(value)
        elif key == "max_rss_kb":
            out[key] = int(value)
        else:
            out[key] = value
    return out


def count_vcf_records(path: Path) -> int:
    n = 0
    with path.open("rt", errors="replace") as handle:
        for line in handle:
            if line and not line.startswith("#"):
                n += 1
    return n


def require_file(path: Path, label: str, executable: bool = False) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if executable and not os.access(path, os.X_OK):
        raise PermissionError(f"{label} is not executable: {path}")


def run_one(
    *,
    dataset: str,
    tool: str,
    command: list[str],
    out_dir: Path,
    timeout: int,
    is_vcf: bool,
    keep_output: bool,
) -> BenchResult:
    """Run one command under /usr/bin/time -v.

    stdout is always streamed to a file so VCF records can be counted; for
    deconstruct runs (is_vcf) the file is deleted afterwards unless keep_output.
    """
    suffix = "vcf" if is_vcf else "out"
    stdout_path = out_dir / f"{dataset}.{tool}.{suffix}"
    stderr_path = out_dir / f"{dataset}.{tool}.stderr.txt"
    time_log_path = out_dir / f"{dataset}.{tool}.time.txt"
    combined_stderr_path = out_dir / f"{dataset}.{tool}.combined_stderr.txt"

    timed_out = False
    exit_code: int | None = None
    start = time.perf_counter()
    try:
        with stdout_path.open("wb") as stdout_handle, combined_stderr_path.open("wb") as stderr_handle:
            full_cmd = ["/usr/bin/time", "-v", *command]
            try:
                proc = subprocess.run(
                    full_cmd,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    timeout=timeout,
                    check=False,
                )
                exit_code = proc.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
                exit_code = None
    finally:
        wall_seconds = time.perf_counter() - start

    combined = combined_stderr_path.read_text(errors="replace")
    time_log_path.write_text(combined)
    stderr_path.write_text(strip_time_v(combined))
    parsed = parse_time_v(combined)

    record_count: int | None = None
    stdout_bytes: int | None = stdout_path.stat().st_size if stdout_path.exists() else None
    if is_vcf and not timed_out and exit_code == 0 and stdout_path.exists():
        record_count = count_vcf_records(stdout_path)

    final_stdout_path = stdout_path
    if is_vcf and not keep_output and stdout_path.exists():
        stdout_path.unlink()
        final_stdout_path = Path(os.devnull)

    return BenchResult(
        dataset=dataset,
        tool=tool,
        command=" ".join(command),
        exit_code=exit_code,
        timed_out=timed_out,
        wall_seconds=wall_seconds,
        time_elapsed=parsed["time_elapsed"],
        user_seconds=parsed["user_seconds"],
        system_seconds=parsed["system_seconds"],
        max_rss_kb=parsed["max_rss_kb"],
        record_count=record_count,
        stdout_path=str(final_stdout_path),
        stdout_bytes=stdout_bytes,
        stderr_path=str(stderr_path),
        time_log_path=str(time_log_path),
    )


def strip_time_v(text: str) -> str:
    marker = "\tCommand being timed:"
    idx = text.find(marker)
    if idx == -1:
        return text
    return text[:idx].rstrip() + "\n"


def write_reports(results: Iterable[BenchResult], out_dir: Path) -> None:
    rows = [asdict(r) for r in results]
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (out_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_report(results: list[BenchResult], ref: str, gfa: Path, gfaz: Path, out_dir: Path) -> None:
    print(f"dataset_ref\t{ref}")
    print(f"gfa\t{gfa}")
    print(f"gfaz\t{gfaz}")
    print(f"out_dir\t{out_dir}")
    print("tool\texit\ttimeout\twall_s\tmax_rss_gib\trecords\tuser_s\tsys_s")
    for r in results:
        rss_gib = "NA" if r.max_rss_kb is None else f"{r.max_rss_kb / (1024 ** 2):.3f}"
        user_s = "NA" if r.user_seconds is None else f"{r.user_seconds:.3f}"
        sys_s = "NA" if r.system_seconds is None else f"{r.system_seconds:.3f}"
        exit_s = "NA" if r.exit_code is None else str(r.exit_code)
        records = "NA" if r.record_count is None else str(r.record_count)
        print(
            f"{r.tool}\t{exit_s}\t{int(r.timed_out)}\t{r.wall_seconds:.3f}"
            f"\t{rss_gib}\t{records}\t{user_s}\t{sys_s}"
        )

    ok = {
        r.tool: r
        for r in results
        if not r.timed_out and r.exit_code == 0 and r.max_rss_kb
    }
    # Compare vg against each gfaz deconstruct mode (skip the compress row).
    if "vg" in ok:
        vg = ok["vg"]
        gfaz_tools = [t for t in ok if t.startswith("gfaz-decon")]
        if gfaz_tools:
            print("\nratios (vg / gfaz)")
        for t in gfaz_tools:
            g = ok[t]
            speedup = vg.wall_seconds / g.wall_seconds if g.wall_seconds else float("nan")
            rss_red = vg.max_rss_kb / g.max_rss_kb if g.max_rss_kb else float("nan")
            print(f"{t}\tspeedup\t{speedup:.3f}x\trss_reduction\t{rss_red:.3f}x")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--preset", choices=sorted(PRESETS), help="Seed gfa/gfaz/ref defaults.")
    parser.add_argument("--gfa", type=Path, help="Input GFA. Required unless --preset is given.")
    parser.add_argument(
        "--gfaz",
        type=Path,
        help="Reuse an existing .gfaz. If omitted, the GFA is compressed into --out-dir.",
    )
    parser.add_argument("--ref", help="Reference path name. Default: preset, then a grch38/chm13 P/W line, then the first.")
    parser.add_argument("--vg-bin", type=Path, default=DEFAULT_VG)
    parser.add_argument("--gfaz-bin", type=Path, default=DEFAULT_GFAZ)
    parser.add_argument(
        "--gfaz-modes",
        default="snarl",
        help="Comma-separated gfaz deconstruct modes to time (%s), or 'all'." % ",".join(GFAZ_MODES),
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=3600, help="Per-command timeout in seconds.")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--keep-output", action="store_true", help="Keep VCF stdout files (default: count then delete).")
    parser.add_argument("--skip-vg", action="store_true")
    parser.add_argument("--skip-gfaz", action="store_true")
    args = parser.parse_args(argv)

    preset = PRESETS.get(args.preset, {}) if args.preset else {}
    gfa = args.gfa or preset.get("gfa")
    if gfa is None:
        parser.error("provide --gfa (or --preset)")
    gfa = Path(gfa)

    if args.gfaz_modes.strip().lower() == "all":
        modes = list(GFAZ_MODES)
    else:
        modes = [m.strip() for m in args.gfaz_modes.split(",") if m.strip()]
    bad = [m for m in modes if m not in GFAZ_MODES]
    if bad:
        parser.error(f"unknown --gfaz-modes: {bad}. choose from {list(GFAZ_MODES)} or 'all'")

    require_file(Path("/usr/bin/time"), "/usr/bin/time", executable=True)
    require_file(gfa, "GFA input")
    if not args.skip_vg:
        require_file(args.vg_bin, "vg binary", executable=True)
    if not args.skip_gfaz:
        require_file(args.gfaz_bin, "gfaz binary", executable=True)

    ref = args.ref or preset.get("ref") or preferred_reference_name(gfa)

    dataset = args.preset or gfa.stem.split(".")[0]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (REPO_ROOT / "benchmark_results" / f"deconstruct_{dataset}_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[BenchResult] = []

    # --- gfaz: ensure a .gfaz exists, timing compression if we build it. ---
    gfaz = Path(args.gfaz) if args.gfaz else (preset.get("gfaz") if preset else None)
    gfaz = Path(gfaz) if gfaz else None
    if not args.skip_gfaz:
        if gfaz is None or not gfaz.exists():
            gfaz = out_dir / f"{gfa.stem}.gfaz"
            compress_cmd = [str(args.gfaz_bin), "compress", str(gfa), str(gfaz)]
            results.append(
                run_one(
                    dataset=dataset,
                    tool="gfaz-compress",
                    command=compress_cmd,
                    out_dir=out_dir,
                    timeout=args.timeout,
                    is_vcf=False,
                    keep_output=True,
                )
            )
            if not gfaz.exists():
                raise RuntimeError(f"compression did not produce {gfaz}; see logs in {out_dir}")

    # --- vg deconstruct (reads the GFA directly). ---
    if not args.skip_vg:
        vg_cmd = [str(args.vg_bin), "deconstruct", "-p", ref, "-t", str(args.threads), str(gfa)]
        results.append(
            run_one(
                dataset=dataset,
                tool="vg",
                command=vg_cmd,
                out_dir=out_dir,
                timeout=args.timeout,
                is_vcf=True,
                keep_output=args.keep_output,
            )
        )

    # --- gfaz deconstruct, one run per selected mode. ---
    if not args.skip_gfaz:
        for mode in modes:
            gfaz_cmd = [
                str(args.gfaz_bin), "deconstruct", "-i", str(gfaz), "-r", ref,
                "-t", str(args.threads), *GFAZ_MODES[mode],
            ]
            results.append(
                run_one(
                    dataset=dataset,
                    tool=f"gfaz-decon-{mode}",
                    command=gfaz_cmd,
                    out_dir=out_dir,
                    timeout=args.timeout,
                    is_vcf=True,
                    keep_output=args.keep_output,
                )
            )

    if not results:
        raise RuntimeError("No tools selected")
    write_reports(results, out_dir)
    print_report(results, ref, gfa, gfaz if gfaz else Path("-"), out_dir)
    return 0 if all((not r.timed_out and r.exit_code == 0) for r in results) else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
