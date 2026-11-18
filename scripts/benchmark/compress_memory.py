#!/usr/bin/env python3
"""Compare peak RSS, wall time, and output identity of `gfaz compress` builds.

Runs are interleaved across configurations (A, B, A, B, ...) so that drifting
machine load affects every configuration equally.

Example:
    python3 scripts/benchmark/compress_memory.py \
        --config base=build_base/bin/gfaz \
        --config "lowmem=build/bin/gfaz --memory-budget 1G --tmp-dir /tmp" \
        --threads 16 --repeat 3 --out results.csv \
        /data1/work/kurty/gfa/chr6.pan.fa.a2fb268.4030258.bc221f9.smooth.gfa
"""

import argparse
import csv
import hashlib
import os
import re
import shlex
import subprocess
import sys
import tempfile


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_time_v(text):
    rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", text)
    wall = re.search(r"Elapsed \(wall clock\) time.*: ([\d:.]+)", text)
    status = re.search(r"Exit status: (\d+)", text)
    seconds = None
    if wall:
        seconds = 0.0
        for part in wall.group(1).split(":"):
            seconds = seconds * 60 + float(part)
    return (
        int(rss.group(1)) / 1024.0 if rss else None,
        seconds,
        int(status.group(1)) if status else None,
    )


def warm_page_cache(path):
    with open(path, "rb") as handle:
        while handle.read(1 << 26):
            pass


def split_config(spec):
    label, _, rest = spec.partition("=")
    parts = shlex.split(rest)
    if not label or not parts:
        raise argparse.ArgumentTypeError(f"bad --config '{spec}'")
    return label, parts[0], parts[1:]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("datasets", nargs="+")
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        type=split_config,
        help="label='binary [extra compress options]'; repeatable",
    )
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--out", help="CSV file to append results to")
    parser.add_argument(
        "--out-dir", help="directory for compressed outputs (default: temp dir)"
    )
    parser.add_argument("--no-warm", action="store_true")
    args = parser.parse_args()

    rows = []
    with tempfile.TemporaryDirectory(dir=args.out_dir) as out_dir:
        for dataset in args.datasets:
            name = os.path.basename(dataset)
            input_gb = os.path.getsize(dataset) / 1e9
            if not args.no_warm:
                warm_page_cache(dataset)
            for rep in range(args.repeat):
                for label, binary, extra in args.config:
                    out_path = os.path.join(out_dir, "out.gfaz")
                    argv = [
                        "/usr/bin/time", "-v", binary, "compress",
                        "-j", str(args.threads), *extra, dataset, out_path,
                    ]
                    proc = subprocess.run(argv, capture_output=True, text=True)
                    rss_mb, wall_s, status = parse_time_v(proc.stderr)
                    if proc.returncode != 0 or status not in (0, None):
                        sys.stderr.write(proc.stderr[-2000:])
                        raise RuntimeError("command failed: " + " ".join(argv))
                    row = {
                        "dataset": name,
                        "config": label,
                        "threads": args.threads,
                        "rep": rep,
                        "input_gb": round(input_gb, 3),
                        "wall_s": round(wall_s, 3),
                        "gb_per_s": round(input_gb / wall_s, 3),
                        "peak_rss_mb": round(rss_mb, 1),
                        "out_bytes": os.path.getsize(out_path),
                        "sha256": sha256_of(out_path)[:16],
                    }
                    os.remove(out_path)
                    rows.append(row)
                    print(
                        "{dataset:28.28s} {config:10s} rep{rep} "
                        "wall={wall_s:8.2f}s  {gb_per_s:6.3f} GB/s  "
                        "rss={peak_rss_mb:10.1f} MB  out={out_bytes}  "
                        "{sha256}".format(**row),
                        flush=True,
                    )

    print("\n=== summary (median wall, max RSS) ===")
    keys = []
    for row in rows:
        key = (row["dataset"], row["config"])
        if key not in keys:
            keys.append(key)
    for dataset, config in keys:
        group = [
            r for r in rows if r["dataset"] == dataset and r["config"] == config
        ]
        walls = sorted(r["wall_s"] for r in group)
        median = walls[len(walls) // 2]
        digests = {r["sha256"] for r in group}
        print(
            f"{dataset:28.28s} {config:10s} wall={median:8.2f}s "
            f"rss={max(r['peak_rss_mb'] for r in group):10.1f} MB "
            f"out={group[0]['out_bytes']} sha={'/'.join(sorted(digests))}"
        )

    if args.out:
        new_file = not os.path.exists(args.out)
        with open(args.out, "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            if new_file:
                writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
