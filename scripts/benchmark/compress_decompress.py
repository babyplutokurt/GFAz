#!/usr/bin/env python3
"""Hot-cache throughput and peak RSS of `gfaz compress` and `gfaz decompress`.

For each GFA: read it into the page cache, compress it (default, and with
`--memory-budget` when `--budget` is given), then decompress the result to
/dev/null and to a file. Every run goes through `/usr/bin/time -v`. Throughput
is the GFA size over end-to-end wall time.

The input is re-read before every compression run, and the fraction of it that
is resident in the page cache is recorded before and after, so a run that was
not actually hot shows up in the results.

Example:
    python3 scripts/benchmark/compress_decompress.py \
        --budget 2G --work-dir /path/on/nvme --repeat 3 --out results.csv \
        /data/chr1.gfa /data/hprc-v1.1-mc-chm13.gfa

A file close to the size of RAM only stays cached if nothing else competes for
the page cache. `--evict` drops other files from the cache first (repeatable;
directories are expanded to the `*.gfa` files inside):
    ... --evict /data --repeat 1 /data/hprc-v2.1-mc-chm13.gfa
"""

import argparse
import csv
import ctypes
import glob
import hashlib
import mmap
import os
import re
import subprocess
import sys
import time

FIELDS = [
    "dataset", "op", "config", "threads", "rep", "input_gb", "wall_s",
    "gb_per_s", "peak_rss_mb", "out_bytes", "sha256", "cached_before",
    "cached_after", "load1",
]


def log(message):
    print(time.strftime("%H:%M:%S ") + message, flush=True)


def warm_page_cache(path):
    buffer = bytearray(1 << 26)
    with open(path, "rb", buffering=0) as handle:
        while handle.readinto(buffer):
            pass


def evict_from_page_cache(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def cached_fraction(path):
    """Fraction of the file resident in the page cache (mincore)."""
    size = os.path.getsize(path)
    if size == 0:
        return 1.0
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                          ctypes.c_int, ctypes.c_int, ctypes.c_long]
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    page = mmap.PAGESIZE
    window = 1 << 30
    resident = 0
    fd = os.open(path, os.O_RDONLY)
    try:
        for offset in range(0, size, window):
            length = min(window, size - offset)
            pages = (length + page - 1) // page
            address = libc.mmap(None, length, mmap.PROT_READ, mmap.MAP_SHARED,
                                fd, offset)
            if address in (None, ctypes.c_void_p(-1).value):
                raise OSError(ctypes.get_errno(), "mmap failed for " + path)
            vector = ctypes.create_string_buffer(pages)
            libc.mincore(address, length, vector)
            resident += sum(1 for byte in vector.raw if byte & 1)
            libc.munmap(address, length)
    finally:
        os.close(fd)
    return min(1.0, resident * page / size)


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def run_timed(argv):
    """Returns (wall seconds, peak RSS in MB)."""
    proc = subprocess.run(["/usr/bin/time", "-v", *argv],
                          capture_output=True, text=True)
    rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", proc.stderr)
    wall = re.search(r"Elapsed \(wall clock\) time.*: ([\d:.]+)", proc.stderr)
    if proc.returncode != 0 or not rss or not wall:
        sys.stderr.write(proc.stderr[-3000:])
        raise RuntimeError("command failed: " + " ".join(argv))
    seconds = 0.0
    for part in wall.group(1).split(":"):
        seconds = seconds * 60 + float(part)
    return seconds, int(rss.group(1)) / 1024.0


def expand_evict_targets(targets, keep):
    paths = []
    for target in targets:
        if os.path.isdir(target):
            paths.extend(sorted(glob.glob(os.path.join(target, "*.gfa"))))
        else:
            paths.append(target)
    keep = os.path.realpath(keep)
    return [p for p in paths if os.path.realpath(p) != keep]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--gfaz-bin", default="build/bin/gfaz")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--budget",
        help="also compress with --memory-budget <SIZE> (e.g. 2G)")
    parser.add_argument(
        "--budget-first", action="store_true",
        help="run the budget configuration before the default one; use it "
        "when the default run's memory would push the input out of the cache")
    parser.add_argument(
        "--work-dir", default=".",
        help="directory for the .gfaz, the decompressed GFA and the spill "
        "file; put it on fast local storage (default: current directory)")
    parser.add_argument(
        "--evict", action="append", default=[],
        help="file or directory of *.gfa to drop from the page cache before "
        "each dataset; repeatable")
    parser.add_argument("--skip-decompress", action="store_true")
    parser.add_argument("--out", help="CSV file to append results to")
    args = parser.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    gfaz_path = os.path.join(args.work_dir, "bench.gfaz")
    gfa_path = os.path.join(args.work_dir, "bench.gfa")
    configs = [("default", [])]
    if args.budget:
        budget = ("budget" + args.budget,
                  ["--memory-budget", args.budget, "--tmp-dir", args.work_dir])
        configs = [budget] + configs if args.budget_first else configs + [budget]

    rows = []

    def record(**row):
        rows.append(row)
        log("  {op:10s} {config:10s} rep{rep} wall={wall_s:9.2f}s "
            "{gb_per_s:6.3f} GB/s rss={peak_rss_mb:10.1f} MB "
            "cached={cached_before}->{cached_after} {sha256}".format(**row))

    try:
        for dataset in args.datasets:
            name = os.path.basename(dataset)
            input_gb = os.path.getsize(dataset) / 1e9
            log(f"=== {name} ({input_gb:.2f} GB)")
            for other in expand_evict_targets(args.evict, dataset):
                evict_from_page_cache(other)
            warm_page_cache(dataset)

            reference = None
            for label, extra in configs:
                for rep in range(args.repeat):
                    warm_page_cache(dataset)
                    before = cached_fraction(dataset)
                    wall, rss = run_timed([
                        args.gfaz_bin, "compress", "-j", str(args.threads),
                        *extra, dataset, gfaz_path])
                    digest = sha256_of(gfaz_path)
                    reference = reference or digest
                    if digest != reference:
                        log(f"  WARNING: output differs from {reference}")
                    record(dataset=name, op="compress", config=label,
                           threads=args.threads, rep=rep,
                           input_gb=round(input_gb, 3), wall_s=round(wall, 3),
                           gb_per_s=round(input_gb / wall, 3),
                           peak_rss_mb=round(rss, 1),
                           out_bytes=os.path.getsize(gfaz_path), sha256=digest,
                           cached_before=round(before, 3),
                           cached_after=round(cached_fraction(dataset), 3),
                           load1=round(os.getloadavg()[0], 2))

            if not args.skip_decompress:
                warm_page_cache(gfaz_path)
                for label, target in (("to_null", "/dev/null"),
                                      ("to_file", gfa_path)):
                    for rep in range(args.repeat):
                        if os.path.exists(gfa_path):
                            os.remove(gfa_path)
                        wall, rss = run_timed([
                            args.gfaz_bin, "decompress", "-j",
                            str(args.threads), gfaz_path, target])
                        record(dataset=name, op="decompress", config=label,
                               threads=args.threads, rep=rep,
                               input_gb=round(input_gb, 3),
                               wall_s=round(wall, 3),
                               gb_per_s=round(input_gb / wall, 3),
                               peak_rss_mb=round(rss, 1),
                               out_bytes=(os.path.getsize(gfa_path)
                                          if target == gfa_path else 0),
                               sha256="", cached_before="", cached_after="",
                               load1=round(os.getloadavg()[0], 2))
    finally:
        for path in (gfa_path, gfaz_path):
            if os.path.exists(path):
                os.remove(path)
        if args.out and rows:
            new_file = not os.path.exists(args.out)
            with open(args.out, "a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                if new_file:
                    writer.writeheader()
                writer.writerows(rows)

    print("\n=== summary (median wall, max RSS) ===")
    keys = []
    for row in rows:
        key = (row["dataset"], row["op"], row["config"])
        if key not in keys:
            keys.append(key)
    for dataset, op, config in keys:
        group = [r for r in rows
                 if (r["dataset"], r["op"], r["config"]) == (dataset, op, config)]
        walls = sorted(r["wall_s"] for r in group)
        median = walls[len(walls) // 2]
        print(f"{dataset:34.34s} {op:10s} {config:10s} wall={median:9.2f}s "
              f"{group[0]['input_gb'] / median:6.3f} GB/s "
              f"rss={max(r['peak_rss_mb'] for r in group) / 1024:7.2f} GB")


if __name__ == "__main__":
    main()
