#!/usr/bin/env python3
"""Benchmark compressors on GFA files and write CSV.

Measures:
- compression ratio = input_size / compressed_size
- compression throughput (MB/s) = input_size / comp_time
- decompression throughput (MB/s) = output_size / decomp_time
"""

import argparse
import csv
import os
import shutil
import subprocess
import time
from pathlib import Path

MB = 1024 * 1024

DEFAULT_TOOLS = {
    "gzip": "gzip",
    "bgzip": "bgzip",
    "zstd": "zstd",  # override with --zstd if your binary is named differently
    "sqz": "sqz",
    "gbz": "gfa2gbwt",
    "gfaz": "gfaz",
}


def run_cmd(cmd, *, shell=False, stdout_path=None, env=None, time_cmd=None, cwd=None):
    start = time.perf_counter()
    stderr_data = b""
    if time_cmd:
        if shell:
            timed_cmd = f"{time_cmd} -v {cmd}"
            proc = subprocess.run(
                timed_cmd,
                shell=True,
                check=True,
                env=env,
                stderr=subprocess.PIPE,
                cwd=cwd,
            )
        else:
            timed_cmd = [time_cmd, "-v"] + cmd
            if stdout_path is None:
                proc = subprocess.run(
                    timed_cmd,
                    shell=False,
                    check=True,
                    env=env,
                    stderr=subprocess.PIPE,
                    cwd=cwd,
                )
            else:
                with open(stdout_path, "wb") as f:
                    proc = subprocess.run(
                        timed_cmd,
                        shell=False,
                        check=True,
                        stdout=f,
                        env=env,
                        stderr=subprocess.PIPE,
                        cwd=cwd,
                    )
        stderr_data = proc.stderr or b""
    else:
        if stdout_path is None:
            subprocess.run(cmd, shell=shell, check=True, env=env, cwd=cwd)
        else:
            with open(stdout_path, "wb") as f:
                subprocess.run(cmd, shell=shell, check=True, stdout=f, env=env, cwd=cwd)
    end = time.perf_counter()
    return end - start, stderr_data


def parse_max_rss_kb(stderr_bytes: bytes):
    if not stderr_bytes:
        return None
    text = stderr_bytes.decode(errors="ignore")
    for line in text.splitlines():
        if "Maximum resident set size" in line:
            parts = line.split(":")
            if len(parts) >= 2:
                try:
                    return int(parts[1].strip())
                except ValueError:
                    return None
    return None


def kb_to_gb(kb: int):
    return kb / 1024.0 / 1024.0


def tool_exists(path_or_name: str) -> bool:
    return shutil.which(path_or_name) is not None or Path(path_or_name).exists()


def file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def cleanup_files(*paths):
    """Remove files/directories to free disk space."""
    for p in paths:
        if p is None:
            continue
        path = Path(p)
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="List of .gfa files")
    ap.add_argument("--csv", required=True, help="Output CSV path")
    ap.add_argument("--tmpdir", default="/tmp/gfaz_eval", help="Working directory for outputs")
    ap.add_argument("--gzip", default=DEFAULT_TOOLS["gzip"])
    ap.add_argument("--bgzip", default=DEFAULT_TOOLS["bgzip"])
    ap.add_argument("--zstd", default=DEFAULT_TOOLS["zstd"])
    ap.add_argument("--sqz", default=DEFAULT_TOOLS["sqz"])
    ap.add_argument("--gbz", default=DEFAULT_TOOLS["gbz"])
    ap.add_argument(
        "--gbz-libpath",
        default="",
        help="Path to add to LD_LIBRARY_PATH for gfa2gbwt",
    )
    ap.add_argument("--gbz-threads", type=int, default=16, help="Threads for gfa2gbwt (-P)")
    ap.add_argument(
        "--time-cmd",
        default="",
        help="Path to /usr/bin/time for peak RSS (e.g., /usr/bin/time)",
    )
    ap.add_argument("--gfaz", default=DEFAULT_TOOLS["gfaz"])
    ap.add_argument("--skip-bgzip", action="store_true", help="Skip sqz+bgzip tests")
    ap.add_argument("--skip-sqz", action="store_true", help="Skip all sqz-related tests")
    ap.add_argument("--only-sqz", action="store_true", help="Only run sqz related tests")
    args = ap.parse_args()

    tmpdir = Path(args.tmpdir)
    ensure_dir(tmpdir)

    rows = []

    for inp_str in args.inputs:
        inp = Path(inp_str)
        if not inp.exists():
            raise FileNotFoundError(inp)

        dataset = inp.stem
        input_size = file_size(inp)

        row = {"dataset": dataset}

        # gzip (work in isolated directory)
        if not args.only_sqz:
            gzip_work = tmpdir / f"{dataset}_gzip"
            ensure_dir(gzip_work)
            gzip_input = gzip_work / f"{dataset}.gfa"
            shutil.copy2(inp, gzip_input)
            gz_out = gzip_work / f"{dataset}.gz"
            gz_dec = gzip_work / f"{dataset}.dec.gfa"
            comp_time, comp_err = run_cmd([args.gzip, "-c", str(gzip_input)], stdout_path=gz_out, time_cmd=args.time_cmd or None)
            comp_size = file_size(gz_out)
            decomp_time, decomp_err = run_cmd([args.gzip, "-cd", str(gz_out)], stdout_path=gz_dec, time_cmd=args.time_cmd or None)
            decomp_size = file_size(gz_dec)
            comp_rss = parse_max_rss_kb(comp_err)
            decomp_rss = parse_max_rss_kb(decomp_err)
            row.update({
                "gzip_ratio": (input_size / comp_size) if comp_size else "",
                "gzip_comp_MBps": (input_size / MB) / comp_time if comp_time else "",
                "gzip_decomp_MBps": (decomp_size / MB) / decomp_time if decomp_time else "",
                "gzip_mem_GB": kb_to_gb(comp_rss) if comp_rss is not None else "",
                "gzip_decomp_mem_GB": kb_to_gb(decomp_rss) if decomp_rss is not None else "",
            })
            cleanup_files(gzip_work)

        # zstd (work in isolated directory)
        if not args.only_sqz:
            zstd_work = tmpdir / f"{dataset}_zstd"
            ensure_dir(zstd_work)
            zstd_input = zstd_work / f"{dataset}.gfa"
            shutil.copy2(inp, zstd_input)
            z_out = zstd_work / f"{dataset}.zst"
            z_dec = zstd_work / f"{dataset}.dec.gfa"
            comp_time, comp_err = run_cmd([args.zstd, "-q", "-c", str(zstd_input)], stdout_path=z_out, time_cmd=args.time_cmd or None)
            comp_size = file_size(z_out)
            decomp_time, decomp_err = run_cmd([args.zstd, "-q", "-d", "-c", str(z_out)], stdout_path=z_dec, time_cmd=args.time_cmd or None)
            decomp_size = file_size(z_dec)
            comp_rss = parse_max_rss_kb(comp_err)
            decomp_rss = parse_max_rss_kb(decomp_err)
            row.update({
                "zstd_ratio": (input_size / comp_size) if comp_size else "",
                "zstd_comp_MBps": (input_size / MB) / comp_time if comp_time else "",
                "zstd_decomp_MBps": (decomp_size / MB) / decomp_time if decomp_time else "",
                "zstd_mem_GB": kb_to_gb(comp_rss) if comp_rss is not None else "",
                "zstd_decomp_mem_GB": kb_to_gb(decomp_rss) if decomp_rss is not None else "",
            })
            cleanup_files(zstd_work)

        # Consolidated sqz block (runs sqz once, then reuses for bgzip/zstd)
        if not args.skip_sqz:
            sqz_work = tmpdir / f"{dataset}_sqz_all"
            ensure_dir(sqz_work)
            
            # --- 1. Base SQZ Compression ---
            sqz_input = sqz_work / f"{dataset}.gfa"
            shutil.copy2(inp, sqz_input)
            sqz_out = sqz_work / f"{dataset}.sqz"
            
            sqz_comp_start = time.perf_counter()
            _, sqz_comp_err = run_cmd(f"{args.sqz} compress {sqz_input} > {sqz_out}", shell=True, time_cmd=args.time_cmd or None)
            sqz_comp_dur = time.perf_counter() - sqz_comp_start
            
            sqz_size = file_size(sqz_out)
            sqz_rss = parse_max_rss_kb(sqz_comp_err)
            
            # --- 2. SQZ Decompression ---
            sqz_dec = sqz_work / f"{dataset}.sqz.dec.gfa"
            sqz_decomp_time, sqz_decomp_err = run_cmd(f"{args.sqz} decompress {sqz_out} > {sqz_dec}", shell=True, time_cmd=args.time_cmd or None)
            sqz_decomp_size = file_size(sqz_dec)
            sqz_decomp_rss = parse_max_rss_kb(sqz_decomp_err)
            
            row.update({
                "sqz_ratio": (input_size / sqz_size) if sqz_size else "",
                "sqz_comp_MBps": (input_size / MB) / sqz_comp_dur if sqz_comp_dur else "",
                "sqz_decomp_MBps": (sqz_decomp_size / MB) / sqz_decomp_time if sqz_decomp_time else "",
                "sqz_mem_GB": kb_to_gb(sqz_rss) if sqz_rss is not None else "",
                "sqz_decomp_mem_GB": kb_to_gb(sqz_decomp_rss) if sqz_decomp_rss is not None else "",
            })

            # --- 3. SQZ + BGZIP ---
            if (not args.skip_bgzip) and tool_exists(args.bgzip):
                sqz_bgz_out = sqz_work / f"{dataset}.sqz.bgz"
                
                # Compress existing .sqz with bgzip
                bgz_comp_start = time.perf_counter()
                _, bgz_comp_err = run_cmd(f"{args.bgzip} -c {sqz_out} > {sqz_bgz_out}", shell=True, time_cmd=args.time_cmd or None)
                bgz_comp_dur = time.perf_counter() - bgz_comp_start
                
                # Total stats
                total_comp_time = sqz_comp_dur + bgz_comp_dur
                total_comp_size = file_size(sqz_bgz_out)
                bgz_rss = parse_max_rss_kb(bgz_comp_err) 
                total_rss = max(sqz_rss or 0, bgz_rss or 0) # estimate

                # Decompress: Pipeline (real measurement needed)
                sqz_bgz_dec = sqz_work / f"{dataset}.sqz.bgz.dec.gfa"
                bgz_decomp_time, bgz_decomp_err = run_cmd(
                    f"{args.bgzip} -cd {sqz_bgz_out} | {args.sqz} decompress /dev/stdin > {sqz_bgz_dec}",
                    shell=True,
                    time_cmd=args.time_cmd or None
                )
                bgz_decomp_size = file_size(sqz_bgz_dec)
                bgz_decomp_rss_val = parse_max_rss_kb(bgz_decomp_err)

                row.update({
                    "sqz_bgzip_ratio": (input_size / total_comp_size) if total_comp_size else "",
                    "sqz_bgzip_comp_MBps": (input_size / MB) / total_comp_time if total_comp_time else "",
                    "sqz_bgzip_decomp_MBps": (bgz_decomp_size / MB) / bgz_decomp_time if bgz_decomp_time else "",
                    "sqz_bgzip_mem_GB": kb_to_gb(total_rss),
                    "sqz_bgzip_decomp_mem_GB": kb_to_gb(bgz_decomp_rss_val) if bgz_decomp_rss_val is not None else "",
                })
            else:
                 row.update({
                    "sqz_bgzip_ratio": "", "sqz_bgzip_comp_MBps": "", "sqz_bgzip_decomp_MBps": "",
                    "sqz_bgzip_mem_GB": "", "sqz_bgzip_decomp_mem_GB": "",
                })

            # --- 4. SQZ + ZSTD ---
            if tool_exists(args.zstd):
                sqz_zst_out = sqz_work / f"{dataset}.sqz.zst"
                
                # Compress existing .sqz with zstd
                zst_comp_start = time.perf_counter()
                _, zst_comp_err = run_cmd(f"{args.zstd} -q -c {sqz_out} > {sqz_zst_out}", shell=True, time_cmd=args.time_cmd or None)
                zst_comp_dur = time.perf_counter() - zst_comp_start
                
                # Total stats
                total_comp_time = sqz_comp_dur + zst_comp_dur
                total_comp_size = file_size(sqz_zst_out)
                zst_rss = parse_max_rss_kb(zst_comp_err)
                total_rss = max(sqz_rss or 0, zst_rss or 0)

                # Decompress: Pipeline
                sqz_zst_dec = sqz_work / f"{dataset}.sqz.zst.dec.gfa"
                zst_decomp_time, zst_decomp_err = run_cmd(
                    f"{args.zstd} -q -d -c {sqz_zst_out} | {args.sqz} decompress /dev/stdin > {sqz_zst_dec}",
                    shell=True,
                    time_cmd=args.time_cmd or None
                )
                zst_decomp_size = file_size(sqz_zst_dec)
                zst_decomp_rss_val = parse_max_rss_kb(zst_decomp_err)

                row.update({
                    "sqz_zstd_ratio": (input_size / total_comp_size) if total_comp_size else "",
                    "sqz_zstd_comp_MBps": (input_size / MB) / total_comp_time if total_comp_time else "",
                    "sqz_zstd_decomp_MBps": (zst_decomp_size / MB) / zst_decomp_time if zst_decomp_time else "",
                    "sqz_zstd_mem_GB": kb_to_gb(total_rss),
                    "sqz_zstd_decomp_mem_GB": kb_to_gb(zst_decomp_rss_val) if zst_decomp_rss_val is not None else "",
                })
            else:
                row.update({
                    "sqz_zstd_ratio": "", "sqz_zstd_comp_MBps": "", "sqz_zstd_decomp_MBps": "",
                    "sqz_zstd_mem_GB": "", "sqz_zstd_decomp_mem_GB": "",
                })

            cleanup_files(sqz_work)
        else:
             # Fill empty if skipped
             for p in ["sqz", "sqz_bgzip", "sqz_zstd"]:
                 for s in ["ratio", "comp_MBps", "decomp_MBps", "mem_GB", "decomp_mem_GB"]:
                     row[f"{p}_{s}"] = ""

        # gbz (work in isolated directory)
        if not args.only_sqz:
            gbz_work = tmpdir / f"{dataset}_gbz"
            ensure_dir(gbz_work)
            gbz_input = gbz_work / f"{dataset}.gfa"
            shutil.copy2(inp, gbz_input)
            # gfa2gbwt uses the base path (without .gfa) and creates .gbz in same directory
            base = gbz_input.with_suffix("")
            gbz_env = dict(os.environ)
            gbz_env["LD_LIBRARY_PATH"] = f"{args.gbz_libpath}:{gbz_env.get('LD_LIBRARY_PATH', '')}"
            comp_time, comp_err = run_cmd(
                [args.gbz, "-P", str(args.gbz_threads), "-c", str(base)],
                env=gbz_env,
                time_cmd=args.time_cmd or None,
            )
            gbz_file = base.with_suffix(".gbz")
            comp_size = file_size(gbz_file)
            decomp_time, decomp_err = run_cmd(
                [args.gbz, "-P", str(args.gbz_threads), "-d", str(base)],
                env=gbz_env,
                time_cmd=args.time_cmd or None,
            )
            gbz_dec = base.with_suffix(".gfa")
            decomp_size = file_size(gbz_dec)
            comp_rss = parse_max_rss_kb(comp_err)
            decomp_rss = parse_max_rss_kb(decomp_err)
            row.update({
                "gbz_ratio": (input_size / comp_size) if comp_size else "",
                "gbz_comp_MBps": (input_size / MB) / comp_time if comp_time else "",
                "gbz_decomp_MBps": (decomp_size / MB) / decomp_time if decomp_time else "",
                "gbz_mem_GB": kb_to_gb(comp_rss) if comp_rss is not None else "",
                "gbz_decomp_mem_GB": kb_to_gb(decomp_rss) if decomp_rss is not None else "",
            })
            cleanup_files(gbz_work)

        # gfaz (work in isolated directory)
        if not args.only_sqz:
            gfaz_work = tmpdir / f"{dataset}_gfaz"
            ensure_dir(gfaz_work)
            gfaz_input = gfaz_work / f"{dataset}.gfa"
            shutil.copy2(inp, gfaz_input)
            gfaz_out = gfaz_work / f"{dataset}.gfaz"
            gfaz_dec = gfaz_work / f"{dataset}.dec.gfa"
            comp_time, comp_err = run_cmd([args.gfaz, "compress", str(gfaz_input), str(gfaz_out)], time_cmd=args.time_cmd or None)
            comp_size = file_size(gfaz_out)
            decomp_time, decomp_err = run_cmd([args.gfaz, "decompress", str(gfaz_out), str(gfaz_dec)], time_cmd=args.time_cmd or None)
            decomp_size = file_size(gfaz_dec)
            comp_rss = parse_max_rss_kb(comp_err)
            decomp_rss = parse_max_rss_kb(decomp_err)
            row.update({
                "gfaz_ratio": (input_size / comp_size) if comp_size else "",
                "gfaz_comp_MBps": (input_size / MB) / comp_time if comp_time else "",
                "gfaz_decomp_MBps": (decomp_size / MB) / decomp_time if decomp_time else "",
                "gfaz_mem_GB": kb_to_gb(comp_rss) if comp_rss is not None else "",
                "gfaz_decomp_mem_GB": kb_to_gb(decomp_rss) if decomp_rss is not None else "",
            })
            cleanup_files(gfaz_work)

        rows.append(row)

    fields = [
        "dataset",
        "gzip_ratio", "gzip_comp_MBps", "gzip_decomp_MBps", "gzip_mem_GB", "gzip_decomp_mem_GB",
        "zstd_ratio", "zstd_comp_MBps", "zstd_decomp_MBps", "zstd_mem_GB", "zstd_decomp_mem_GB",
        "sqz_ratio", "sqz_comp_MBps", "sqz_decomp_MBps", "sqz_mem_GB", "sqz_decomp_mem_GB",
        "sqz_bgzip_ratio", "sqz_bgzip_comp_MBps", "sqz_bgzip_decomp_MBps", "sqz_bgzip_mem_GB", "sqz_bgzip_decomp_mem_GB",
        "sqz_zstd_ratio", "sqz_zstd_comp_MBps", "sqz_zstd_decomp_MBps", "sqz_zstd_mem_GB", "sqz_zstd_decomp_mem_GB",
        "gbz_ratio", "gbz_comp_MBps", "gbz_decomp_MBps", "gbz_mem_GB", "gbz_decomp_mem_GB",
        "gfaz_ratio", "gfaz_comp_MBps", "gfaz_decomp_MBps", "gfaz_mem_GB", "gfaz_decomp_mem_GB",
    ]

    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
