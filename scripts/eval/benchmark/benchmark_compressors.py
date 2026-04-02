#!/usr/bin/env python3
"""Benchmark compressors on GFA files and write a summary CSV."""

import argparse
import csv
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

MB = 1024 * 1024
EMPTY_METRICS = {
    "ratio": "",
    "comp_MBps": "",
    "decomp_MBps": "",
    "mem_GB": "",
    "decomp_mem_GB": "",
}
CSV_FIELDS = [
    "dataset",
    "gzip_ratio", "gzip_comp_MBps", "gzip_decomp_MBps", "gzip_mem_GB", "gzip_decomp_mem_GB",
    "zstd_ratio", "zstd_comp_MBps", "zstd_decomp_MBps", "zstd_mem_GB", "zstd_decomp_mem_GB",
    "sqz_ratio", "sqz_comp_MBps", "sqz_decomp_MBps", "sqz_mem_GB", "sqz_decomp_mem_GB",
    "sqz_bgzip_ratio", "sqz_bgzip_comp_MBps", "sqz_bgzip_decomp_MBps", "sqz_bgzip_mem_GB", "sqz_bgzip_decomp_mem_GB",
    "sqz_zstd_ratio", "sqz_zstd_comp_MBps", "sqz_zstd_decomp_MBps", "sqz_zstd_mem_GB", "sqz_zstd_decomp_mem_GB",
    "gbz_ratio", "gbz_comp_MBps", "gbz_decomp_MBps", "gbz_mem_GB", "gbz_decomp_mem_GB",
    "gfaz_ratio", "gfaz_comp_MBps", "gfaz_decomp_MBps", "gfaz_mem_GB", "gfaz_decomp_mem_GB",
]
DEFAULT_TOOLS = {
    "gzip": "gzip",
    "bgzip": "bgzip",
    "zstd": "zstd",
    "sqz": "sqz",
    "gbz": "gfa2gbwt",
    "gfaz": "gfaz",
}


@dataclass
class CommandResult:
  elapsed_seconds: float
  stderr_bytes: bytes


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Benchmark compressor ratio, throughput, and peak RSS on GFA files."
  )
  parser.add_argument("--inputs", nargs="+", required=True, help="Input .gfa files")
  parser.add_argument("--csv", required=True, help="Output CSV path")
  parser.add_argument(
      "--tmpdir",
      default="/tmp/gfaz_eval",
      help="Working directory for temporary benchmark files",
  )
  parser.add_argument("--gzip", default=DEFAULT_TOOLS["gzip"], help="Path to gzip")
  parser.add_argument("--bgzip", default=DEFAULT_TOOLS["bgzip"], help="Path to bgzip")
  parser.add_argument("--zstd", default=DEFAULT_TOOLS["zstd"], help="Path to zstd")
  parser.add_argument("--sqz", default=DEFAULT_TOOLS["sqz"], help="Path to sqz")
  parser.add_argument("--gbz", default=DEFAULT_TOOLS["gbz"], help="Path to gfa2gbwt")
  parser.add_argument("--gfaz", default=DEFAULT_TOOLS["gfaz"], help="Path to gfaz")
  parser.add_argument(
      "--gbz-libpath",
      default="",
      help="Path prefix to prepend to LD_LIBRARY_PATH for gfa2gbwt",
  )
  parser.add_argument(
      "--gbz-threads",
      type=int,
      default=16,
      help="Threads for gfa2gbwt (-P)",
  )
  parser.add_argument(
      "--time-cmd",
      default="",
      help="Path to /usr/bin/time for peak RSS capture",
  )
  parser.add_argument("--skip-bgzip", action="store_true", help="Skip sqz+bgzip tests")
  parser.add_argument("--skip-gbz", action="store_true", help="Skip GBZ tests")
  parser.add_argument("--skip-sqz", action="store_true", help="Skip all sqz-related tests")
  parser.add_argument("--only-sqz", action="store_true", help="Only run sqz-related tests")
  return parser.parse_args()


def tool_exists(path_or_name: str) -> bool:
  return shutil.which(path_or_name) is not None or Path(path_or_name).exists()


def file_size(path: Path) -> int:
  return path.stat().st_size if path.exists() else 0


def ensure_dir(path: Path) -> None:
  path.mkdir(parents=True, exist_ok=True)


def cleanup_paths(*paths: Path) -> None:
  for path in paths:
    if path.exists():
      if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
      else:
        path.unlink(missing_ok=True)


def parse_max_rss_kb(stderr_bytes: bytes) -> int | None:
  if not stderr_bytes:
    return None
  text = stderr_bytes.decode(errors="ignore")
  for line in text.splitlines():
    if "Maximum resident set size" in line:
      _, _, value = line.partition(":")
      try:
        return int(value.strip())
      except ValueError:
        return None
  return None


def kb_to_gb(kb: int | None) -> float | str:
  if kb is None:
    return ""
  return kb / 1024.0 / 1024.0


def ratio_bytes(input_size: int, compressed_size: int) -> float | str:
  if compressed_size == 0:
    return ""
  return input_size / compressed_size


def throughput_mb_per_s(size_bytes: int, elapsed_seconds: float) -> float | str:
  if elapsed_seconds == 0:
    return ""
  return (size_bytes / MB) / elapsed_seconds


def metrics_dict(
    *,
    input_size: int,
    compressed_size: int,
    decompressed_size: int,
    compress_seconds: float,
    decompress_seconds: float,
    compress_rss_kb: int | None,
    decompress_rss_kb: int | None,
) -> dict[str, float | str]:
  return {
      "ratio": ratio_bytes(input_size, compressed_size),
      "comp_MBps": throughput_mb_per_s(input_size, compress_seconds),
      "decomp_MBps": throughput_mb_per_s(decompressed_size, decompress_seconds),
      "mem_GB": kb_to_gb(compress_rss_kb),
      "decomp_mem_GB": kb_to_gb(decompress_rss_kb),
  }


def update_row(
    row: dict[str, object],
    prefix: str,
    metrics: dict[str, float | str],
) -> None:
  for suffix, value in metrics.items():
    row[f"{prefix}_{suffix}"] = value


def run_cmd(
    cmd: list[str] | str,
    *,
    shell: bool = False,
    stdout_path: Path | None = None,
    env: dict[str, str] | None = None,
    time_cmd: str | None = None,
    cwd: Path | None = None,
) -> CommandResult:
  start = time.perf_counter()
  if time_cmd:
    if shell:
      proc = subprocess.run(
          f"{time_cmd} -v {cmd}",
          shell=True,
          check=True,
          env=env,
          stderr=subprocess.PIPE,
          cwd=cwd,
      )
    else:
      timed_cmd = [time_cmd, "-v", *cmd]
      if stdout_path is None:
        proc = subprocess.run(
            timed_cmd,
            check=True,
            env=env,
            stderr=subprocess.PIPE,
            cwd=cwd,
        )
      else:
        with stdout_path.open("wb") as handle:
          proc = subprocess.run(
              timed_cmd,
              check=True,
              stdout=handle,
              env=env,
              stderr=subprocess.PIPE,
              cwd=cwd,
          )
    stderr_bytes = proc.stderr or b""
  else:
    if stdout_path is None:
      subprocess.run(cmd, shell=shell, check=True, env=env, cwd=cwd)
    else:
      with stdout_path.open("wb") as handle:
        subprocess.run(cmd, shell=shell, check=True, stdout=handle, env=env, cwd=cwd)
    stderr_bytes = b""
  return CommandResult(time.perf_counter() - start, stderr_bytes)


def shell_quote(path: Path | str) -> str:
  return subprocess.list2cmdline([str(path)])


def validate_args(args: argparse.Namespace) -> tuple[list[Path], str | None]:
  input_paths = [Path(path) for path in args.inputs]
  for input_path in input_paths:
    if not input_path.is_file():
      raise FileNotFoundError(f"Input GFA not found: {input_path}")
  if args.only_sqz and args.skip_sqz:
    raise ValueError("--only-sqz and --skip-sqz cannot be used together")
  if args.gbz_threads < 1:
    raise ValueError("--gbz-threads must be >= 1")
  if args.time_cmd and not tool_exists(args.time_cmd):
    raise FileNotFoundError(f"time command not found: {args.time_cmd}")
  return input_paths, args.time_cmd or None


def require_tool(path_or_name: str, label: str) -> None:
  if not tool_exists(path_or_name):
    raise FileNotFoundError(f"{label} tool not found: {path_or_name}")


def bench_simple_codec(
    *,
    input_path: Path,
    work_dir: Path,
    compressed_path: Path,
    decompressed_path: Path,
    compress_cmd: list[str],
    decompress_cmd: list[str],
    time_cmd: str | None,
) -> dict[str, float | str]:
  local_input = work_dir / input_path.name
  shutil.copy2(input_path, local_input)

  comp_result = run_cmd(compress_cmd, stdout_path=compressed_path, time_cmd=time_cmd)
  decomp_result = run_cmd(
      decompress_cmd,
      stdout_path=decompressed_path,
      time_cmd=time_cmd,
  )

  return metrics_dict(
      input_size=file_size(local_input),
      compressed_size=file_size(compressed_path),
      decompressed_size=file_size(decompressed_path),
      compress_seconds=comp_result.elapsed_seconds,
      decompress_seconds=decomp_result.elapsed_seconds,
      compress_rss_kb=parse_max_rss_kb(comp_result.stderr_bytes),
      decompress_rss_kb=parse_max_rss_kb(decomp_result.stderr_bytes),
  )


def bench_gzip(
    *,
    input_path: Path,
    work_dir: Path,
    gzip_bin: str,
    time_cmd: str | None,
) -> dict[str, float | str]:
  local_input = work_dir / f"{input_path.stem}.gfa"
  compressed = work_dir / f"{input_path.stem}.gz"
  decompressed = work_dir / f"{input_path.stem}.dec.gfa"
  return bench_simple_codec(
      input_path=input_path,
      work_dir=work_dir,
      compressed_path=compressed,
      decompressed_path=decompressed,
      compress_cmd=[gzip_bin, "-c", str(local_input)],
      decompress_cmd=[gzip_bin, "-cd", str(compressed)],
      time_cmd=time_cmd,
  )


def bench_zstd(
    *,
    input_path: Path,
    work_dir: Path,
    zstd_bin: str,
    time_cmd: str | None,
) -> dict[str, float | str]:
  local_input = work_dir / f"{input_path.stem}.gfa"
  compressed = work_dir / f"{input_path.stem}.zst"
  decompressed = work_dir / f"{input_path.stem}.dec.gfa"
  return bench_simple_codec(
      input_path=input_path,
      work_dir=work_dir,
      compressed_path=compressed,
      decompressed_path=decompressed,
      compress_cmd=[zstd_bin, "-q", "-c", str(local_input)],
      decompress_cmd=[zstd_bin, "-q", "-d", "-c", str(compressed)],
      time_cmd=time_cmd,
  )


def bench_sqz(
    *,
    input_path: Path,
    work_dir: Path,
    sqz_bin: str,
    bgzip_bin: str,
    zstd_bin: str,
    skip_bgzip: bool,
    time_cmd: str | None,
) -> dict[str, dict[str, float | str]]:
  local_input = work_dir / f"{input_path.stem}.gfa"
  sqz_path = work_dir / f"{input_path.stem}.sqz"
  sqz_dec = work_dir / f"{input_path.stem}.sqz.dec.gfa"
  shutil.copy2(input_path, local_input)

  sqz_comp = run_cmd(
      f"{shell_quote(sqz_bin)} compress {shell_quote(local_input)} > {shell_quote(sqz_path)}",
      shell=True,
      time_cmd=time_cmd,
  )
  sqz_decomp = run_cmd(
      f"{shell_quote(sqz_bin)} decompress {shell_quote(sqz_path)} > {shell_quote(sqz_dec)}",
      shell=True,
      time_cmd=time_cmd,
  )

  input_size = file_size(local_input)
  sqz_rss = parse_max_rss_kb(sqz_comp.stderr_bytes)
  results = {
      "sqz": metrics_dict(
          input_size=input_size,
          compressed_size=file_size(sqz_path),
          decompressed_size=file_size(sqz_dec),
          compress_seconds=sqz_comp.elapsed_seconds,
          decompress_seconds=sqz_decomp.elapsed_seconds,
          compress_rss_kb=sqz_rss,
          decompress_rss_kb=parse_max_rss_kb(sqz_decomp.stderr_bytes),
      )
  }

  if skip_bgzip or not tool_exists(bgzip_bin):
    results["sqz_bgzip"] = dict(EMPTY_METRICS)
  else:
    sqz_bgzip_path = work_dir / f"{input_path.stem}.sqz.bgz"
    sqz_bgzip_dec = work_dir / f"{input_path.stem}.sqz.bgz.dec.gfa"
    bgzip_comp = run_cmd(
        f"{shell_quote(bgzip_bin)} -c {shell_quote(sqz_path)} > {shell_quote(sqz_bgzip_path)}",
        shell=True,
        time_cmd=time_cmd,
    )
    bgzip_decomp = run_cmd(
        f"{shell_quote(bgzip_bin)} -cd {shell_quote(sqz_bgzip_path)} | "
        f"{shell_quote(sqz_bin)} decompress /dev/stdin > {shell_quote(sqz_bgzip_dec)}",
        shell=True,
        time_cmd=time_cmd,
    )
    results["sqz_bgzip"] = metrics_dict(
        input_size=input_size,
        compressed_size=file_size(sqz_bgzip_path),
        decompressed_size=file_size(sqz_bgzip_dec),
        compress_seconds=sqz_comp.elapsed_seconds + bgzip_comp.elapsed_seconds,
        decompress_seconds=bgzip_decomp.elapsed_seconds,
        compress_rss_kb=max(sqz_rss or 0, parse_max_rss_kb(bgzip_comp.stderr_bytes) or 0),
        decompress_rss_kb=parse_max_rss_kb(bgzip_decomp.stderr_bytes),
    )

  if not tool_exists(zstd_bin):
    results["sqz_zstd"] = dict(EMPTY_METRICS)
  else:
    sqz_zstd_path = work_dir / f"{input_path.stem}.sqz.zst"
    sqz_zstd_dec = work_dir / f"{input_path.stem}.sqz.zst.dec.gfa"
    zstd_comp = run_cmd(
        f"{shell_quote(zstd_bin)} -q -c {shell_quote(sqz_path)} > {shell_quote(sqz_zstd_path)}",
        shell=True,
        time_cmd=time_cmd,
    )
    zstd_decomp = run_cmd(
        f"{shell_quote(zstd_bin)} -q -d -c {shell_quote(sqz_zstd_path)} | "
        f"{shell_quote(sqz_bin)} decompress /dev/stdin > {shell_quote(sqz_zstd_dec)}",
        shell=True,
        time_cmd=time_cmd,
    )
    results["sqz_zstd"] = metrics_dict(
        input_size=input_size,
        compressed_size=file_size(sqz_zstd_path),
        decompressed_size=file_size(sqz_zstd_dec),
        compress_seconds=sqz_comp.elapsed_seconds + zstd_comp.elapsed_seconds,
        decompress_seconds=zstd_decomp.elapsed_seconds,
        compress_rss_kb=max(sqz_rss or 0, parse_max_rss_kb(zstd_comp.stderr_bytes) or 0),
        decompress_rss_kb=parse_max_rss_kb(zstd_decomp.stderr_bytes),
    )

  return results


def gbz_env(libpath: str) -> dict[str, str]:
  env = dict(os.environ)
  if libpath:
    previous = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{libpath}:{previous}" if previous else libpath
  return env


def bench_gbz(
    *,
    input_path: Path,
    work_dir: Path,
    gbz_bin: str,
    gbz_threads: int,
    gbz_libpath: str,
    time_cmd: str | None,
) -> dict[str, float | str]:
  local_input = work_dir / f"{input_path.stem}.gfa"
  shutil.copy2(input_path, local_input)
  base = local_input.with_suffix("")
  env = gbz_env(gbz_libpath)

  comp = run_cmd(
      [gbz_bin, "-P", str(gbz_threads), "-c", str(base)],
      env=env,
      time_cmd=time_cmd,
  )
  decomp = run_cmd(
      [gbz_bin, "-P", str(gbz_threads), "-d", str(base)],
      env=env,
      time_cmd=time_cmd,
  )

  return metrics_dict(
      input_size=file_size(local_input),
      compressed_size=file_size(base.with_suffix(".gbz")),
      decompressed_size=file_size(base.with_suffix(".gfa")),
      compress_seconds=comp.elapsed_seconds,
      decompress_seconds=decomp.elapsed_seconds,
      compress_rss_kb=parse_max_rss_kb(comp.stderr_bytes),
      decompress_rss_kb=parse_max_rss_kb(decomp.stderr_bytes),
  )


def bench_gfaz(
    *,
    input_path: Path,
    work_dir: Path,
    gfaz_bin: str,
    time_cmd: str | None,
) -> dict[str, float | str]:
  local_input = work_dir / f"{input_path.stem}.gfa"
  compressed = work_dir / f"{input_path.stem}.gfaz"
  decompressed = work_dir / f"{input_path.stem}.dec.gfa"
  shutil.copy2(input_path, local_input)

  comp = run_cmd(
      [gfaz_bin, "compress", str(local_input), str(compressed)],
      time_cmd=time_cmd,
  )
  decomp = run_cmd(
      [gfaz_bin, "decompress", str(compressed), str(decompressed)],
      time_cmd=time_cmd,
  )

  return metrics_dict(
      input_size=file_size(local_input),
      compressed_size=file_size(compressed),
      decompressed_size=file_size(decompressed),
      compress_seconds=comp.elapsed_seconds,
      decompress_seconds=decomp.elapsed_seconds,
      compress_rss_kb=parse_max_rss_kb(comp.stderr_bytes),
      decompress_rss_kb=parse_max_rss_kb(decomp.stderr_bytes),
  )


def benchmark_dataset(
    *,
    input_path: Path,
    tmpdir: Path,
    args: argparse.Namespace,
    time_cmd: str | None,
) -> dict[str, object]:
  row: dict[str, object] = {"dataset": input_path.stem}

  if not args.only_sqz:
    require_tool(args.gzip, "gzip")
    gzip_work = tmpdir / f"{input_path.stem}_gzip"
    ensure_dir(gzip_work)
    try:
      update_row(
          row,
          "gzip",
          bench_gzip(
              input_path=input_path,
              work_dir=gzip_work,
              gzip_bin=args.gzip,
              time_cmd=time_cmd,
          ),
      )
    finally:
      cleanup_paths(gzip_work)

    require_tool(args.zstd, "zstd")
    zstd_work = tmpdir / f"{input_path.stem}_zstd"
    ensure_dir(zstd_work)
    try:
      update_row(
          row,
          "zstd",
          bench_zstd(
              input_path=input_path,
              work_dir=zstd_work,
              zstd_bin=args.zstd,
              time_cmd=time_cmd,
          ),
      )
    finally:
      cleanup_paths(zstd_work)

    if args.skip_gbz:
      update_row(row, "gbz", dict(EMPTY_METRICS))
    else:
      require_tool(args.gbz, "gfa2gbwt")
      gbz_work = tmpdir / f"{input_path.stem}_gbz"
      ensure_dir(gbz_work)
      try:
        update_row(
            row,
            "gbz",
            bench_gbz(
                input_path=input_path,
                work_dir=gbz_work,
                gbz_bin=args.gbz,
                gbz_threads=args.gbz_threads,
                gbz_libpath=args.gbz_libpath,
                time_cmd=time_cmd,
            ),
        )
      finally:
        cleanup_paths(gbz_work)

    require_tool(args.gfaz, "gfaz")
    gfaz_work = tmpdir / f"{input_path.stem}_gfaz"
    ensure_dir(gfaz_work)
    try:
      update_row(
          row,
          "gfaz",
          bench_gfaz(
              input_path=input_path,
              work_dir=gfaz_work,
              gfaz_bin=args.gfaz,
              time_cmd=time_cmd,
          ),
      )
    finally:
      cleanup_paths(gfaz_work)

  if args.skip_sqz:
    for prefix in ["sqz", "sqz_bgzip", "sqz_zstd"]:
      update_row(row, prefix, dict(EMPTY_METRICS))
  else:
    require_tool(args.sqz, "sqz")
    sqz_work = tmpdir / f"{input_path.stem}_sqz"
    ensure_dir(sqz_work)
    try:
      sqz_results = bench_sqz(
          input_path=input_path,
          work_dir=sqz_work,
          sqz_bin=args.sqz,
          bgzip_bin=args.bgzip,
          zstd_bin=args.zstd,
          skip_bgzip=args.skip_bgzip,
          time_cmd=time_cmd,
      )
      for prefix, metrics in sqz_results.items():
        update_row(row, prefix, metrics)
    finally:
      cleanup_paths(sqz_work)

  return row


def write_rows(output_csv: Path, rows: list[dict[str, object]]) -> None:
  output_csv.parent.mkdir(parents=True, exist_ok=True)
  with output_csv.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
    writer.writeheader()
    writer.writerows(rows)


def main() -> int:
  args = parse_args()
  input_paths, time_cmd = validate_args(args)
  tmpdir = Path(args.tmpdir)
  ensure_dir(tmpdir)

  rows = []
  for input_path in input_paths:
    print(f"Benchmarking {input_path}")
    rows.append(
        benchmark_dataset(
            input_path=input_path,
            tmpdir=tmpdir,
            args=args,
            time_cmd=time_cmd,
        )
    )

  write_rows(Path(args.csv), rows)
  print(f"Wrote {args.csv}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
