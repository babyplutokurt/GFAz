#!/usr/bin/env python3

import argparse
import csv
import hashlib
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark whole-path PAV on a path-only GFA using GFAz and ODGI. "
            "The script runs: gfaz compress -> gfaz pav, then odgi build -> odgi pav."
        )
    )
    parser.add_argument("gfa", help="Input path-only GFA file")
    parser.add_argument(
        "--gfaz-bin",
        required=True,
        help="Path to the gfaz executable",
    )
    parser.add_argument(
        "--odgi-bin",
        required=True,
        help="Path to the odgi executable",
    )
    parser.add_argument(
        "--workdir",
        default="",
        help="Directory for generated .gfaz, .og, BED, outputs, and logs "
        "(default: pav_compare_<GFA stem> next to the current directory)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=32,
        help="Threads passed to gfaz pav and odgi build/pav (default: 32)",
    )
    parser.add_argument(
        "--time-bin",
        default="/usr/bin/time",
        help="Path to GNU time executable (default: /usr/bin/time)",
    )
    parser.add_argument(
        "--max-paths",
        type=int,
        default=0,
        help="Use only the first N P-lines when generating the BED "
        "(default: 0 = all paths)",
    )
    parser.add_argument(
        "--bed-end",
        type=int,
        default=1_000_000_000,
        help="End coordinate used for whole-path BED ranges "
        "(default: 1000000000)",
    )
    parser.add_argument(
        "--no-sample-group",
        action="store_true",
        help="Do not pass -S/--group-by-sample to pav",
    )
    parser.add_argument(
        "--no-matrix",
        action="store_true",
        help="Do not pass -M/--matrix-output to pav",
    )
    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"error: {label} does not exist: {path}")
    if not path.is_file():
        raise SystemExit(f"error: {label} is not a file: {path}")


def default_workdir(gfa_path: Path) -> Path:
    stem = gfa_path.name
    if stem.endswith(".gfa"):
        stem = stem[: -len(".gfa")]
    return Path.cwd() / f"pav_compare_{stem}"


def collect_path_names(gfa_path: Path, max_paths: int) -> tuple[list[str], int]:
    path_names: list[str] = []
    walk_count = 0
    with gfa_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("P\t"):
                fields = line.rstrip("\n").split("\t")
                if len(fields) >= 2 and (max_paths == 0 or len(path_names) < max_paths):
                    path_names.append(fields[1])
            elif line.startswith("W\t"):
                walk_count += 1
    return path_names, walk_count


def write_whole_path_bed(path_names: list[str], bed_path: Path, bed_end: int) -> None:
    with bed_path.open("w", encoding="utf-8") as out:
        for name in path_names:
            out.write(f"{name}\t0\t{bed_end}\t{name}\n")


def parse_max_rss_kb(stderr_text: str) -> int | None:
    for line in stderr_text.splitlines():
        if "Maximum resident set size" in line:
            _, _, value = line.partition(":")
            try:
                return int(value.strip())
            except ValueError:
                return None
    return None


def format_command(cmd: list[str]) -> str:
    return " ".join(str(part) for part in cmd)


def run_timed(
    *,
    label: str,
    cmd: list[str],
    stdout_path: Path | None,
    stderr_path: Path,
) -> dict[str, object]:
    print(f"[run] {label}", flush=True)
    print(f"      {format_command(cmd)}", flush=True)

    start = time.perf_counter()
    with stderr_path.open("w", encoding="utf-8") as err:
        if stdout_path is None:
            proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=err)
        else:
            with stdout_path.open("w", encoding="utf-8") as out:
                proc = subprocess.run(cmd, stdout=out, stderr=err)
    elapsed = time.perf_counter() - start

    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
    max_rss_kb = parse_max_rss_kb(stderr_text)
    if proc.returncode != 0:
        tail = "\n".join(stderr_text.strip().splitlines()[-20:])
        raise RuntimeError(
            f"{label} failed with exit status {proc.returncode}. "
            f"See {stderr_path}.\n{tail}"
        )

    return {
        "label": label,
        "elapsed_seconds": elapsed,
        "max_rss_kb": max_rss_kb,
        "command": format_command(cmd),
        "stderr_log": str(stderr_path),
        "stdout_path": str(stdout_path) if stdout_path else "",
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ratio(numerator: float | int | None, denominator: float | int | None) -> str:
    if numerator is None or denominator is None or denominator == 0:
        return ""
    return f"{float(numerator) / float(denominator):.6g}"


def write_summary(
    summary_path: Path,
    rows: list[dict[str, object]],
    metadata: dict[str, object],
) -> None:
    fieldnames = [
        "dataset",
        "path_count",
        "walk_count",
        "threads",
        "label",
        "elapsed_seconds",
        "max_rss_kb",
        "max_rss_gb",
        "command",
        "stdout_path",
        "stderr_log",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            rss = row.get("max_rss_kb")
            out = {
                "dataset": metadata["dataset"],
                "path_count": metadata["path_count"],
                "walk_count": metadata["walk_count"],
                "threads": metadata["threads"],
                "label": row["label"],
                "elapsed_seconds": f"{float(row['elapsed_seconds']):.6f}",
                "max_rss_kb": rss if rss is not None else "",
                "max_rss_gb": f"{float(rss) / 1024.0 / 1024.0:.6f}"
                if rss is not None
                else "",
                "command": row["command"],
                "stdout_path": row["stdout_path"],
                "stderr_log": row["stderr_log"],
            }
            writer.writerow(out)


def main() -> int:
    args = parse_args()
    gfa_path = Path(args.gfa).resolve()
    gfaz_bin = Path(args.gfaz_bin).resolve()
    odgi_bin = Path(args.odgi_bin).resolve()
    time_bin = Path(args.time_bin).resolve()

    require_file(gfa_path, "GFA")
    require_file(gfaz_bin, "gfaz executable")
    require_file(odgi_bin, "odgi executable")
    require_file(time_bin, "time executable")

    workdir = Path(args.workdir).resolve() if args.workdir else default_workdir(gfa_path).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    dataset = gfa_path.name[:-4] if gfa_path.name.endswith(".gfa") else gfa_path.stem
    gfaz_path = workdir / f"{dataset}.gfaz"
    og_path = workdir / f"{dataset}.og"
    bed_path = workdir / f"{dataset}.whole_paths.bed"
    gfaz_out = workdir / f"{dataset}.gfaz.pav.tsv"
    odgi_out = workdir / f"{dataset}.odgi.pav.tsv"
    summary_path = workdir / f"{dataset}.pav_compare.tsv"

    path_names, walk_count = collect_path_names(gfa_path, args.max_paths)
    if not path_names:
        raise SystemExit("error: no P-lines found; this script currently expects path-only GFA input")
    if walk_count:
        raise SystemExit(
            f"error: found {walk_count} W-lines; this benchmark intentionally "
            "requires path-only GFA input so GFAz and ODGI compare the same "
            "PAV query set"
        )

    write_whole_path_bed(path_names, bed_path, args.bed_end)
    print(f"[info] wrote BED with {len(path_names)} ranges: {bed_path}", flush=True)

    pav_flags: list[str] = []
    if not args.no_sample_group:
        pav_flags.append("-S")
    if not args.no_matrix:
        pav_flags.append("-M")

    rows: list[dict[str, object]] = []
    try:
        rows.append(
            run_timed(
                label="gfaz_compress",
                cmd=[
                    str(time_bin),
                    "-v",
                    str(gfaz_bin),
                    "compress",
                    str(gfa_path),
                    str(gfaz_path),
                ],
                stdout_path=None,
                stderr_path=workdir / "gfaz_compress.time.log",
            )
        )
        rows.append(
            run_timed(
                label="gfaz_pav",
                cmd=[
                    str(time_bin),
                    "-v",
                    str(gfaz_bin),
                    "pav",
                    "-i",
                    str(gfaz_path),
                    "-b",
                    str(bed_path),
                    *pav_flags,
                    "-t",
                    str(args.threads),
                ],
                stdout_path=gfaz_out,
                stderr_path=workdir / "gfaz_pav.time.log",
            )
        )
        rows.append(
            run_timed(
                label="odgi_build",
                cmd=[
                    str(time_bin),
                    "-v",
                    str(odgi_bin),
                    "build",
                    "-g",
                    str(gfa_path),
                    "-o",
                    str(og_path),
                    "-t",
                    str(args.threads),
                ],
                stdout_path=None,
                stderr_path=workdir / "odgi_build.time.log",
            )
        )
        rows.append(
            run_timed(
                label="odgi_pav",
                cmd=[
                    str(time_bin),
                    "-v",
                    str(odgi_bin),
                    "pav",
                    "-i",
                    str(og_path),
                    "-b",
                    str(bed_path),
                    *pav_flags,
                    "-t",
                    str(args.threads),
                ],
                stdout_path=odgi_out,
                stderr_path=workdir / "odgi_pav.time.log",
            )
        )
    except RuntimeError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    metadata = {
        "dataset": dataset,
        "path_count": len(path_names),
        "walk_count": walk_count,
        "threads": args.threads,
    }
    write_summary(summary_path, rows, metadata)

    by_label = {str(row["label"]): row for row in rows}
    gfaz_pav = by_label["gfaz_pav"]
    odgi_pav = by_label["odgi_pav"]
    gfaz_loop = float(by_label["gfaz_compress"]["elapsed_seconds"]) + float(
        gfaz_pav["elapsed_seconds"]
    )
    odgi_loop = float(by_label["odgi_build"]["elapsed_seconds"]) + float(
        odgi_pav["elapsed_seconds"]
    )

    print("\n[result]")
    print(f"workdir: {workdir}")
    print(f"summary_tsv: {summary_path}")
    print(f"gfaz_pav_seconds: {float(gfaz_pav['elapsed_seconds']):.6f}")
    print(f"odgi_pav_seconds: {float(odgi_pav['elapsed_seconds']):.6f}")
    print(
        "pav_speedup_odgi_over_gfaz: "
        f"{ratio(odgi_pav['elapsed_seconds'], gfaz_pav['elapsed_seconds'])}"
    )
    print(f"gfaz_pav_max_rss_kb: {gfaz_pav['max_rss_kb'] or ''}")
    print(f"odgi_pav_max_rss_kb: {odgi_pav['max_rss_kb'] or ''}")
    print(
        "pav_rss_ratio_odgi_over_gfaz: "
        f"{ratio(odgi_pav['max_rss_kb'], gfaz_pav['max_rss_kb'])}"
    )
    print(f"gfaz_full_loop_seconds: {gfaz_loop:.6f}")
    print(f"odgi_full_loop_seconds: {odgi_loop:.6f}")
    print(f"full_loop_speedup_odgi_over_gfaz: {ratio(odgi_loop, gfaz_loop)}")

    if gfaz_out.exists() and odgi_out.exists():
        gfaz_hash = sha256_file(gfaz_out)
        odgi_hash = sha256_file(odgi_out)
        print(f"gfaz_output_sha256: {gfaz_hash}")
        print(f"odgi_output_sha256: {odgi_hash}")
        print(f"outputs_byte_identical: {str(gfaz_hash == odgi_hash).lower()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
