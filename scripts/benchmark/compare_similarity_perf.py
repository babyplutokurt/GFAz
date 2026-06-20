#!/usr/bin/env python3
"""Benchmark `gfaz similarity` against `odgi similarity`.

odgi materializes the graph (`odgi build` then `odgi similarity`); gfaz streams
the compressed `.gfaz`. Each command runs under `/usr/bin/time -v` capturing wall
time and peak RSS. Optionally checks value concordance (sorted body, exact at
%.6f) since both tools emit the same group-name/bp columns.

Grouping correspondence: gfaz `-p` (per path/walk) == odgi default (no -D);
gfaz `-S` (per sample) == odgi `-D '#' -p 1`.

Examples:
  python scripts/benchmark/compare_similarity_perf.py \
      --gfa /data/chrY.gfa --gfaz chrY.gfaz --grouping sample --threads 16 \
      --concordance
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ODGI = Path("/home/kurty/Release/odgi/bin/odgi")
DEFAULT_GFAZ = REPO_ROOT / "build" / "bin" / "gfaz"


def timed(cmd, stdout_path=None):
  """Run `cmd` under /usr/bin/time -v; return (wall_s, rss_kb, exit, elapsed)."""
  full = ["/usr/bin/time", "-v", *cmd]
  start = time.perf_counter()
  with tempfile.NamedTemporaryFile("w+", suffix=".err") as err:
    out = open(stdout_path, "wb") if stdout_path else subprocess.DEVNULL
    try:
      rc = subprocess.run(full, stdout=out, stderr=err, check=False).returncode
    finally:
      if stdout_path:
        out.close()
    wall = time.perf_counter() - start
    err.seek(0)
    log = err.read()
  rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", log)
  el = re.search(r"Elapsed \(wall clock\) time .*: (.+)", log)
  return wall, int(rss.group(1)) if rss else None, rc, el.group(1) if el else None


def body_sorted(path: Path) -> list[str]:
  lines = [ln for ln in path.read_text(errors="replace").splitlines()
           if ln.strip()]
  return sorted(lines[1:]) if lines else []


def main(argv):
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--gfa", type=Path, required=True, help="GFA for odgi.")
  ap.add_argument("--gfaz", type=Path,
                  help="Existing .gfaz; if omitted, compress --gfa.")
  ap.add_argument("--grouping", choices=["per-path", "sample", "sample-hap"],
                  default="sample")
  ap.add_argument("--threads", type=int, default=16)
  ap.add_argument("--odgi-bin", type=Path, default=DEFAULT_ODGI)
  ap.add_argument("--gfaz-bin", type=Path, default=DEFAULT_GFAZ)
  ap.add_argument("--out-dir", type=Path,
                  default=REPO_ROOT / "benchmark_results")
  ap.add_argument("--concordance", action="store_true",
                  help="Compare sorted output bodies for exact value agreement.")
  args = ap.parse_args(argv)

  args.out_dir.mkdir(parents=True, exist_ok=True)
  gfaz_flag = {"per-path": ["-p"], "sample": ["-S"],
               "sample-hap": ["-H"]}[args.grouping]
  odgi_flag = {"per-path": [], "sample": ["-D", "#", "-p", "1"],
               "sample-hap": ["-D", "#", "-p", "2"]}[args.grouping]
  t = str(args.threads)

  # gfaz: ensure a .gfaz.
  gfaz = args.gfaz
  if gfaz is None or not gfaz.exists():
    gfaz = args.out_dir / (args.gfa.stem + ".gfaz")
    print(f"[compress] {args.gfa} -> {gfaz}")
    subprocess.run([str(args.gfaz_bin), "compress", str(args.gfa), str(gfaz)],
                   check=True)

  gfaz_out = args.out_dir / "gfaz.similarity.tsv"
  odgi_out = args.out_dir / "odgi.similarity.tsv"
  og = args.out_dir / (args.gfa.stem + ".og")

  print(f"[gfaz similarity {' '.join(gfaz_flag)}]")
  g_wall, g_rss, g_rc, g_el = timed(
      [str(args.gfaz_bin), "similarity", "-i", str(gfaz), *gfaz_flag, "-t", t],
      gfaz_out)

  print("[odgi build]")
  b_wall, b_rss, b_rc, _ = timed(
      [str(args.odgi_bin), "build", "-g", str(args.gfa), "-o", str(og), "-t", t])
  print(f"[odgi similarity {' '.join(odgi_flag)}]")
  o_wall, o_rss, o_rc, o_el = timed(
      [str(args.odgi_bin), "similarity", "-i", str(og), *odgi_flag, "-t", t],
      odgi_out)

  def gib(kb):
    return "NA" if kb is None else f"{kb / (1024 ** 2):.2f}"

  print("\ntool\twall_s\tpeak_gib\texit")
  print(f"gfaz\t{g_wall:.2f}\t{gib(g_rss)}\t{g_rc}")
  print(f"odgi-build\t{b_wall:.2f}\t{gib(b_rss)}\t{b_rc}")
  print(f"odgi-similarity\t{o_wall:.2f}\t{gib(o_rss)}\t{o_rc}")
  odgi_total = b_wall + o_wall
  odgi_peak = max(x for x in (b_rss, o_rss) if x is not None)
  print(f"odgi-total\t{odgi_total:.2f}\t{gib(odgi_peak)}\t-")
  if g_wall > 0:
    print(f"\nspeedup (odgi-total / gfaz)\t{odgi_total / g_wall:.2f}x")
  if g_rss and odgi_peak:
    print(f"rss reduction (odgi-peak / gfaz)\t{odgi_peak / g_rss:.2f}x")

  if args.concordance:
    same = body_sorted(gfaz_out) == body_sorted(odgi_out)
    print(f"\nconcordance (sorted body, exact %.6f): {'MATCH' if same else 'DIFFER'}")
    return 0 if same else 3
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv[1:]))
