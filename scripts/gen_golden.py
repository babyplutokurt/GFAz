#!/usr/bin/env python3
"""
Regenerate the committed concordance golden files in tests/golden/.

This is the ONLY code that runs the external reference tools (odgi, panacus, vg).
The concordance tests themselves read only the committed goldens, so they stay
hermetic. Run this whenever a fixture changes or a reference tool is upgraded:

    python3 scripts/gen_golden.py

Tool paths come from env (GFAZ_ODGI_BIN / GFAZ_PANACUS_BIN / GFAZ_VG_BIN) with
defaults under /home/kurty/Release. A missing tool skips its goldens with a note
rather than failing the whole run.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tests.regression.regression_utils import SkipTest
from tests.concordance.concordance_utils import (
    ODGI_BIN,
    PANACUS_BIN,
    VG_BIN,
    normalize_pav,
    normalize_vcf_for_golden,
    parse_growth_panacus,
    tool_or_skip,
    write_golden,
)

FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures"


def run(cmd):
  return subprocess.run(cmd, text=True, capture_output=True, check=True)


def tool_version(path: Path) -> str:
  junk = ("error", "unrecognized", "usage", "type `", "available command",
          "help", "options:")
  for flag in ("version", "--version", "-V", "-v"):
    try:
      r = subprocess.run([str(path), flag], text=True, capture_output=True)
      for line in (r.stdout + "\n" + r.stderr).splitlines():
        line = line.strip()
        low = line.lower()
        if line and any(c.isdigit() for c in line) and not any(j in low for j in junk):
          return line
    except Exception:
      continue
  return "unknown"


def gen_pav():
  odgi = tool_or_skip(ODGI_BIN, "odgi")
  ver = tool_version(odgi)
  gfa = FIXTURE_DIR / "pav_pathonly_concordance.gfa"
  bed = FIXTURE_DIR / "pav_pathonly_concordance.bed"
  with tempfile.TemporaryDirectory() as d:
    og = Path(d) / "g.og"
    run([str(odgi), "build", "-g", str(gfa), "-o", str(og), "-t", "2"])
    for form, extra, name in (
        ("long", [], "pav_pathonly_concordance.long.golden"),
        ("matrix", ["-M"], "pav_pathonly_concordance.matrix.golden"),
        ("sample-long", ["-S"], "pav_pathonly_concordance.sample.long.golden"),
    ):
      r = run([str(odgi), "pav", "-i", str(og), "-b", str(bed), "-t", "2"] + extra)
      body = normalize_pav(r.stdout)
      cmd = f"odgi pav -i <og> -b {bed.name} -t 2 {' '.join(extra)}".strip()
      write_golden(
          name,
          [f"tool: {ver}", f"command: {cmd}", f"fixture: {gfa.name}, {bed.name}",
           "normalize: header verbatim + body lines sorted"],
          body,
      )
      print(f"  wrote tests/golden/{name}")


def gen_growth():
  panacus = tool_or_skip(PANACUS_BIN, "panacus")
  ver = tool_version(panacus)
  gfa = FIXTURE_DIR / "growth_fixture.gfa"
  for mode, flags, name in (
      ("sample-hap-seq", [], "growth_fixture.sample-hap-seq.golden"),
      ("sample-hap", ["-H"], "growth_fixture.sample-hap.golden"),
      ("sample", ["-S"], "growth_fixture.sample.golden"),
  ):
    r = run([str(panacus), "growth", *flags, str(gfa)])
    curve = parse_growth_panacus(r.stdout)  # {k: int}
    body = "\n".join(f"{k}\t{curve[k]}" for k in sorted(curve))
    cmd = f"panacus growth {' '.join(flags)} {gfa.name}".strip()
    write_golden(
        name,
        [f"tool: {ver}", f"command: {cmd}", f"gfaz mode: -G {mode}",
         "predicate: panacus[k] == floor(gfaz[k])"],
        body,
    )
    print(f"  wrote tests/golden/{name}")


def gen_deconstruct():
  vg = tool_or_skip(VG_BIN, "vg")
  ver = tool_version(vg)
  gfa = FIXTURE_DIR / "deconstruct_links_fixture.gfa"
  r = run([str(vg), "deconstruct", "-p", "ref", str(gfa)])
  body = normalize_vcf_for_golden(r.stdout)
  name = "deconstruct_links_fixture.golden"
  write_golden(
      name,
      [f"tool: {ver}", f"command: vg deconstruct -p ref {gfa.name}",
       "normalize: (CHROM,POS,REF,ALT)->sample=GT, sorted; ID/QUAL/INFO ignored"],
      body,
  )
  print(f"  wrote tests/golden/{name}")


def main():
  any_written = False
  for label, fn in (("pav/odgi", gen_pav),
                    ("growth/panacus", gen_growth),
                    ("deconstruct/vg", gen_deconstruct)):
    print(f"[{label}]")
    try:
      fn()
      any_written = True
    except SkipTest as e:
      print(f"  SKIP: {e}")
    except subprocess.CalledProcessError as e:
      print(f"  ERROR running tool: {e}\n  STDERR: {e.stderr}")
      raise
  if not any_written:
    print("No goldens written (no external tools found).")
  print("Done.")


if __name__ == "__main__":
  main()
