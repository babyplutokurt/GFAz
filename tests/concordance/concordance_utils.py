"""
Shared helpers for the compute-engine concordance tests.

Strategy: gfaz's compute engine (pav / growth / deconstruct) is meant to
reproduce odgi / panacus / vg. Rather than invoke those tools at test time, we
commit *golden* files holding their normalized output (regenerate with
`scripts/gen_golden.py`). The tests here read only the committed goldens and
compare them against gfaz's normalized output, so they are hermetic and
deterministic.

The normalizers below are the single source of truth for "what counts as equal"
and are reused by `scripts/gen_golden.py` so goldens and tests agree by
construction. They encode the divergences discovered against the real tools:

  - pav (odgi):   gfaz emits rows in BED order, odgi in path-id order -> compare
                  header line exactly, body lines after sorting. odgi also drops
                  W-lines on `odgi build`, so pav fixtures must be path-only.
  - growth (panacus): panacus[k] == floor(gfaz[k]); panacus prefixes 4 metadata
                  rows and a `0  NaN` row that we strip.
  - deconstruct (vg): vg adds ID / QUAL / INFO `AT=` and orders INFO differently;
                  compare only (CHROM, POS, REF, ALT) -> per-sample GT vector.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DIR = REPO_ROOT / "tests" / "golden"

# External tool locations (override via env). Only `scripts/gen_golden.py` uses
# these; the tests themselves never touch the external binaries.
ODGI_BIN = Path(os.environ.get("GFAZ_ODGI_BIN", "/home/kurty/Release/odgi/bin/odgi"))
PANACUS_BIN = Path(
    os.environ.get(
        "GFAZ_PANACUS_BIN", "/home/kurty/Release/panacus/target/release/panacus"
    )
)
VG_BIN = Path(os.environ.get("GFAZ_VG_BIN", "/home/kurty/Release/vg/bin/vg"))


# --------------------------------------------------------------------------
# Golden file IO
# --------------------------------------------------------------------------
def golden_path(name: str) -> Path:
  return GOLDEN_DIR / name


def load_golden(name: str):
  """Return the golden file's text, or raise SkipTest if it has not been
  generated yet (run scripts/gen_golden.py)."""
  from tests.regression.regression_utils import SkipTest

  path = golden_path(name)
  if not path.exists():
    raise SkipTest(
        f"golden {name} missing; run `python3 scripts/gen_golden.py` to create it"
    )
  return path.read_text()


def write_golden(name: str, provenance_lines, body: str):
  """Write a golden file: provenance as '## ' comment lines (stripped before
  comparison) followed by the normalized body."""
  GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
  prov = "\n".join(f"## {ln}" for ln in provenance_lines)
  text = prov + "\n" + body
  if not text.endswith("\n"):
    text += "\n"
  golden_path(name).write_text(text)


def tool_or_skip(path: Path, name: str) -> Path:
  """Used by gen_golden.py: ensure an external binary exists, else SkipTest."""
  from tests.regression.regression_utils import SkipTest

  if not path.exists() or not os.access(path, os.X_OK):
    raise SkipTest(f"{name} not found/executable at {path}")
  return path


# --------------------------------------------------------------------------
# Golden bodies carry a leading '##' provenance block; strip it before
# comparing. Single-'#' lines (e.g. the VCF '#SAMPLES' body header) are kept.
# --------------------------------------------------------------------------
def strip_golden_comments(text: str) -> str:
  return "\n".join(
      ln for ln in text.splitlines() if not ln.startswith("##")
  ).strip("\n")


# --------------------------------------------------------------------------
# pav normalization
# --------------------------------------------------------------------------
def normalize_pav(text: str) -> str:
  """Header line verbatim + body lines sorted, joined by '\n'. Empty/blank
  lines dropped. Comparable across gfaz (BED order) and odgi (path-id order)."""
  lines = [ln for ln in text.splitlines() if ln.strip() != ""]
  if not lines:
    return ""
  header, body = lines[0], sorted(lines[1:])
  return "\n".join([header] + body)


# --------------------------------------------------------------------------
# similarity normalization
# --------------------------------------------------------------------------
def normalize_similarity(text: str) -> str:
  """Header line verbatim + body lines sorted. gfaz emits pairs in (a,b)
  ascending order, odgi in hash-map order, so compare after sorting the body.
  Both tools print %.6f, so values compare as exact strings."""
  lines = [ln for ln in text.splitlines() if ln.strip() != ""]
  if not lines:
    return ""
  header, body = lines[0], sorted(lines[1:])
  return "\n".join([header] + body)


# --------------------------------------------------------------------------
# stats / depth normalization
# --------------------------------------------------------------------------
def normalize_stats(text: str) -> str:
  """Non-blank lines verbatim. gfaz and odgi emit the same fixed row order
  (summary: header + one line; -b: A/C/G/T), so no sorting is needed."""
  return "\n".join(ln for ln in text.splitlines() if ln.strip() != "")


def normalize_depth(text: str) -> str:
  """Header line verbatim + body lines sorted. gfaz emits the -d table in node-id
  order; odgi emits it in internal (thread-dependent) order, so compare after
  sorting the body. The summary is header + one line (sorting is a no-op). Node
  ids match only when the GFA segment names are 1..N (true for the fixtures)."""
  lines = [ln for ln in text.splitlines() if ln.strip() != ""]
  if not lines:
    return ""
  header, body = lines[0], sorted(lines[1:])
  return "\n".join([header] + body)


# --------------------------------------------------------------------------
# growth normalization
# --------------------------------------------------------------------------
def parse_growth_gfaz(text: str) -> dict:
  """{k: float} from a gfaz growth table (skip '#' and the 'k\tgrowth' row)."""
  curve = {}
  in_body = False
  for line in text.splitlines():
    if not line or line.startswith("#"):
      continue
    if line.split("\t")[0] == "k":
      in_body = True
      continue
    if not in_body:
      continue
    f = line.split("\t")
    curve[int(f[0])] = float(f[1])
  return curve


def parse_growth_panacus(text: str) -> dict:
  """{m: int} from panacus growth output. Skips '#' comments, the metadata
  rows (panacus/growth, count/node, coverage/<n>, quorum/<n>), and the `0 NaN`
  row. Keeps only rows whose first column is a positive integer."""
  curve = {}
  for line in text.splitlines():
    if not line or line.startswith("#"):
      continue
    f = line.split("\t")
    if len(f) < 2:
      continue
    key = f[0].strip()
    if not key.isdigit():
      continue
    m = int(key)
    if m == 0:
      continue
    try:
      curve[m] = int(float(f[1]))
    except ValueError:
      continue
  return curve


# --------------------------------------------------------------------------
# VCF / deconstruct normalization
# --------------------------------------------------------------------------
def parse_vcf_records(text: str):
  """Return (sample_columns, {(chrom,pos,ref,alt): gt_tuple}). Ignores ID,
  QUAL, FILTER and INFO so vg's AT=/ID/QUAL extras don't cause spurious
  mismatches. GT is indexed by sample name to be robust to column order."""
  samples = []
  records = {}
  for line in text.splitlines():
    if line.startswith("##") or not line:
      continue
    f = line.split("\t")
    if line.startswith("#CHROM"):
      samples = f[9:] if len(f) > 9 else []
      continue
    if len(f) < 5:
      continue
    chrom, pos, _id, ref, alt = f[0], f[1], f[2], f[3], f[4]
    gts = f[9:] if len(f) > 9 else []
    gt_by_sample = {samples[i]: gts[i] for i in range(min(len(samples), len(gts)))}
    records[(chrom, pos, ref, alt)] = gt_by_sample
  return samples, records


def vcf_concordant(gfaz_text: str, vg_text: str, min_key_overlap=1.0,
                   min_gt_match=1.0, max_count_delta=0.0):
  """Compare two VCFs at (CHROM,POS,REF,ALT)->GT level. Defaults are strict
  (exact) for the tiny fixture; loosen the thresholds for large real datasets
  where vg/gfaz concord at ~99.99% rather than 100%."""
  _, g = parse_vcf_records(gfaz_text)
  _, v = parse_vcf_records(vg_text)
  if not v:
    return False, "reference (vg) VCF has no records"
  shared = set(g) & set(v)
  overlap = len(shared) / max(len(g), len(v))
  if overlap < min_key_overlap:
    return False, (
        f"site overlap {overlap:.5f} < {min_key_overlap} "
        f"(gfaz={len(g)} vg={len(v)} shared={len(shared)})"
    )
  count_delta = abs(len(g) - len(v)) / len(v)
  if count_delta > max_count_delta:
    return False, f"record-count delta {count_delta:.5f} > {max_count_delta}"
  gt_ok = sum(1 for k in shared if g[k] == v[k])
  gt_frac = gt_ok / len(shared) if shared else 0.0
  if gt_frac < min_gt_match:
    return False, f"GT match {gt_frac:.5f} < {min_gt_match} over {len(shared)} sites"
  return True, "ok"


def vcf_pos_keys(text: str):
  """Set of (CHROM, POS) over a VCF's records. Used for position-level
  concordance on large graphs, where REF/ALT spelling legitimately differs at
  complex sites but positions should match closely."""
  keys = set()
  for line in text.splitlines():
    if line.startswith("#") or not line.strip():
      continue
    f = line.split("\t")
    if len(f) >= 2:
      keys.add((f[0], f[1]))
  return keys


def normalize_vcf_for_golden(text: str) -> str:
  """Canonical, sorted, comparable VCF body for storage as a golden:
  one line per record `CHROM\tPOS\tREF\tALT\t<sample=gt;...>`."""
  samples, records = parse_vcf_records(text)
  lines = []
  for (chrom, pos, ref, alt), gt_by_sample in records.items():
    gtstr = ";".join(f"{s}={gt_by_sample.get(s, '.')}" for s in samples)
    lines.append(f"{chrom}\t{pos}\t{ref}\t{alt}\t{gtstr}")
  return "\n".join(["#SAMPLES\t" + "\t".join(samples)] + sorted(lines))
