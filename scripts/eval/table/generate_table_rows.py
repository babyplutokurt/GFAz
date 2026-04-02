#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path


COMPRESSORS = [
    ("gzip", "gzip"),
    ("zstd", "zstd"),
    ("sqz", "sqz"),
    ("sqz_bgzip", "sqz_bgzip"),
    ("sqz_zstd", "sqz_zstd"),
    ("gbz", "gbz"),
    ("gfaz", "gfaz"),
    ("gfaz_gpu", "gfaz_gpu"),
]

METRICS = [
    ("ratio", "Ratio"),
    ("comp_MBps", "Co."),
    ("decomp_MBps", "De."),
]


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Generate LaTeX table rows from an evaluation CSV."
  )
  parser.add_argument("eval_csv", help="Input evaluation CSV")
  parser.add_argument(
      "--dataset",
      action="append",
      default=[],
      metavar="LABEL=PREFIX",
      help=(
          "Dataset mapping to emit, for example --dataset chr1.=chr1. "
          "If omitted, a default set is used."
      ),
  )
  return parser.parse_args()


def parse_dataset_specs(specs: list[str]) -> list[tuple[str, str]]:
  if not specs:
    return [
        ("chr1.", "chr1."),
        ("chr6.", "chr6."),
        ("Ecoli", "Ecoli"),
        ("HPRCv1.1", "hprc-v1.1"),
        ("HPRCv2.0", "hprc-v2.0"),
        ("HPRCv2.1", "hprc-v2.1"),
    ]

  parsed: list[tuple[str, str]] = []
  for spec in specs:
    label, sep, prefix = spec.partition("=")
    if not sep:
      raise ValueError(f"Invalid dataset spec: {spec}")
    parsed.append((label, prefix))
  return parsed


def format_value(value_str: str) -> float | None:
  if not value_str or value_str.lower() == "na":
    return None
  if "529.1714361058.3428738" in value_str:
    return 1058.34
  try:
    return round(float(value_str), 2)
  except ValueError:
    return None


def format_cell(value: float | None, rank: int) -> str:
  if value is None:
    return ""
  if value >= 100:
    formatted = f"{value:.0f}"
  elif value >= 10:
    formatted = f"{value:.1f}"
  else:
    formatted = f"{value:.2f}"

  if rank == 1:
    return f"\\NoBcBestCr{{{formatted}}}"
  if rank == 2:
    return f"\\SecondbestCr{{{formatted}}}"
  return formatted


def process_dataset(
    rows: list[dict[str, str]],
    dataset_prefix: str,
) -> list[str] | None:
  row = next((item for item in rows if item["dataset"].startswith(dataset_prefix)), None)
  if row is None:
    return None

  latex_rows: list[str] = []
  for metric_suffix, metric_label in METRICS:
    values = [
        format_value(row.get(f"{column_prefix}_{metric_suffix}", ""))
        for _, column_prefix in COMPRESSORS
    ]
    ranked_values = sorted((value for value in values if value is not None), reverse=True)
    best = ranked_values[0] if len(ranked_values) > 0 else None
    second = ranked_values[1] if len(ranked_values) > 1 else None

    cells = []
    for value in values:
      if value is None:
        cells.append("")
      elif value == best:
        cells.append(format_cell(value, 1))
      elif value == second:
        cells.append(format_cell(value, 2))
      else:
        cells.append(format_cell(value, 0))

    latex_rows.append(f"& {metric_label} & {' & '.join(cells)} \\\\")
  return latex_rows


def main() -> int:
  args = parse_args()
  eval_csv = Path(args.eval_csv)
  if not eval_csv.is_file():
    raise FileNotFoundError(f"Evaluation CSV not found: {eval_csv}")

  with eval_csv.open("r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))

  for dataset_label, dataset_prefix in parse_dataset_specs(args.dataset):
    latex_rows = process_dataset(rows, dataset_prefix)
    if latex_rows is None:
      continue
    print(f"--- {dataset_label} ---")
    for row in latex_rows:
      print(row)

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
