#!/usr/bin/env python3

import argparse
import csv
import re
from pathlib import Path


COMPRESSORS = [
    "Gzip",
    "Zstd",
    "sqz",
    "sqz+bgzip",
    "sqz+Zstd",
    "GBZ",
    "gfaz(CPU)",
    "gfaz(GPU)",
]


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description=(
          "Extract per-dataset CSV files from the LaTeX compressor evaluation "
          "table."
      )
  )
  parser.add_argument("latex_table", help="Path to the LaTeX table file")
  parser.add_argument("output_dir", help="Directory for generated CSV files")
  return parser.parse_args()


def strip_latex_markup(value: str) -> str:
  value = value.split("\\\\")[0]
  value = value.split("%")[0]
  value = re.sub(r"\\NoBcBestCr\{([^\}]+)\}", r"\1", value)
  value = re.sub(r"\\SecondbestCr\{([^\}]+)\}", r"\1", value)
  return value.strip()


def dataset_key(raw_name: str) -> str:
  name = raw_name.replace(".", "")
  if "E.coli" in raw_name:
    return "Ecoli"
  return name


def parse_table(latex_path: Path) -> dict[str, dict[str, list[str]]]:
  data: dict[str, dict[str, list[str]]] = {}
  current_dataset: str | None = None

  for raw_line in latex_path.read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if not line:
      continue

    match = re.search(r"\\TabDataName\{(.*?)\}", line)
    if match:
      current_dataset = dataset_key(match.group(1))
      data[current_dataset] = {}
      continue

    if not current_dataset:
      continue

    metric = None
    if "& Ratio" in line:
      metric = "Ratio"
    elif "& Co." in line:
      metric = "Compression_Speed_MBps"
    elif "& De." in line:
      metric = "Decompression_Speed_MBps"

    if metric is None:
      continue

    parts = line.split("&")
    if len(parts) < 10:
      raise ValueError(f"Malformed table row for {current_dataset}: {line}")

    data[current_dataset][metric] = [
        strip_latex_markup(value) for value in parts[2:10]
    ]

  return data


def write_csvs(
    table_data: dict[str, dict[str, list[str]]],
    output_dir: Path,
) -> None:
  output_dir.mkdir(parents=True, exist_ok=True)

  for dataset, metrics in table_data.items():
    output_path = output_dir / f"{dataset}.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
      writer = csv.writer(handle)
      writer.writerow(
          [
              "Compressor",
              "Ratio",
              "Compression_Speed_MBps",
              "Decompression_Speed_MBps",
          ]
      )
      for index, compressor in enumerate(COMPRESSORS):
        writer.writerow(
            [
                compressor,
                metrics.get("Ratio", [""] * len(COMPRESSORS))[index],
                metrics.get("Compression_Speed_MBps", [""] * len(COMPRESSORS))[index],
                metrics.get("Decompression_Speed_MBps", [""] * len(COMPRESSORS))[index],
            ]
        )
    print(f"Wrote {output_path}")


def main() -> int:
  args = parse_args()
  latex_path = Path(args.latex_table)
  output_dir = Path(args.output_dir)

  if not latex_path.is_file():
    raise FileNotFoundError(f"LaTeX table not found: {latex_path}")

  table_data = parse_table(latex_path)
  write_csvs(table_data, output_dir)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
