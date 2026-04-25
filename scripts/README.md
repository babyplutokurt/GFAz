# Scripts Layout

`scripts/benchmark/`
- Benchmark drivers and performance comparison helpers.
- `decompress_modes.py`: compare CPU streaming vs legacy decompression.
- `extract_queries.py`: benchmark `extract-path` / `extract-walk` queries.
- `add_haplotypes.py`: benchmark `gfaz add-haplotypes`.
- `pav_compare_gfaz_odgi.py`: run a path-only GFA whole-path PAV comparison
  between GFAz and ODGI.

`scripts/eval/`
- Evaluation workflow helpers split into `benchmark/`, `plot/`, `table/`, and `validate/`.

`scripts/data/`
- Dataset preparation and batch-processing helpers.
- `compress_directory.py`: compress all `.gfa` files in a directory.
- `split_traversal_tail.py`: split off trailing `P/W` records from a GFA.
- `split_traversal_tail.sh`: shell wrapper for extracting the last 50 `P/W` lines.
- `analyze_gfa.py`: summarize record counts and byte sizes by GFA line type.
