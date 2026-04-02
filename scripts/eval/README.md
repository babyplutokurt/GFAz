# Eval Workflow

`benchmark/`
- `benchmark_compressors.py`: run the compressor benchmark suite and write the main evaluation CSV.

`plot/`
- `plot_grouped_bars.py`: grouped bar charts from per-dataset CSV files.
- `plot_normalized_bars.py`: normalized bar charts relative to Zstd.
- `plot_pareto.py`: normalized throughput-vs-ratio scatter plots.

`table/`
- `extract_table_csv.py`: convert the paper LaTeX table into per-dataset CSV files.
- `generate_table_rows.py`: generate LaTeX table rows from the benchmark CSV.

`validate/`
- `compare_gfa.py`: compare two GFA files using the project parser and verifier.

Suggested flow:
1. Run `benchmark/benchmark_compressors.py` to produce the main CSV.
2. Convert or prepare per-dataset CSV files as needed.
3. Use the scripts under `plot/` to generate figures.
4. Use the scripts under `table/` to regenerate table content.
