# Tests Layout

`tests/cpu/`
- CPU regression tests for in-memory and streaming decompression paths.

`tests/gpu/`
- GPU round-trip regression tests.

`tests/regression/`
- Fixture-driven end-to-end regressions. The main entry point is
  `tests/regression/test_example_regression.py`, which runs CPU, GPU, parity,
  and CLI `--stats` checks against `example.gfa`.

Legacy root-level test scripts remain as wrappers so older commands still
run, but new commands should use the files under `tests/`.
