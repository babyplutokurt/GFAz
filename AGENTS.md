# Repository Guidelines

## Project Structure & Module Organization
Core implementation lives in `src/` with public headers in `include/`, mirrored across three modules: `core/` (shared foundation — `model`, `codec`, `utils`, and `core/defaults.hpp`), `compress/` (the .gfa ⇄ .gfaz compressor — `io`, `grammar`, the compression/decompression workflows, and the GPU backend under `compress/gpu/`), and `compute/` (the compute engine that runs directly on `.gfaz` — `growth`, `pav`, `deconstruct`, `snarl_finder`, `traversal_query`). These build into three static libraries — `gfaz_core`, `gfaz_compress`, `gfaz_compute` — with `gfaz_compress` and `gfaz_compute` each depending only on `gfaz_core` (the compressor and compute engine never link each other). The CLI (`src/cli/`, binary `build/bin/gfaz`) and the Python extension module (`src/python/bindings.cpp`, imported as `gfa_compression`) link all three. Both backends serialize the same `CompressedData` schema to the same `.gfaz` container, so backend-specific code should not introduce a second file format. Tests live under `tests/`: path-specific round-trip scripts in `tests/cpu/` and `tests/gpu/`, fixture-driven regressions in `tests/regression/`, and golden-file concordance tests in `tests/concordance/`; `tests/run_all.py` is the single runner. Helper scripts are organized under `scripts/benchmark/`, `scripts/eval/`, and `scripts/data/`. Documentation lives under `docs/` (build/internals at the top level, command specs in `docs/workflows/`, forward-looking notes in `docs/design/`; see `docs/README.md` for the index); only `README.md` and this `AGENTS.md` stay in the repo root. Superseded notes are parked in `deprecated/`. Third-party code is vendored in `extern/`; treat it as upstream unless a dependency update is intentional.

## Build, Test, and Development Commands
Initialize dependencies first:

```bash
conda activate gfa
git submodule update --init --recursive
```

Configure a CPU build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
```

Enable CUDA when needed:

```bash
cmake -S . -B build -DENABLE_CUDA=ON -DCUDA_PATH=/usr/local/cuda-12.8
cmake --build build -j"$(nproc)"
```

The CLI binary is `build/bin/gfaz`. Run the full suite from one entry point:

```bash
conda activate gfa
python3 tests/run_all.py            # PASS / SKIP / FAIL summary
ctest --test-dir build              # same, via the `gfaz_tests` CTest entry
```

## Coding Style & Naming Conventions
Use C++17 and follow the existing style: 2-space indentation, braces on the same line, and compact helper comments only where the code is not obvious. Keep header/source pairs aligned by name and module, for example `include/core/codec/codec.hpp` and `src/core/codec/codec.cpp`. Use `snake_case` for functions and variables, `PascalCase` for data types, and preserve existing namespace names such as `Codec`. Python additions should follow PEP 8 and keep imports simple enough to run from the repo root.

## Testing Guidelines
Tests run through `python3 tests/run_all.py` (also registered as the `gfaz_tests` CTest entry, so `ctest --test-dir build` works too). Each suite reports PASS, SKIP, or FAIL; a suite that cannot run (compiled bindings not importable, or a missing external-tool golden) is SKIPped rather than failed, and GPU paths are skipped unless a CUDA build is present. The concordance tests under `tests/concordance/` compare against committed golden files and do not invoke odgi/panacus/vg at test time — regenerate goldens with `python3 scripts/gen_golden.py` when intentionally changing reference behavior. For targeted work: `tests/regression/test_compression_regression.py` for shared serialization / CPU/GPU compatibility, the `tests/cpu/` and `tests/gpu/` scripts for path-specific round-trips, and the `tests/regression/` CLI suites for command behavior. See `tests/README.md` for the full layout. Include the exact command used in your PR notes.

## Commit & Pull Request Guidelines
Recent history uses short imperative subjects (`Initial release commit`, `Compare use sorting`). Follow that pattern: one-line, present-tense summaries under about 72 characters. Pull requests should explain the behavioral change, list validation commands, and call out CPU/GPU impact explicitly. Include logs or screenshots only when changing user-facing CLI behavior or plots in `scripts/`.

## Configuration Tips
Use the project Conda environment before building or testing: `conda activate gfa`. Prefer out-of-tree builds in `build/`. Keep generated artifacts and large benchmark outputs out of version control. For GPU work, document any required environment variables or toolkit paths in the PR so others can reproduce the build. When documenting or testing workflows, assume a single shared `.gfaz` format: CPU-compressed files should be readable through GPU decompression, and GPU-compressed files should be readable through CPU decompression.
