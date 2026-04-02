# Tests Layout

`tests/cpu/`
- CPU regression tests for in-memory and streaming decompression paths.

`tests/gpu/`
- GPU round-trip regression tests.

Legacy root-level test scripts remain as wrappers so older commands still
run, but new commands should use the files under `tests/`.
