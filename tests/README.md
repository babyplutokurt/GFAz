# Tests Layout

`tests/cpu/`
- CPU path-specific tests:
- `test_cpu_legacy_roundtrip.py` covers the legacy materialized path.
- `test_cpu_streaming_roundtrip.py` covers the direct-writer path.

`tests/gpu/`
- GPU path-specific tests:
- `test_gpu_legacy_roundtrip.py` covers the legacy path.
- `test_gpu_host_roundtrip.py` covers the rolling host-graph path.
- `test_gpu_streaming_roundtrip.py` covers the rolling direct-writer path.

`tests/regression/`
- Fixture-driven end-to-end regressions. The main entry point is
- `tests/regression/test_example_regression.py`, which runs the explicit CPU/GPU
- path matrix and cross-backend compatibility checks against `example.gfa`.

Legacy root-level test scripts remain as wrappers so older commands still
run, but new commands should use the files under `tests/`.
