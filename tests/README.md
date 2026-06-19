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
- Fixture-driven end-to-end regressions.
- `test_compression_regression.py` runs the explicit CPU/GPU path matrix and
  cross-backend compatibility checks against `example.gfa` (requires the
  compiled Python bindings).
- `test_cli_commands.py` covers `extract-path`, `extract-walk`, `add-haplotypes`
  and `pav` using compact fixtures under `tests/fixtures/`.
- `test_deconstruct.py` covers `deconstruct` (GFA -> VCF) with hand-verified VCF
  records.
- `test_growth.py` covers `growth` (pangenome growth curve) with a hand-verified
  expected curve for every grouping mode.
- `test_compression_optional_fields.py` is a pure-CLI round-trip exercising
  J-lines, C-lines, and S-line optional fields of every type (i/f/A/Z/B).
- `test_cpu_roundtrip.py` is a pure-CLI compress->decompress matrix over several
  fixtures, both CPU decompress paths (default streaming and `--legacy`), and a
  `--delta`/`--rounds`/`--threshold` parameter sweep. It always runs (no
  bindings) and is the floor under the deeper binding-based suite below.
- `test_thread_determinism.py` asserts pav/growth/deconstruct produce
  byte-identical output at 1/4/8 threads (the OpenMP reductions must be
  thread-count invariant).
- `test_degenerate_inputs.py` covers degenerate-but-valid graphs (segments-only,
  walks-only, header-only) round-tripping cleanly, and invalid invocations
  (missing/malformed BED, unresolved pav/deconstruct reference) failing with a
  non-zero exit.

`tests/concordance/`
- Golden-file concordance tests that lock gfaz's compute engine to the reference
  tools it reproduces. The external tools are NOT run at test time: the golden
  files under `tests/golden/` hold their *normalized* output and are regenerated
  with `python3 scripts/gen_golden.py` (which is the only code that invokes the
  external binaries; tool paths come from `GFAZ_ODGI_BIN` / `GFAZ_PANACUS_BIN` /
  `GFAZ_VG_BIN`).
- `test_pav_vs_odgi.py` — `gfaz pav` vs `odgi pav` (header verbatim + sorted body;
  fixtures are path-only because `odgi build` drops W-lines).
- `test_growth_vs_panacus.py` — `panacus[k] == floor(gfaz[k])` across the paired
  grouping modes.
- `test_deconstruct_vs_vg.py` — `gfaz deconstruct` (default snarl mode) vs
  `vg deconstruct` compared at (CHROM,POS,REF,ALT) -> per-sample GT.
- Shared normalizers live in `tests/concordance/concordance_utils.py`.

## Running

The hermetic CPU-only suite (CLI regressions + concordance, no bindings needed):

    python3 tests/run_all.py            # PASS / SKIP / FAIL summary
    ctest --test-dir build              # same, via the `gfaz_tests` CTest entry

A suite that cannot run (e.g. a missing golden, or the bindings module not being
importable) exits 77 and is reported as SKIP rather than FAIL.

`run_all.py` also runs the binding-based structural round-trip suite
(`test_compression_regression.py --skip-gpu`). The compiled `gfa_compression`
module is ABI-tied to the Python it was built against, so the runner discovers a
binding-capable interpreter (from `build/CMakeCache.txt`, falling back to the
current one) and runs the suite with it; if none is found, the suite is skipped.
GPU is skipped because it is experimental and often not built. The `tests/cpu`
and `tests/gpu` path-specific scripts are still run directly when desired.

Legacy root-level test scripts remain as wrappers so older commands still
run, but new commands should use the files under `tests/`.
