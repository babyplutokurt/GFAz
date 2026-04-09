  1. Baseline + Safety Net

  - Freeze a baseline by running current CPU/GPU smoke tests and recording pass/fail.
  - Capture current CLI behavior (compress/decompress options, defaults, output naming).
  - Define “no-regression” checks for parser fidelity, round-trip correctness, and .gfaz read/write compatibility.

  2. Repository Structure Cleanup

  - Classify files into: active, legacy, experimental.
  - Remove or isolate dead/stale entrypoints (likely src/main.cpp) after confirming no references.
  - Move old docs/scripts into deprecated/ or tools/ with clear labels.
  - Add a short PROJECT_LAYOUT.md describing where core CPU/GPU/CLI/bindings code lives.

  3. Build System Consolidation

  - Refactor CMakeLists.txt to reduce duplication between Python module and CLI source lists.
  - Centralize compile options, include paths, and feature flags (ENABLE_CUDA, ENABLE_PROFILING).
  - Ensure CPU-only builds do not pull GPU-only codepaths accidentally.
  - Verify clean builds for both modes from scratch.

  4. Public API/CLI Surface Cleanup

  - Normalize naming and defaults across Python API, bindings, and CLI.
  - Separate stable APIs from experimental GPU helper bindings in src/bindings.cpp.
  - Improve CLI help text consistency with actual behavior and defaults.
  - Keep backward compatibility where possible; document intentional API changes.

  5. Code Hygiene Pass

  - Remove unused includes/helpers and stale comments.
  - Standardize error handling/messages across parser/workflows/serialization.
  - Improve file-level organization for very large files (src/bindings.cpp, workflow .cpp files) by grouping related sections.

  6. Documentation Alignment

  - Update BUILD_GUIDE.md and workflow.md to match actual build flags, features, and file format version.
  - Add concise docs for:
      - CPU vs GPU backend responsibilities
      - Python binding capabilities by build mode
      - CLI usage and examples
  - Document known limitations and optional dependencies (OpenMP, CUDA).

  7. Test Coverage Strengthening

  - Add focused tests for:
      - .gfaz compatibility/version handling
      - CPU round-trip with optional fields/J/C/W lines
      - CLI round-trip
      - GPU round-trip parity (when CUDA enabled)
  - Add a small deterministic fixture set for fast CI-like local validation.

  8. Final Verification + Change Summary

  - Re-run full test matrix (CPU-only and GPU-enabled).
  - Compare compression/decompression outputs against baseline.
  - Produce a cleanup report listing:
      - Removed files
      - Refactored modules
      - API/CLI changes
      - Remaining technical debt.

  Execution order recommendation

  1. Baseline/tests, 2) structure/build cleanup, 3) API/CLI cleanup, 4) docs/tests hardening.
