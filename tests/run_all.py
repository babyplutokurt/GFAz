#!/usr/bin/env python3
"""
Single entry point for the gfaz test suite: the CLI regressions, the golden-file
concordance tests, and (best-effort) the binding-based structural round-trip
suite.

Each suite runs as its own subprocess. Exit codes:
  0   -> PASS
  77  -> SKIP (the suite raised SkipTest, e.g. a golden or the bindings module
         is missing)
  else-> FAIL

A non-zero overall exit is returned only if at least one suite FAILED; SKIPs do
not fail the run.

The hermetic suites need only the `gfaz` CLI binary and run under this
interpreter. The binding-based round-trip suite
(test_compression_regression.py) needs the compiled `gfa_compression` module,
which is ABI-tied to the Python it was built against; the runner discovers a
matching interpreter and runs it with --skip-gpu (GPU is experimental and often
not built). If no binding-capable interpreter is found, that suite is skipped.

Usage:
    python3 tests/run_all.py
    ctest --test-dir build            # via the registered `gfaz_tests` test
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = REPO_ROOT / "build"
SKIP_EXIT_CODE = 77

# CLI-only / golden-file suites: hermetic, run under this interpreter.
HERMETIC_SUITES = [
    "tests/regression/test_cli_commands.py",
    "tests/regression/test_deconstruct.py",
    "tests/regression/test_growth.py",
    "tests/regression/test_cpu_roundtrip.py",
    "tests/regression/test_compression_optional_fields.py",
    "tests/regression/test_thread_determinism.py",
    "tests/regression/test_degenerate_inputs.py",
    "tests/regression/test_input_robustness.py",
    "tests/regression/test_pansn_grouping.py",
    "tests/concordance/test_pav_vs_odgi.py",
    "tests/concordance/test_growth_vs_panacus.py",
    "tests/concordance/test_deconstruct_vs_vg.py",
    # Opt-in chrY-scale concordance; SKIPs unless GFAZ_LARGE_CONCORDANCE is set.
    "tests/concordance/test_deconstruct_vs_vg_large.py",
]

# Binding-dependent suites: (path, extra_args). Run with a discovered
# binding-capable interpreter. GPU is skipped (experimental / often not built).
BINDING_SUITES = [
    ("tests/regression/test_compression_regression.py", ["--skip-gpu"]),
]


def _candidate_interpreters():
  """Interpreters to probe for a working `gfa_compression`, best first."""
  seen = []
  cache = BUILD_DIR / "CMakeCache.txt"
  if cache.exists():
    for line in cache.read_text(errors="ignore").splitlines():
      for key in ("_Python3_EXECUTABLE:INTERNAL=",
                  "PYBIND11_PYTHON_EXECUTABLE_LAST:INTERNAL=",
                  "PYTHON3_EXECUTABLE:FILEPATH="):
        if line.startswith(key):
          seen.append(line.split("=", 1)[1].strip())
  seen.append(sys.executable)
  out = []
  for p in seen:
    if p and p not in out and Path(p).exists():
      out.append(p)
  return out


def find_binding_python():
  probe = (f"import sys; sys.path.insert(0, {str(BUILD_DIR)!r}); "
           "import gfa_compression")
  for interp in _candidate_interpreters():
    r = subprocess.run([interp, "-c", probe], cwd=str(REPO_ROOT),
                       capture_output=True)
    if r.returncode == 0:
      return interp
  return None


def run_suite(cmd):
  proc = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True, capture_output=True)
  out = (proc.stdout + proc.stderr).strip()
  if proc.returncode == 0:
    return "PASS", out
  if proc.returncode == SKIP_EXIT_CODE:
    return "SKIP", out
  return "FAIL", out


def report(label, status, output):
  marker = {"PASS": "✅", "SKIP": "⏭️", "FAIL": "❌", "MISSING": "❓"}[status]
  print(f"{marker} {status:5} {label}")
  if status in ("FAIL", "MISSING") and output:
    for line in output.splitlines():
      print(f"      | {line}")


def main():
  results = []

  for rel in HERMETIC_SUITES:
    path = REPO_ROOT / rel
    if not path.exists():
      results.append("MISSING")
      report(rel, "MISSING", "")
      continue
    status, output = run_suite([sys.executable, str(path)])
    results.append(status)
    report(rel, status, output)

  binding_py = find_binding_python()
  for rel, extra in BINDING_SUITES:
    path = REPO_ROOT / rel
    label = f"{rel} {' '.join(extra)}".strip()
    if not path.exists():
      results.append("MISSING")
      report(label, "MISSING", "")
      continue
    if binding_py is None:
      results.append("SKIP")
      report(label + "  (no binding-capable interpreter found)", "SKIP", "")
      continue
    status, output = run_suite([binding_py, str(path), *extra])
    results.append(status)
    report(f"{label}  [{binding_py}]", status, output)

  npass = results.count("PASS")
  nskip = results.count("SKIP")
  nfail = results.count("FAIL") + results.count("MISSING")
  print(f"\nSummary: {npass} passed, {nskip} skipped, {nfail} failed "
        f"({len(results)} suites)")
  sys.exit(1 if nfail else 0)


if __name__ == "__main__":
  main()
