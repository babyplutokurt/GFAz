#!/usr/bin/env python3
"""
Single entry point for the hermetic (CPU-only, no-bindings) gfaz test suite:
the CLI regressions plus the golden-file concordance tests.

Each suite is run as its own subprocess. Exit codes:
  0   -> PASS
  77  -> SKIP (the suite raised SkipTest, e.g. a golden is missing)
  else-> FAIL

A non-zero overall exit is returned only if at least one suite FAILED; SKIPs do
not fail the run. Binding-dependent round-trip suites
(tests/regression/test_compression_regression.py and tests/cpu, tests/gpu) are
NOT included here because they require the compiled Python bindings; run those
directly when the bindings are built.

Usage:
    python3 tests/run_all.py
    ctest --test-dir build            # via the registered `gfaz_tests` test
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SKIP_EXIT_CODE = 77

SUITES = [
    "tests/regression/test_cli_commands.py",
    "tests/regression/test_deconstruct.py",
    "tests/regression/test_growth.py",
    "tests/regression/test_compression_optional_fields.py",
    "tests/concordance/test_pav_vs_odgi.py",
    "tests/concordance/test_growth_vs_panacus.py",
    "tests/concordance/test_deconstruct_vs_vg.py",
]


def run_suite(rel_path: str):
  path = REPO_ROOT / rel_path
  if not path.exists():
    return "MISSING", ""
  proc = subprocess.run(
      [sys.executable, str(path)], cwd=str(REPO_ROOT),
      text=True, capture_output=True)
  out = (proc.stdout + proc.stderr).strip()
  if proc.returncode == 0:
    return "PASS", out
  if proc.returncode == SKIP_EXIT_CODE:
    return "SKIP", out
  return "FAIL", out


def main():
  results = []
  for rel in SUITES:
    status, output = run_suite(rel)
    results.append((rel, status))
    marker = {"PASS": "✅", "SKIP": "⏭️", "FAIL": "❌", "MISSING": "❓"}[status]
    print(f"{marker} {status:5} {rel}")
    if status in ("FAIL", "MISSING") and output:
      for line in output.splitlines():
        print(f"      | {line}")

  npass = sum(1 for _, s in results if s == "PASS")
  nskip = sum(1 for _, s in results if s == "SKIP")
  nfail = sum(1 for _, s in results if s in ("FAIL", "MISSING"))
  print(f"\nSummary: {npass} passed, {nskip} skipped, {nfail} failed "
        f"({len(results)} suites)")
  sys.exit(1 if nfail else 0)


if __name__ == "__main__":
  main()
