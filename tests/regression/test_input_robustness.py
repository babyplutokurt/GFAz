#!/usr/bin/env python3
"""
Input-robustness regressions for the parser and the deserializer.

Covers gaps hardened in the audit:
  - CRLF (Windows) line endings round-trip identically to LF (no stray '\\r'
    leaking into the last field of each line);
  - a 0-byte .gfa is a valid empty graph (compress+decompress succeed);
  - a truncated .gfaz fails with a clean non-zero exit, never a crash/signal.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.regression.regression_utils import (
    CLI_PATH,
    ensure_cli_exists,
    require_success,
    run_command,
)

GFA_LF = (
    "H\tVN:Z:1.1\n"
    "S\t1\tACGT\tLN:i:4\n"
    "S\t2\tTT\n"
    "L\t1\t+\t2\t+\t0M\n"
    "P\tp1\t1+,2+\t*\n"
    "W\ts1\t0\tc1\t0\t6\t>1>2\n"
)


def _round_trip(cli: Path, gfa: Path, d: Path, tag: str) -> str:
  gfaz = d / f"{tag}.gfaz"
  out = d / f"{tag}.out.gfa"
  require_success(run_command([str(cli), "compress", str(gfa), str(gfaz)]),
                  f"compress {tag}")
  require_success(run_command([str(cli), "decompress", str(gfaz), str(out)]),
                  f"decompress {tag}")
  return out.read_text()


def test_crlf_matches_lf(cli: Path):
  with tempfile.TemporaryDirectory() as dd:
    d = Path(dd)
    lf = d / "lf.gfa"
    lf.write_bytes(GFA_LF.encode())
    crlf = d / "crlf.gfa"
    crlf.write_bytes(GFA_LF.replace("\n", "\r\n").encode())

    lf_out = _round_trip(cli, lf, d, "lf")
    crlf_out = _round_trip(cli, crlf, d, "crlf")
    if lf_out != crlf_out:
      raise AssertionError(
          "CRLF input did not round-trip identically to LF input:\n"
          f"--- LF ---\n{lf_out}\n--- CRLF ---\n{crlf_out}")
    if "\r" in crlf_out:
      raise AssertionError("decompressed CRLF output still contains a '\\r'")


def test_empty_file_is_empty_graph(cli: Path):
  with tempfile.TemporaryDirectory() as dd:
    d = Path(dd)
    empty = d / "empty.gfa"
    empty.write_bytes(b"")
    gfaz = d / "empty.gfaz"
    out = d / "empty.out.gfa"
    require_success(run_command([str(cli), "compress", str(empty), str(gfaz)]),
                    "compress empty file")
    require_success(run_command([str(cli), "decompress", str(gfaz), str(out)]),
                    "decompress empty file")


def test_truncated_gfaz_fails_cleanly(cli: Path):
  with tempfile.TemporaryDirectory() as dd:
    d = Path(dd)
    gfa = d / "g.gfa"
    gfa.write_bytes(GFA_LF.encode())
    gfaz = d / "g.gfaz"
    require_success(run_command([str(cli), "compress", str(gfa), str(gfaz)]),
                    "compress for truncation")

    blob = gfaz.read_bytes()
    # Keep the magic+version header but lop off the body so length fields point
    # past EOF.
    truncated = d / "trunc.gfaz"
    truncated.write_bytes(blob[: max(8, len(blob) // 2)])

    result = run_command([str(cli), "decompress", str(truncated),
                          str(d / "trunc.out.gfa")])
    if result.returncode <= 0:
      raise AssertionError(
          "truncated .gfaz should fail with a clean positive exit code, got "
          f"{result.returncode} (<=0 indicates a crash/signal)")


def main():
  cli = Path(CLI_PATH)
  ensure_cli_exists(cli)
  test_crlf_matches_lf(cli)
  test_empty_file_is_empty_graph(cli)
  test_truncated_gfaz_fails_cleanly(cli)
  print("✅ PASS input_robustness")


if __name__ == "__main__":
  main()
