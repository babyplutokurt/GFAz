#!/usr/bin/env python3
"""
Input-robustness regressions for the parser and the deserializer.

Covers gaps hardened in the audit:
  - CRLF (Windows) line endings round-trip identically to LF (no stray '\\r'
    leaking into the last field of each line);
  - a parser that starts in sequential-numeric mode can fall back to string IDs;
  - record families may be arbitrarily interleaved, including references that
    appear before their segment definitions;
  - a 0-byte .gfa is a valid empty graph (compress+decompress succeed);
  - a truncated .gfaz fails with a clean non-zero exit, never a crash/signal;
  - a record that references an undefined segment is a clean error by default
    and is accepted with --allow-missing-segments, never a crash/signal.
"""

import subprocess
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


def _compress(cli: Path, gfa: Path, gfaz: Path, tag: str):
  require_success(run_command([str(cli), "compress", str(gfa), str(gfaz)]),
                  f"compress {tag}")


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


def test_numeric_to_string_id_fallback(cli: Path):
  mixed_ids = (
      "H\tVN:Z:1.1\n"
      "S\t1\tA\n"
      "S\t2\tC\n"
      "S\talpha\tG\n"
      "L\t1\t+\talpha\t+\t0M\n"
      "P\tmixed\t1+,2+,alpha+\t*\n"
  )
  with tempfile.TemporaryDirectory() as dd:
    d = Path(dd)
    gfa = d / "mixed_ids.gfa"
    gfa.write_text(mixed_ids)
    out = _round_trip(cli, gfa, d, "mixed_ids")
    for expected in (
        "S\t1\tA",
        "S\t2\tC",
        "S\talpha\tG",
        "L\t1\t+\talpha\t+\t0M",
        "P\tmixed\t1+,2+,alpha+\t*",
    ):
      if expected not in out:
        raise AssertionError(
            f"numeric-to-string fallback lost {expected!r}:\n{out}")


def test_interleaved_record_families(cli: Path):
  grouped_numeric = (
      "H\tVN:Z:1.1\n"
      "S\t1\tA\tLN:i:1\n"
      "S\t2\tC\tLN:i:1\n"
      "L\t2\t+\t1\t-\t0M\n"
      "J\t2\t+\t1\t-\t*\n"
      "C\t2\t+\t1\t-\t0\t0M\n"
      "P\tp\t2-,1+\t*\n"
      "W\ts\t0\tchr\t0\t2\t>1<2\n"
  )
  interleaved_numeric = (
      "H\tVN:Z:1.1\n"
      "P\tp\t2-,1+\t*\n"
      "L\t2\t+\t1\t-\t0M\n"
      "S\t1\tA\tLN:i:1\n"
      "W\ts\t0\tchr\t0\t2\t>1<2\n"
      "J\t2\t+\t1\t-\t*\n"
      "S\t2\tC\tLN:i:1\n"
      "C\t2\t+\t1\t-\t0\t0M\n"
  )

  grouped_named = (
      "H\tVN:Z:1.1\n"
      "S\talpha\tA\n"
      "S\tbeta\tC\n"
      "L\tbeta\t+\talpha\t-\t0M\n"
      "P\tnamed\tbeta+,alpha-\t*\n"
      "W\ts\t0\tchr\t0\t2\t>beta<alpha\n"
  )
  interleaved_named = (
      "H\tVN:Z:1.1\n"
      "P\tnamed\tbeta+,alpha-\t*\n"
      "L\tbeta\t+\talpha\t-\t0M\n"
      "S\talpha\tA\n"
      "W\ts\t0\tchr\t0\t2\t>beta<alpha\n"
      "S\tbeta\tC\n"
  )

  grouped_out_of_order = (
      "H\tVN:Z:1.1\n"
      "S\t2\tC\n"
      "S\t1\tA\n"
      "P\tout-of-order\t2+,1+\t*\n"
  )
  interleaved_out_of_order = (
      "H\tVN:Z:1.1\n"
      "P\tout-of-order\t2+,1+\t*\n"
      "S\t2\tC\n"
      "S\t1\tA\n"
  )

  with tempfile.TemporaryDirectory() as dd:
    d = Path(dd)
    for label, grouped, interleaved in (
        ("numeric", grouped_numeric, interleaved_numeric),
        ("named", grouped_named, interleaved_named),
        ("out-of-order", grouped_out_of_order,
         interleaved_out_of_order),
    ):
      grouped_gfa = d / f"{label}.grouped.gfa"
      interleaved_gfa = d / f"{label}.interleaved.gfa"
      grouped_gfaz = d / f"{label}.grouped.gfaz"
      interleaved_gfaz = d / f"{label}.interleaved.gfaz"
      grouped_gfa.write_text(grouped)
      interleaved_gfa.write_text(interleaved)

      _compress(cli, grouped_gfa, grouped_gfaz, f"{label} grouped")
      _compress(cli, interleaved_gfa, interleaved_gfaz,
                f"{label} interleaved")
      if grouped_gfaz.read_bytes() != interleaved_gfaz.read_bytes():
        raise AssertionError(
            f"{label} interleaving changed the compressed archive")

      grouped_out = _round_trip(cli, grouped_gfa, d, f"{label}.grouped.rt")
      interleaved_out = _round_trip(
          cli, interleaved_gfa, d, f"{label}.interleaved.rt")
      if grouped_out != interleaved_out:
        raise AssertionError(
            f"{label} interleaving changed decompressed output:\n"
            f"--- grouped ---\n{grouped_out}\n"
            f"--- interleaved ---\n{interleaved_out}")


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


def test_unseekable_stdin_round_trip_and_truncation(cli: Path):
  with tempfile.TemporaryDirectory() as dd:
    d = Path(dd)
    gfa = d / "stdin.gfa"
    gfa.write_bytes(GFA_LF.encode())
    gfaz = d / "stdin.gfaz"
    require_success(run_command([str(cli), "compress", str(gfa), str(gfaz)]),
                    "compress for stdin")

    blob = gfaz.read_bytes()
    file_based = d / "file.out.gfa"
    require_success(
        run_command([str(cli), "decompress", str(gfaz), str(file_based)]),
        "file-based decompression for stdin comparison",
    )
    for mode, extra_args in (("streaming", []), ("legacy", ["--legacy"])):
      streamed = d / f"stdin.{mode}.out.gfa"
      result = subprocess.run(
          [str(cli), "decompress", *extra_args, "-", str(streamed)],
          input=blob,
          capture_output=True,
          check=False,
      )
      if result.returncode != 0:
        raise AssertionError(
            f"{mode} stdin decompression failed with exit code "
            f"{result.returncode}\nSTDOUT:\n"
            f"{result.stdout.decode(errors='replace')}\nSTDERR:\n"
            f"{result.stderr.decode(errors='replace')}")
      if streamed.read_bytes() != file_based.read_bytes():
        raise AssertionError(
            f"{mode} stdin decompression differs from file-based decompression")

    truncated = d / "stdin.truncated.out.gfa"
    result = subprocess.run(
        [str(cli), "decompress", "-", str(truncated)],
        input=blob[:max(8, len(blob) // 2)],
        capture_output=True,
        check=False,
    )
    if result.returncode <= 0:
      raise AssertionError(
          "truncated stdin should fail with a clean positive exit code, got "
          f"{result.returncode} (<=0 indicates a crash/signal)")


def _write_graph(path: Path, segment_names, records):
  path.write_text(
      "H\tVN:Z:1.1\n" +
      "".join(f"S\t{name}\tACGT\n" for name in segment_names) + records)


def test_missing_segment_references(cli: Path):
  # A record that references a segment no S-line defines is an error by
  # default. --allow-missing-segments accepts the input as before the check:
  # a reference by an unknown name is dropped, and with segments named 1..N a
  # number above N is kept. Covered for the three ways the parser names
  # segments (1..N, other numbers, strings) at 1 and 4 threads. The overlong
  # and non-numeric step names used to throw inside a parallel loop and abort
  # the process when segments are 1..N.
  filler = ">{a}>{b}" * 16
  cases = {
      "dense": ([str(i) for i in range(1, 9)], "1", "2", "9"),
      "sparse": (["5", "9", "40", "77"], "5", "9", "6"),
      "named": (["sa", "sb", "sc"], "sa", "sb", "zz"),
  }
  with tempfile.TemporaryDirectory(prefix="gfaz_missing_segment_") as tmp:
    d = Path(tmp)
    for tag, (names, a, b, missing) in cases.items():
      body = filler.format(a=a, b=b)
      good_records = (
          f"L\t{a}\t+\t{b}\t+\t0M\n"
          f"P\tp1\t{a}+,{b}-\t*\n"
          f"W\ts1\t0\tc1\t0\t9\t{body}>{a}{body}\n"
          f"W\ts2\t0\tc1\t0\t9\t>{a}<{b}\n")
      # Names that no naming mode can resolve.
      never_records = (
          f"L\t{a}\t+\t{b}\t+\t0M\n"
          f"L\t{a}\t+\tx1\t+\t0M\n"
          f"P\tp1\t{a}+,x1+,{b}-\t*\n"
          f"W\ts1\t0\tc1\t0\t9\t{body}>999999999999<x1"
          f">{a}{body}\n"
          f"W\ts2\t0\tc1\t0\t9\t>{a}<{b}>99999999999999999999<0\n")
      # A plausible name that just is not defined.
      undefined_records = (
          good_records +
          f"L\t{a}\t+\t{missing}\t+\t0M\n"
          f"P\tp2\t{a}+,{missing}+,{b}-\t*\n"
          f"W\ts3\t0\tc1\t0\t9\t{body}>{missing}{body}\n")
      good = d / f"{tag}.good.gfa"
      never = d / f"{tag}.never.gfa"
      undefined = d / f"{tag}.undefined.gfa"
      _write_graph(good, names, good_records)
      _write_graph(never, names, never_records)
      _write_graph(undefined, names, undefined_records)
      for threads in ("1", "4"):
        for bad in (never, undefined):
          strict = run_command([str(cli), "compress", "-j", threads, str(bad),
                                str(d / "strict.gfaz")])
          if (strict.returncode != 1 or
              "no S-line defines" not in strict.stderr):
            raise AssertionError(
                f"{bad.name} -j {threads}: expected a clean error for "
                f"missing segments, got exit {strict.returncode}:\n"
                f"{strict.stderr}")
        allow = ["-j", threads, "--allow-missing-segments"]
        expected = _round_trip_with(cli, good, d, f"{tag}.good{threads}",
                                    ["-j", threads])
        dropped = _round_trip_with(cli, never, d, f"{tag}.never{threads}",
                                   allow)
        if dropped != expected:
          raise AssertionError(
              f"{tag} -j {threads}: --allow-missing-segments did not drop "
              "exactly the unresolvable references")
        kept = _round_trip_with(cli, undefined, d,
                                f"{tag}.undefined{threads}", allow)
        has_missing_step = f">{missing}>" in kept
        if has_missing_step != (tag == "dense"):
          raise AssertionError(
              f"{tag} -j {threads}: --allow-missing-segments should keep an "
              "undefined number only when segments are named 1..N")


def _round_trip_with(cli: Path, gfa: Path, d: Path, tag: str, extra) -> str:
  gfaz = d / f"{tag}.gfaz"
  out = d / f"{tag}.out.gfa"
  require_success(
      run_command([str(cli), "compress", *extra, str(gfa), str(gfaz)]),
      f"compress {tag}")
  require_success(run_command([str(cli), "decompress", str(gfaz), str(out)]),
                  f"decompress {tag}")
  return out.read_text()


def main():
  cli = Path(CLI_PATH)
  ensure_cli_exists(cli)
  test_crlf_matches_lf(cli)
  test_numeric_to_string_id_fallback(cli)
  test_interleaved_record_families(cli)
  test_empty_file_is_empty_graph(cli)
  test_truncated_gfaz_fails_cleanly(cli)
  test_unseekable_stdin_round_trip_and_truncation(cli)
  test_missing_segment_references(cli)
  print("✅ PASS input_robustness")


if __name__ == "__main__":
  main()
