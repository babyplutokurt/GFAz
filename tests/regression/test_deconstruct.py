#!/usr/bin/env python3
"""
Regression suite for the `gfaz deconstruct` subcommand (GFA -> VCF).

The fixtures are designed so every VCF field is hand-verifiable:

  deconstruct_fixture.gfa
    Reference `ref` = nodes 1..9 spelling AAAA C GGGG TT A CCCC GG AAAT TTTT
    (total length 26). Two diploid samples each carry exactly one variant on
    one haplotype:
      - HG001#0 : SNP   node2 (C) -> node10 (T)          => substitution @ POS 5
      - HG002#0 : deletion of node5 (A)                  => indel @ POS 11
      - HG002#1 : insertion of node11 (GGG) before node8 => indel @ POS 18
    All other haplotypes follow the reference.

  deconstruct_revcomp_fixture.gfa
    Reference `ref` = 1+,2+,3+ (TT GG CC). HGX#0 takes node4 in REVERSE
    orientation between the flanks, so the ALT must be revcomp(AC) = GT.
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


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures"
MAIN_FIXTURE = FIXTURE_DIR / "deconstruct_fixture.gfa"
REVCOMP_FIXTURE = FIXTURE_DIR / "deconstruct_revcomp_fixture.gfa"
INVERSION_FIXTURE = FIXTURE_DIR / "deconstruct_inversion_fixture.gfa"
PANSN_REF_FIXTURE = FIXTURE_DIR / "deconstruct_pansn_ref_fixture.gfa"
REVERSE_PATH_FIXTURE = FIXTURE_DIR / "deconstruct_reverse_path_fixture.gfa"


def temp_gfaz(prefix: str) -> Path:
  handle = tempfile.NamedTemporaryFile(
      mode="w", suffix=".gfaz", prefix=prefix, delete=False
  )
  path = Path(handle.name)
  handle.close()
  return path


def compress(cli: Path, fixture: Path) -> Path:
  out = temp_gfaz("gfaz_dec_")
  require_success(
      run_command([str(cli), "compress", str(fixture), str(out)]),
      f"compress {fixture.name}",
  )
  return out


def data_lines(stdout: str):
  return [ln for ln in stdout.splitlines() if ln and not ln.startswith("##")]


def assert_lines(actual, expected, step):
  if actual != expected:
    raise AssertionError(
        f"{step}: VCF body mismatch.\n"
        f"Expected:\n" + "\n".join(expected) + "\n"
        f"Actual:\n" + "\n".join(actual)
    )


def test_main_records(cli: Path, gfaz: Path):
  result = run_command(
      [str(cli), "deconstruct", "-i", str(gfaz), "-r", "ref", "-S", "--linear"]
  )
  require_success(result, "deconstruct main")
  expected = [
      "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tHG001\tHG002",
      "ref\t5\t.\tC\tT\t.\t.\tAC=1;AN=4;AF=0.25;NS=2\tGT\t1|0\t0|0",
      "ref\t11\t.\tTA\tT\t.\t.\tAC=1;AN=4;AF=0.25;NS=2\tGT\t0|0\t1|0",
      "ref\t18\t.\tG\tGGGG\t.\t.\tAC=1;AN=4;AF=0.25;NS=2\tGT\t0|0\t0|1",
  ]
  assert_lines(data_lines(result.stdout), expected, "deconstruct main")


def test_no_gt(cli: Path, gfaz: Path):
  result = run_command(
      [str(cli), "deconstruct", "-i", str(gfaz), "-r", "ref", "-S", "-G", "--linear"]
  )
  require_success(result, "deconstruct --no-gt")
  expected = [
      "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO",
      "ref\t5\t.\tC\tT\t.\t.\tAC=1;AN=4;AF=0.25;NS=2",
      "ref\t11\t.\tTA\tT\t.\t.\tAC=1;AN=4;AF=0.25;NS=2",
      "ref\t18\t.\tG\tGGGG\t.\t.\tAC=1;AN=4;AF=0.25;NS=2",
  ]
  assert_lines(data_lines(result.stdout), expected, "deconstruct --no-gt")


def test_contig_header(cli: Path, gfaz: Path):
  result = run_command(
      [str(cli), "deconstruct", "-i", str(gfaz), "-r", "ref", "-S", "--linear"]
  )
  require_success(result, "deconstruct header")
  if "##contig=<ID=ref,length=26>" not in result.stdout:
    raise AssertionError(
        "deconstruct header: missing/incorrect contig line.\n" + result.stdout
    )


def test_reverse_complement(cli: Path, gfaz: Path):
  # node4 = AC, traversed reverse -> ALT must be revcomp(AC) = GT.
  result = run_command(
      [str(cli), "deconstruct", "-i", str(gfaz), "-r", "ref", "-S", "--linear"]
  )
  require_success(result, "deconstruct revcomp")
  expected = [
      "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tHGX",
      "ref\t3\t.\tGG\tGT\t.\t.\tAC=1;AN=1;AF=1;NS=1\tGT\t1",
  ]
  assert_lines(data_lines(result.stdout), expected, "deconstruct revcomp")


def test_inversion_matches_vg_fixture(cli: Path, gfaz: Path):
  result = run_command(
      [str(cli), "deconstruct", "-i", str(gfaz), "-r", "x", "-S", "--linear"]
  )
  require_success(result, "deconstruct inversion")
  expected = [
      "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ty",
      "x\t10\t.\tCTTGGAAATTTTCTGGAGTT\tAACTCCAGAAAATTTCCAAG\t.\t.\tAC=1;AN=1;AF=1;NS=1\tGT\t1",
  ]
  assert_lines(data_lines(result.stdout), expected, "deconstruct inversion")


def test_cpx_header_without_gt(cli: Path, gfaz: Path):
  result = run_command(
      [str(cli), "deconstruct", "-i", str(gfaz), "-r", "x", "-S", "-G", "-m", "1", "--linear"]
  )
  require_success(result, "deconstruct CPX no-GT")
  if "##ALT=<ID=CPX" not in result.stdout:
    raise AssertionError("deconstruct CPX no-GT: missing CPX ALT header")
  expected = [
      "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO",
      "x\t10\t.\tCTTGGAAATTTTCTGGAGTT\t<CPX>\t.\t.\tAC=1;AN=1;AF=1;NS=1",
  ]
  assert_lines(data_lines(result.stdout), expected, "deconstruct CPX no-GT")


def test_pansn_reference_contig_name(cli: Path, gfaz: Path):
  result = run_command(
      [str(cli), "deconstruct", "-i", str(gfaz), "-r", "REF#0#chrQ", "-S", "--linear"]
  )
  require_success(result, "deconstruct PanSN reference")
  if "##contig=<ID=chrQ,length=3>" not in result.stdout:
    raise AssertionError(
        "deconstruct PanSN reference: missing vg-style contig header.\n"
        + result.stdout
    )
  expected = [
      "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tHG001",
      "chrQ\t2\t.\tC\tT\t.\t.\tAC=1;AN=1;AF=1;NS=1\tGT\t1",
  ]
  assert_lines(data_lines(result.stdout), expected, "deconstruct PanSN reference")


def test_reverse_path_block(cli: Path, gfaz: Path):
  result = run_command(
      [str(cli), "deconstruct", "-i", str(gfaz), "-r", "ref", "-S", "--linear"]
  )
  require_success(result, "deconstruct reverse path block")
  expected = [
      "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tHGX",
      "ref\t2\t.\tC\tT\t.\t.\tAC=1;AN=1;AF=1;NS=1\tGT\t1",
  ]
  assert_lines(data_lines(result.stdout), expected, "deconstruct reverse path block")


def test_snarl_inversion(cli: Path, gfaz: Path):
  # Snarl mode must call the inversion that the strict superbubble detector
  # rejects (node2 traversed reverse between flanks node1/node3), matching
  # `vg deconstruct -p x`. The interior is captured on the opposite strand and
  # reverse-complemented back to the reference frame.
  expected = [
      "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ty",
      "x\t10\t.\tCTTGGAAATTTTCTGGAGTT\tAACTCCAGAAAATTTCCAAG\t.\t.\tAC=1;AN=1;AF=1;NS=1\tGT\t1",
  ]
  for flag in ("--snarl", "--vg-compat", "--vg-compact"):
    result = run_command(
        [str(cli), "deconstruct", "-i", str(gfaz), "-r", "x", "-S", flag]
    )
    require_success(result, f"deconstruct {flag} inversion")
    assert_lines(data_lines(result.stdout), expected, f"deconstruct {flag} inversion")


def test_missing_reference_errors(cli: Path, gfaz: Path):
  result = run_command(
      [str(cli), "deconstruct", "-i", str(gfaz), "-r", "nope", "-S"]
  )
  if result.returncode == 0:
    raise AssertionError(
        "deconstruct should fail when the reference name is absent"
    )


def main():
  cli = Path(CLI_PATH)
  ensure_cli_exists(cli)

  main_gfaz = compress(cli, MAIN_FIXTURE)
  rc_gfaz = compress(cli, REVCOMP_FIXTURE)
  inv_gfaz = compress(cli, INVERSION_FIXTURE)
  pansn_gfaz = compress(cli, PANSN_REF_FIXTURE)
  reverse_path_gfaz = compress(cli, REVERSE_PATH_FIXTURE)
  try:
    test_main_records(cli, main_gfaz)
    test_no_gt(cli, main_gfaz)
    test_contig_header(cli, main_gfaz)
    test_missing_reference_errors(cli, main_gfaz)
    test_reverse_complement(cli, rc_gfaz)
    test_inversion_matches_vg_fixture(cli, inv_gfaz)
    test_snarl_inversion(cli, inv_gfaz)
    test_cpx_header_without_gt(cli, inv_gfaz)
    test_pansn_reference_contig_name(cli, pansn_gfaz)
    test_reverse_path_block(cli, reverse_path_gfaz)
    print("✅ PASS deconstruct_regressions")
  finally:
    for p in (main_gfaz, rc_gfaz, inv_gfaz, pansn_gfaz, reverse_path_gfaz):
      if p.exists():
        p.unlink()


if __name__ == "__main__":
  main()
