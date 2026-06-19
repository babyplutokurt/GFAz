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
LINKS_FIXTURE = FIXTURE_DIR / "deconstruct_links_fixture.gfa"
SUBRANGE_FIXTURE = FIXTURE_DIR / "deconstruct_subrange_fixture.gfa"


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


def test_group_by_haplotype(cli: Path, gfaz: Path):
  # Default (snarl) mode on a links-bearing graph; -H -> one column per
  # (sample, hap), each haploid.
  result = run_command(
      [str(cli), "deconstruct", "-i", str(gfaz), "-r", "ref", "-H"]
  )
  require_success(result, "deconstruct -H")
  expected = [
      "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tHG001#0\tHG001#1\tHG002#0\tHG002#1",
      "ref\t5\t.\tC\tT\t.\t.\tAC=1;AN=4;AF=0.25;NS=4\tGT\t1\t0\t0\t0",
      "ref\t11\t.\tTA\tT\t.\t.\tAC=1;AN=4;AF=0.25;NS=4\tGT\t0\t0\t1\t0",
      "ref\t18\t.\tG\tGGGG\t.\t.\tAC=1;AN=4;AF=0.25;NS=4\tGT\t0\t0\t0\t1",
  ]
  assert_lines(data_lines(result.stdout), expected, "deconstruct -H")


def test_per_path(cli: Path, gfaz: Path):
  # -p -> one haploid column per path/walk.
  result = run_command(
      [str(cli), "deconstruct", "-i", str(gfaz), "-r", "ref", "-p"]
  )
  require_success(result, "deconstruct -p")
  expected = [
      "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tHG001#0#chr1\tHG001#1#chr1\tHG002#0#chr1\tHG002#1#chr1",
      "ref\t5\t.\tC\tT\t.\t.\tAC=1;AN=4;AF=0.25;NS=4\tGT\t1\t0\t0\t0",
      "ref\t11\t.\tTA\tT\t.\t.\tAC=1;AN=4;AF=0.25;NS=4\tGT\t0\t0\t1\t0",
      "ref\t18\t.\tG\tGGGG\t.\t.\tAC=1;AN=4;AF=0.25;NS=4\tGT\t0\t0\t0\t1",
  ]
  assert_lines(data_lines(result.stdout), expected, "deconstruct -p")


def test_subrange_reference(cli: Path, gfaz: Path):
  """Reference paths covering a non-zero subrange of their contig (vg parity).

  The fixture has two whole-chromosome references as W-lines with subranges:
  REF#0#chr1 over chr1[1000,1009) (AAAA C GGGG) and REF#0#chr2 over chr2[500,505)
  (GG A CC); HG001 carries a SNP on each (C->T at chr1, A->T at chr2). Checks:
    - non-zero-start POS offset: chr1 variant at walk-offset 5 -> POS 1000+5=1005;
      chr2 variant at walk-offset 3 -> POS 500+3=503; ##contig length = subrange end;
    - `-P REF` selects BOTH reference chromosomes in one VCF (so REF is not a
      sample, only HG001 is);
    - the PanSN base name resolves the same slice as the exact ":start-end" name.
  """
  expected_prefix = [
      "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tHG001",
      "chr1\t1005\t.\tC\tT\t.\t.\tAC=1;AN=1;AF=1;NS=1\tGT\t1",
      "chr2\t503\t.\tA\tT\t.\t.\tAC=1;AN=1;AF=1;NS=1\tGT\t1",
  ]
  result = run_command(
      [str(cli), "deconstruct", "-i", str(gfaz), "-P", "REF", "-S"]
  )
  require_success(result, "deconstruct -P REF")
  for contig in ("##contig=<ID=chr1,length=1009>", "##contig=<ID=chr2,length=505>"):
    if contig not in result.stdout:
      raise AssertionError(
          "deconstruct -P: contig length must be the subrange end.\n"
          + result.stdout)
  assert_lines(data_lines(result.stdout), expected_prefix, "deconstruct -P REF")

  # Base name (no ":start-end") must resolve the same slice as the exact name,
  # in both snarl (default) and --linear modes.
  for mode in ([], ["--linear"]):
    base = run_command(
        [str(cli), "deconstruct", "-i", str(gfaz), "-r", "REF#0#chr1", "-S", *mode])
    exact = run_command(
        [str(cli), "deconstruct", "-i", str(gfaz), "-r", "REF#0#chr1:1000-1009",
         "-S", *mode])
    require_success(base, f"deconstruct base-name {mode}")
    require_success(exact, f"deconstruct exact-name {mode}")
    if base.stdout != exact.stdout:
      raise AssertionError(
          f"deconstruct {mode}: base name and exact ':start-end' name differ.")
    if "\t1005\t" not in base.stdout:
      raise AssertionError(
          f"deconstruct {mode}: subrange-start POS offset not applied (no 1005).")

  # Reference selection goes through one name-sorted index (exact + raw-prefix).
  # A prefix matching a single contig selects only that contig...
  one = run_command(
      [str(cli), "deconstruct", "-i", str(gfaz), "-P", "REF#0#chr2", "-S"])
  require_success(one, "deconstruct -P REF#0#chr2")
  chroms = {ln.split("\t", 1)[0] for ln in data_lines(one.stdout)
            if not ln.startswith("#")}
  if chroms != {"chr2"}:
    raise AssertionError(
        f"deconstruct -P REF#0#chr2 should select only chr2, got {chroms}")
  # ...and a prefix matching nothing is a clean error, not an empty VCF.
  none = run_command(
      [str(cli), "deconstruct", "-i", str(gfaz), "-P", "NOPE", "-S"])
  if none.returncode == 0:
    raise AssertionError("deconstruct -P with no match should fail")


def test_graph_info_at(cli: Path, gfaz: Path):
  """`-a` adds graph annotations for vg parity without changing the variant call.

  On the subrange fixture, gfaz renumbers segments 1,2,3,10,21,22,23,24 to
  1..8 in declaration order, so chr1's snarl is >1..>3 with REF interior node 2
  (C) and ALT node 10->4 (T), and chr2's is >5..>8 with REF node 6 (A) / ALT
  node 23->7 (T). With `-a` we expect, relative to the default output:
    - CHROM = full PanSN base name (REF#0#chr1) and matching ##contig ids;
    - ID    = snarl boundary id (>src>sink) in gfaz's node space;
    - INFO gains ;AT=<ref-path>,<alt-path> (REF first), plus the AT header line;
    - everything else (POS/REF/ALT/AC/AN/AF/NS/GT) identical to the default.
  """
  expected = [
      "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tHG001",
      "REF#0#chr1\t1005\t>1>3\tC\tT\t.\t.\tAC=1;AN=1;AF=1;NS=1;AT=>1>2>3,>1>4>3\tGT\t1",
      "REF#0#chr2\t503\t>5>8\tA\tT\t.\t.\tAC=1;AN=1;AF=1;NS=1;AT=>5>6>8,>5>7>8\tGT\t1",
  ]
  result = run_command(
      [str(cli), "deconstruct", "-i", str(gfaz), "-P", "REF", "-S", "-a"]
  )
  require_success(result, "deconstruct -P REF -a")
  for need in ("##contig=<ID=REF#0#chr1,length=1009>",
               "##contig=<ID=REF#0#chr2,length=505>",
               "##INFO=<ID=AT,"):
    if need not in result.stdout:
      raise AssertionError(
          f"deconstruct -a: expected header line {need!r}.\n" + result.stdout)
  assert_lines(data_lines(result.stdout), expected, "deconstruct -P REF -a")

  # The default (no -a) must stay lean: no AT, ID='.', bare CHROM. (Guards the
  # byte-identical default contract.)
  plain = run_command(
      [str(cli), "deconstruct", "-i", str(gfaz), "-P", "REF", "-S"])
  require_success(plain, "deconstruct -P REF (default)")
  if ";AT=" in plain.stdout or "##INFO=<ID=AT," in plain.stdout:
    raise AssertionError("default deconstruct must not emit AT.\n" + plain.stdout)
  # --at / --graph-info long aliases behave identically to -a.
  for alias in ("--at", "--graph-info"):
    aliased = run_command(
        [str(cli), "deconstruct", "-i", str(gfaz), "-P", "REF", "-S", alias])
    require_success(aliased, f"deconstruct {alias}")
    if aliased.stdout != result.stdout:
      raise AssertionError(f"deconstruct {alias} differs from -a output.")


def test_linear_at_header_gated(cli: Path, gfaz: Path):
  """`--linear -a` must not advertise AT: only the snarl writer emits it.

  The legacy linear writer produces no AT values or snarl IDs, so emitting the
  ##INFO=<ID=AT,...> header (or any ;AT=) there would describe a field that
  never appears. The graph-info CHROM naming still applies to both writers.
  """
  result = run_command(
      [str(cli), "deconstruct", "-i", str(gfaz), "-P", "REF", "-S",
       "--linear", "-a"]
  )
  require_success(result, "deconstruct -P REF --linear -a")
  if "##INFO=<ID=AT," in result.stdout:
    raise AssertionError(
        "--linear -a must not emit the AT header (linear writer has no AT).\n"
        + result.stdout)
  if ";AT=" in result.stdout:
    raise AssertionError(
        "--linear -a must not emit any ;AT= field.\n" + result.stdout)
  # CHROM PanSN naming still applies under -a (both writers).
  if "REF#0#chr1" not in result.stdout:
    raise AssertionError(
        "--linear -a should still use the PanSN CHROM name.\n" + result.stdout)


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
  links_gfaz = compress(cli, LINKS_FIXTURE)
  subrange_gfaz = compress(cli, SUBRANGE_FIXTURE)
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
    test_group_by_haplotype(cli, links_gfaz)
    test_per_path(cli, links_gfaz)
    test_subrange_reference(cli, subrange_gfaz)
    test_graph_info_at(cli, subrange_gfaz)
    test_linear_at_header_gated(cli, subrange_gfaz)
    print("✅ PASS deconstruct_regressions")
  finally:
    for p in (main_gfaz, rc_gfaz, inv_gfaz, pansn_gfaz, reverse_path_gfaz,
              links_gfaz, subrange_gfaz):
      if p.exists():
        p.unlink()


if __name__ == "__main__":
  main()
