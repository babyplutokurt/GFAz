#!/usr/bin/env python3
"""
Concordance: `gfaz deconstruct` (default top-level-snarl mode) vs
`vg deconstruct` (committed golden).

Producing output identical to vg is the goal of gfaz's default mode. The fixture
carries L-lines (vg and gfaz's snarl mode both need links). Comparison is at the
(CHROM, POS, REF, ALT) -> per-sample GT level; vg's extra ID/QUAL and INFO AT=
fields are ignored (see normalize_vcf_for_golden). The golden holds vg's
normalized VCF; this test reads the golden and runs only gfaz.

On this tiny hand-built fixture the match is exact (strict). For large real
graphs vg/gfaz concord at ~99.99%; vcf_concordant() in concordance_utils
supports tolerant thresholds for an optional large-dataset test.
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
    run_main,
)
from tests.concordance.concordance_utils import (
    load_golden,
    normalize_vcf_for_golden,
    strip_golden_comments,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
GFA = REPO_ROOT / "tests" / "fixtures" / "deconstruct_links_fixture.gfa"
GOLDEN = "deconstruct_links_fixture.golden"


def compress(cli: Path) -> Path:
  handle = tempfile.NamedTemporaryFile(
      mode="w", suffix=".gfaz", prefix="gfaz_deccon_", delete=False)
  out = Path(handle.name)
  handle.close()
  require_success(run_command([str(cli), "compress", str(GFA), str(out)]),
                  "compress deconstruct fixture")
  return out


def main():
  cli = Path(CLI_PATH)
  ensure_cli_exists(cli)
  gfaz = compress(cli)
  try:
    result = run_command(
        [str(cli), "deconstruct", "-i", str(gfaz), "-r", "ref", "-S"])
    require_success(result, "gfaz deconstruct")
    actual = normalize_vcf_for_golden(result.stdout)
    expected = strip_golden_comments(load_golden(GOLDEN))
    if actual != expected:
      raise AssertionError(
          "deconstruct concordance mismatch.\n"
          f"--- expected (vg golden) ---\n{expected}\n"
          f"--- actual (gfaz) ---\n{actual}")
    print("✅ PASS deconstruct_vs_vg")
  finally:
    if gfaz.exists():
      gfaz.unlink()


if __name__ == "__main__":
  run_main(main)
