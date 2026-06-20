#!/usr/bin/env python3
"""
Concordance: `gfaz depth` vs `odgi depth` (committed golden).

The depth summary (#node.count graph.length step.count path.length ...) is a pure
aggregate, and the per-node `-d` table is keyed by node id. gfaz uses its own
1-based node ids; on this fixture the GFA segment names are already 1..N, so the
ids coincide and the two tools agree byte-for-byte. Goldens hold odgi's output
(regenerate with scripts/gen_golden.py); this test never invokes odgi.

Correspondence: gfaz depth (default) == odgi depth -S ; gfaz depth -d == odgi
depth -d. odgi drops W-lines on `odgi build`, so the fixture is path-only.
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
    normalize_depth,
    strip_golden_comments,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
GFA = REPO_ROOT / "tests" / "fixtures" / "similarity_fixture.gfa"


def compress(cli: Path) -> Path:
  handle = tempfile.NamedTemporaryFile(
      mode="w", suffix=".gfaz", prefix="gfaz_depthcon_", delete=False)
  out = Path(handle.name)
  handle.close()
  require_success(run_command([str(cli), "compress", str(GFA), str(out)]),
                  "compress depth fixture")
  return out


def check(cli: Path, gfaz: Path, extra, golden_name: str):
  result = run_command(
      [str(cli), "depth", "-i", str(gfaz), "-t", "2"] + extra)
  require_success(result, f"gfaz depth {' '.join(extra) or '(summary)'}")
  actual = normalize_depth(result.stdout)
  expected = strip_golden_comments(load_golden(golden_name))
  if actual != expected:
    raise AssertionError(
        f"depth concordance mismatch for {golden_name}.\n"
        f"--- expected (odgi golden) ---\n{expected}\n"
        f"--- actual (gfaz) ---\n{actual}")


def main():
  cli = Path(CLI_PATH)
  ensure_cli_exists(cli)
  gfaz = compress(cli)
  try:
    check(cli, gfaz, [], "depth_fixture.summarize.golden")
    check(cli, gfaz, ["-d"], "depth_fixture.per_node.golden")
    print("✅ PASS depth_vs_odgi")
  finally:
    if gfaz.exists():
      gfaz.unlink()


if __name__ == "__main__":
  run_main(main)
