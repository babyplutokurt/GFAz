#!/usr/bin/env python3
"""
Runs the libhandlegraph adapter cross-check (tests/cpp/test_handle_graph.cpp)
on the fixtures and on the generated multi-block graph, at several index block
sizes. The checker compares every interface method of GfazHandleGraph with a
brute-force answer from the decompressed GFA.

SKIPs unless the build has the adapter: configure with
  -DGFAZ_BUILD_HANDLEGRAPH=ON -DGFAZ_HANDLEGRAPH_INCLUDE_DIR=<dir>
  -DGFAZ_HANDLEGRAPH_LIBRARY=<libhandlegraph.a>
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.regression.regression_utils import (
    BUILD_DIR,
    CLI_PATH,
    SkipTest,
    ensure_cli_exists,
    require_success,
    run_command,
    run_main,
)
from tests.regression.test_extract_region import FIXTURES, make_synthetic

CHECKER = BUILD_DIR / "gfaz_handle_graph_test"


def main():
  ensure_cli_exists(CLI_PATH)
  if not CHECKER.exists():
    raise SkipTest("libhandlegraph adapter not built (GFAZ_BUILD_HANDLEGRAPH)")
  cli = str(CLI_PATH)
  with tempfile.TemporaryDirectory(prefix="gfaz_handle_") as tmp_name:
    tmp = Path(tmp_name)
    synthetic = tmp / "synthetic.gfa"
    make_synthetic(synthetic)
    graphs = [synthetic] + [FIXTURES / name for name in (
        "deconstruct_links_fixture.gfa",
        "growth_fixture_walks.gfa",
        "deconstruct_inversion_fixture.gfa",
        "roundtrip_orientations_fixture.gfa",
    )]
    for gfa in graphs:
      gfaz = tmp / (gfa.stem + ".gfaz")
      full = tmp / (gfa.stem + ".full.gfa")
      require_success(run_command([cli, "compress", str(gfa), str(gfaz)]),
                      f"compress {gfa.name}")
      require_success(run_command([cli, "decompress", str(gfaz), str(full)]),
                      f"decompress {gfa.name}")
      for block_steps in (1, 7, 1024):
        require_success(
            run_command([str(CHECKER), str(gfaz), str(full), str(block_steps)]),
            f"adapter check {gfa.name} block_steps={block_steps}")
  print("✅ libhandlegraph adapter checks passed")


if __name__ == "__main__":
  run_main(main)
