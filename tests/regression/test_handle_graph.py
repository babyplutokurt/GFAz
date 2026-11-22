#!/usr/bin/env python3
"""
Runs the libhandlegraph adapter cross-check (tests/cpp/test_handle_graph.cpp)
on the fixtures and on the generated multi-block graph (under three segment
namings), at several index block sizes. The checker compares every interface method of GfazHandleGraph with a
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
from tests.regression.test_extract_region import (
    FIXTURES,
    SEGMENT_NAMINGS,
    make_synthetic,
)

CHECKER = BUILD_DIR / "gfaz_handle_graph_test"


def main():
  ensure_cli_exists(CLI_PATH)
  if not CHECKER.exists():
    raise SkipTest("libhandlegraph adapter not built (GFAZ_BUILD_HANDLEGRAPH)")
  cli = str(CLI_PATH)
  with tempfile.TemporaryDirectory(prefix="gfaz_handle_") as tmp_name:
    tmp = Path(tmp_name)
    # The generated graph with its segments named 1..N, by other numbers
    # (which become the node ids) and by strings (ids 1..N plus names).
    synthetic = []
    for label, name_of in SEGMENT_NAMINGS.items():
      synthetic.append(tmp / f"synthetic_{label}.gfa")
      make_synthetic(synthetic[-1], name_of)
    graphs = synthetic + [FIXTURES / name for name in (
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
      # A tiny block cache forces constant recycling, which is where a stale
      # reference to a decoded block would show up.
      for block_steps, cached_blocks in ((1, 1), (7, 2), (1024, 1024)):
        require_success(
            run_command([str(CHECKER), str(gfaz), str(full), str(block_steps),
                         str(cached_blocks)]),
            f"adapter check {gfa.name} block_steps={block_steps} "
            f"cached_blocks={cached_blocks}")
      # The fixtures have no zone crowded enough for the repeat index; a
      # threshold of 0 or 1 blocks puts all or some of their zones in it.
      for repeat_zone_blocks in (0, 1):
        require_success(
            run_command([str(CHECKER), str(gfaz), str(full), "7", "2",
                         str(repeat_zone_blocks)]),
            f"adapter check {gfa.name} "
            f"repeat_zone_blocks={repeat_zone_blocks}")
      # A saved index next to the container is used when its blocks are no
      # longer than the ones asked for, and passed over otherwise.
      require_success(run_command([cli, "index", "-i", str(gfaz), "-b", "4"]),
                      f"index {gfa.name}")
      for block_steps in (2, 7):
        require_success(
            run_command([str(CHECKER), str(gfaz), str(full), str(block_steps)]),
            f"adapter check {gfa.name} saved index, block_steps={block_steps}")
  print("✅ libhandlegraph adapter checks passed")


if __name__ == "__main__":
  run_main(main)
