from pathlib import Path
import sys


def repo_root() -> Path:
  return Path(__file__).resolve().parents[1]


def add_build_to_syspath() -> Path:
  build_dir = repo_root() / "build"
  sys.path.insert(0, str(build_dir))
  return build_dir
