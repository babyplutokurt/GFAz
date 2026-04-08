#!/usr/bin/env python3
"""
Compatibility wrapper for the split GPU round-trip tests.

Use one of:
  tests/gpu/test_gpu_legacy_roundtrip.py
  tests/gpu/test_gpu_streaming_roundtrip.py

This wrapper preserves the old entry point and defaults to the legacy test.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.gpu.test_gpu_legacy_roundtrip import main


if __name__ == "__main__":
    sys.exit(main())
