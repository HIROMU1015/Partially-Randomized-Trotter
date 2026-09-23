#!/usr/bin/env python3
"""CLI entry point for bounded, resumable validation task batches."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from trotterlib.parallel_validation_executor import cli_main


if __name__ == "__main__":
    raise SystemExit(cli_main())
