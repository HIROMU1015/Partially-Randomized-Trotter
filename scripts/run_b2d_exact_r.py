#!/usr/bin/env python3
"""Read-only Exact-R inspection/preflight or explicit one-attempt execution."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from trottertracks.b2d_evaluator.exact_r import main

if __name__ == "__main__":
    raise SystemExit(main())
