#!/usr/bin/env python3
"""Fixed Full DF-PF: inspect/preflight only, or explicit one-shot execution."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from trottertracks.b2d_evaluator.full_df import main

if __name__ == "__main__":
    raise SystemExit(main())
