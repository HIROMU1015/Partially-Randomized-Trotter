#!/usr/bin/env python3
"""Independent A0 high-precision check; --inspect never starts arithmetic."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from trottertracks.b2d_evaluator.followup_io import main
if __name__ == "__main__":
    raise SystemExit(main("a0"))
