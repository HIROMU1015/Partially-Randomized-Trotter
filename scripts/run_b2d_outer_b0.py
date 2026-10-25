#!/usr/bin/env python3
"""Fixed B0 native pilot; --inspect never builds circuits or operators."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from trottertracks.b2d_evaluator.followup_io import main
if __name__ == "__main__":
    raise SystemExit(main("b0"))
