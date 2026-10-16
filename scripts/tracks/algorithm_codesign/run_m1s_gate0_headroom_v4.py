#!/usr/bin/env python3
"""Isolated M1-S Gate0 v4 CLI. No trotterlib import or scientific default action."""
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from trottertracks.algorithm_codesign.m1s_gate0_v4.control import main

if __name__ == "__main__":
    raise SystemExit(main())
