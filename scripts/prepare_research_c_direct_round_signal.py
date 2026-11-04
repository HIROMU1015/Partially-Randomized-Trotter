#!/usr/bin/env python3
"""Passive technical SOURCE/draft checks only; no science execution flag."""
import json
from pathlib import Path
from trottertracks.c_direct_round_signal import preflight


if __name__ == '__main__':
    print(json.dumps(preflight(Path(__file__).resolve().parents[1]),sort_keys=True))
