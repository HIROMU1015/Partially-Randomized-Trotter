#!/usr/bin/env python3
"""Bytes-only preparation check; no scientific execution command is exposed."""
import json
from pathlib import Path
from trottertracks.c_direct_m0_runner import preflight


if __name__=='__main__':
    print(json.dumps(preflight(Path(__file__).resolve().parents[1]),sort_keys=True))
