#!/usr/bin/env python3
"""Passive SOURCE/input/fresh-output preflight only; no science launch option."""
import json
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'src'))
from trottertracks.c_h4_relocated_execution import preflight
c=preflight(root)
print(json.dumps(dict(status='PASSIVE_PREFLIGHT_PASS',scope=c['scope'],arrays_decoded=0,
    science_executed=0,fresh_output=c['fresh_output'],execution_permission=False)))
