#!/usr/bin/env python3
"""Passive byte/package preflight only; no scientific execution option."""
import argparse
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'src'))
from trottertracks.c_cdf0_relocated import CONTRACT, preflight

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--contract', default=CONTRACT)
args = parser.parse_args()
c = preflight(root, args.contract)
print(json.dumps(dict(status='PASSIVE_PREFLIGHT_PASS', scope=c['scope'],
    candidate_id=c['candidate']['id'], arrays_decoded=0, science_executed=0,
    execution_permission=False, fresh_output=c['fresh_output'])))
