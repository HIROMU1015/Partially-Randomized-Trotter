#!/usr/bin/env python3
"""Passive SOURCE/input/fresh-output preflight only; no science launch option."""
import argparse
import json
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'src'))
from trottertracks.c_h4_relocated_execution import preflight
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--contract',default='docs/research/research_c_h4_relocated_downstream_contract_unapproved_20261106.json')
args=parser.parse_args()
c=preflight(root,args.contract)
print(json.dumps(dict(status='PASSIVE_PREFLIGHT_PASS',scope=c['scope'],arrays_decoded=0,
    science_executed=0,fresh_output=c['fresh_output'],execution_permission=False)))
