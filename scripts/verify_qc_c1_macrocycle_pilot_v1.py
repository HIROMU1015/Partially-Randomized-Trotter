#!/usr/bin/env python3
"""Audit a saved, separately authorized C1 attempt. No fresh eigensolvers."""
import json
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(root/'src'))
if __name__=='__main__':
    from trottertracks.representation_exploration.c1_macrocycle_checker_v1 import audit
    result=audit(root)
    out=root/'artifacts/qc_c1_macrocycle_pilot_v1/2026-10-13/attempt_1'
    (out/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
