#!/usr/bin/env python3
"""Authorized saved-value/descriptor audit only;no new H4 operator evaluation."""
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
if __name__=='__main__':
    from trottertracks.c_block_witness_audit import audit
    def read(path):return json.loads((ROOT/path).read_text())
    family='artifacts/research_c_cx_structural_blocks/2026-11-05/block-pilot01/'
    original='artifacts/research_c_head_bridge_stage_p/2026-11-04/P-run01/'
    saved=read('artifacts/research_c_compiled_raw_acceleration/2026-11-05/local-pilot01/structural_plan.json')['plan']
    value=audit(saved,read(family+'structural_plan.json')['plan'],read(family+'compiled_raw.json'),
                read(original+'compiled_record.json'),read(original+'raw_record.json'))
    print(json.dumps(value,ensure_ascii=False,indent=2))
