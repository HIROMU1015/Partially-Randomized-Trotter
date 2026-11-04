#!/usr/bin/env python3
"""Saved-descriptor exact Boolean Clifford gate only. No rotations or bound."""
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
if __name__=='__main__':
    from trottertracks.c_relative_pauli_enclosure import normal_form
    saved=json.loads((ROOT/'artifacts/research_c_compiled_raw_acceleration/2026-11-05/local-pilot01/structural_plan.json').read_text())['plan']
    forms=[normal_form(e,collect_rotations=False) for e in saved['expansions']]
    equal=forms[0]['inverse_tableau']==forms[1]['inverse_tableau']
    print(json.dumps(dict(kind='EXACT_DESCRIPTOR_CLIFFORD_SIGNATURE_ONLY',source_primitive_plan_id=saved['plan_id'],
        projective_Clifford_equal=equal,forms=[dict(inverse_tableau=f['inverse_tableau'],phase=f['phase'],
            declared_Clifford_gate_count=len(f['clifford_record']['gates'])) for f in forms],
        new_rotation_plan_formed=False,new_error_bound=None,new_H4_Arb_evaluations=0,new_H4_build=0,new_H4_compile=0,
        note='exact integer/tableau and rational/pi descriptor transformation;no gate matrices,trig,peeling,partition optimization or numerical operator evaluation'),indent=2))
