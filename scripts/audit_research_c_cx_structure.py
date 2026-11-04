#!/usr/bin/env python3
"""Saved descriptors only: no Arb, circuit construction, compilation or arrays."""
from collections import Counter
import difflib
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def audit():
    from trottertracks import c_cx_structural_blocks as b,c_compiled_raw_local as local
    family=ROOT/'artifacts/research_c_compiled_raw_acceleration/2026-11-05/local-pilot01'
    original=ROOT/'artifacts/research_c_head_bridge_stage_p/2026-11-04/P-run01'
    paths=[family/'structural_plan.json',original/'compiled_record.json',original/'raw_record.json']
    saved,C,R=[json.loads(p.read_text()) for p in paths];saved=saved['plan']
    b.verify_saved_plan(saved,C,R)
    keys=[[tuple(e['gate']['qubits']) for e in s['events']] for s in saved['skeletons']]
    edits=[]
    for tag,i,k,j,l in difflib.SequenceMatcher(None,*keys,autojunk=False).get_opcodes():
        if tag!='equal':edits.append(dict(tag=tag,compiled_range=[i,k],raw_range=[j,l],compiled_CX=keys[0][i:k],raw_CX=keys[1][j:l]))
    variants=[]
    for exact in (False,True):
        for guard in (0,1):
            p=b.partition(saved,max_support=4,guard=guard,exact_clifford=exact)
            blocks=[f for f in p['factors'] if f['kind']=='block']
            bodies={local._digest(dict(kind=f['kind'],n=len(f['support']),left=f['left'],right=f['right']))
                    for f in p['factors'] if f['kind']!='shared_CX'}
            variants.append(dict(exact_clifford=exact,guard=guard,plan_id=p['plan_id'],
                applicable=p['applicable'],complete_partition=p['complete_partition'],
                normalized_CX=p['normalized_CX_counts'],window_count=len(p['windows']),
                factor_kinds=dict(Counter(f['kind'] for f in p['factors'])),
                internal_support_histogram=dict(Counter(len(f['support']) for f in blocks)),
                max_internal_gate_count=max(max(len(f['left']),len(f['right'])) for f in blocks),
                unique_numeric_blocks=len(bodies),unsupported=p['unsupported'],
                rewrite_counts=[x['counts'] for x in p['rewrites']],
                internal_blocks=[dict(support=f['support'],source_item_spans=f['source_item_spans'],
                    gate_counts=[len(f['left']),len(f['right'])]) for f in blocks]))
    commute=[len(local.skeleton(e,commute_CX=True)['events']) for e in saved['expansions']]
    return dict(kind='STRUCTURAL_ONLY_SAVED_RECORD_AUDIT',inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        source_primitive_plan_id=saved['plan_id'],matched_CX=2892,unmatched_compiled=2,unmatched_raw=20,
        alignment_note='one deterministic token alignment,not unique pass history',edits=edits,
        old_registered_commutation_CX_counts=commute,variants=variants,
        selected_policy=dict(max_support=4,guard=1,exact_clifford=True,terminal_boundary='pool all terminal wires if reached'),
        numerical_upper=None,real_H4_Arb_evaluations=0,new_build=0,new_compile=0,
        historical_transform_pass_identity='UNRECOVERED_NO_PASS_TRACE')


if __name__=='__main__':print(json.dumps(audit(),ensure_ascii=False,indent=2))
