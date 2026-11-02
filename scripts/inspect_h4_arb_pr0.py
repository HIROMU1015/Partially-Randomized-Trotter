#!/usr/bin/env python3
"""PR0 saved evidence -> exact schedules. Static inspection, never evolution."""
import hashlib
import json
from pathlib import Path
import struct
import sys
import types

ROOT=Path(__file__).resolve().parents[1]
pkg=types.ModuleType('pr0_arb_private');pkg.__path__=[str(ROOT/'src/trotterlib')];sys.modules['pr0_arb_private']=pkg
from pr0_arb_private import h4_arb_pr0_preparation as P
from pr0_arb_private import h4_r0_empirical as R
C,B=P.C,P.B
BASE=ROOT/'artifacts/non_df_h4_central_arb_pr0_preparation/2026-10-25'


def inspect():
    inputs=C.read(BASE/'inputs.json')['files']
    C.hashes({str(ROOT/k):v for k,v in inputs.items()})
    t=next(t for t in C.read(ROOT/'artifacts/non_df_h4_central_rho_preparation/2026-10-24/targets.json')['targets'] if t['key']=='PR0_cosine')
    model=C.read(ROOT/t['state_model']);join=C.read(ROOT/t['price_join'])
    C.require(C.digest({k:v for k,v in model.items() if k!='binding_hash'})==model['binding_hash'],'C_MODEL_BINDING')
    array_hash=P.extract('src/trotterlib/h4_ball_evidence.py','array_hash',dict(struct=struct,hashlib=hashlib,json=json))
    state_id=C.read(ROOT/'artifacts/non_df_h4_central_r0_empirical/2026-10-24/run01/state_identity.json')
    C.require(array_hash(model['state'],vector=True)==model['state_hash']==t['state_hash']==state_id['state_hash'],'EXACT_RAW_RAY_IDENTITY')
    plan=P.build_plan(model,join)
    C.require(all(t[k]==join[k] for k in ('trajectory_id','trajectory_record_hash','event_record_hash')),'FIXED_TARGET_JOIN')
    a=R.api();loaded={}
    for role in ('source_body','controlled_body','input_qpy','compiled_qpy'):
        path=ROOT/t[role]
        with path.open('rb') as stream:values=a.qpy.load(stream)
        C.require(len(values)==1,'SINGLE_SAVED_PROGRAM');loaded[role]=values[0]
    expected,_=a.recipe(model,'PR',join)
    C.require([a.token(x,loaded['source_body']) for x in loaded['source_body'].data]==expected,'SOURCE_BODY_C_RECIPE')
    folder=(ROOT/t['input_qpy']).parent;inp=C.read(folder/'input.json');metrics=C.read(folder/'metrics.json')
    R.P.check_saved_tags(inp,metrics,R.P.D2)
    binding=R.validate_controlled(loaded['source_body'],loaded['controlled_body'],loaded['input_qpy'],inp['sidecar'])
    for role,fp in [('input_qpy','input_fingerprint'),('compiled_qpy','output_fingerprint')]:
        R.validate_program(loaded[role],compiled=role=='compiled_qpy',n=8)
        C.require(a.fingerprint(loaded[role])==t[fp],'QPY_FINGERPRINT')
    program=B.descriptor_from_circuit(loaded['compiled_qpy'],axis='cosine')
    program['origin']=dict(qpy_sha256=C.sha(ROOT/t['compiled_qpy']),fingerprint=t['output_fingerprint'],decode_only=True,state_actions=0,compile=0)
    C.require(dict(loaded['compiled_qpy'].count_ops())==t['saved_post_histogram']==metrics['posttranspile_histogram'],'SAVED_NATIVE_HISTOGRAM')
    C.save(BASE/'PR0_selected_C_plan_NO_EXECUTION.json',plan)
    C.save(BASE/'PR0_compiled_program_NO_EXECUTION.json',program)
    C.save(BASE/'state_identity_QUOTED.json',state_id)
    C.save(BASE/'structural_inspection.json',dict(status='PASS_PR0_STATIC_SOURCE_EVENT_QPY_BINDING',QPY_decodes=4,
        state_actions=0,Arb_H4_target_actions=0,new_compile=0,trajectory_id=t['trajectory_id'],trajectory_record_hash=join['trajectory_record_hash'],
        event_record_hash=join['event_record_hash'],selected_events=len(plan['events']),event_orders=[e['taylor_order'] for e in plan['events']],
        retained_blocks=len(plan['retained_blocks']),retained_rotations=2*plan['q']*len(plan['retained_blocks']),
        native_instructions=len(program['instructions']),native_histogram=t['saved_post_histogram'],source_control_binding=binding,
        state_hash=state_id['state_hash'],C_target_plan_sha256=C.sha(BASE/'PR0_selected_C_plan_NO_EXECUTION.json'),
        compiled_descriptor_sha256=C.sha(BASE/'PR0_compiled_program_NO_EXECUTION.json'),
        target_interval=None,compiled_signal_interval=None,rho_interval=None,PR_law_mean_used=False))
    print(json.dumps(dict(status='PASS_STATIC_ONLY',QPY_decodes=4,state_actions=0,events=len(plan['events']))))


if __name__=='__main__':inspect()
