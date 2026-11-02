"""G9 v2 future runner; pending separate authorization, no execution in preparation."""
import argparse,hashlib,json,time,sys
from pathlib import Path
from fractions import Fraction as F
from trottertracks.algorithm_codesign.g7_launch import protected_check,consume_marker,sha
from trottertracks.algorithm_codesign.g9_v2_launch import verify_launch
from trottertracks.algorithm_codesign.rte_reallocation.launch import BudgetGuard
from trottertracks.algorithm_codesign.synthesis_placement.wrapper_launch import verify_runtime
from trottertracks.algorithm_codesign.rte_reallocation.numeric import configure,synthesize,validate_saved
from trottertracks.algorithm_codesign.rte_reallocation.native import Angle
from trottertracks.algorithm_codesign.g9_comparison import generators,plan,row,event_lists,P,X
from trottertracks.algorithm_codesign.g9_native import cts_events
from trottertracks.algorithm_codesign.g9_p5 import audit_p5

ROOT=Path(__file__).resolve().parents[3]
PREP=ROOT/'artifacts/track_b_g9_v2_api_boundary_preparation/2026-10-10'


def serial(v):
    if isinstance(v,F):return str(v)
    if isinstance(v,dict):return {str(k):serial(w) for k,w in v.items()}
    if isinstance(v,(list,tuple)):return [serial(w) for w in v]
    return v


def acquire_fixed_primitive(angle,contract):
    # Reuse validation uses Fraction; the fixed synthesis API accepts a string.
    # Preserve the exact registered value and key identity; do not use float.
    epsilon=contract['primitive_error']
    if not isinstance(epsilon,str):
        raise TypeError('fixed primitive_error must remain the contract string')
    return synthesize(angle,epsilon,contract['synthesizer_options'],
                      max_characters=contract['caps']['sequence_characters'])


def execute(source):
    c,auth,head,before=verify_launch(ROOT,PREP/'contract_v2.json',source)
    out=ROOT/c['result_directory'];runtime=verify_runtime(ROOT,c)
    if sha(Path(sys.executable).resolve())!=c['runtime_executable_sha256']:
        raise PermissionError('fixed runtime executable mismatch')
    inventory=json.loads((ROOT/c['synthesis_inventory']).read_text())
    marker=consume_marker(out,{'kind':'G9_V2_SEPARATELY_AUTHORIZED_MATCHED_NATIVE_ONE_SHOT','source_commit':source,
        'contract_sha256':sha(PREP/'contract_v2.json'),'execution_HEAD':head,'authorization':auth,'authorization_sha256':sha(ROOT/c['authorization_path']),
        'runs':1,'retries':0,'mandatory_STOP':True})
    result={'status':'G9_TECHNICAL_INCONCLUSIVE','technical_reason':None,'source_commit':source,'execution_HEAD':head,'authorization_commit':head,
        'authorization_sha256':sha(ROOT/c['authorization_path']),
        'contract_sha256':sha(PREP/'contract_v2.json'),'marker_sha256':sha(marker),'runtime':runtime,
        'rows':[],'synthesis_cache':{},'new_synthesis_calls':0,'reused_G7_sequence_keys':0,'runs':1,'retries':0,
        'new_inputs_grid_LP_DF_molecule_NPZ_GPU_quantum_measurement_trajectory':0,
        'native_circuit_description_and_small_matrix_reference_authorized':True,'actual_quantum_shots':0,
        'protected_before':before,'mandatory_STOP':True,'next_science_authorized':False,'method_or_novelty_adopted':False,
        'prefix_rows_usable_for_final_research_decision':False}
    guard=BudgetGuard(c['caps']);wall=time.monotonic()
    try:
        with guard:
            configure(c['interval_dps']);old=json.loads((ROOT/c['reuse_G7_result']).read_text())['synthesis_cache']
            eps=F(c['primitive_error'])
            for key in inventory['keys']:
                guard.check();ratio=key['ratio'];angle=Angle('atan',F(ratio))
                if key['acquisition']=='reuse_G7':
                    saved=old[ratio];validate_saved(saved,angle,eps)
                    if saved['sequence_sha256']!=key['sequence_sha256']:raise PermissionError('reuse identity mismatch')
                    result['synthesis_cache'][ratio]=saved;result['reused_G7_sequence_keys']+=1
                else:
                    if result['new_synthesis_calls']>=c['caps']['synthesis_keys']:raise RuntimeError('acquisition call cap')
                    guard.begin_key();result['new_synthesis_calls']+=1
                    acquired=acquire_fixed_primitive(angle,c)
                    acquired['resource']=guard.end_key()
                    if not acquired['error_pass']:raise ArithmeticError('strict new-key guard failed')
                    result['synthesis_cache'][ratio]=acquired
                cache_bytes=len(json.dumps(result['synthesis_cache']).encode())
                if len(result['synthesis_cache'])>32 or cache_bytes>1024*1024:
                    raise RuntimeError('fixed shared cache capacity')
            result['synthesis_cache_bytes']=cache_bytes
            result['formal_P5_audit']=audit_p5(P,X)
            gs=generators();plans=[plan(g) for g in gs]  # No event table input to these plans.
            ce,cert=cts_events(P,X);cp=plan(cts=ce)
            if any(b['N_per_axis']>c['caps']['shot_cap_per_axis'] for b in plans+[cp]):
                raise RuntimeError('shot cap')
            result['CTS_operator_certificate']=cert
            phase=time.monotonic();events=event_lists(gs)
            actualkeys={str(e['ratio']) for es in events+[ce] for e in es if e['ratio']}
            if actualkeys!=set(result['synthesis_cache']):raise PermissionError('result-prior inventory mismatch')
            for g,es,b in zip(gs,events,plans):
                guard.check();result['rows'].append(row(g.arm,es,b,result['synthesis_cache']))
            result['rows'].append(row('matched_CTS',ce,cp,result['synthesis_cache']))
            for g,es,b in zip(gs,events,plans):
                guard.check();result['rows'].append(row(g.arm,es,b,result['synthesis_cache'],True))
            result['reference_and_native_accounting_wall_seconds']=time.monotonic()-phase
            result['resource']=guard.usage();guard.check()
            result['status']='G9_MATCHED_NATIVE_RESOURCE_MAP_COMPLETE'
            result['prefix_rows_usable_for_final_research_decision']=True
    except Exception as exc:
        result['technical_reason']=type(exc).__name__+': '+str(exc)
        result['resource']=guard.usage()
    result['wall_seconds_including_launch']=time.monotonic()-wall
    result['protected_after']=protected_check(ROOT,c)
    if result['protected_after']['violations']:
        result.update(status='G9_TECHNICAL_INCONCLUSIVE',technical_reason='protected history mismatch',prefix_rows_usable_for_final_research_decision=False)
    raw=json.dumps(serial(result),indent=2,ensure_ascii=False)+'\n'
    if len(raw.encode())>c['caps']['output_bytes']:
        result.update(status='G9_TECHNICAL_INCONCLUSIVE',technical_reason='output cap; no prefix outcome',prefix_rows_usable_for_final_research_decision=False,rows=[])
        raw=json.dumps(serial(result),indent=2,ensure_ascii=False)+'\n'
    with (out/'result_v1.json').open('x') as f:f.write(raw)
    with (out/'STOP.json').open('x') as f:json.dump({'mandatory_STOP':True,'status':result['status'],'next_science_authorized':False,'research_owner':'GPT / user'},f,indent=2);f.write('\n')
    print(json.dumps({'status':result['status'],'new_synthesis_calls':result['new_synthesis_calls'],'reused_keys':result['reused_G7_sequence_keys'],'rows':len(result['rows']),'mandatory_STOP':True}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-commit',required=True)
    execute(parser.parse_args().source_commit)
