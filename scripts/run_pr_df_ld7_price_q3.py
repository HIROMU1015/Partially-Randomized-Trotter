"""At most two unique frozen LD7 own-price allocations; original Q3 worker."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import run_pr_df_ld_split_q3_resume as W
ROOT=W.ROOT
PUBLIC='docs/research/d_pr_df_ld7_price_results_20261116'
CALLER='scripts/run_pr_df_ld7_price_q3.py'
if hasattr(sys,'set_int_max_str_digits'):sys.set_int_max_str_digits(0)
def read(c,p):return json.loads(W.blob(c,p))
def fixed(source):
    if W.blob(source,CALLER)!=(ROOT/CALLER).read_bytes():raise ValueError('Q3 caller changed')

def prepare(source,allocation):
    fixed(source);runtime=W.runtime_check();inputs=read(allocation,PUBLIC+'/ALLOCATION_INPUTS.json');cases=inputs['cases']
    if len(cases)>2 or len({tuple((r['N_X'],r['N_Y'])for r in c['allocation']['shots'])for c in cases})!=len(cases):raise ValueError('Duplicate/too many Q3')
    checks=[];held=[];accepted=[]
    for c in cases:
        W.U.metadata(c);b=W.Q.bounds(c);o=inputs['options'];reason=None
        if b['maximum_round_outcomes']>o['maximum_round_outcomes']or b['verification_pairs_upper']>o['maximum_verification_pairs']or b['transition_pairs_upper']>o['total_state_refinement']['maximum_transition_pairs']:reason='COUNT_SCOPE_EXCEEDED'
        checks.append(dict(label=c['label'],bounds=b,held_reason=reason))
        if reason:held.append(dict(label=c['label'],reason=reason,scientific_calls=0))
        else:accepted.append(c)
    inputs.update(cases=accepted,held=held,preflight=checks)
    W.write(ROOT/PUBLIC/'Q3_INPUTS.json',dict(source=source,allocation_commit=allocation,inputs=inputs,runtime=runtime,maximum_new_Q3=2,output=PUBLIC+'/q3'))
    print([(x['label'],x['held_reason'],x['bounds']['transition_pairs_upper'])for x in checks],flush=True)

def execute():
    p=json.loads((ROOT/PUBLIC/'Q3_INPUTS.json').read_text());fixed(p['source'])
    if p['runtime']!=W.runtime_check():raise ValueError('Runtime mismatch')
    delta=json.loads((ROOT/PUBLIC/'Q3_DELTA_CHECK.json').read_text())
    if delta['equivalent_results']or delta['matching_jobs']or delta['unresolved']:raise ValueError('Unreconciled duplicate work')
    inputs=p['inputs'];out=ROOT/PUBLIC/'q3';out.mkdir()
    for c in inputs['cases']:W.U.metadata(c);(out/c['label']).mkdir()
    src=dict(commit=p['source'],path=CALLER,inputs=dict(commit=W.head(),path=PUBLIC+'/Q3_INPUTS.json'),runtime=p['runtime'])
    W.write(out/'EXECUTION.json',dict(source=src,maximum_scientific_calls=2,held=inputs['held']))
    result=W.supervise(inputs['cases'],inputs,src,out)
    ok=(not result['errors']and not result['monitor_errors']and len(result['receipts'])==len(inputs['cases'])and all(r['status']!='UNKNOWN'and r['scientific_evaluation_calls']==1 for r in result['receipts'])and all(p['exitcode']==0 for p in result['process_exitcodes']))
    W.write(out/'COMPLETION.json',dict(status='FINITE_Q3_SAVED'if ok else'STOP_UNKNOWN_NO_RETRY',**result,held=inputs['held'],mandatory_stop=True))
    return 0 if ok else 2

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','execute'));p.add_argument('--source');p.add_argument('--allocation');a=p.parse_args()
    if a.mode=='prepare':prepare(a.source,a.allocation)
    else:raise SystemExit(execute())
