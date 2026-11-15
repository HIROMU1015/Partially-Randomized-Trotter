"""At most four unique adopted PR-DF split allocations; immutable existing Q3 route."""
import argparse
from copy import deepcopy
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
VIEW=Path('/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/d2-pr-df-ld-split-q3-source-20261115')
Q3_SOURCE='27e70efe533862f38f4080aabb9cdd3ad1749fe9'
PUBLIC='docs/research/d_pr_df_ld_split_results_20261115'
CALLER='scripts/run_pr_df_ld_split_q3.py'
if hasattr(sys,'set_int_max_str_digits'):sys.set_int_max_str_digits(0)
sys.path[:0]=[str(VIEW/'src'),str(VIEW/'scripts')]
spec=importlib.util.spec_from_file_location('_split_fixed_q3_route',VIEW/'scripts/run_pr_df_fullh_total_q3_fixed.py')
U=importlib.util.module_from_spec(spec);sys.modules[spec.name]=U;spec.loader.exec_module(U)

def blob(commit,path):return subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT)
def write(path,value):
    with Path(path).open('x')as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def enc(v):return dict(numerator=v.numerator,denominator=v.denominator)
def count_bounds(shots):
    cells=[(r['N_X']+1)*(r['N_Y']+1)for r in shots]
    pairs=cells[0];states=min(cells[0],8192)
    for c in cells[1:]:pairs+=states*c;states=min(8192,states*c)
    return dict(maximum_round_outcomes=max(cells),verification_pairs_upper=sum(a*b for a,b in zip(cells,cells[1:])),transition_pairs_upper=pairs)

def bounds(case):
    r=count_bounds(case['allocation']['shots'])
    # Reuse saved full m0 decoder-state count ONLY when this new vector's m0
    # is exactly the saved m0 count lattice. Other rounds use their own counts.
    shots=case['allocation']['shots']
    if shots[0]['N_X']==shots[0]['N_Y']==64:
        old=U.finite_bounds_from_saved_support([64,60,56,52,47,43,38,34,30,25,21,17,12])
        cells=[(s['N_X']+1)*(s['N_Y']+1)for s in shots]
        r.update(uncompressed_transition_pairs_upper=r['transition_pairs_upper'],
            transition_pairs_upper=cells[0]+old['first_round_full_support_states']*cells[1]+8192*sum(cells[2:]),
            support_source=old['support_source'],rule='SAME_M0_COUNT_LATTICE_AND_DECODER; subsequent new count supports and frontier bound')
    return r

def prepare(source,allocation_commit):
    if (ROOT/CALLER).read_bytes()!=blob(source,CALLER):raise ValueError('Unfixed Q3 caller')
    U.check_source()
    # Constructor SOURCE/AST gate is metadata-only; exercise it before any
    # process is submitted, rather than discovering a runtime mismatch there.
    U.R.Observed(ROOT/PUBLIC,'preflight',workers=1,outcome_batch=8)
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=VIEW,text=True).strip()!=Q3_SOURCE:raise ValueError('Q3 SOURCE view changed')
    inputs=json.loads(blob(allocation_commit,PUBLIC+'/ALLOCATION_INPUTS.json'));cases=inputs['cases']
    if len(cases)>4 or len({(c['L_D'],tuple((r['N_X'],r['N_Y'])for r in c['allocation']['shots']))for c in cases})!=len(cases):raise ValueError('Nonunique/too many Q3')
    held=[];accepted=[];checks=[]
    for c in cases:
        b=bounds(c);opts=inputs['options'];reason=None
        if b['maximum_round_outcomes']>opts['maximum_round_outcomes']or b['verification_pairs_upper']>opts['maximum_verification_pairs']or b['transition_pairs_upper']>opts['total_state_refinement']['maximum_transition_pairs']:reason='COUNT_SCOPE_EXCEEDED'
        if any(not 1<=r[a]<=cap for r,cap in zip(c['allocation']['shots'],[64,60,56,52,47,43,38,34,30,25,21,17,12])for a in ('N_X','N_Y')):raise ValueError('Vector scope')
        # Metadata/canonical/signal intake only. Fresh closure is checked in the
        # original scientific API and persisted per new vector before total DP.
        binding,view=U.metadata(c)
        checks.append(dict(label=c['label'],bounds=b,binding=binding['id'],held_reason=reason))
        if reason:held.append(dict(label=c['label'],reason=reason,scientific_calls=0))
        else:accepted.append(c)
    inputs.update(cases=accepted,held=held,source=source,allocation_commit=allocation_commit,concurrent_configurations=len(accepted),
        reused_route=dict(commit=Q3_SOURCE,path='scripts/run_pr_df_fullh_total_q3_fixed.py'),preflight=checks,
        scientific_scope='AT_MOST_FOUR_PR_DF_CANDIDATE_OWN_ALLOCATIONS_NO_SIGNAL_MC_OR_PRICE_CALLS',
        duplicate_delta='No equivalent candidate/vector in saved V0 or finite menu; rechecked remote related branch publication metadata before run. See DELTA_CHECK_Q3.json.')
    write(ROOT/PUBLIC/'Q3_CALL_INPUTS.json',inputs)
    print('PREFLIGHT',[(r['label'],r['held_reason'],r['bounds']['transition_pairs_upper'])for r in checks],flush=True)

def execute():
    inputs=json.loads((ROOT/PUBLIC/'Q3_CALL_INPUTS.json').read_text());source=inputs['source']
    if blob(source,CALLER)!=(ROOT/CALLER).read_bytes():raise ValueError('Q3 SOURCE mutation')
    U.check_source()
    out=ROOT/PUBLIC/'q3_calls';out.mkdir()
    for c in inputs['cases']:(out/c['label']).mkdir()
    input_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    source_ref=dict(commit=source,path=CALLER,inputs=dict(commit=input_commit,path=PUBLIC+'/Q3_CALL_INPUTS.json'),reused_route=inputs['reused_route'])
    U.R.write_new(out/'EXECUTION.json',dict(source=source_ref,started_UTC=U.R.utc(),maximum_scientific_calls=4,held=inputs['held']))
    result=U.supervise(inputs['cases'],inputs,source_ref,out)
    status='FINITE_Q3_SAVED'if not result['errors']and not result['monitor_errors']and all(r['status']!='UNKNOWN'for r in result['receipts'])else'UNKNOWN_STOP_NO_RETRY'
    U.R.write_new(out/'COMPLETION.json',dict(status=status,**result,held=inputs['held'],finished_UTC=U.R.utc(),mandatory_stop=True))
    return 0 if status=='FINITE_Q3_SAVED'else 2

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('prepare','execute'));p.add_argument('--source');p.add_argument('--allocation-commit');a=p.parse_args()
    if a.stage=='prepare':prepare(a.source,a.allocation_commit)
    else:raise SystemExit(execute())
