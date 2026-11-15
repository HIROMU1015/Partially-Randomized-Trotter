"""Join one saved LD7 V0 outcome to the two saved LD1/LD2 V0 outcomes.

No signal/risk recomputation, prices, fits, allocation generation or ranking.
"""
from copy import deepcopy
import csv
import json
from pathlib import Path
import subprocess
import sys
from summarize_pr_df_ld_split_q3_resume import dec,display

ROOT=Path(__file__).resolve().parents[1]
PUBLIC='docs/research/d_pr_df_ld7_v0_results_20261116'
CALLER='scripts/summarize_pr_df_ld7_v0.py'
if hasattr(sys,'set_int_max_str_digits'):sys.set_int_max_str_digits(0)
def read(commit,path):return json.loads(subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT))
def write(name,value):
    with(ROOT/PUBLIC/name).open('x')as f:json.dump(value,f,indent=2,allow_nan=False,ensure_ascii=False);f.write('\n')

def summarize(raw,source):
    raw=subprocess.check_output(['git','rev-parse',raw+'^{commit}'],cwd=ROOT,text=True).strip()
    source=subprocess.check_output(['git','rev-parse',source+'^{commit}'],cwd=ROOT,text=True).strip()
    if (ROOT/CALLER).read_bytes()!=subprocess.check_output(['git','show',source+':'+CALLER],cwd=ROOT):raise ValueError('Unfixed join SOURCE')
    completion=read(raw,PUBLIC+'/q3/COMPLETION.json')
    if completion['status']!='Q3_SAVED':raise ValueError('Partial outcome requires missing-data report, no retry')
    folder=PUBLIC+'/q3/LD7_V0';r=read(raw,folder+'/RESULT.json');api=read(raw,folder+'/API_RETURN.json')
    coarse=read(raw,folder+'/PARTIAL_COARSE.json');dp=read(raw,folder+'/PARTIAL_TOTAL_DP.json')
    package=read(raw,PUBLIC+'/Q3_INPUTS.json');case=package['inputs']['cases'][0]
    if r['record']!=api[0]or r['scientific_evaluation_calls']!=1 or r['allocation']!=case['allocation']:raise ValueError('API/call/allocation mismatch')
    if any(p['exitcode']!=0 for p in completion['process_exitcodes']):raise ValueError('Worker process failed')
    if coarse['decoder_certificate']['status']!='VERIFIED_FINITE_BINARY64_SUPPORT':raise ValueError('Closure missing')
    if dp['transition_pairs']>141559036 or any(v['retained_states']>8192 for v in dp['rounds'][:-1]):raise ValueError('Q3 finite-scope violation')
    rows=deepcopy(read(raw,PUBLIC+'/REUSED_V0.json')['rows']);risk=r['record']['diagnostic']['deterministic_risk']
    row=dict(L_D=7,rule='V0',shots=case['allocation']['shots'],status=r['status'],MSE_lower=risk['MSE_lower'],MSE_upper=risk['MSE_upper'],
        MSE_lower_display=display(dec(risk['MSE_lower'])),MSE_upper_display=display(dec(risk['MSE_upper'])),
        Q3_source=dict(commit=raw,path=folder+'/RESULT.json'),Q3_disposition='NEW_ONE_CALL_COMPLETED')
    rows.append(row)
    if any(v['shots']!=case['allocation']['shots']for v in rows):raise ValueError('Different V0 vectors')
    next_step=('LD7_OWN_PRICE_CALIBRATION_AND_CANDIDATE_OWN_ALLOCATION_COMPARISON'if r['status']=='MODEL_BOUND_MEETS_ADOPTED_THRESHOLD'
        else'REASSESS_THIS_V0_ONLY_NO_AUTOMATIC_ALLOCATION_OR_RETRY')
    write('SUMMARY.json',dict(schema='PR_DF_THREE_SPLITS_SAME_V0_Q3_V1',status='THREE_V0_MODEL_COMPARISON_COMPLETE',source=source,raw_publication=raw,rows=rows,
        threshold=package['inputs']['adopted_evaluation'],checks=dict(API_equals_RESULT=True,scientific_Q3_calls=1,closure=coarse['decoder_certificate'],transition_pairs=dp['transition_pairs'],
        mapper_13_rows_R1_single_piece=True,signals_13_valid=True,worker_exitcodes=completion['process_exitcodes']),
        counts=dict(new_configurations=1,new_signal_rounds=13,reused_FullH_reference_pairs=13,new_reference_actions=0,new_Q3=1,reused_V0_Q3=2,sampling=0,circuit_build=0,compile=0,fit=0,price_prediction=0,new_allocation=0,scientific_retry=0),
        next_decision=next_step,qualifications=dict(target_law='ideal_finite_RTE_law',measurement='IID_SHOTS_AXES_ROUNDS',ground_state_certified=False,physical_total_error_guarantee=False,
        E_C=None,CI=None,G=None,winner=None,global_LD_optimality=False),mandatory_stop=True))
    with(ROOT/PUBLIC/'V0_COMPARISON.csv').open('x')as f:
        w=csv.DictWriter(f,fieldnames=['L_D','rule','MSE_lower_display','MSE_upper_display','status','Q3_disposition']);w.writeheader();w.writerows({k:v[k]for k in w.fieldnames}for v in rows)
    for v in rows:print(v['L_D'],v['MSE_lower_display'],v['MSE_upper_display'],v['status'])

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--raw',required=True);p.add_argument('--source',required=True);a=p.parse_args();summarize(a.raw,a.source)
