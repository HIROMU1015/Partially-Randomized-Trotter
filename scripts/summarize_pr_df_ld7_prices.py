"""Saved nine-entry model-cost/Q3 join; preserve older LD1/LD2 estimates."""
from copy import deepcopy
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from summarize_pr_df_ld_split_q3_resume import dec,enc,display
if hasattr(sys,'set_int_max_str_digits'):sys.set_int_max_str_digits(0)
ROOT=Path(__file__).resolve().parents[1]
PUBLIC='docs/research/d_pr_df_ld7_price_results_20261116'
CALLER='scripts/summarize_pr_df_ld7_prices.py'
OLD='8ec7c882806809460d38a638a891bafcf18fafd1'
def read(c,p):return json.loads(subprocess.check_output(['git','show',c+':'+p],cwd=ROOT))
def write(n,v):
    with(ROOT/PUBLIC/n).open('x')as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')

def summarize(source,raw):
    if (ROOT/CALLER).read_bytes()!=subprocess.check_output(['git','show',source+':'+CALLER],cwd=ROOT):raise ValueError('Unfixed summary')
    pack=read(raw,PUBLIC+'/Q3_INPUTS.json');inputs=pack['inputs'];completion=read(raw,PUBLIC+'/q3/COMPLETION.json')
    if completion['status']!='FINITE_Q3_SAVED':raise ValueError('Partial Q3 requires explicit missing-result report')
    old=read(OLD,'docs/research/d_pr_df_ld_split_q3_resume_20261116/SUMMARY.json');menu=deepcopy(old['menu']);model=inputs['model_commit'];prices=read(model,PUBLIC+'/AXIS_PRICES.json')['rows'];own=[];bylabel={}
    v0=read('0defa286c53a1b9acf513be453907bcad83102b3','docs/research/d_pr_df_ld7_v0_results_20261116/SUMMARY.json')['rows'][2]
    checks=[]
    for item in inputs['menu']:
        r=deepcopy(item);label=r['label'];r['prices_source']=dict(commit=model,path=PUBLIC+'/AXIS_PRICES.json')
        if r['rule']=='V0':
            for k in ('status','MSE_lower','MSE_upper','Q3_source'):r[k]=v0[k]
            r['Q3_disposition']='REUSED_SAVED_V0'
        elif r.get('reuse_label'):
            prior=bylabel[r['reuse_label']]
            for k in ('status','MSE_lower','MSE_upper','Q3_source'):r[k]=prior[k]
            r['Q3_disposition']='REUSED_IDENTICAL_VECTOR'
        elif any(h['label']==label for h in inputs['held']):
            r.update(status='HELD_COUNT_SCOPE',MSE_lower=None,MSE_upper=None,Q3_source=None,Q3_disposition='ZERO_CALL_HELD')
        else:
            folder=PUBLIC+'/q3/'+label;res=read(raw,folder+'/RESULT.json');api=read(raw,folder+'/API_RETURN.json');coarse=read(raw,folder+'/PARTIAL_COARSE.json');dp=read(raw,folder+'/PARTIAL_TOTAL_DP.json')
            if res['record']!=api[0]or res['scientific_evaluation_calls']!=1 or res['allocation']['shots']!=r['shots']:raise ValueError('Q3 API/vector/call mismatch')
            if coarse['decoder_certificate']['status']!='VERIFIED_FINITE_BINARY64_SUPPORT'or dp['transition_pairs']>141559036:raise ValueError('Closure/scope')
            risk=res['record']['diagnostic']['deterministic_risk'];r.update(status=res['status'],MSE_lower=risk['MSE_lower'],MSE_upper=risk['MSE_upper'],Q3_source=dict(commit=raw,path=folder+'/RESULT.json'),Q3_disposition='NEW_ONE_CALL_COMPLETED')
            checks.append(dict(label=label,calls=1,API_equals_RESULT=True,closure=coarse['decoder_certificate']['status'],transition_pairs=dp['transition_pairs']))
        weighted=[]
        for i,p in enumerate(prices):
            n=r['shots'][p['m']]['N_'+p['axis']];weighted.append(dict(L_D=7,m=p['m'],axis=p['axis'],physical_id=p['physical_id'],configuration_id=p['configuration_id'],shots=n,C_ref=p['C_ref'],weighted=enc(n*dec(p['C_ref'])),price_source=dict(commit=model,path=PUBLIC+'/AXIS_PRICES.json',selector=['rows',i]),application=p['application']))
        total=sum(dec(t['weighted'])for t in weighted);early=sum(dec(t['weighted'])for t in weighted if t['m']<=6);late=total-early
        if total!=dec(r['cost']):raise ValueError('Frozen total mismatch')
        r.update(cost_display=display(total),m0_m6=enc(early),m7_m12=enc(late),m7_m12_fraction=enc(late/total),m7_m12_percent_display=display(100*late/total),weighted_terms=weighted,
            MSE_lower_display=None if r['MSE_lower']is None else display(dec(r['MSE_lower'])),MSE_upper_display=None if r['MSE_upper']is None else display(dec(r['MSE_upper'])))
        own.append(r);bylabel[label]=r
    menu+=own;selected=deepcopy(old['minimum_admitted_per_LD']);admitted=[r for r in own if r['status']=='MODEL_BOUND_MEETS_ADOPTED_THRESHOLD'];minimum=min(dec(r['cost'])for r in admitted);ties=[r for r in admitted if dec(r['cost'])==minimum]
    selected.append(dict(L_D=7,cost=enc(minimum),cost_display=display(minimum),rules=[r['rule']for r in ties],rows=ties,scope='MINIMUM_AMONG_V0_A_SHAPE_B_PRICE_ONLY'))
    next_LD=9 if minimum<dec(selected[1]['cost'])else 3
    write('SUMMARY.json',dict(schema='PR_DF_LD1_LD2_LD7_NINE_MODEL_MENU_V1',source=source,raw_publication=raw,status='FINITE_NINE_ENTRY_COMPARISON_COMPLETE',
        model_commit=model,allocation_commit=pack['allocation_commit'],LD1_LD2_saved_source=dict(commit=OLD,path='docs/research/d_pr_df_ld_split_q3_resume_20261116/SUMMARY.json'),
        menu=menu,minimum_admitted_per_LD=selected,checks=checks,Q3_completion=completion,next_candidate_proposal=next_LD,
        counts=dict(new_paths=10,compile=20,axis_fits=2,new_references=26,new_allocations=2,new_Q3=len(checks),reused_V0_Q3=3,reused_LD1_LD2_menu_entries=6,signal=0,configuration=0,scientific_retry=0),
        qualifications=dict(prices='MODEL_REFERENCES_NOT_FORMAL_E_C',LD1='STRUCTURE_PLUS_RESIDUAL',LD2='THREE_POINT_FULL_C',LD7='TWO_POINT_FULL_C_ALPHA_PLUS_BETA_Q_R1',long_round='m7..12 EXTRAPOLATION',E_C=None,CI=None,G=None,winner=None,global_LD_optimality=False),mandatory_stop=True))
    write('C_HANDOFF.json',dict(schema='PR_DF_LD7_MODEL_COST_Q3_JOIN_FOR_C_V1',configuration=read('0defa286c53a1b9acf513be453907bcad83102b3','docs/research/d_pr_df_ld7_v0_results_20261116/Q3_INPUTS.json')['inputs']['cases'][0]['configuration'],
        configuration_id=prices[0]['configuration_id'],original_law='ideal_finite_RTE_law',compiled_price_quantity='MEASURED_FULL_CONTROLLED_HADAMARD_WRAPPER_COMPILED_RZ_PER_SHOT',
        models=dict(commit=model,path=PUBLIC+'/FROZEN_MODELS.json'),axis_prices=dict(commit=model,path=PUBLIC+'/AXIS_PRICES.json'),
        holdout=dict(commit=inputs['holdout_commit'],path=PUBLIC+'/HOLDOUT_COMPARISON.json'),allocations=dict(commit=pack['allocation_commit'],path=PUBLIC+'/ALLOCATION_INPUTS.json'),
        rows=own,price_acceptance='SOURCE_RECEIPT_CHECKED_EXPLORATORY_MODEL_NOT_LAW_EXPECTATION',C_independent_admission='NOT_PERFORMED_BY_D2',model_holdout_verdict='NOT_ASSESSED_NO_ADOPTED_TOLERANCE',mandatory_stop=True))
    with(ROOT/PUBLIC/'MENU.csv').open('x')as f:
        keys=['L_D','rule','cost_display','MSE_lower_display','MSE_upper_display','status','m7_m12_percent_display'];w=csv.DictWriter(f,fieldnames=keys+['N_X','N_Y']);w.writeheader()
        for r in menu:w.writerow({k:r[k]for k in keys}|dict(N_X=' '.join(str(s['N_X'])for s in r['shots']),N_Y=' '.join(str(s['N_Y'])for s in r['shots'])))
    for r in selected:print('SELECTED',r['L_D'],r['rules'],r['cost_display'])
    print('NEXT_PROPOSAL',next_LD)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--raw',required=True);a=p.parse_args();summarize(a.source,a.raw)
