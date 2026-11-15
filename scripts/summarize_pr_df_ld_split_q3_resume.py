"""Join saved Q3 outcomes to frozen costs; never regenerate prices or allocations."""
from copy import deepcopy
from decimal import Decimal,localcontext
from fractions import Fraction
import csv
import json
from pathlib import Path
import subprocess
import sys
if hasattr(sys,'set_int_max_str_digits'):sys.set_int_max_str_digits(0)
ROOT=Path(__file__).resolve().parents[1]
PUBLIC='docs/research/d_pr_df_ld_split_q3_resume_20261116'
PRIOR='docs/research/d_pr_df_ld_split_results_20261115'
OLD='e0b3cad2a024070389f757160f3a20d7da02c5ff'
ALLOC='4ef70274261cf18bd9356a15d780eb5b6b61c60c'
CALLER='scripts/summarize_pr_df_ld_split_q3_resume.py'
PASS='MODEL_BOUND_MEETS_ADOPTED_THRESHOLD'

def read(commit,path):return json.loads(subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT))
def dec(v):return Fraction(v['numerator'],v['denominator'])if isinstance(v,dict)else Fraction(v)
def enc(v):v=Fraction(v);return dict(numerator=v.numerator,denominator=v.denominator)
def display(v):
    with localcontext()as c:c.prec=24;return str(Decimal(v.numerator)/Decimal(v.denominator))
def write(path,value):
    with Path(path).open('x')as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')

def summarize(source,raw):
    source=subprocess.check_output(['git','rev-parse',source],cwd=ROOT,text=True).strip()
    if (ROOT/CALLER).read_bytes()!=subprocess.check_output(['git','show',source+':'+CALLER],cwd=ROOT):raise ValueError('Unfixed summary SOURCE')
    old=read(OLD,PRIOR+'/SUMMARY.json');alloc=read(ALLOC,PRIOR+'/ALLOCATION_INPUTS.json');completion=read(raw,PUBLIC+'/run01/COMPLETION.json')
    if completion['status']!='FOUR_FIXED_Q3_SAVED':raise ValueError('Partial outcome needs explicit missing-data report')
    rows=[];checks=[]
    for i,item in enumerate(alloc['menu']):
        r=deepcopy(item);r['allocation_disposition_at_freeze']=r.pop('disposition');prior=next(x for x in old['menu']if (x['L_D'],x['rule'])==(r['L_D'],r['rule']))
        if r['cost']!=prior['cost']or r['shots']!=prior['shots']:raise ValueError('Frozen costs/vectors changed')
        r.update(cost_source=dict(commit=ALLOC,path=PRIOR+'/ALLOCATION_INPUTS.json',selector=['menu',i]),
            cost_display=display(dec(r['cost'])),m0_m6=prior['m0_m6'],m7_m12=prior['m7_m12'],
            weighted_terms_source=dict(commit=OLD,path=PRIOR+'/SUMMARY.json',selector=['weighted_terms']),
            m7_m12_fraction=enc(dec(prior['m7_m12'])/dec(r['cost'])),m7_m12_percent_display=display(100*dec(prior['m7_m12'])/dec(r['cost'])))
        if r['rule']=='V0':
            r.update(status=prior['status'],MSE_lower=prior['MSE_lower'],MSE_upper=prior['MSE_upper'],
                Q3_source=prior['Q3_source'],Q3_disposition='REUSED_SAVED_V0')
        else:
            folder=PUBLIC+'/run01/'+r['label'];res=read(raw,folder+'/RESULT.json');api=read(raw,folder+'/API_RETURN.json')
            coarse=read(raw,folder+'/PARTIAL_COARSE.json');dp=read(raw,folder+'/PARTIAL_TOTAL_DP.json')
            if res['record']!=api[0]or res['scientific_evaluation_calls']!=1 or res['allocation']['shots']!=r['shots']:raise ValueError('API/call/vector mismatch')
            risk=res['record']['diagnostic']['deterministic_risk']
            if (coarse.get('decoder_certificate')or{}).get('status')!='VERIFIED_FINITE_BINARY64_SUPPORT':raise ValueError('Closure missing')
            if dp['transition_pairs']>141559036 or any(x['retained_states']>8192 for x in dp['rounds'][:-1]):raise ValueError('Finite scope')
            r.update(status=res['status'],MSE_lower=risk['MSE_lower'],MSE_upper=risk['MSE_upper'],
                Q3_source=dict(commit=raw,path=folder+'/RESULT.json'),Q3_disposition='NEW_ONE_CALL_COMPLETED',
                MSE_upper_method=risk.get('MSE_upper_method'),closure=coarse['decoder_certificate'],transition_pairs=dp['transition_pairs'],resources=read(raw,folder+'/RESOURCE_FINAL.json'))
            checks.append(dict(label=r['label'],API_equals_RESULT=True,calls=1,closure_status=coarse['decoder_certificate']['status'],transition_pairs=dp['transition_pairs']))
        r['MSE_lower_display']=display(dec(r['MSE_lower']));r['MSE_upper_display']=display(dec(r['MSE_upper']))
        if not 0<=dec(r['MSE_lower'])<=dec(r['MSE_upper']):raise ValueError('Inconsistent saved risk')
        rows.append(r)
    selected=[]
    for L in (1,2):
        admitted=[r for r in rows if r['L_D']==L and r['status']==PASS]
        if not admitted:raise ValueError('Saved admitted V0 missing')
        minimum=min(dec(r['cost'])for r in admitted);ties=[r for r in admitted if dec(r['cost'])==minimum]
        selected.append(dict(L_D=L,cost=enc(minimum),cost_display=display(minimum),rules=[r['rule']for r in ties],
            rows=ties,scope='MINIMUM_AMONG_V0_A_SHAPE_B_PRICE_ONLY'))
    out=dict(schema='PR_DF_LD_SPLIT_SIX_ENTRY_Q3_COMPARISON_V1',status='FINITE_SIX_ENTRY_COMPARISON_COMPLETE',source=source,raw_publication=raw,
        fixed_price_allocation_source=ALLOC,fixed_Q3_input_source='4d3aa8a666e153188addfb3a4f2b8ef10733a3dd',
        menu=rows,minimum_admitted_per_LD=selected,checks=checks,completion=completion,
        counts=dict(new_Q3=4,reused_V0_Q3=2,new_sampling=0,new_compile=0,new_fit=0,new_price_prediction=0,new_allocation=0,additional_Q3=0,retries=0),
        qualifications=dict(target='SAVED_IDEAL_FINITE_RTE_RAW_IID_MODEL_MSE',prices='FROZEN_EXPLORATORY_MODELS_LD1_STRUCTURE_PLUS_RESIDUAL_LD2_FULL_WRAPPER_THREE_POINT',
            LD2_m7_m12='UNVALIDATED_EXTRAPOLATION',all_law_E_C=None,CI=None,G=None,winner=None,global_LD_optimality=False),mandatory_stop=True)
    write(ROOT/PUBLIC/'SUMMARY.json',out)
    with (ROOT/PUBLIC/'MENU.csv').open('x')as f:
        names=['L_D','rule','status','cost_display','MSE_lower_display','MSE_upper_display','m7_m12_percent_display','N_X','N_Y'];w=csv.DictWriter(f,fieldnames=names);w.writeheader()
        for r in rows:w.writerow({k:r[k]for k in names[:-2]}|dict(N_X=' '.join(str(s['N_X'])for s in r['shots']),N_Y=' '.join(str(s['N_Y'])for s in r['shots'])))
    for r in rows:print(r['L_D'],r['rule'],r['status'],r['cost_display'],r['MSE_lower_display'],r['MSE_upper_display'],r['m7_m12_percent_display'],flush=True)
    for r in selected:print('SELECTED',r['L_D'],r['rules'],r['cost_display'],flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--raw',required=True);a=p.parse_args();summarize(a.source,a.raw)
