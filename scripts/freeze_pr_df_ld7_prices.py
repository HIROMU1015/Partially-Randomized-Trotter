"""Exact two-point full-C model, independent descriptive holdout, own budgets."""
import argparse
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
import freeze_pr_df_ld_split as F
ROOT=F.ROOT
PUBLIC='docs/research/d_pr_df_ld7_price_results_20261116'
VDIR='docs/research/d_pr_df_ld7_v0_results_20261116'
BASE='0defa286c53a1b9acf513be453907bcad83102b3'
CALLER='scripts/freeze_pr_df_ld7_prices.py'
read,write,enc,dec=F.read,F.write,F.enc,F.dec
if hasattr(sys,'set_int_max_str_digits'):sys.set_int_max_str_digits(0)
def fixed(source):
    if F.blob(source,CALLER)!=(ROOT/CALLER).read_bytes():raise ValueError('Unfixed model/allocation SOURCE')

def means(raw,stage):
    inputs=read(raw,PUBLIC+'/ACQUISITION_INPUTS.json');acq=read(raw,PUBLIC+'/'+stage+'/acquisition/RESULT.json')
    role='CALIBRATION'if stage=='calibration'else'HOLDOUT';cells=[c for c in inputs['cells']if c['role']==role];n=4 if stage=='calibration'else 2
    if acq['status']!='ACQUIRED'or acq['completed_compiles']!=2*n*len(cells):raise ValueError('Incomplete price receipts; no reduced cohort')
    out=[]
    for c in cells:
        m=c['m'];rows=[r for r in acq['rows']if r['m']==m]
        if [r['index']for r in rows]!=list(range(n)):raise ValueError('Missing path')
        for axis in ('X','Y'):
            paths=[]
            for r in rows:
                v=r['record'];a=next(a for a in v['records']if a['axis']==axis)
                if r['status']!='VALID'or v['request']!=c['request']or v['cohort']!=c['cohort']or v['seeds']!=c['seeds'][r['index']]or v['index']!=r['index']:raise ValueError('Path identity')
                locator=dict(commit=raw,path=PUBLIC+f'/{stage}/acquisition/m{m}/p{r["index"]}/'+{'X':'cosine','Y':'sine'}[axis]+'.json')
                receipt=read(locator['commit'],locator['path'])
                if receipt['metrics']!=a['metrics']or receipt['fingerprint']!=a['fingerprint']or receipt['compiler']!=c['request']['physical_context']['price_profile']['compiler']or a['C']!=a['metrics']['rz_count']or a['D']!=a['C']-a['S']:raise ValueError('Axis receipt metrics/fingerprint/compiler')
                paths.append(dict(index=r['index'],axis=axis,cohort=v['cohort'],seeds=v['seeds'],S=a['S'],C=a['C'],D=a['D'],fingerprint=a['fingerprint'],receipt=locator,
                    path_receipt=dict(commit=raw,path=PUBLIC+f'/{stage}/acquisition/m{m}/p{r["index"]}/PATH_RECEIPT.json'),event_digest=v['event_digest']))
            pc=c['request']['physical_context']['physical_candidate']
            out.append(dict(m=m,axis=axis,q=pc['q'],R=pc['R_outer'],physical_id=pc['id'],configuration_id=c['request']['configuration_id'],n_paths=n,
                mean_C=enc(sum(Fraction(p['C'])for p in paths)/n),paths=paths,request=c['request'],role=role))
    return out

def fit_two(cells):
    if [r['m']for r in cells]!=[0,6]or[(r['q'],r['R'])for r in cells]!=[(1,1),(64,1)]:raise ValueError('Fixed two-feature cells')
    co=F.solve([[Fraction(1),Fraction(r['q'])]for r in cells],[dec(r['mean_C'])for r in cells])
    if any(co[0]+co[1]*r['q']!=dec(r['mean_C'])for r in cells):raise ValueError('Exact solve residual')
    return co

def freeze(source,raw):
    fixed(source);cells=means(raw,'calibration');config=read(BASE,VDIR+'/generation/CONFIGURATION.json');models={};rows=[]
    for axis in ('X','Y'):
        own=[c for c in cells if c['axis']==axis];co=fit_two(own)
        models[axis]=dict(alpha=enc(co[0]),beta=enc(co[1]),features=['one','q'],offset=0,calibration=own,
            calibration_residuals=[enc(0),enc(0)],interpretation='FULL_WRAPPER_HEAD_PLUS_TAIL_COMBINED_SLOPE_R1',independent_validation='NOT_YET_OPENED',
            solver=dict(commit=F.SOLVER[0],path=F.SOLVER[1],selector=['_solve']))
        for r in config['rounds']:
            c=r['physical_candidate'];price=co[0]+co[1]*c['q']
            if c['R_outer']!=1 or price<=0:raise ValueError('Nonpositive price or model domain; no clip/refit')
            rows.append(dict(L_D=7,configuration_id=config['id'],m=r['m'],axis=axis,physical_id=c['id'],q=c['q'],R=1,C_ref=enc(price),law=r['law_binding'],
                application='CALIBRATION'if r['m']in(0,6)else'LONG_ROUND_EXTRAPOLATION'if r['m']>6 else'INTERPOLATION',
                configuration_source=dict(commit=BASE,path=VDIR+'/generation/CONFIGURATION.json',selector=['rounds',r['m']]),model_axis=axis))
    rows.sort(key=lambda r:(r['m'],r['axis']))
    write(ROOT/PUBLIC/'FROZEN_MODELS.json',dict(status='FROZEN_2_AXIS_MODELS',source=source,raw_commit=raw,formula='C_ref=alpha+beta*q; full wrapper; offset0; no S addition',models=models,
        model_verdict='NOT_ASSESSED_NO_ADOPTED_TOLERANCE',finite_cohort_is_law_expectation=False,holdout_not_opened=True,extrapolation='m7..m12 beyond q64; two-point fit is not predictive validation'))
    write(ROOT/PUBLIC/'AXIS_PRICES.json',dict(source=source,rows=rows,positive_finite_axes=26,new_references=26))

def holdout(source,raw,model):
    fixed(source);cells=means(raw,'holdout');prices=read(model,PUBLIC+'/AXIS_PRICES.json');rows=[]
    for c in cells:
        p=next(r for r in prices['rows']if r['m']==3 and r['axis']==c['axis']);P=dec(p['C_ref']);O=dec(c['mean_C']);d=P-O
        rows.append(dict(**c,prediction=enc(P),observed=enc(O),signed_error=enc(d),absolute_error=enc(abs(d)),
            signed_relative_error=None if O==0 else enc(d/O),absolute_relative_error=None if O==0 else enc(abs(d)/O),
            relative_reason='ZERO_OBSERVED_PRICE'if O==0 else None,prediction_source=dict(commit=model,path=PUBLIC+'/AXIS_PRICES.json',selector=['rows',6 if c['axis']=='X'else 7])))
    write(ROOT/PUBLIC/'HOLDOUT_COMPARISON.json',dict(status='DESCRIPTIVE_COMPLETE_RELATIVE_UNDEFINED'if any(dec(c['observed'])==0 for c in rows)else'DESCRIPTIVE_COMPLETE',source=source,model_commit=model,raw_commit=raw,rows=rows,
        model_verdict='NOT_ASSESSED_NO_ADOPTED_TOLERANCE',interpolation_only=True,refit=False))

def allocate(source,model,hold):
    fixed(source);prices=read(model,PUBLIC+'/AXIS_PRICES.json');h=read(hold,PUBLIC+'/HOLDOUT_COMPARISON.json')
    if not h['status'].startswith('DESCRIPTIVE_COMPLETE'):raise ValueError('Holdout prerequisite')
    rows=prices['rows'];caps=[n for n in F.V0 for _ in ('X','Y')];p=[dec(r['C_ref'])for r in rows];total=sum(n*c for n,c in zip(caps,p));budget=Fraction(3,4)*total
    prior=read(BASE,VDIR+'/Q3_INPUTS.json')['inputs'];baseline=deepcopy(prior['cases'][0]);v0=read(BASE,VDIR+'/SUMMARY.json')['rows'][2]
    menu=[dict(L_D=7,rule='V0',label='LD7_V0',shots=baseline['allocation']['shots'],cost=enc(total),budget=None,unused_budget=None,status=v0['status'],Q3_source=v0['Q3_source'],disposition='REUSED_V0')];used={tuple(caps):'LD7_V0'};cases=[]
    for rule in ('A_SHAPE','B_PRICE'):
        g=F.generate(caps,p,budget,rule)
        if g is None:raise ValueError('Below one-shot floor')
        vector=g['shots'];cost=sum(n*c for n,c in zip(vector,p));shots=[dict(m=m,N_X=vector[2*m],N_Y=vector[2*m+1])for m in range(13)]
        if cost!=g['conditional_cost']or cost>budget or any(not 1<=n<=cap for n,cap in zip(vector,caps)):raise ValueError('Allocation contract')
        label='LD7_'+rule;reuse=used.get(tuple(vector));menu.append(dict(L_D=7,rule=rule,label=label,shots=shots,cost=enc(cost),budget=enc(budget),unused_budget=enc(budget-cost),
            lambda_value=enc(g['lambda_value']),next_breakpoint_cost=None if g['next_breakpoint_cost']is None else enc(g['next_breakpoint_cost']),reuse_label=reuse,
            disposition='REUSED_DUPLICATE_VECTOR'if reuse else'NEW_Q3_PENDING'))
        if not reuse:
            case=deepcopy(baseline);case.update(label=label,L_D=7,rule=rule,allocation=dict(allocation_id=label+'_3O4_01',shot_policy='USER_SUPPLIED',shots=shots),reference_cost=enc(cost),budget=enc(budget));cases.append(case)
        used[tuple(vector)]=label
    write(ROOT/PUBLIC/'ALLOCATION_INPUTS.json',dict(source=source,model_commit=model,holdout_commit=hold,prices=rows,menu=menu,cases=cases,maximum_new_Q3=2,
        **{k:prior[k]for k in ('statistical_contract','options','adopted_evaluation','acceleration')},generator=dict(commit=F.GENERATOR[0],path=F.GENERATOR[1]),selection='ACTUAL_MODEL_COST_MINIMUM_AMONG_ADMITTED_V0_A_B'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('freeze','holdout','allocate'));p.add_argument('--source',required=True);p.add_argument('--raw');p.add_argument('--model');p.add_argument('--holdout');a=p.parse_args()
    if a.mode=='freeze':freeze(a.source,a.raw)
    elif a.mode=='holdout':holdout(a.source,a.raw,a.model)
    else:allocate(a.source,a.model,a.holdout)
