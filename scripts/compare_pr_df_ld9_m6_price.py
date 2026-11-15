"""Exact descriptive m6 means/difference/ratio from two independent cohorts.

Only direct receipts; no fitted/extrapolated price, model or13-round total.
"""
import argparse
import csv
from decimal import Decimal,localcontext
from fractions import Fraction
import json
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]
PUBLIC='docs/research/d_pr_df_ld9_m6_price_20261116'
CALLER='scripts/compare_pr_df_ld9_m6_price.py'
OLD='5c5d87ab32d03bb107a9a6bfed3cba253aef2aa0'
OLD_DIR='docs/research/d_pr_df_ld7_price_results_20261116'
def blob(c,p):return subprocess.check_output(['git','show',c+':'+p],cwd=ROOT)
def read(c,p):return json.loads(blob(c,p))
def enc(x):x=Fraction(x);return dict(numerator=x.numerator,denominator=x.denominator)
def display(x):
    with localcontext()as c:c.prec=24;return str(Decimal(x.numerator)/Decimal(x.denominator))
def write(n,v):
    with(ROOT/PUBLIC/n).open('x')as f:json.dump(v,f,indent=2,allow_nan=False,ensure_ascii=False);f.write('\n')

def collect(commit,directory,acq_path,cell):
    acq=read(commit,acq_path)
    if acq['status']!='ACQUIRED':raise ValueError('Partial acquisition; do not shrink cohort')
    rows=[r for r in acq['rows']if r['m']==6]
    if [r['index']for r in rows]!=list(range(4)):raise ValueError('Missing m6 path')
    out=[];means={}
    for axis in ('X','Y'):
        own=[]
        for r in rows:
            rec=r['record'];a=next(a for a in rec['records']if a['axis']==axis)
            if r['status']!='VALID'or rec['request']!=cell['request']or rec['cohort']!=cell['cohort']or rec['index']!=r['index']or rec['seeds']!=cell['seeds'][r['index']]:raise ValueError('Path identity/cohort/seed mismatch')
            axis_path=directory+f'/m6/p{r["index"]}/'+{'X':'cosine','Y':'sine'}[axis]+'.json';receipt=read(commit,axis_path)
            if receipt['metrics']!=a['metrics']or receipt['fingerprint']!=a['fingerprint']or receipt['compiler']!=cell['request']['physical_context']['price_profile']['compiler']or a['C']!=a['metrics']['rz_count']or a['D']!=a['C']-a['S']:raise ValueError('Axis/metrics/fingerprint/compiler mismatch')
            own.append(dict(m=6,axis=axis,index=r['index'],cohort=rec['cohort'],seeds=rec['seeds'],physical_id=rec['request']['physical_context']['physical_candidate']['id'],configuration_id=rec['request']['configuration_id'],
                S=a['S'],C=a['C'],D=a['D'],fingerprint=a['fingerprint'],event_digest=rec['event_digest'],events=r['events'],status=r['status'],
                receipt=dict(commit=commit,path=axis_path),path_receipt=dict(commit=commit,path=directory+f'/m6/p{r["index"]}/PATH_RECEIPT.json'),execution_source=rec['request']['physical_context']['execution_source'],gate_status=r['gate_status'],
                numerical_action_gate=r['numerical_action_gate'],full_wrapper_statevector_validation=r['full_wrapper_statevector_validation']))
        means[axis]=sum(Fraction(r['C'])for r in own)/4;out+=own
    return out,means

def compare(source,raw):
    if blob(source,CALLER)!=(ROOT/CALLER).read_bytes():raise ValueError('Unfixed comparison SOURCE')
    inputs=read(raw,PUBLIC+'/INPUTS.json');oldinputs=read(OLD,OLD_DIR+'/ACQUISITION_INPUTS.json');oldcell=next(c for c in oldinputs['cells']if c['m']==6);newcell=inputs['cell']
    oldpc=oldcell['request']['physical_context'];newpc=newcell['request']['physical_context']
    for k in ('system','geometry','basis','hamiltonian','state','snapshot','rank','rank_policy'):
        if oldpc['binding'][k]!=newpc['binding'][k]:raise ValueError('H/state/snapshot conditions differ')
    if oldpc['price_profile']!=newpc['price_profile']or oldcell['cohort']==newcell['cohort']:raise ValueError('Profile/cohort conditions')
    for pc,prefix in ((oldpc,7),(newpc,9)):
        c=pc['physical_candidate']
        if(c['prefix'],c['T'],c['q'],c['R_outer'],c['K'],c['formula'],c['delta'])!=(prefix,6.4,64,1,2,'2nd',.1):raise ValueError('Wrong physical point')
    oldrows,oldmeans=collect(OLD,OLD_DIR+'/calibration/acquisition',OLD_DIR+'/calibration/acquisition/RESULT.json',oldcell)
    newrows,newmeans=collect(raw,PUBLIC+'/acquisition',PUBLIC+'/acquisition/RESULT.json',newcell)
    if any(x!=Fraction(1575349,4)for x in oldmeans.values()):raise ValueError('Saved LD7 reference mean differs')
    comparisons=[]
    for axis in ('X','Y'):
        a,b=oldmeans[axis],newmeans[axis];comparisons.append(dict(axis=axis,n_paths_LD7=4,n_paths_LD9=4,LD7_mean=enc(a),LD9_mean=enc(b),difference_LD9_minus_LD7=enc(b-a),ratio_LD9_over_LD7=enc(b/a),
            LD7_mean_display=display(a),LD9_mean_display=display(b),difference_display=display(b-a),ratio_display=display(b/a)))
    cohorts=[dict(L_D=L,cohort=cell['cohort'],configuration_id=cell['request']['configuration_id'],request=cell['request'],rows=rows)for L,cell,rows in ((7,oldcell,oldrows),(9,newcell,newrows))]
    write('COMPARISON.json',dict(schema='PR_DF_LD7_LD9_M6_DIRECT_PRICE_V1',status='DESCRIPTIVE_COMPLETE',source=source,LD9_raw_commit=raw,LD7_saved_commit=OLD,
        quantity='MEASURED_FULL_CONTROLLED_HADAMARD_WRAPPER_COMPILED_RZ_PER_SHOT',cohorts=cohorts,comparisons=comparisons,
        scope='TWO_SEPARATE_NORMAL_FINITE_COHORTS_NOT_PAIRED_BY_INDEX',counts=dict(reused_LD7_paths=4,reused_LD7_axis_receipts=8,new_LD9_paths=4,new_compiles=8,new_signals=0,new_Q3=0,fit=0,additional_price_points=0,retry=0),
        LD9_allround_prices=None,LD9_total_model_cost=None,formal_E_C=None,CI=None,G=None,winner=None,mandatory_stop=True))
    write('C_HANDOFF.json',dict(source=source,LD9_configuration=newpc['input_source']['configuration'],LD9_configuration_id=newcell['request']['configuration_id'],m=6,physical_id=newpc['physical_candidate']['id'],
        LD9_raw=dict(commit=raw,path=PUBLIC+'/acquisition/RESULT.json'),LD7_reference=dict(commit=OLD,path=OLD_DIR+'/calibration/acquisition/RESULT.json',m=6),cohorts=cohorts,comparisons=comparisons,
        classification='DIRECT_FINITE_COHORT_DESCRIPTIVE_PRICE_ONLY',fit=None,LD9_total_model_cost=None,formal_E_C=None,G=None,C_independent_admission='NOT_PERFORMED_BY_D2',mandatory_stop=True))
    with(ROOT/PUBLIC/'PATH_PRICES.csv').open('x')as f:
        keys=['L_D','m','axis','index','cohort','physical_id','S','C','D','fingerprint','status'];w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for L,rows in ((7,oldrows),(9,newrows)):w.writerows(dict(L_D=L,**{k:r[k]for k in keys[1:]})for r in rows)
    for c in comparisons:print(c['axis'],c['LD7_mean_display'],c['LD9_mean_display'],c['difference_display'],c['ratio_display'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--raw',required=True);a=p.parse_args();compare(a.source,a.raw)
