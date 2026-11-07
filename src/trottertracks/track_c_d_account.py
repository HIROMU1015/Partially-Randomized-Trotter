"""Strong full-PF controls and exact-modular count accounting.

Native precision qualification remains distinct from source diagnostics.
"""
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from .track_c_d_internal import OUT,ROOT,ORDERS,common,save_json
from .track_c_d_validate import H_VALUES,signal_task
from .track_c_d_resources import r_key,SAMPLES
from trotterlib.rpe_four_round_phase_validation import reconstruct_rpe_phase

def strong():
    target=OUT/'STRONG_FULL.json'
    if target.exists():raise RuntimeError('Do not overwrite full-PF control')
    cases=[dict(name='strong_'+formula+'_'+''.join(map(str,p)),kind='full_'+formula,order=[*p,*range(4,13)]) for p in ORDERS for formula in ('S2','Y4') if p!=(0,1,2,3)]
    rows=[]
    with ProcessPoolExecutor(max_workers=min(len(os.sched_getaffinity(0)),len(cases)*len(H_VALUES))) as pool:
        for f in as_completed([pool.submit(signal_task,(c,h)) for c in cases for h in H_VALUES]):rows.extend(f.result())
    choices=[]
    for case in cases:
        for m in range(18):
            good=[x for x in rows if x['name']==case['name'] and x['m']==m and x['feasible']]
            choices.append(dict(name=case['name'],m=m,best=min(good,key=lambda x:x['surrogate']) if good else None))
    save_json(target,dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),choices=choices,
        rows=len(rows),order_opportunity='Same 24 permutations of first four deterministic blocks as partial; other full blocks retained. Not globally optimal over 13! orders.'))
    print('STRONG_FULL_COMPLETE',len(rows),flush=True)

def choices():
    return [x['best'] for file in ('SIGNAL_SUMMARY.json','STRONG_FULL.json') for x in json.loads((OUT/file).read_text())['choices'] if x['best'] and x['name']!='partial_Y4']

def run():
    target=OUT/'G_RESULTS.json'
    if target.exists():raise RuntimeError('Do not overwrite G results')
    rows=choices();dets={}
    for p in (OUT/'det_price').glob('*.json'):
        d=json.loads(p.read_text());dets[(d['case']['name'],d['h'])]=d
    samples={}
    for p in (OUT/'rte_price').glob('*.json'):
        x=json.loads(p.read_text());key=r_key(x['t'],x['n']);samples.setdefault(key,{}).setdefault(x['rare'],[]).append(x)
    moments={};replicates={}
    for key,strata in samples.items():
        mean=np.zeros(4);reps={}
        for rare,rs in strata.items():
            rs.sort(key=lambda x:x['j']);assert len(rs)==SAMPLES
            ps=rs[0]['probability'];assert all(x['probability']==ps for x in rs)
            values=np.array([x['metrics'] for x in rs],float);reps[rare]=ps*values;mean+=ps*values.mean(axis=0)
        moments[key]=mean;replicates[key]=reps
    def price_row(x):
        d=dets[(x['name'],x['h'])];g={k:np.array(v,float) for k,v in d['groups'].items()};n=x['steps']
        if x['case']['kind'].startswith('full_'):det=g['prefix']+g['suffix']+(n-1)*g['middle']
        else:det=g['first']+g['last']+n*g['inside']+(n-1)*g['junction']
        stochastic=sum((n*moments[r_key(q['time'],q['r'])] for q in x['recipe']),np.zeros(4))
        vectors=np.array([det+stochastic+np.array(d['overhead'][axis]) for axis in ('X','Y')])
        return vectors,det
    individual=[];by_name={}
    for x in rows:
        mean,det=price_row(x);record=dict(signal=x,price_mean=mean.tolist(),round_G=(x['shots_per_axis']*mean.sum(axis=0)).tolist(),deterministic_piece_counts=det.tolist())
        individual.append(record);by_name.setdefault(x['name'],[]).append(record)
    def policy_result(name,selected):
        selected=sorted(selected,key=lambda x:x['signal']['m']);assert len(selected)==18
        total=np.sum([x['round_G'] for x in selected],axis=0);reps={rare:np.zeros((SAMPLES,4)) for rare in (False,True)}
        for record in selected:
            x=record['signal'];factor=2*x['shots_per_axis']*x['steps']
            for q in x['recipe']:
                key=r_key(q['time'],q['r'])
                for rare in (False,True):reps[rare]+=factor*replicates[key][rare]
        covariance=sum((np.cov(v,rowvar=False,ddof=1)/SAMPLES for v in reps.values()),np.zeros((4,4)))
        phases=[math.atan2(x['signal']['raw']['imag'],x['signal']['raw']['real']) for x in selected]
        decode=reconstruct_rpe_phase(phases,[1<<m for m in range(18)]);energy=-decode['final_phase_estimate']/.02;error=abs(energy-common()['energy'])
        branch=all(abs(x['base_phase_estimate']+.02*common()['energy'])<=.4/(1<<m)+1e-10 for m,x in enumerate(decode['rounds']))
        return dict(name=name,G=total.tolist(),covariance_MC_mean=covariance.tolist(),MC_SE=np.sqrt(np.maximum(0,np.diag(covariance))).tolist(),
            selected=selected,shots_total=sum(2*x['signal']['shots_per_axis'] for x in selected),branch_correct=branch,decoded_energy=energy,energy_error=error,
            energy_pass=error<=1.5936e-4,decoder=decode,replicates_for_paired_comparison={str(k):v.tolist() for k,v in reps.items()})
    policies=[policy_result(name,rs) for name,rs in by_name.items()]
    for primary,label in ((0,'RZ'),(1,'CX')):
        for family,valid in [('PR',lambda x:not x['signal']['case']['kind'].startswith('full_')),('Full',lambda x:x['signal']['case']['kind'].startswith('full_'))]:
            selected=[min((x for x in individual if valid(x) and x['signal']['m']==m),key=lambda x:x['round_G'][primary]) for m in range(18)]
            policies.append(policy_result(f'{family}_adaptive_{label}',selected))
    original=next(x for x in policies if x['name']=='original');comparisons=[]
    for policy in policies:
        variance=np.zeros(4)
        for rare in ('False','True'):
            diff=np.array(policy['replicates_for_paired_comparison'][rare])-np.array(original['replicates_for_paired_comparison'][rare])
            variance+=np.var(diff,axis=0,ddof=1)/SAMPLES
        comparisons.append(dict(name=policy['name'],ratio_to_original=(np.array(policy['G'])/np.array(original['G'])).tolist(),paired_difference_MC_SE=np.sqrt(variance).tolist()))
    save_json(target,dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),metrics=['RZ','CX','serial_depth_upper','size'],policies=policies,comparisons=comparisons,
        qualification='G for a concrete modular compiler-A implementation and conditional float64 source-shot budgets. Counts exactly additive; all used R occurrence lengths/laws directly sampled. No whole-compile extrapolation. Native 18-round precision remains pending. Serial depth is an upper bound, not full DAG depth.',
        same_opportunities='Same h/r grid and surrogate selection; full controls additionally receive same 24 first-four permutations. Adaptive selection finite-menu estimates, not certified global optimum; MC selection bias not removed.',
        statistical='MC covariance includes shared price caches/CRN across rounds and families. Distinct from missing native precision qualification or whole-versus-modular systematic difference.'))
    print('G_COMPLETE',len(policies),flush=True)

if __name__=='__main__':strong() if sys.argv[1:] == ['strong'] else run()
