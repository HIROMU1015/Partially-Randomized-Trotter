"""Saved-point resource attribution and alpha allocation; unchanged RTE law.

Exactly additive native RZ/CX/size of the existing modular implementation.
Depth is its serial upper bound. Negative boundary corrections are explicit.
"""
from fractions import Fraction
from functools import lru_cache
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import numpy as np
import mpmath as mp
from .track_c_h4 import ROOT,shot_rows,save_json
from .track_c_d_resources import r_key
from .track_c_source_guarantee import I,lower,up

OUT=ROOT/'artifacts/track_c_bottleneck/2026-11-08'
PRIOR=ROOT/'.venv/prior'
PUBLIC=ROOT/'artifacts/track_c_d_internal/2026-11-08'
NAMES=('original','PR_nonresonant_RZ','Full_nonresonant_RZ')

def identity(x):return x['name'],x['m'],x['h'],x['rate']

@lru_cache(None)
def saved():
    raw=json.loads((PRIOR/'G_RESULTS.json').read_text());nr=json.loads((PUBLIC/'NONRESONANT.json').read_text())
    policies={p['name']:p for p in raw['policies']};policies.update({p['name']:p for p in nr['policies']})
    qualified={}
    for path in (PUBLIC/'QUALIFICATION.json',PRIOR/'QUALIFICATION_EXTENSION.json',PRIOR/'REGISTERED_QUALIFICATION.json'):
        for r in json.loads(path.read_text())['rows']:qualified[identity(r)]=r
    dets={}
    for path in (PRIOR/'det_price').glob('*.json'):
        r=json.loads(path.read_text());dets[(r['case']['name'],r['h'])]=r
    samples={}
    for path in (PRIOR/'rte_price').glob('*.json'):
        r=json.loads(path.read_text());samples.setdefault(r_key(r['t'],r['n']),{}).setdefault(str(r['rare']),[]).append(r)
    for strata in samples.values():
        for rs in strata.values():rs.sort(key=lambda r:r['j'])
    return policies,qualified,dets,samples

def components(record):
    _,qs,dets,samples=saved();x=record['signal'];q=qs[identity(x)];d=dets[(x['name'],x['h'])];g={k:np.array(v,float) for k,v in d['groups'].items()};n=x['steps']
    if x['case']['kind'].startswith('full_'):
        bulk=n*(g['prefix']+g['suffix']);boundary=(n-1)*(g['middle']-g['prefix']-g['suffix'])
    else:
        bulk=n*(g['first']+g['last']+g['inside']);boundary=(n-1)*(g['junction']-g['first']-g['last'])
    stochastic=np.zeros(4);reps={str(rare):np.zeros((16,4)) for rare in (False,True)}
    for recipe in x['recipe']:
        for rare,rs in samples[r_key(recipe['time'],recipe['r'])].items():
            value=n*rs[0]['probability']*np.array([r['metrics'] for r in rs]);reps[rare]+=value;stochastic+=value.mean(axis=0)
    overhead=np.array([d['overhead'][axis] for axis in ('X','Y')]);mean=bulk+boundary+stochastic+overhead
    assert np.allclose(mean,record['price_mean'],rtol=1e-13,atol=1e-6)
    return dict(signal=x,qualification=q,D_bulk=bulk,D_boundary=boundary,D=bulk+boundary,R=stochastic,overhead=overhead,mean=mean,reps=reps)

def rigorous_shots(rho,beta,alpha):
    mp.iv.prec=192;value=4*mp.iv.log(2/I(alpha))/(I(rho)**2*mp.iv.sin(I(beta))**2)
    with mp.workprec(192):return int(mp.ceil(mp.mpf(value._mpi_[1])))

def allocate(points,primary=0):
    rows=[dict(feasible=True,radius_lower=p['qualification']['rho_lower'],beta_stat=p['qualification']['beta_stat_lower'],range=1.) for p in points]
    prices=np.array([p['mean'] for p in points]);request=shot_rows(rows,prices=prices[:,:,primary])[-1]
    alphas=np.array(request['alpha']).reshape(18,2)
    while sum(Fraction.from_float(float(a)) for a in alphas.flat)>Fraction(1,20):
        i=np.unravel_index(int(np.argmax(alphas)),alphas.shape);alphas[i]=math.nextafter(alphas[i],0.)
    ns=np.array([[rigorous_shots(p['qualification']['rho_lower'],p['qualification']['beta_stat_lower'],a) for a in row] for p,row in zip(points,alphas)],dtype=np.int64)
    return alphas,ns

def account(points,ns):
    byround=[];reps={str(rare):np.zeros((16,4)) for rare in (False,True)}
    for p,n in zip(points,ns):
        weight=int(n.sum());row={name:(weight*p[name]).tolist() for name in ('D_bulk','D_boundary','D','R')}
        row['wrapper']=(n[:,None]*p['overhead']).sum(axis=0).tolist();row['total']=(n[:,None]*p['mean']).sum(axis=0).tolist();row['m']=p['signal']['m'];row['name']=p['signal']['name'];row['shots']=n.tolist()
        assert np.allclose(np.array(row['D'])+row['R']+row['wrapper'],row['total'],rtol=1e-13)
        byround.append(row)
        for rare in reps:reps[rare]+=weight*p['reps'][rare]
    totals={k:np.sum([r[k] for r in byround],axis=0).tolist() for k in ('D_bulk','D_boundary','D','R','wrapper','total')}
    covariance=sum(np.cov(v,rowvar=False,ddof=1)/16 for v in reps.values())
    return dict(rounds=byround,totals=totals,MC_SE=np.sqrt(np.maximum(0,np.diag(covariance))).tolist(),covariance_MC=covariance.tolist(),reps={k:v.tolist() for k,v in reps.items()},shots_total=int(ns.sum()))

def run():
    OUT.mkdir(parents=True,exist_ok=False);policies,*_=saved();results=[];sensitivities=[];flat=[]
    for name in NAMES:
        records=sorted(policies[name]['selected'],key=lambda r:r['signal']['m']);points=[components(r) for r in records]
        ns=np.array([[p['qualification']['shots_per_axis']]*2 for p in points]);base=account(points,ns)
        allocations=[]
        for primary in (0,1):
            alpha,newn=allocate(points,primary);allocations.append(dict(primary=['RZ','CX'][primary],alpha=alpha.tolist(),**account(points,newn)))
        counter=[]
        for label in ('no_normalization','no_PF_bias','no_other_error'):
            newn=[]
            for p in points:
                q=p['qualification'];x=p['signal'];rho=q['rho_lower'];beta=q['beta_stat_lower']
                if label=='no_normalization':rho=min(1.,rho*math.exp(x['log_normalizer']))
                if label=='no_PF_bias':beta=min(.4,beta+q['PF_phase_upper'])
                if label=='no_other_error':beta=.4-q['PF_phase_upper']
                newn.append([rigorous_shots(rho,beta,1/720)]*2)
            counter.append(dict(name=label,**account(points,np.array(newn))))
        for component in ('D','R','wrapper'):
            for fraction in (.1,.25,.5,1.):
                sensitivities.append(dict(policy=name,component=component,fraction=fraction,G=(np.array(base['totals']['total'])-fraction*np.array(base['totals'][component])).tolist(),scope='fixed shots, hypothetical component scaling; zero is a sensitivity only'))
        results.append(dict(name=name,baseline=base,alpha_allocations=allocations,counterfactual_shots=counter))
        for r in base['rounds']:
            for i,metric in enumerate(['RZ','CX','serial_depth_upper','size']):flat.append(dict(policy=name,m=r['m'],metric=metric,D=r['D'][i],RTE=r['R'][i],wrapper=r['wrapper'][i],D_D_boundary_signed=r['D_boundary'][i],total=r['total'][i]))
    reference=np.array(next(r for r in results if r['name']=='Full_nonresonant_RZ')['baseline']['totals']['total']);pr=next(r for r in results if r['name']=='PR_nonresonant_RZ')['baseline']['totals']
    thresholds={k:((np.array(pr['total'])-reference)/np.array(pr[k])).tolist() for k in ('D','R')}
    save_json(OUT/'DECOMPOSITION.json',dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),base='b9dacb639eabdd3cf499e89f8807ff50b68e23ef',policies=results,
        qualification='Exactly additive counts of the same existing modular protocol. D/D boundary is signed and included in D, never added twice. D/R cuts have no separate measured price in prior results; basis/control attribution needs local template measurements.',
        metrics=['RZ','CX','serial_depth_upper','size'],alpha_scope='same fixed source points in both arms, exact alpha sum <=1/20; no RTE law or return-aggregation change'))
    save_json(OUT/'SENSITIVITY.json',dict(fixed_shot=sensitivities,required_reduction_fraction_to_match_Full=thresholds,qualification='Not algorithmic feasibility or a universal lower bound. MC SE is separate from structural assumptions.'))
    with (OUT/'ROUND_COMPONENTS.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]),lineterminator='\n');w.writeheader();w.writerows(flat)
    print('SAVED_ANALYSIS_COMPLETE',flush=True)

if __name__=='__main__':run()
