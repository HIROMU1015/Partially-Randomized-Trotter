"""New finite C-N price cohort under reference A; never resume old run01."""
from __future__ import annotations
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache, partial
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from .track_c_h4 import ROOT, common, save_json
from .track_c_compiler_a import compile_reference, PROFILE
from . import track_c_h4_price as legacy

BASE='3258a89f225d9530fba38e9d2343bc0165dd6d7d'
OUT=ROOT/'artifacts/track_c_guarantee_cn_price/2026-11-06/B_cn_price'
METRICS=('rz','cx','depth','size')
CASES=(dict(pilot='C-N',formula='S2',h=.02,ld=3,name='CN_S2_h0.02_ld3'),
       dict(pilot='C-N',formula='Y4',h=.02,ld=12,name='CN_Y4_h0.02_ld12'))


def reject_old_compile(*a,**kw):raise RuntimeError('Old opt2 compiler forbidden in new A cohort')
legacy.native_compile=reject_old_compile


def metrics(q):
    count=q.count_ops()
    return [int(count.get('rz',0)),int(count.get('cx',0)),int(q.depth()),int(q.size())]


def concat(pieces):
    result=QuantumCircuit(9)
    for q in pieces:result.compose(q,inplace=True)
    return result


def maximal_groups(parts):
    assert len(parts)%3==0
    groups=[];pending=[]
    for i,p in enumerate(parts):
        if i%3==1:
            if pending:groups.append(('D',concat(pending)));pending=[]
            groups.append(('R',p))
        else:pending.append(p)
    if pending:groups.append(('D',concat(pending)))
    return groups


@lru_cache(None)
def boundary(axis):
    prep=QuantumCircuit(9);prep.h(8)
    read=QuantumCircuit(9,1)
    if axis=='Y':read.sdg(8)
    read.h(8);read.measure(8,0)
    return compile_reference(prep),compile_reference(read)


@lru_cache(None)
def d_native(which):
    forward=legacy.d_part(3,.01).copy();p=common()['preps'][3]
    forward.p(-.02*(p.constant_coefficient+p.extracted_identity_coefficient),8)
    reverse=legacy.d_part(3,.01,True)
    source={'first':forward,'middle':reverse.compose(forward),'last':reverse}[which]
    return compile_reference(source)


def strong(pieces,axis):
    out=QuantumCircuit(9,1);serial=0
    for p in [boundary(axis)[0],*pieces,boundary(axis)[1]]:
        out.compose(p,qubits=range(9),clbits=range(p.num_clbits),inplace=True)
        serial+=p.depth()
    return metrics(out),serial


def sample(request):
    case=request['case'];q=request['q'];r=request['r'];rare=request['rare'];j=request['j']
    path=OUT/'raw'/request['key']/f'{j:04d}.json'
    if path.exists():raise RuntimeError('Duplicate scientific sample: '+str(path))
    seed=int.from_bytes(hashlib.sha256(f'trackC_AB_3258_20261106:{case["name"]}:{r}:{q}:{rare}:{j}'.encode()).digest()[:8],'big')
    start=time.monotonic()
    parts,_,prob=legacy.trajectory_parts(case,q,r,rare,seed,
        d_override=partial(legacy.d_part,case['ld']),price_additive=False)
    full=[metrics(compile_reference(legacy.wrapper(parts,axis))) for axis in ('X','Y')]
    if case['ld']==12:
        additive=full;serial=[x[2] for x in full]
    else:
        groups=maximal_groups(parts);native=[];di=0
        for role,p in groups:
            if role=='R':native.append(compile_reference(p))
            else:
                which='first' if di==0 else ('last' if di==q else 'middle')
                native.append(d_native(which));di+=1
        assert di==q+1
        paired=[strong(native,axis) for axis in ('X','Y')]
        additive=[x[0] for x in paired];serial=[x[1] for x in paired]
    record=dict(request=request,seed=seed,probability=prob if rare else 1-prob,
        full=full,strong=additive,strong_serial_block_depth=serial,
        strong_depth_definition='depth of actual composed native pieces, with no final global reoptimization',
        full_tail_free_baseline_same_by_definition=case['ld']==12,
        wall_seconds=time.monotonic()-start)
    save_json(path,record)
    return request['key'],record


def requests():
    result=[]
    for case in CASES:
        for r in ((0,) if case['ld']==12 else (1,2,4,8,16,32)):
            for q,n in ((1,32),(2,32),(8,128)):
                for rare in ((False,) if case['ld']==12 else (False,True)):
                    key=f'{case["name"]}_r{r}_q{q}_rare{int(rare)}'
                    for j in range(1 if case['ld']==12 else n):
                        result.append(dict(case=case,r=r,q=q,rare=rare,n_registered=n,j=j,key=key))
    return result


def technical_test():
    parts=[]
    for j in range(3):
        d=QuantumCircuit(9);d.p(.017*(j+1),8);d.crz(.00001,8,0)
        r=QuantumCircuit(9);r.cz(8,1)
        v=QuantumCircuit(9);v.crz(-.019,8,0)
        parts.extend([d,r,v])
    grouped=maximal_groups(parts)
    # Exact literal instruction order, including global phases, is preserved.
    a=concat(parts);b=concat([x[1] for x in grouped])
    assert a==b and [x[0] for x in grouped]==['D','R','D','R','D','R','D']
    return dict(maximal_D_literal_order=True,old_opt2_forbidden=True,old_44_or_H4_q1_tests_repeated=False)


def moments(records):
    probability=records[0]['probability']
    assert all(x['probability']==probability for x in records)
    a=np.array([x['full']+x['strong'] for x in records],float).reshape(len(records),16)
    covariance=np.cov(a,rowvar=False,ddof=1)/len(a) if len(a)>1 else np.zeros((16,16))
    return probability,a.mean(axis=0),covariance


def account(groups,output=None):
    output=OUT if output is None else Path(output)
    combined={};cells=[]
    for case in CASES:
        for r in ((0,) if case['ld']==12 else (1,2,4,8,16,32)):
            for q in (1,2,8):
                mean=np.zeros(16);cov=np.zeros((16,16));ps=0.
                for rare in ((False,) if case['ld']==12 else (False,True)):
                    key=f'{case["name"]}_r{r}_q{q}_rare{int(rare)}'
                    p,m,v=moments(groups[key]);mean+=p*m;cov+=p*p*v;ps+=p
                    cells.append(dict(key=key,n=len(groups[key]),probability=p))
                assert abs(ps-1)<1e-12
                combined[(case['name'],r,q)]=(mean,cov)
    save_json(output/'CELLS.json',cells)
    result=[]
    prior=json.loads(subprocess.check_output(['git','show',BASE+':artifacts/track_c_compiler_a_18round/2026-11-06/run01/ROUNDS.json'],cwd=ROOT))
    for case,shotcase in zip(CASES,prior):
        assert case['name']==shotcase['name']
        fits={};holdouts=[];predictions=[]
        for row in shotcase['rows']:
            r=0 if case['ld']==12 else row['r'];q=row['q']
            one,v1=combined[(case['name'],r,1)];two,v2=combined[(case['name'],r,2)]
            fits[r]=(one,two,v1,v2)
            prediction=(2-q)*one+(q-1)*two
            predictions.append(dict(m=row['m'],q=q,r=r,shots_per_axis=row['shots_per_axis_conditional'],price=prediction.reshape(2,2,4).tolist()))
        for r,(one,two,v1,v2) in fits.items():
            observed,v8=combined[(case['name'],r,8)];pred=7*two-6*one
            holdouts.append(dict(r=r,actual=observed.reshape(2,2,4).tolist(),prediction=pred.reshape(2,2,4).tolist(),
                point_relative_error=(abs(pred-observed)/np.maximum(abs(observed),1)).reshape(2,2,4).tolist(),
                predicted_relative_95_halfwidth=(1.96*np.sqrt(np.maximum(0,np.diag(36*v1+49*v2)))/np.maximum(abs(pred),1)).reshape(2,2,4).tolist(),
                observed_MC_covariance=v8.tolist(),status='q8 local diagnostic only; not a long-round certificate'))
        totals=np.zeros(16);ws={r:(np.zeros(16),np.zeros(16)) for r in fits}
        for row,p in zip(shotcase['rows'],predictions):
            n=row['shots_per_axis_conditional'];q=row['q'];r=p['r'];value=np.array(p['price']).reshape(16)
            totals+=n*value
            ws[r][0][:]+=n*(2-q);ws[r][1][:]+=n*(q-1)
        variance=np.zeros((16,16))
        for r,(_,_,v1,v2) in fits.items():
            a,b=ws[r];variance+=np.outer(a,a)*v1+np.outer(b,b)*v2
        direct=np.zeros(16);direct_cov=np.zeros((16,16));direct_rounds=[]
        for row in shotcase['rows']:
            if row['q'] not in (1,2,8):continue
            r=0 if case['ld']==12 else row['r'];n=row['shots_per_axis_conditional']
            mean,cov=combined[(case['name'],r,row['q'])]
            direct+=n*mean;direct_cov+=n*n*cov;direct_rounds.append(row['m'])
        metrics_out={};direct_out={}
        for j,name in enumerate(METRICS):
            i=[j,4+j];k=[8+j,12+j];weights=np.zeros(16);weights[i]=1;weights[k]=-1
            full=float(totals[i].sum());strong_total=float(totals[k].sum())
            fweight=np.zeros(16);fweight[i]=1;sweight=np.zeros(16);sweight[k]=1
            metrics_out[name]=dict(full_conditional_G=full,strong_conditional_G=strong_total,
                full_calibration_MC_SE=float(math.sqrt(max(0,fweight@variance@fweight))),
                strong_calibration_MC_SE=float(math.sqrt(max(0,sweight@variance@sweight))),
                paired_difference_full_minus_strong=full-strong_total,
                paired_calibration_MC_SE=float(math.sqrt(max(0,weights@variance@weights))),
                long_round_model_error_bound=None)
            direct_out[name]=dict(full_fixed_shot_subtotal=float(direct[i].sum()),strong_fixed_shot_subtotal=float(direct[k].sum()),
                full_MC_SE=float(math.sqrt(max(0,fweight@direct_cov@fweight))),
                paired_difference_MC_SE=float(math.sqrt(max(0,weights@direct_cov@weights))))
        result.append(dict(case=case,metrics=metrics_out,holdouts=holdouts,round_price_predictions=predictions,
            all_affine_predictions_nonnegative=all(min(np.array(x['price']).reshape(-1))>=0 for x in predictions),
            fixed_total_shots=shotcase['total_shots_conditional'],direct_measured_RPE_rounds=direct_rounds,
            direct_measured_fixed_shot_subtotals=direct_out,all_RPE_cost_status='CONDITIONAL_UNDETERMINED',
            interpretation='Fixed saved shots; paired MC uncertainty includes reuse of anchors across rounds. Long-q model discrepancy is unbounded.',
            scientific_success_status='independent of A; no unconditional success or cost winner'))
    save_json(output/'ACCOUNTING.json',result)
    return result


def run():
    if (OUT/'STOP.json').exists():raise RuntimeError('Stopped cohort; do not resume')
    jobs=requests();assert len({x['key'] for x in jobs})==39 and len(jobs)==2307
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'raw').exists():raise RuntimeError('Existing raw cohort; no automatic retry/resume')
    save_json(OUT/'TECHNICAL_TEST.json',technical_test())
    common()
    for which in ('first','middle','last'):d_native(which)
    for axis in ('X','Y'):boundary(axis)
    workers=min(len(jobs),len(os.sched_getaffinity(0)))
    save_json(OUT/'SCOPE.json',dict(base_commit=BASE,cells=39,samples=2307,workers=workers,profile=PROFILE,
        calibration_q=[1,2],holdout_q=8,stratum_n_calibration=32,stratum_n_holdout=128,
        seed_namespace='trackC_AB_3258_20261106',new_seed_cohort=True,old_run_restarted=False,
        pricing_metrics=list(METRICS),strong_baseline='maximal D runs and complete R occurrences; actual composed depth',
        A_independent=True,additional_sampling_batches=False))
    groups=defaultdict(list);start=time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(sample,x) for x in jobs]
        for count,f in enumerate(as_completed(futures),1):
            key,record=f.result();groups[key].append(record)
            if count%100==0 or count==len(jobs):print('B samples',count,'/',len(jobs),'seconds',round(time.monotonic()-start,1),flush=True)
    for key in groups:groups[key].sort(key=lambda x:x['request']['j'])
    result=account(groups)
    save_json(OUT/'SUMMARY.json',dict(cells=39,samples_completed=len(jobs),profile=PROFILE,
        all_RPE_cost_status='CONDITIONAL_UNDETERMINED',winner=None,T_synthesis=0,old_run_restarted=False,
        additional_price_grid=False,wall_seconds=time.monotonic()-start))
    save_json(OUT/'STOP.json',dict(status='MANDATORY_STOP',science_workers_remaining=0,automatic_resume=False))
    print('B complete; mandatory STOP; no long-round winner',flush=True)
