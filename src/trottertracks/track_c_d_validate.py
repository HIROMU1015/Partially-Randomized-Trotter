"""D-only joint cancellation and held-out full-RPE signals.

Coefficients are selected from saved BCH data, never long-round phases.
Genuine finite-time data are held out of this coefficient selection.
"""
from concurrent.futures import ProcessPoolExecutor,as_completed
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from scipy.optimize import brentq
from .track_c_d_internal import OUT,ROOT,schedule,kernel,analyse,common,save_json
from .track_c_mixed_formula import CACHE_SHA

H_VALUES=(.005,.01,.02,.04,.08,.16,.32,.64,1.28)
RATES=(1,2,4,8,16,32)

def candidates():
    a=json.loads((OUT/'STAGE_A.json').read_text());a2=json.loads((OUT/'STAGE_A2.json').read_text())
    ordered=sorted([x for x in a['rows'] if x['case']['kind']=='ordered'],key=lambda x:abs(x['H2_ground']))
    cases=[dict(name='original',kind='ordered',order=[0,1,2,3]),ordered[0]['case'],dict(name='half_control',kind='half_control',order=[0,1,2,3]),
        dict(name='D2',kind='D2',order=[0,1,2,3]),dict(name='D4',kind='D4',order=[0,1,2,3]),dict(name='partial_Y4',kind='partial_Y4',order=[0,1,2,3]),
        dict(name='full_S2',kind='full_S2',order=list(range(13))),dict(name='full_Y4',kind='full_Y4',order=list(range(13)))]
    vertices=[]
    for j in range(4):
        eligible=[x for x in a2['rows'] if x['fragment']==j and x['polynomial_ascending'][2]>1e-8]
        proposals=[]
        for row in eligible:
            coef=row['polynomial_ascending'];av=-coef[1]/(2*coef[2]);value=float(np.polynomial.polynomial.polyval(av,coef))
            proposals.append((value,abs(av),av,row))
        if proposals:
            _,_,av,row=min(proposals,key=lambda x:(x[0],x[1]))
            case=dict(name=f'vertex_j{j}',kind='redistributed',order=row['order'],fragment=j,a=av)
            vertices.append(analyse(case));cases.append(case)
    tuned=[]
    for row in (next(x for x in ordered if x['case']['order']==[0,1,2,3]),ordered[0]):
        order=row['case']['order'];internal=row['D_internal_ground'];outer=next(x for x in a['rows'] if x['case']['kind']=='D4')['H2_ground']
        eta=-4*outer/internal
        v=brentq(lambda v:2*v**3+(1-2*v)**3-eta,1.3512071919596578,2.,xtol=5e-15)
        case=dict(name='tuned_'+''.join(map(str,order)),kind='Dtuned',order=order,v=v)
        detail=analyse(case);detail.update(eta_target=eta,v=v);tuned.append(detail);cases.append(case)
    return cases,vertices,tuned

def signal_task(task):
    case,requested_h=task;c=common();rows=[];full=case['kind'].startswith('full_');ld=12 if full else 3
    for m in range(18):
        t=.02*(1<<m);steps=max(1,math.ceil(t/requested_h));h=t/steps;seq=schedule(case,h)
        for rate in ((0,) if full else RATES):
            u,f,lb,le,recipe=kernel(case,h,rate)
            z=complex(np.vdot(c['state'],np.linalg.matrix_power(u,steps)@c['state']));zf=complex(np.vdot(c['state'],np.linalg.matrix_power(f,steps)@c['state']))
            target=np.exp(-1j*c['energy']*t);pf=abs(float(np.angle(z*np.conj(target))))
            defect=float(np.linalg.norm(u.conj().T@u-np.eye(70),2))
            numerical=1e-10+steps*(defect+np.finfo(float).eps*70*len(seq)*4)
            input_disk=t*(9.480991703075102e-15+c['preps'][ld].threshold_operator_error_bound)+2*1.9069976415519017e-15/.23263021512018023
            rte=math.expm1(steps*le) if steps*le<700 else None
            other=None if rte is None else rte+numerical+input_disk
            logb=steps*lb;atten=math.exp(-logb) if logb<745 else 0.;rho=0. if other is None else max(0.,abs(z)-other)*atten
            beta=None if other is None or other>=abs(z) else .4-pf-math.asin(other/abs(z))
            feasible=bool(beta is not None and beta>0 and rho>1e-100)
            shots=math.ceil(4*math.log(1440)/(rho*rho*math.sin(beta)**2)) if feasible else None
            nd=steps*sum(k=='D' for k,_,_ in seq)-(steps-1 if seq[0][:2]==seq[-1][:2] else 0)
            ne=steps*sum(x['r'] for x in recipe)
            rows.append(dict(name=case['name'],case=case,requested_h=requested_h,h=h,m=m,t=t,steps=steps,rate=rate,PF_phase=pf,
                finite_phase=abs(float(np.angle(zf*np.conj(target)))),ideal_radius=abs(z),finite_radius=abs(zf),raw=dict(real=zf.real*atten,imag=zf.imag*atten),
                log_normalizer=logb,rho=rho,beta_stat=beta,shots_per_axis=shots,feasible=feasible,finite_error_upper=rte,numerical_disk=numerical,input_disk=input_disk,
                D_exponentials=nd,RTE_events=ne,RTE_occurrences=steps*len(recipe),recipe=recipe,
                surrogate=None if shots is None else 2*shots*(nd+ne),qualification='conditional float64 source, new-angle compiler interval pending'))
    save_json(OUT/'signals'/f'{case["name"]}_{requested_h}.json',dict(rows=rows));return rows

def run():
    (OUT/'signals').mkdir(exist_ok=False);cache=Path(os.environ['TRACK_C_H4_COMMON_CACHE']);assert hashlib.sha256(cache.read_bytes()).hexdigest()==CACHE_SHA
    cases,vertices,tuned=candidates();source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    save_json(OUT/'DESIGN.json',dict(source_commit=source,cases=cases,vertices=vertices,tuned=tuned,
        identity='H2_total=H2_outer+(2v^3+(1-2v)^3) H2_D/4 for D_symmetric triple composition in each half. Ground cancellation only.',
        holdout='h grid and all long-time/finite-RTE signals are not used to select coefficients.'))
    tasks=[(case,h) for case in cases for h in H_VALUES];rows=[];start=time.monotonic();workers=min(len(tasks),len(os.sched_getaffinity(0)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(signal_task,t) for t in tasks]):rows.extend(f.result())
    rows.sort(key=lambda x:(x['name'],x['requested_h'],x['m'],x['rate']))
    choices=[]
    for case in cases:
        for m in range(18):
            good=[x for x in rows if x['name']==case['name'] and x['m']==m and x['feasible']]
            choices.append(dict(name=case['name'],m=m,best=None if not good else min(good,key=lambda x:x['surrogate'])))
    save_json(OUT/'SIGNALS.json',dict(rows=rows))
    save_json(OUT/'SIGNAL_SUMMARY.json',dict(source_commit=source,input_cache_sha256=CACHE_SHA,choices=choices,workers=workers,wall_seconds=time.monotonic()-start,
        count=len(rows),unique_conditions=len({(x['name'],x['h'],x['m'],x['rate']) for x in rows}),qualification='surrogate selection only; gate price and native guarantee pending'))
    columns=['name','requested_h','h','m','rate','steps','PF_phase','finite_phase','ideal_radius','log_normalizer','rho','beta_stat','shots_per_axis','feasible','D_exponentials','RTE_events','surrogate']
    with (OUT/'SIGNALS.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore',lineterminator='\n');writer.writeheader();writer.writerows(rows)
    print('SIGNALS_COMPLETE',len(rows),flush=True)

if __name__=='__main__':run()
