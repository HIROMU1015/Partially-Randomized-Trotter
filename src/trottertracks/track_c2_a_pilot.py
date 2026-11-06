"""C2-A finite PF crossover screen, isolated from the stopped C-N run.

Short-wrapper prices are diagnostics. An affine long-q prediction is never
promoted to a qualified full-RPE price or a scientific winner.
"""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import hashlib,json,math,os,time
import numpy as np
from .track_c_h4 import ROOT,common,kernel,save_json,phase_distance
from .track_c_h4_price import trajectory_parts,wrapper,d_part
from .track_c_compiler_a import compile_reference
OUT=ROOT/'artifacts/track_c2_a_pilot/2026-11-06'
ROUNDS=(0,8,17)
R=(1,2,4,8,16,32)

def signal_job(task):
    c=common();rows=[]
    for r in (R if task['ld']==3 else (0,)):
        ideal,finite,lb,le,recipe=kernel(c,task['formula'],task['h'],task['ld'],r)
        defect=float(np.linalg.norm(ideal.conj().T@ideal-np.eye(70),2))
        for m in ROUNDS:
            q=2**m;steps=round(q*.02/task['h']);u=np.linalg.matrix_power(ideal,steps)
            z=complex(np.vdot(c['state'],u@c['state']));pf=phase_distance(z,np.exp(-1j*c['energy']*q*.02))
            num=1e-10+steps*(defect+np.finfo(float).eps*70*len(recipe)*4)
            inp=q*.02*(9.480991703075102e-15+c['preps'][task['ld']].threshold_operator_error_bound)+2*1.9069976415519017e-15/.23263021512018023
            err=math.expm1(steps*le) if steps*le<700 else None
            other=None if err is None else err+num+inp
            beta=None if other is None or other>=abs(z) else .4-pf-math.asin(other/abs(z))
            logb=steps*lb;atten=math.exp(-logb) if logb<745 else 0.
            rho=0. if other is None else max(0.,abs(z)-other)*atten
            valid=beta is not None and beta>0 and rho>0
            n=math.ceil(4*math.log(1440)/(rho*rho*math.sin(beta)**2)) if valid and rho>1e-100 else None
            rows.append(dict(m=m,q=q,r=r,steps=steps,PF_phase=pf,ideal_radius=abs(z),log_normalizer=logb,
                rho_lower=rho,beta_stat=beta,shots_per_axis=n,finite_error_upper=err,numerical_disk=num,input_disk=inp,
                feasible=bool(valid and n is not None),qualification='conditional_float64_screen; new compiler bound not certified'))
    save_json(OUT/'signals'/f"{task['name']}.json",dict(task=task,rows=rows))
    return task,rows

def metric(qc):
    a=qc.count_ops();return [int(a.get('rz',0)),int(a.get('cx',0)),int(qc.depth()),int(qc.size())]

def price_job(request):
    task,r,q,rare=request
    seed=int.from_bytes(hashlib.sha256(f'C2A:{task["name"]}:{r}:{q}:{rare}'.encode()).digest()[:8],'big')
    parts,_,mass=trajectory_parts(task,q,r,rare,seed,d_override=partial(d_part,task['ld']),price_additive=False)
    prices=[metric(compile_reference(wrapper(parts,b))) for b in ('X','Y')]
    result=dict(task=task,r=r,q=q,rare=rare,seed=seed,conditional_probability=mass if rare else 1-mass,prices=prices,
        samples=1,qualification='one deterministic reproducible representative per conditional stratum; NOT an MC mean certificate')
    save_json(OUT/'prices'/f"{task['name']}_r{r}_q{q}_rare{int(rare)}.json",result)
    return result

def run():
    if (OUT/'STOP.json').exists():raise RuntimeError('Stopped pilot')
    start=time.monotonic();tasks=[dict(pilot='C2-A',formula=f,h=h,ld=ld,name=f'{f}_h{h}_ld{ld}') for ld in (3,12) for f in ('S2','Y4') for h in (.02,.01)]
    workers=min(len(tasks),len(os.sched_getaffinity(0)))
    with ProcessPoolExecutor(max_workers=workers) as pool:data=[x.result() for x in [pool.submit(signal_job,t) for t in tasks]]
    # Candidate r values use a transparent primitive-count screening surrogate.
    # Each fixed PF family receives exactly the same h/r search opportunity.
    selected={};requests=[]
    for task,rows in data:
        rs=set()
        for m in ROUNDS:
            candidates=[x for x in rows if x['m']==m and x['feasible']]
            if candidates:
                best=min(candidates,key=lambda x:x['shots_per_axis']*x['steps']*(len(x.get('recipe',[])) or (1 if task['formula']=='S2' else 3))*(task['ld']+1+(x['r'] if task['ld']==3 else 0)))
                rs.add(best['r'])
        selected[task['name']]=sorted(rs)
        for r in sorted(rs):
            for q in (1,2,4):
                for rare in ((False,True) if task['ld']==3 else (False,)):
                    requests.append((task,r,q,rare))
    save_json(OUT/'PRICE_REQUESTS.json',dict(selected_r=selected,requests=requests,scope='q1/q2 affine diagnostic, q4 holdout; one representative per conditional stratum'))
    print('A signal cells',sum(len(rows) for _,rows in data),'short price cells',len(requests),flush=True)
    prices=[]
    with ProcessPoolExecutor(max_workers=min(len(requests),len(os.sched_getaffinity(0)))) as pool:
        futures=[pool.submit(price_job,x) for x in requests]
        for future in as_completed(futures):
            prices.append(future.result())
            if len(prices)%12==0:print('A prices',len(prices),'/',len(requests),flush=True)
    lookup={}
    for task in tasks:
        for r in selected[task['name']]:
            for q in (1,2,4):
                records=[x for x in prices if x['task']['name']==task['name'] and x['r']==r and x['q']==q]
                value=np.zeros((2,4))
                for x in records:
                    prob=x['conditional_probability'] if task['ld']==3 else 1.
                    value+=prob*np.array(x['prices'])
                lookup[(task['name'],r,q)]=value
    holdings=[];rankings=[]
    for task,rows in data:
        for r in selected[task['name']]:
            a=lookup[(task['name'],r,1)];b=lookup[(task['name'],r,2)];hold=lookup[(task['name'],r,4)]
            pred=3*b-2*a
            holdings.append(dict(name=task['name'],r=r,q4_actual=hold.tolist(),q4_affine=pred.tolist(),
                relative_point_error=(abs(pred-hold)/np.maximum(1,hold)).tolist()))
            for row in rows:
                if row['r']!=r or not row['feasible']:continue
                value=(2-row['q'])*a+(row['q']-1)*b
                for k,metric_name in enumerate(('RZ','CX')):
                    rankings.append(dict(**task,**row,metric=metric_name,diagnostic_shot_price=float(row['shots_per_axis']*value[:,k].sum()),
                        price_status='UNQUALIFIED_LONG_Q_AFFINE_REPRESENTATIVE',price_per_axis=value[:,k].tolist()))
    winners=[]
    for ld in (3,12):
        for metric_name in ('RZ','CX'):
            chosen=[]
            for m in ROUNDS:
                rows=[x for x in rankings if x['ld']==ld and x['metric']==metric_name and x['m']==m]
                best=min(rows,key=lambda x:x['diagnostic_shot_price'])
                family=[]
                for formula in ('S2','Y4'):
                    rr=[x for x in rows if x['formula']==formula]
                    if rr:
                        z=min(rr,key=lambda x:x['diagnostic_shot_price']);family.append(dict(formula=formula,h=z['h'],r=z['r'],cost=z['diagnostic_shot_price']))
                chosen.append(dict(m=m,formula=best['formula'],h=best['h'],r=best['r'],family_minima=family))
            winners.append(dict(ld=ld,metric=metric_name,rows=chosen,diagnostic_family_crossover=len(set(x['formula'] for x in chosen))>1))
    save_json(OUT/'CROSSOVER_SCREEN.json',dict(winners=winners,all_candidates=rankings,q4_holdouts=holdings,
        qualification='finite pilot diagnostics only; representative stratum paths are not an expected full-price estimate',
        r_search='all six r signals; only surrogate-selected r values receive short prices; compiled global r optimum NOT established'))
    save_json(OUT/'SUMMARY.json',dict(status='STOP_PRICE_UNCERTAINTY_NO_QUALIFIED_CROSSOVER',
        signal_cells=sum(len(rows) for _,rows in data),short_price_cells=len(prices),rounds=list(ROUNDS),
        all18_cost_computed=False,winners=winners,wall_seconds=time.monotonic()-start,
        interpretation='No qualified PF switching benefit. Diagnostic ranking is not a scientific no-go or a certified fixed-family optimum.',
        next_required='A qualified expected long-round price mechanism and, if justified, a new bounded execution scope; no automatic continuation'))
    save_json(OUT/'STOP.json',dict(status='MANDATORY_STOP',reason='price uncertainty prevents qualified crossover/total-G claim',automatic_resume=False,workers_remaining=0))
    print('A mandatory STOP: price qualification',flush=True)

if __name__=='__main__':run()
