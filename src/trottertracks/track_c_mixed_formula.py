"""Fixed Track C H4 mixed-splitting experiment; no B2 P1 inputs or old runs.

Chronological endpoints are the original one-sided D sweeps. The middle D
sweep is reverse/forward symmetric, so a=1/4 is EXACTLY two old half steps.
This is a refinement control, never an algorithmic improvement claim.
"""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy.linalg import expm
from .track_c_h4 import ROOT, common, evolution, kernel as old_kernel, tail_operator, weights, save_json

OUT=ROOT/'artifacts/track_c_mixed_formula/2026-11-08'
CACHE_SHA='68269aa0415f048a0220beb4054daa1bba715c6eb29fbbe9d5ba779baf0346f2'
PARAMETERS={'mixed_2MN':.1931833275037836,'mixed_zero_D':(3-math.sqrt(3))/6,
            'mixed_zero_R':1/6,'refinement_control':.25,'old_endpoint':.5}
H_VALUES=(.01,.02,.04,.08,.16,.32)
RATES=(1,2,4,8,16,32)
SCREEN_ROUNDS=(0,4,8)


def comm(a,b):return a@b-b@a
def hermitian(a):return (a+a.conj().T)/2
def expected(a,c):return float(np.vdot(c['state'],a@c['state']).real)


def coefficients(a):
    return (1-6*a+6*a*a)/12,(1-6*a)/24


def derive_bch():
    """Exact free-associative expansion through degree three, not a fitted BCH."""
    import sympy as s
    a=s.Symbol('a',real=True)
    def product(x,y):
        z={}
        for u,c in x.items():
            for v,d in y.items():
                if len(u+v)<=3:z[u+v]=z.get(u+v,0)+c*d
        return {w:s.expand(c) for w,c in z.items()}
    out={'':s.Integer(1)}
    for letter,c in [('A',a),('B',s.Rational(1,2)),('A',1-2*a),('B',s.Rational(1,2)),('A',a)]:
        out=product(out,{letter*k:c**k/s.factorial(k) for k in range(4)})
    x=dict(out);x['']-=1
    x2=product(x,x);x3=product(x2,x)
    log={w:s.expand(x.get(w,0)-x2.get(w,0)/2+x3.get(w,0)/3) for w in out}
    alpha=s.factor(log['AAB']);beta=s.factor(-log['ABB'])
    assert s.simplify(alpha-(1-6*a+6*a*a)/12)==0
    assert s.simplify(beta-(1-6*a)/24)==0
    c1={'AAB':1,'ABA':-2,'BAA':1};c2={'BAB':2,'BBA':-1,'ABB':-1}
    assert all(s.simplify(v-alpha*c1.get(w,0)-beta*c2.get(w,0))==0 for w,v in log.items() if len(w)==3)
    return dict(alpha=str(alpha),beta=str(beta),ground_difference=str(s.factor(alpha-beta)),
        completed_square=str(s.expand(alpha-beta-s.Rational(1,96)-(a-s.Rational(1,4))**2/2)),
        degree3_words={w:str(s.factor(v)) for w,v in log.items() if len(w)==3},
        convention='A=-i H_D, B=-i H_R; H_eff=H-h^2(alpha C_D+beta C_R)+O(h^4)',
        ground_identity='expect(C_D)+expect(C_R)=0 for an exact eigenstate of H_D+H_R',
        consequence='alpha-beta=1/96+(a-1/4)^2/2; no real a removes the ground leading mixed term generically.',
        refinement_identity='a=1/4 is exactly two S2 half steps, including the registered one-sided D sweeps and finite RTE.')


def compact(sequence):
    result=[]
    for kind,index,t in sequence:
        if t==0:continue
        if result and result[-1][:2]==(kind,index):
            old=result.pop();t+=old[2]
        if t!=0:result.append((kind,index,t))
    return result


def sequence(name, h=1., ld=3):
    """Append order. Scalar evolution is applied once, separately."""
    n=ld+1
    def sweep(time,reverse=False):return [('D',j,time) for j in (reversed(range(n)) if reverse else range(n))]
    def s2(time):
        return sweep(time/2)+([('R',-1,time)] if ld==3 else [])+sweep(time/2,True)
    if name in ('registered_S2','full_S2'):return compact(s2(h))
    if name in ('partial_Y4','full_Y4'):
        return compact([item for w in weights('Y4') for item in s2(float(w*h))])
    a=PARAMETERS[name];central=1-2*a
    return compact(sweep(a*h)+[('R',-1,h/2)]+sweep(central*h/2,True)
        +sweep(central*h/2)+[('R',-1,h/2)]+sweep(a*h,True))


@lru_cache(None)
def spectrum(index):return common()['spectra'][index]


@lru_cache(None)
def aggregate_D_spectrum():
    return np.linalg.eigh(hermitian(sum(common()['block_matrices'][:4])))


@lru_cache(None)
def matrix_kernel(name,h,rate,*,exact_D=False):
    c=common();ld=12 if name.startswith('full_') else 3
    exact=np.eye(70,dtype=complex);finite=exact.copy();logb=logerr=0.;occurrences=[]
    if exact_D:
        a=PARAMETERS[name]
        seq=[('D',-2,a*h),('R',-1,h/2),('D',-2,(1-2*a)*h),('R',-1,h/2),('D',-2,a*h)]
        ds=aggregate_D_spectrum()
    else:seq=sequence(name,h,ld)
    for kind,index,t in seq:
        if kind=='D':u=evolution(ds if exact_D else spectrum(index),t);v=u
        else:
            count=max(1,math.ceil(abs(t)*rate/.02))
            u=evolution(c['tail_spectra'][3],t)
            v,lb,le=tail_operator(c,t,count);logb+=lb;logerr+=le
            occurrences.append(dict(time=t,r=count,dimensionless_time=c['preps'][3].exact_rte_lambda_r*t/count))
        exact=u@exact;finite=v@finite
    p=c['preps'][ld];scalar=p.constant_coefficient+p.extracted_identity_coefficient
    phase=np.exp(-1j*h*scalar)
    return phase*exact,phase*finite,logb,logerr,occurrences


def cubic_generator(seq,matrices):
    """Polynomial log of the actual multiblock sequence, retaining internal PF."""
    n=matrices[0].shape[0];u=[np.eye(n,dtype=complex),*[np.zeros((n,n),complex) for _ in range(3)]]
    h=np.zeros((n,n),complex)
    for kind,index,t in seq:
        generator=matrices[-1] if kind=='R' else matrices[index]
        x=-1j*t*generator;h+=t*generator
        e=[np.eye(n,dtype=complex),x,x@x/2,x@x@x/6]
        u=[sum((e[j]@u[k-j] for j in range(k+1)),np.zeros((n,n),complex)) for k in range(4)]
    l3=u[3]-(u[1]@u[2]+u[2]@u[1])/2+u[1]@u[1]@u[1]/3
    result=1j*l3
    assert np.linalg.norm(result-result.conj().T,'fro')<1e-9
    return hermitian(result)


def local_analysis():
    c=common();blocks=[hermitian(x) for x in c['block_matrices'][:4]];d=sum(blocks);r=hermitian(c['tails'][3])
    cd=comm(d,comm(d,r));cr=comm(r,comm(d,r));p=c['preps'][3]
    model=d+r+(p.constant_coefficient+p.extracted_identity_coefficient)*np.eye(70)
    residual=(model-c['energy']*np.eye(70))@c['state']
    report=dict(BCH=derive_bch(),D_commutator_expectation=expected(cd,c),R_commutator_expectation=expected(cr,c),
        ground_commutator_sum=expected(cd+cr,c),ground_identity_residual_bound=2*float(np.linalg.norm(residual))*float(np.linalg.norm(comm(d,r)@c['state'])),
        saved_state_model_residual=float(np.linalg.norm(residual)),saved_state_norm=float(np.linalg.norm(c['state'])),physical_model_difference_fro=float(np.linalg.norm(model-c['physical_H'],'fro')),
        outer_norms=dict(C_D=float(np.linalg.norm(cd,2)),C_R=float(np.linalg.norm(cr,2))),candidates=[])
    raw={'C_D':cd,'C_R':cr,'model':model}
    for name in ['registered_S2',*PARAMETERS]:
        a=PARAMETERS.get(name,.5);alpha,beta=coefficients(a)
        outer=-alpha*cd-beta*cr
        full=cubic_generator(sequence(name),[*blocks,r]);raw[name+'_H2']=full
        internal=cubic_generator(sequence(name),[*blocks,np.zeros_like(r)])
        raw[name+'_internal_H2']=internal
        row=dict(name=name,a=a,alpha=alpha,beta=beta,ground_coefficient=expected(full,c),
            outer_ground_coefficient=expected(outer,c),internal_and_cross_ground_coefficient=expected(full-outer,c),
            pure_D_internal_coefficient=expected(internal,c),additional_mixed_internal_coefficient=expected(full-outer-internal,c),
            H2_operator_norm=float(np.linalg.norm(full,2)),outer_H2_norm=float(np.linalg.norm(outer,2)),local_checks=[])
        for h in (.02,.01,.005):
            u,_,_,_,_=matrix_kernel(name,h,1)
            exact=expm(-1j*h*model)
            z=np.vdot(c['state'],u@c['state']);zref=np.vdot(c['state'],exact@c['state'])
            row['local_checks'].append(dict(h=h,energy_phase_coefficient=-float(np.angle(z*np.conj(zref)))/(h**3),
                operator_cubic_error=float(np.linalg.norm(u-exact,2))/(h**3)))
        row['exact_D_checks']=[]
        for h in (.02,.01):
            u,_,_,_,_=matrix_kernel('old_endpoint' if name=='registered_S2' else name,h,1,exact_D=True)
            exact=expm(-1j*h*model)
            z=np.vdot(c['state'],u@c['state']);zref=np.vdot(c['state'],exact@c['state'])
            row['exact_D_checks'].append(dict(h=h,energy_phase_coefficient=-float(np.angle(z*np.conj(zref)))/(h**3)))
        report['candidates'].append(row)
    # Full native-state and finite-RTE identities, including scalar phases.
    identities=[]
    for h in (.02,.04):
        for rate in (1,4):
            old=old_kernel(c,'S2',h,3,rate)
            end=matrix_kernel('old_endpoint',h,rate)
            refined=old_kernel(c,'S2',h/2,3,rate)
            mixed=matrix_kernel('refinement_control',h,rate)
            errors=dict(endpoint_ideal=float(np.max(abs(old[0]-end[0]))),endpoint_finite=float(np.max(abs(old[1]-end[1]))),
                refined_ideal=float(np.max(abs(mixed[0]-refined[0]@refined[0]))),
                refined_finite=float(np.max(abs(mixed[1]-refined[1]@refined[1]))),normalizer=abs(mixed[2]-2*refined[2]))
            assert max(errors.values())<1e-11
            identities.append(dict(h=h,rate=rate,errors=errors))
    report['identity_checks']=identities
    OUT.mkdir(parents=True,exist_ok=True);np.savez_compressed(OUT/'BCH_OPERATORS.npz',**raw)
    save_json(OUT/'THEORY_AND_LOCAL.json',report);return report


def signal_task(task, rounds=SCREEN_ROUNDS, output_subdir="screen", fair_counts=False):
    name,proposed_h=task;c=common();rows=[];ld=12 if name.startswith('full_') else 3
    for m in rounds:
        time_=.02*(1<<m);steps=max(1,math.ceil(time_/proposed_h));h=time_/steps
        for rate in ((0,) if ld==12 else RATES):
            u,f,lb,le,recipe=matrix_kernel(name,h,rate)
            ideal=np.linalg.matrix_power(u,steps);finite=np.linalg.matrix_power(f,steps)
            z=complex(np.vdot(c['state'],ideal@c['state']));zf=complex(np.vdot(c['state'],finite@c['state']))
            target=np.exp(-1j*c['energy']*time_)
            pf=abs(float(np.angle(z*np.conj(target))));finite_phase=abs(float(np.angle(zf*np.conj(target))))
            seq=sequence(name,h,ld);count=len(seq);defect=float(np.linalg.norm(u.conj().T@u-np.eye(70),2))
            numerical=1e-10+steps*(defect+np.finfo(float).eps*70*count*4)
            input_disk=time_*(9.480991703075102e-15+c['preps'][ld].threshold_operator_error_bound)+2*1.9069976415519017e-15/.23263021512018023
            rte_error=math.expm1(steps*le) if steps*le<700 else None
            other=None if rte_error is None else rte_error+numerical+input_disk
            logb=steps*lb;atten=math.exp(-logb) if logb<745 else 0.
            rho=0. if other is None else max(0.,abs(z)-other)*atten
            beta=None if other is None or other>=abs(z) else .4-pf-math.asin(other/abs(z))
            feasible=bool(beta is not None and beta>0 and rho>1e-100)
            shots=math.ceil(4*math.log(1440)/(rho*rho*math.sin(beta)**2)) if feasible else None
            calls_D=steps*sum(kind=='D' for kind,_,_ in seq)
            if fair_counts and seq[0][:2]==seq[-1][:2] and seq[0][0]=='D':calls_D-=steps-1
            events=steps*sum(x['r'] for x in recipe)
            rows.append(dict(name=name,requested_h=proposed_h,h=h,m=m,t=time_,steps=steps,rate=rate,
                PF_phase=pf,finite_phase=finite_phase,ideal_radius=abs(z),corrected_finite_radius=abs(zf),
                raw_signal=dict(real=zf.real*atten,imag=zf.imag*atten),log_normalizer=logb,rho_lower=rho,beta_stat=beta,
                shots_per_axis=shots,feasible=feasible,finite_error_upper=rte_error,numerical_disk=numerical,input_disk=input_disk,
                deterministic_exponentials=calls_D,RTE_events=events,RTE_occurrences=steps*len(recipe),
                primitive_shot_surrogate=None if shots is None else 2*shots*(calls_D+events),
                qualification='conditional float64 signal screen; primitive surrogate is not compiled price; compiler disk pending'))
    save_json(OUT/output_subdir/f'{name}_{proposed_h}.json',dict(task=task,rows=rows));return rows


def run():
    OUT.mkdir(parents=True,exist_ok=False)
    cache=Path(os.environ['TRACK_C_H4_COMMON_CACHE'])
    assert hashlib.sha256(cache.read_bytes()).hexdigest()==CACHE_SHA
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    paths=['src/trottertracks/track_c_mixed_formula.py','src/trottertracks/track_c_h4.py','src/trotterlib/rte.py']
    hashes={}
    for path in paths:
        raw=(ROOT/path).read_bytes()
        assert raw==subprocess.check_output(['git','show',source+':'+path],cwd=ROOT),path
        hashes[path]=hashlib.sha256(raw).hexdigest()
    start=time.monotonic();theory=local_analysis()
    names=['registered_S2','mixed_2MN','mixed_zero_D','mixed_zero_R','refinement_control','partial_Y4','full_S2','full_Y4']
    tasks=[(name,h) for name in names for h in H_VALUES];rows=[]
    with ProcessPoolExecutor(max_workers=min(len(tasks),len(os.sched_getaffinity(0)))) as pool:
        for f in as_completed([pool.submit(signal_task,t) for t in tasks]):rows.extend(f.result())
    rows.sort(key=lambda x:(x['name'],x['requested_h'],x['m'],x['rate']))
    save_json(OUT/'SCREEN.json',dict(rows=rows))
    choices=[]
    for name in names:
        for m in SCREEN_ROUNDS:
            eligible=[x for x in rows if x['name']==name and x['m']==m and x['feasible']]
            choices.append(dict(name=name,m=m,best=None if not eligible else min(eligible,key=lambda x:x['primitive_shot_surrogate'])))
    assert hashlib.sha256(cache.read_bytes()).hexdigest()==CACHE_SHA
    for path,h in hashes.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h,path
    save_json(OUT/'SCREEN_SUMMARY.json',dict(status='SCREEN_COMPLETE_NOT_FINAL_STOP',source_commit=source,input_cache_sha256=CACHE_SHA,
        source_sha256=hashes,workers=min(len(tasks),len(os.sched_getaffinity(0))),
        inherited_base='ce84fa8e2d2f1d487df3f05e63d519ebb03f83ad',families=names,step_menu=list(H_VALUES),rate_menu=list(RATES),rounds=list(SCREEN_ROUNDS),
        choices=choices,local_coefficients=theory['candidates'],wall_seconds=time.monotonic()-start,
        new_SCF=0,new_DF_fit=0,new_ground_solve=0,price_MC=0,compiled_prices=0,
        qualification='Mechanism screen. Continue only non-equivalent scientifically promising candidates; no RPE winner from primitive surrogate.'))
    print('SCREEN_COMPLETE',len(rows),flush=True)
