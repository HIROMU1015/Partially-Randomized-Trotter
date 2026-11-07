"""D-internal PF experiment at the unchanged Track C H4 input.

Stage A enumerates all 24 orders, rather than fitting a selected time signal.
All output is new. Existing STOPs and preparation functions are untouched.
"""
from concurrent.futures import ProcessPoolExecutor,as_completed
from functools import lru_cache
import hashlib
import itertools
import math
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from scipy.linalg import expm
from .track_c_h4 import ROOT,common,evolution,tail_operator,weights,save_json
from .track_c_mixed_formula import compact,cubic_generator,coefficients,comm,hermitian,expected,CACHE_SHA

OUT=ROOT/'artifacts/track_c_d_internal/2026-11-08'
ORDERS=tuple(itertools.permutations(range(4)))

def s2(order,h=1.,tail=True):
    return compact([('D',i,h/2) for i in order]+([('R',-1,h)] if tail else [])+[('D',i,h/2) for i in reversed(order)])

def d_symmetric(order,h):return s2(order,h,tail=False)

def schedule(case,h=1.):
    order=tuple(case.get('order',range(4)));kind=case['kind']
    if kind=='ordered':return s2(order,h)
    if kind=='D2':d=d_symmetric(order,h/2)
    elif kind in ('D4','Dtuned'):
        ws=weights('Y4') if kind=='D4' else [case['v'],1-2*case['v'],case['v']]
        d=compact([x for w in ws for x in d_symmetric(order,w*h/2)])
    elif kind=='redistributed':
        j=case['fragment'];a=case['a'];rest=[i for i in order if i!=j]
        left=[('D',j,a*h)]+[('D',i,h/2) for i in rest]+[('D',j,(.5-a)*h)]
        return compact(left+[('R',-1,h)]+list(reversed(left)))
    elif kind=='composed':
        w=case['w'];p=tuple(case['p']);q=tuple(case['q'])
        return compact(s2(p,w*h)+s2(q,(1-2*w)*h)+s2(p,w*h))
    elif kind=='partial_Y4':return compact([x for w in weights('Y4') for x in s2(order,w*h)])
    elif kind=='half_control':return compact(s2(order,h/2)*2)
    elif kind in ('full_S2','full_Y4'):
        order=tuple(case.get('order',range(13)))
        return s2(order,h,False) if kind=='full_S2' else compact([x for w in weights('Y4') for x in s2(order,w*h,False)])
    else:raise ValueError(kind)
    return compact(d+[('R',-1,h)]+d)

@lru_cache(None)
def cached_kernel(case_json,h,rate):
    case=__import__('json').loads(case_json);c=common();ld=12 if case['kind'].startswith('full_') else 3
    seq=schedule(case,h);u=np.eye(70,dtype=complex);f=u.copy();lb=le=0.;recipe=[]
    for kind,index,t in seq:
        if kind=='D':a=evolution(c['spectra'][index],t);b=a
        else:
            n=max(1,math.ceil(abs(t)*rate/.02));a=evolution(c['tail_spectra'][3],t)
            b,l,e=tail_operator(c,t,n);lb+=l;le+=e;recipe.append(dict(time=t,r=n))
        u=a@u;f=b@f
    prep=c['preps'][ld];scalar=prep.constant_coefficient+prep.extracted_identity_coefficient
    phase=np.exp(-1j*h*scalar)
    return phase*u,phase*f,lb,le,recipe

def kernel(case,h,rate):
    import json
    return cached_kernel(json.dumps(case,sort_keys=True),h,rate)

def analyse(case):
    c=common();matrices=[hermitian(x) for x in c['block_matrices'][:4]]+[hermitian(c['tails'][3])]
    seq=schedule(case);h2=cubic_generator(seq,matrices)
    internal=cubic_generator(seq,[*matrices[:4],np.zeros_like(matrices[-1])]);mixed=h2-internal
    d=sum(matrices[:4]);r=matrices[-1];alpha,beta=coefficients(.5);outer=-alpha*comm(d,comm(d,r))-beta*comm(r,comm(d,r))
    row=dict(case=case,H2_ground=expected(h2,c),D_internal_ground=expected(internal,c),mixed_total_ground=expected(mixed,c),
        aggregate_outer_reference_ground=expected(outer,c),additional_mixed_internal_ground=expected(mixed-outer,c),H2_norm=float(np.linalg.norm(h2,2)),
        D_exponentials=sum(k=='D' for k,_,_ in seq),R_occurrences=sum(k=='R' for k,_,_ in seq),negative_D_time=sum(abs(t) for k,_,t in seq if k=='D' and t<0),local=[])
    prep=c['preps'][3];model=sum(matrices)+(prep.constant_coefficient+prep.extracted_identity_coefficient)*np.eye(70)
    for h in (.02,.01):
        u,*_=kernel(case,h,1);z=np.vdot(c['state'],u@c['state']);target=np.exp(-1j*c['energy']*h)
        row['local'].append(dict(h=h,phase_coefficient=-float(np.angle(z*np.conj(target)))/h**3,operator_coefficient=float(np.linalg.norm(u-expm(-1j*h*model),2))/h**3))
    return row

def stage_a():
    OUT.mkdir(parents=True,exist_ok=False);cache=Path(os.environ['TRACK_C_H4_COMMON_CACHE'])
    assert hashlib.sha256(cache.read_bytes()).hexdigest()==CACHE_SHA
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    relative=str(Path(__file__).relative_to(ROOT));assert Path(__file__).read_bytes()==subprocess.check_output(['git','show',source+':'+relative],cwd=ROOT)
    cases=[dict(name='order_'+''.join(map(str,p)),kind='ordered',order=p) for p in ORDERS]
    cases += [dict(name=kind+'_0123',kind=kind,order=list(range(4))) for kind in ('D2','D4','half_control','partial_Y4')]
    start=time.monotonic();rows=[];workers=min(len(cases),len(os.sched_getaffinity(0)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(analyse,t) for t in cases]):rows.append(f.result())
    save_json(OUT/'STAGE_A.json',dict(source_commit=source,input_cache_sha256=CACHE_SHA,rows=sorted(rows,key=lambda x:abs(x['H2_ground'])),workers=workers,wall_seconds=time.monotonic()-start,
        interpretation='All orders screened on the full PR ground BCH coefficient; D-only and mixed terms are separate. No price inference.'))
    print('STAGE_A_COMPLETE',len(rows),flush=True)

if __name__=='__main__':stage_a()
