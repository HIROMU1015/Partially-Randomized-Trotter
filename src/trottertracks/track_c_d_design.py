"""BCH selection of one-fragment symmetric time redistribution.

The free variable changes placement, never the total time of any fragment.
No signal/round observation is used to tune a. Finite h and long t are holdouts.
"""
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from .track_c_d_internal import OUT,ROOT,ORDERS,schedule,analyse,common,save_json
from .track_c_mixed_formula import cubic_generator,hermitian,expected

def design(task):
    order,j=task;c=common();matrices=[hermitian(x) for x in c['block_matrices'][:4]]+[hermitian(c['tails'][3])]
    def val(a):return expected(cubic_generator(schedule(dict(kind='redistributed',order=order,fragment=j,a=float(a))),matrices),c)
    xs=np.array([0.,.25,.5]);ys=np.array([val(a) for a in xs]);coef=np.polynomial.polynomial.polyfit(xs,ys,2)
    checks=[dict(a=a,actual=val(a),polynomial=float(np.polynomial.polynomial.polyval(a,coef))) for a in (-.5,.125,1.)]
    mismatch=max(abs(x['actual']-x['polynomial']) for x in checks)
    assert mismatch<1e-10
    trimmed=coef.copy()
    while len(trimmed)>1 and abs(trimmed[-1])<1e-10*max(abs(coef)):trimmed=trimmed[:-1]
    roots=np.polynomial.polynomial.polyroots(trimmed)
    cases=[]
    for root in roots:
        if abs(root.imag)>1e-8:continue
        a=float(root.real)
        case=dict(name='redistribute_'+''.join(map(str,order))+f'_j{j}_a{a:.9g}',kind='redistributed',order=order,fragment=j,a=a)
        cases.append(dict(case=case,leading_residual=val(a),negative_D_time=2*max(0.,-a)+2*max(0.,a-.5)))
    return dict(order=order,fragment=j,polynomial_ascending=coef.tolist(),quadratic_checks=checks,mismatch=mismatch,roots=cases)

def run():
    target=OUT/'STAGE_A2.json'
    if target.exists():raise RuntimeError('Do not overwrite design')
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    tasks=[(p,j) for p in ORDERS for j in range(4)];workers=min(len(tasks),len(os.sched_getaffinity(0)));rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(design,t) for t in tasks]):rows.append(f.result())
    # Adjacent placements of j produce equivalent schedules; deduplicate literal sequence.
    roots={}
    for row in rows:
        for candidate in row['roots']:
            key=tuple((k,i,round(t,10)) for k,i,t in schedule(candidate['case']))
            roots.setdefault(key,candidate)
    candidates=sorted(roots.values(),key=lambda x:(x['negative_D_time'],abs(x['case']['a'])))
    # Cost/negative-time principle, chosen entirely from BCH: smallest negative D time.
    # Distinct singled-out fragments are retained if available, not fitted on phases.
    selected=[]
    for candidate in candidates:
        if not any(x['case']['fragment']==candidate['case']['fragment'] for x in selected):selected.append(candidate)
    details=[analyse(x['case']) for x in selected]
    save_json(target,dict(source_commit=source,workers=workers,rows=rows,unique_roots=candidates,selected=details,
        selection='One smallest-negative-D-time real root per singled-out fragment; no finite-time signal fitting.',
        positive_order_blend='All 24 unmodified S2 ground H2 coefficients are positive; positive-time compositions cannot cancel their cubic weighted sum.',
        qualification='State/input-specific leading BCH cancellation, not a fourth-order operator formula. Full finite-RTE and held-out time checks required.'))
    print('STAGE_A2_COMPLETE',len(candidates),len(details),flush=True)

if __name__=='__main__':run()
