"""Limited step/resolution joint checks; canonical K2 RTE law unchanged.

The saved grid already covers the D steps. Add only resolution 64 at m17,
where the resource decomposition and r*exp(const/r) tradeoff warrant it.
"""
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import replace
from functools import lru_cache
import hashlib
import json
import math
import os
import subprocess
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from .track_c_bottleneck import OUT,ROOT,PRIOR,saved,identity
from .track_c_bottleneck_basis import partial_ops,circuit
from .track_c_bottleneck_runwise import build_singletons
from .track_c_d_resources import r_part,metrics,r_key
from .track_c_d_internal import kernel
from .track_c_h4 import common,diagonal_primitives,save_json
from .track_c_compiler_a import compile_reference
from trotterlib.df_trotter.ops import append_diagonal_primitives

def diag_values(p):
    states=np.arange(256);phase=np.full(256,p.global_phase,float)
    for i,a in p.rz:phase-=a*(1-2*((states>>i)&1))/2
    for i,j,a in p.rzz:phase-=a*(1-2*((states>>i)&1))*(1-2*((states>>j)&1))/2
    return np.exp(1j*phase)

def full_probe(index):
    block=common()['preps'][12].deterministic_blocks[index];p=diagonal_primitives(block,.02)
    support=sorted({i for i,a in p.rz if a!=0}|{i for i,j,a in p.rzz if a!=0}|{j for i,j,a in p.rzz if a!=0})
    oldops=block.runtime_basis_operations;b=Operator(circuit(oldops)).data;single=b[np.ix_([1<<i for i in range(8)],[1<<i for i in range(8)])]/b[0,0]
    newops=partial_ops(single,support);new=Operator(circuit(newops)).data;mapping={i:j for j,i in enumerate(support)}
    mapped=replace(p,rz=tuple((mapping[i],a) for i,a in p.rz),rzz=tuple((mapping[i],mapping[j],a) for i,j,a in p.rzz if a!=0))
    target=(b*diag_values(p))@b.conj().T;actual=(new*diag_values(mapped))@new.conj().T;err=float(np.linalg.norm(target-actual,'fro'));assert err<1e-10
    counts=[]
    for ops,diagonal in ((oldops,p),(newops,mapped)):
        q=QuantumCircuit(9);q.compose(circuit(ops,9,True),inplace=True);append_diagonal_primitives(q,diagonal,controlled=True,ancilla_qubit=8);q.compose(circuit(ops,9),inplace=True)
        counts.append(metrics(compile_reference(q)))
    return dict(block=index,active_columns=support,old=counts[0],partial_QR=counts[1],operator_error=err)

def new_signal(base,rate):
    x=dict(base);case=x['case'];m=x['m'];h=x['h'];steps=x['steps'];c=common()
    _,f,lb,le,recipe=kernel(case,h,rate);zf=np.vdot(c['state'],np.linalg.matrix_power(f,steps)@c['state'])
    rte=math.expm1(steps*le);other=rte+x['input_disk']+x['numerical_disk'];rho=max(0.,x['ideal_radius']-other)*math.exp(-steps*lb)
    beta=.4-x['PF_phase']-math.asin(other/x['ideal_radius'])
    x.update(rate=rate,recipe=recipe,finite_error_upper=rte,rho=rho,beta_stat=beta,log_normalizer=steps*lb,raw=dict(real=float(zf.real)*math.exp(-steps*lb),imag=float(zf.imag)*math.exp(-steps*lb)),
        finite_phase=abs(float(np.angle(zf*np.exp(1j*c['energy']*x['t'])))),RTE_events=steps*sum(r['r'] for r in recipe),feasible=bool(beta>0 and rho>0),qualification='New resolution diagnostic, native/source certificate pending.')
    return x

def price_new(task):
    t,n,rare,j=task;seed=int.from_bytes(hashlib.sha256(f'TrackC_bottleneck_joint:{t}:{n}:{rare}:{j}'.encode()).digest()[:8],'big')
    q,p,choices=r_part(t,n,rare,seed);old=metrics(compile_reference(q));proposal,_=build_singletons(t,n,choices);new=metrics(compile_reference(proposal));which='singletons' if new[0]<old[0] else 'full'
    r=dict(t=t,n=n,rare=rare,j=j,seed=seed,probability=p if rare else 1-p,choices=choices,old=old,proposal=new,selected=new if which=='singletons' else old,choice=which)
    save_json(OUT/'joint_prices'/f'{r_key(t,n)}_{int(rare)}_{j}.json',r);return r

def run():
    if (OUT/'JOINT_CANDIDATES.json').exists():raise RuntimeError('Do not overwrite joint candidates')
    (OUT/'joint_prices').mkdir(exist_ok=False);raw=json.loads((PRIOR/'SIGNALS.json').read_text())['rows'];cases={x['name']:x['case'] for x in raw};new=[]
    for name in ('tuned_0123','tuned_0321'):
        for h in (.16,.32):
            base=next(x for x in raw if x['name']==name and x['m']==17 and x['requested_h']==h and x['rate']==32);new.append(new_signal(base,64))
    cells={(r['time'],r['r']) for x in new for r in x['recipe']};cells.add((.32,512))
    workers=len(os.sched_getaffinity(0));probes=[];prices=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for f in as_completed(pool.submit(full_probe,j) for j in range(13)):probes.append(f.result())
        for f in as_completed(pool.submit(price_new,(*cell,rare,j)) for cell in cells for rare in (False,True) for j in range(16)):prices.append(f.result())
    save_json(OUT/'FULL_PARTIAL_QR_PROBE.json',dict(rows=probes,interpretation='Same partial-column construction offered to deterministic blocks; exact nonzero active columns only, no pruning or changed Hamiltonian. New representation is not selected without native/source qualification.'))
    save_json(OUT/'JOINT_CANDIDATES.json',dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),new_signals=new,price_cells=sorted(cells),new_price_samples=len(prices),
        boundary='Only the existing RTE resolution parameter is extended at m17. K2 order/probability formula, component law and return aggregation unchanged. Finite m17 signals; no wrapped-energy tuning.'))
    print('JOINT_CANDIDATES_COMPLETE',flush=True)

if __name__=='__main__':run()
