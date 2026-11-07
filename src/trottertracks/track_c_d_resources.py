"""Actual modular compiler-A implementation and direct occurrence MC.

RZ/CX/size add exactly for the concatenated native implementation. This is
not an extrapolation of a whole-compile fit. Serial depth is an upper bound,
not the depth of its full DAG. Short whole-compile comparisons are separate.
"""
from concurrent.futures import ProcessPoolExecutor,as_completed
from functools import lru_cache
from types import SimpleNamespace
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import time
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from .track_c_d_internal import ROOT,OUT,common,schedule,save_json
from .track_c_mixed_formula import compact
from .track_c_h4 import finite_rte_distribution
from .track_c_h4_price import conditional_flags,wrapper,metrics
from .track_c_compiler_a import compile_reference,PROFILE
from trotterlib.df_partial_s2 import QiskitDFPartialS2CircuitBuilder
from trotterlib.df_rte_circuit import DFRTEEventSequenceCircuitRequest
from trotterlib.df_rte_qiskit import QiskitDFRTEEventCircuitBuilder
from trotterlib.rte import _make_event

SAMPLES=16

@lru_cache(None)
def d_piece(seq,ld):
    prep=common()['preps'][ld];q=QuantumCircuit(9);req=SimpleNamespace(preparation=prep,controlled=True,ancilla_qubit=8,step_time=1.)
    builder=QiskitDFPartialS2CircuitBuilder()
    for kind,i,t in seq:
        assert kind=='D';builder._append_block(q,prep.deterministic_blocks[i],req,evolution_time=t)
    return q

def split(case,h):
    seq=schedule(case,h)
    if case['kind'].startswith('full_'):
        k,i,t=seq[0];assert seq[-1][:2]==seq[0][:2] and seq[-1][2]==t
        return dict(prefix=seq[:1],suffix=seq[1:],middle=seq[1:-1]+[(k,i,2*t)],full=True)
    groups=[[]]
    for item in seq:
        if item[0]=='R':groups.append([])
        else:groups[-1].append(item)
    return dict(first=groups[0],last=groups[-1],junction=compact(groups[-1]+groups[0]),inside=[x for g in groups[1:-1] for x in g],full=False)

def overhead(ld,t,axis):
    prep=common()['preps'][ld];p=QuantumCircuit(9);p.p(-t*(prep.constant_coefficient+prep.extracted_identity_coefficient),8)
    return compile_reference(wrapper([p],axis))

def det_task(task):
    case,h=task;ld=12 if case['kind'].startswith('full_') else 3;groups=split(case,h);vectors={}
    for name,seq in groups.items():
        if name=='full':continue
        if not seq:vectors[name]=[0,0,0,0];continue
        vectors[name]=metrics(compile_reference(d_piece(tuple(seq),ld)))
    overheads={axis:metrics(overhead(ld,.02,axis)) for axis in ('X','Y')}
    row=dict(case=case,h=h,groups=vectors,overhead=overheads,policy='No subsequent pass across compiled pieces; actual executable modular circuit, not whole-compile price.')
    save_json(OUT/'det_price'/f'{case["name"]}_{h}.json',row);return row

def r_key(t,n):return f'{float(t).hex()}_{n}'

def r_part(t,n,rare,seed):
    prep=common()['preps'][3];dist=finite_rte_distribution(prep.exact_rte_lambda_r*t/n,2)
    rng=np.random.default_rng(seed);flags=conditional_flags([dist.order_probabilities[1]]*n,rare,rng)
    components=prep.rte_preparation.symbolic_tail.components;ps=np.array([x.probability for x in components]);ps/=ps.sum()
    events=[];choices=[]
    for flag in flags:
        indices=rng.choice(len(components),size=3 if flag else 1,p=ps)
        choices.append(dict(indices=indices.tolist(),order=2 if flag else 0));events.append(_make_event(indices,components,dist,1 if flag else 0))
    req=DFRTEEventSequenceCircuitRequest(events=tuple(events),component_specs=prep.rte_preparation.component_specs,controlled=True,ancilla_qubit=8,
        tail_id=prep.rte_preparation.symbolic_tail.tail_id,tail_hash=prep.rte_preparation.symbolic_tail.tail_hash,occurrence_rte_steps=n)
    q=QiskitDFRTEEventCircuitBuilder(basis_registry=prep.rte_preparation.basis_registry).build_sequence(req).circuit
    p=-math.expm1(n*math.log1p(-dist.order_probabilities[1]));return q,p,choices

def rte_task(task):
    t,n,rare,j=task
    # Shared uniforms across lengths at each dimensionless event time; valid CRN.
    group=f'{float(t/n).hex()}:{int(rare)}:{j}'
    seed=int.from_bytes(hashlib.sha256(('TrackC_D_resources:'+group).encode()).digest()[:8],'big')
    start=time.monotonic();q,p,choices=r_part(t,n,rare,seed);native=compile_reference(q)
    row=dict(t=t,n=n,rare=rare,j=j,seed=seed,probability=p if rare else 1-p,metrics=metrics(native),choices=choices,
        peak_RSS_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,wall_seconds=time.monotonic()-start)
    save_json(OUT/'rte_price'/f'{r_key(t,n)}_rare{int(rare)}_j{j}.json',row);return row

def assemble(case,h,steps,tails):
    seq=compact(schedule(case,h)*steps);q=QuantumCircuit(9);ld=12 if case['kind'].startswith('full_') else 3;cursor=0
    for kind,i,t in seq:
        if kind=='D':q.compose(d_piece(((kind,i,t),),ld),inplace=True)
        else:q.compose(tails[cursor],inplace=True);cursor+=1
    assert cursor==len(tails)
    prep=common()['preps'][ld];q.p(-steps*h*(prep.constant_coefficient+prep.extracted_identity_coefficient),8)
    return q

def modular_parts(case,h,steps,tails):
    g=split(case,h);ld=12 if g['full'] else 3;parts=[]
    def d(seq):return compile_reference(d_piece(tuple(seq),ld))
    if g['full']:parts=[d(g['prefix']),*[d(g['middle'])]*(steps-1),d(g['suffix'])]
    else:
        # All priced partial families have one R, except exact half-step control.
        seq=schedule(case,h);groups=[[]]
        for x in seq:
            if x[0]=='R':groups.append([])
            else:groups[-1].append(x)
        parts=[d(groups[0])];cursor=0
        for step in range(steps):
            for j in range(len(groups)-1):
                parts.append(compile_reference(tails[cursor]));cursor+=1
                next_d=groups[j+1]
                if j==len(groups)-2 and step<steps-1:next_d=compact(next_d+groups[0])
                parts.append(d(next_d))
    return parts

def whole_check(task):
    case,h,steps=task;ld=12 if case['kind'].startswith('full_') else 3;tails=[]
    for cycle in range(steps):
        for kind,_,t in schedule(case,h):
            if kind=='R':
                n=max(1,math.ceil(abs(t)*32/.02));tails.append(r_part(t,n,cycle==0,724712+cycle)[0])
    source=assemble(case,h,steps,tails);parts=modular_parts(case,h,steps,tails);result=[]
    prep=common()['preps'][ld];scalar=QuantumCircuit(9);scalar.p(-steps*h*(prep.constant_coefficient+prep.extracted_identity_coefficient),8)
    # Compile measured wrappers around already-native core, without reoptimizing it.
    pre=QuantumCircuit(9);pre.h(8);native_pre=compile_reference(pre)
    c=common();rev=np.array([int(f'{int(x):08b}'[::-1],2) for x in c['sector'].basis_indices]);initial=np.zeros(512,complex);initial[rev]=c['state']
    for axis in ('X','Y'):
        full=compile_reference(wrapper([source],axis))
        read=QuantumCircuit(9,1);read.compose(scalar,inplace=True)
        if axis=='Y':read.sdg(8)
        read.h(8);read.measure(8,0);native_read=compile_reference(read)
        modular=QuantumCircuit(9,1);modular.compose(native_pre,inplace=True)
        for part in parts:modular.compose(part,inplace=True)
        modular.compose(native_read,inplace=True)
        check=None
        if steps==1:
            a=Statevector(initial.copy()).evolve(wrapper([source],axis).remove_final_measurements(inplace=False)).data.copy()
            b=Statevector(initial.copy()).evolve(modular.remove_final_measurements(inplace=False)).data.copy()
            phase=np.vdot(a,b);phase/=abs(phase)
            readout=lambda v:float(np.vdot(v[:256],v[:256]).real-np.vdot(v[256:],v[256:]).real)
            check=dict(projective_error=float(np.linalg.norm(a-b/phase)),readout_error=abs(readout(a)-readout(b)))
        result.append(dict(axis=axis,whole=metrics(full),modular=metrics(modular),serial_depth_upper=sum(q.depth() for q in [native_pre,*parts,native_read]),state_check=check))
    row=dict(case=case,h=h,steps=steps,results=result)
    save_json(OUT/'whole_checks'/f'{case["name"]}_{steps}.json',row);return row

def run():
    for name in ('det_price','rte_price','whole_checks'):(OUT/name).mkdir(exist_ok=False)
    from .track_c_d_account import choices as get_choices
    choices=get_choices()
    cases={x['name']:x['case'] for x in choices};det_tasks={(x['name'],x['h']):(x['case'],x['h']) for x in choices}
    rte_cells={(r_key(r['time'],r['r'])):(r['time'],r['r']) for x in choices for r in x['recipe']}
    workers=min(len(os.sched_getaffinity(0)),max(len(det_tasks),len(rte_cells)*2*SAMPLES));start=time.monotonic();dets=[];rs=[];ws=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(det_task,t) for t in det_tasks.values()]):dets.append(f.result())
        tasks=[(*x,rare,j) for x in rte_cells.values() for rare in (False,True) for j in range(SAMPLES)]
        for f in as_completed([pool.submit(rte_task,t) for t in tasks]):rs.append(f.result())
        checks=[(cases[name],.02 if name in ('original','order_0321') else .16,steps) for name in ('original','order_0321','tuned_0123','tuned_0321','full_S2','full_Y4') for steps in (1,2,8)]
        for f in as_completed([pool.submit(whole_check,t) for t in checks]):ws.append(f.result())
    save_json(OUT/'PRICE_RAW_SUMMARY.json',dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),compiler=PROFILE,
        samples_per_stratum=SAMPLES,workers=workers,rte_cells=list(rte_cells.values()),deterministic_cells=len(dets),RTE_samples=len(rs),whole_checks=len(ws),wall_seconds=time.monotonic()-start,
        max_worker_RSS_kib=max(x['peak_RSS_kib'] for x in rs),metrics=['RZ','CX','serial_depth_upper','size'],
        interpretation='Exact additive count formula of an actual modular A circuit. No estimated whole-circuit transfer. Full-DAG depth is not summed depth. MC estimates directly price every used occurrence length/law.'))
    print('RESOURCES_COMPLETE',len(rs),flush=True)

if __name__=='__main__':run()
