"""Full measured-wrapper Q prices under the registered conditional strata."""
from __future__ import annotations
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PhaseGate
from .track_c_h4 import *
from trotterlib.rte import _make_event
from trotterlib.df_rte_circuit import DFRTEEventSequenceCircuitRequest
from trotterlib.df_rte_qiskit import QiskitDFRTEEventCircuitBuilder


def conditional_flags(probabilities,rare,rng):
    p=np.asarray(probabilities,float)
    if not rare:return np.zeros(len(p),bool)
    logprefix=np.r_[0.,np.cumsum(np.log1p(-p))[:-1]]
    mass=-math.expm1(float(np.log1p(-p).sum()))
    if mass<=0:raise ValueError('requested zero-probability rare stratum')
    firstprob=np.exp(logprefix)*p/mass
    firstprob/=firstprob.sum()
    first=int(rng.choice(len(p),p=firstprob))
    flags=np.zeros(len(p),bool);flags[first]=True
    flags[first+1:]=rng.random(len(p)-first-1)<p[first+1:]
    return flags


@lru_cache(None)
def d_part(ld,time,reverse=False,processing=False,controlled=True):
    c=common();prep=c['preps'][ld];builder=QiskitDFPartialS2CircuitBuilder()
    qc=QuantumCircuit(9 if controlled else 8)
    req=SimpleNamespace(preparation=prep,controlled=controlled,ancilla_qubit=8,step_time=2*time)
    seq=cp_d_sequence(ld+1,time) if processing else [(i,time) for i in (reversed(range(ld+1)) if reverse else range(ld+1))]
    for i,t in seq:builder._append_block(qc,prep.deterministic_blocks[i],req,evolution_time=t)
    return qc


@lru_cache(None)
def processor_part():
    c=common();prep=c['preps'][3];qc=QuantumCircuit(8)
    builder=QiskitDFPartialS2CircuitBuilder()
    req=SimpleNamespace(preparation=prep,controlled=False,ancilla_qubit=None,step_time=.02)
    for i,t in cp_d_sequence(4,.01,processor=True):
        builder._append_block(qc,prep.deterministic_blocks[i],req,evolution_time=t)
    return qc.inverse()


def native_compile(qc):
    return transpile(qc,basis_gates=['rz','sx','x','cx'],optimization_level=2,seed_transpiler=17,num_processes=1)


def metrics(qc):
    counts=qc.count_ops()
    return [int(counts.get('rz',0)),int(counts.get('cx',0)),int(qc.depth()),int(qc.size())]


@lru_cache(None)
def d_price(ld,time,reverse,processing):
    return np.array(metrics(native_compile(d_part(ld,time,reverse,processing))),float)


@lru_cache(None)
def forward_part(ld,time,processing):
    prep=common()['preps'][ld]
    qc=d_part(ld,time/2,False,processing).copy()
    scalar=-time*(prep.constant_coefficient+prep.extracted_identity_coefficient)
    if scalar:qc.append(PhaseGate(scalar),[8])
    return qc,np.array(metrics(native_compile(qc)),float)


def trajectory_parts(task,q,r,rare,seed,*,d_override=None,tail_override=None,price_additive=True):
    c=common();ld=task.get('ld',3);h=task.get('h',.02);prep=c['preps'][ld]
    processing=task['pilot']=='C-P';stages=[]
    for _ in range(round(q*.02/h)):
        for w in weights(task.get('formula','S2')):
            t=float(w*h);rj=max(1,math.ceil(abs(t)*r/.02)) if ld==3 else 0
            dist=None if ld==12 else finite_rte_distribution(prep.exact_rte_lambda_r*t/rj,2)
            stages.append((t,rj,dist))
    rng=np.random.default_rng(seed)
    ps=[dist.order_probabilities[1] for _,n,dist in stages for _ in range(n)]
    flags=conditional_flags(ps,rare,rng) if ps else []
    cursor=0;parts=[];additive=np.zeros(4)
    if task.get('P'):
        p=QuantumCircuit(9);p.compose(processor_part(),qubits=range(8),inplace=True)
        parts.append(p);additive+=metrics(native_compile(p))
    for t,n,dist in stages:
        if d_override is None:forward,forward_metrics=forward_part(ld,t,processing)
        else:
            forward=d_override(t/2,False).copy()
            scalar=-t*(prep.constant_coefficient+prep.extracted_identity_coefficient)
            if scalar:forward.append(PhaseGate(scalar),[8])
            forward_metrics=np.zeros(4)
        parts.append(forward)
        # Price the actual whole D run including its scalar phase.
        additive+=forward_metrics
        if n:
            components=prep.rte_preparation.symbolic_tail.components
            probabilities=np.array([x.probability for x in components]);probabilities/=probabilities.sum()
            events=[]
            for flag in flags[cursor:cursor+n]:
                order=2 if flag else 0
                indices=rng.choice(len(components),size=order+1,p=probabilities)
                events.append(_make_event(indices,components,dist,1 if flag else 0))
            cursor+=n
            request=DFRTEEventSequenceCircuitRequest(events=tuple(events),
                component_specs=prep.rte_preparation.component_specs,controlled=True,ancilla_qubit=8,
                tail_id=prep.rte_preparation.symbolic_tail.tail_id,tail_hash=prep.rte_preparation.symbolic_tail.tail_hash,
                occurrence_rte_steps=n)
            factory=QiskitDFRTEEventCircuitBuilder if tail_override is None else tail_override
            tail=factory(basis_registry=prep.rte_preparation.basis_registry).build_sequence(request).circuit
            parts.append(tail)
            if price_additive:additive+=metrics(native_compile(tail))
        reverse=d_part(ld,t/2,True,processing) if d_override is None else d_override(t/2,True)
        parts.append(reverse)
        if price_additive:additive+=d_price(ld,t/2,True,processing)
    probability=float(-math.expm1(math.fsum(math.log1p(-p) for p in ps))) if ps else 0.
    return parts,additive,probability


def wrapper(parts,axis):
    qc=QuantumCircuit(9,1);qc.h(8)
    for part in parts:qc.compose(part,qubits=range(9),inplace=True)
    if axis=='Y':qc.sdg(8)
    qc.h(8);qc.measure(8,0)
    return qc


def price_task(request):
    task=request['task'];q=request['q'];r=request['r'];rare=request['rare'];n=request['n']
    key=request['key'];path=OUT/'prices'/f'{key}.json'
    if path.exists():return key,'saved'
    vectors=[];add=[];p_rare=None
    for j in range(n):
        pair_group='CP_shared' if task['pilot']=='C-P' else task['name']
        seed=int.from_bytes(hashlib.sha256(f'{pair_group}:{r}:{q}:{rare}:{j}'.encode()).digest()[:8],'big')
        parts,a,p_rare=trajectory_parts(task,q,r,rare,seed)
        sample=[];asample=[]
        for axis in ('X','Y'):
            qc=wrapper(parts,axis);sample.append(metrics(native_compile(qc)))
            empty=wrapper([],axis);overhead=np.array(metrics(native_compile(empty)),float)
            asample.append((a+overhead).tolist())
        vectors.append(sample);add.append(asample)
        if task['ld']==12:
            # Deterministic exact expectation: all nominal samples are identical.
            break
    v=np.array(vectors,float);a=np.array(add,float)
    save_json(path,dict(request=request,resource_names=list(METRICS),full_samples=v.tolist(),
        strong_additive_samples=a.tolist(),probability=p_rare if rare else 1-p_rare,
        mean=v.mean(axis=0).tolist(),strong_additive_mean=a.mean(axis=0).tolist(),
        samples_actual=len(v),deterministic_exact=task['ld']==12,
        compiler=dict(qiskit='1.3.0',basis=['rz','sx','x','cx'],opt=2,seed=17),
        status='COMPLETE'))
    return key,'complete'


def price_tasks():
    tasks=[]
    for task in signal_tasks():
        data=json.loads((OUT/'signals'/f"{task['name']}.json").read_text())
        if not data['all_rounds_feasible']:continue
        for r in ([0] if task['ld']==12 else sorted(set(RVALUES))):
            for q,n in ((1,32),(2,32),(8,128)):
                for rare in ((False,) if task['ld']==12 else (False,True)):
                    key=f"{task['name']}_r{r}_q{q}_rare{int(rare)}"
                    tasks.append(dict(task=task,r=r,q=q,rare=rare,n=n,key=key))
    return tasks


@lru_cache(None)
def wrapper_correction(axis):
    prep=QuantumCircuit(9);prep.h(8)
    read=QuantumCircuit(9,1)
    if axis=='Y':read.sdg(8)
    read.h(8);read.measure(8,0)
    return (np.array(metrics(native_compile(prep)))+np.array(metrics(native_compile(read)))
            -np.array(metrics(native_compile(wrapper([],axis)))))


@lru_cache(None)
def d_link_correction(previous_time,next_time):
    reverse=d_part(3,previous_time/2,True,False)
    forward,fprice=forward_part(3,next_time,False)
    combined=reverse.compose(forward)
    vector=(np.array(metrics(native_compile(combined)),float)
            -d_price(3,previous_time/2,True,False)-fprice)
    key=hashlib.sha256(f'{previous_time.hex()}:{next_time.hex()}'.encode()).hexdigest()
    save_json(OUT/'strong_D_runs'/f'{key}.json',dict(previous_time=previous_time,next_time=next_time,
        resource_names=list(METRICS),correction=vector.tolist(),policy='maximal D run between actual R occurrences',
        compiler='Qiskit1.3.0 opt2 rz/sx/x/cx seed17'))
    return vector


def strong_correction(task,q):
    from collections import Counter
    if task['ld']==12:return None
    times=[float(w*task['h']) for _ in range(round(q*.02/task['h'])) for w in weights(task['formula'])]
    pairs=Counter(zip(times[:-1],times[1:]));delta=np.zeros(4)
    for (left,right),count in pairs.items():delta+=count*d_link_correction(left,right)
    return np.array([delta+wrapper_correction(axis) for axis in ('X','Y')])


def strong_correction_task(task):
    result={str(q):strong_correction(task,q).tolist() for q in (1,2,8)}
    save_json(OUT/'strong_D_runs'/f"{task['name']}_CORRECTION.json",result)
    return task['name'],'maximal-D correction complete'
