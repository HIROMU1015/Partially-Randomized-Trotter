"""Limited paired reference-A pricing; no long-circuit model is assumed.

All three positive-coefficient families share the same R times, event counts,
flags and component draws. Exact block fusion applies to every family.
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
from .track_c_mixed_formula import OUT,ROOT,CACHE_SHA,sequence,compact,common,save_json
from .track_c_h4 import finite_rte_distribution
from .track_c_h4_price import conditional_flags,wrapper,metrics
from .track_c_compiler_a import compile_reference,PROFILE
from trotterlib.df_partial_s2 import QiskitDFPartialS2CircuitBuilder
from trotterlib.rte import _make_event
from trotterlib.df_rte_circuit import DFRTEEventSequenceCircuitRequest
from trotterlib.df_rte_qiskit import QiskitDFRTEEventCircuitBuilder

NAMES=('refinement_control','mixed_2MN','mixed_zero_D')

def event_parts(h,steps,rate,rare,seed):
    c=common();prep=c['preps'][3];n=max(1,math.ceil(h/2*rate/.02))
    dist=finite_rte_distribution(prep.exact_rte_lambda_r*h/(2*n),2)
    count=steps*2*n;p=dist.order_probabilities[1];rng=np.random.default_rng(seed)
    flags=conditional_flags([p]*count,rare,rng)
    components=prep.rte_preparation.symbolic_tail.components
    probabilities=np.array([x.probability for x in components]);probabilities/=probabilities.sum()
    parts=[];choices=[]
    for i in range(steps*2):
        events=[]
        for flag in flags[i*n:(i+1)*n]:
            indices=rng.choice(len(components),size=3 if flag else 1,p=probabilities)
            choices.append(dict(order=2 if flag else 0,indices=indices.tolist()))
            events.append(_make_event(indices,components,dist,1 if flag else 0))
        req=DFRTEEventSequenceCircuitRequest(events=tuple(events),component_specs=prep.rte_preparation.component_specs,
            controlled=True,ancilla_qubit=8,tail_id=prep.rte_preparation.symbolic_tail.tail_id,
            tail_hash=prep.rte_preparation.symbolic_tail.tail_hash,occurrence_rte_steps=n)
        parts.append(QiskitDFRTEEventCircuitBuilder(basis_registry=prep.rte_preparation.basis_registry).build_sequence(req).circuit)
    return parts,-math.expm1(count*math.log1p(-p)),choices

@lru_cache(None)
def d_block(index,t):
    prep=common()['preps'][3];q=QuantumCircuit(9)
    req=SimpleNamespace(preparation=prep,controlled=True,ancilla_qubit=8,step_time=2*t)
    QiskitDFPartialS2CircuitBuilder()._append_block(q,prep.deterministic_blocks[index],req,evolution_time=t)
    return q

def circuit_parts(name,h,steps,tails):
    seq=compact(sequence(name,h)*steps);parts=[];cursor=0
    for kind,index,t in seq:
        if kind=='D':parts.append(d_block(index,t))
        else:parts.append(tails[cursor]);cursor+=1
    assert cursor==len(tails)
    prep=common()['preps'][3];scalar=QuantumCircuit(9)
    scalar.p(-steps*h*(prep.constant_coefficient+prep.extracted_identity_coefficient),8)
    return [scalar,*parts]

def topology(q):
    value=[(x.operation.name,[q.find_bit(b).index for b in x.qubits],[q.find_bit(b).index for b in x.clbits]) for x in q.data]
    return hashlib.sha256(json.dumps(value,separators=(',',':')).encode()).hexdigest()

def check_state(source,native):
    c=common();reverse=np.array([int(f'{int(x):08b}'[::-1],2) for x in c['sector'].basis_indices])
    initial=np.zeros(512,complex);initial[reverse]=c['state']
    a=Statevector(initial.copy()).evolve(source.remove_final_measurements(inplace=False)).data.copy()
    b=Statevector(initial.copy()).evolve(native.remove_final_measurements(inplace=False)).data.copy()
    phase=np.vdot(a,b);phase/=abs(phase)
    def read(v):return float(np.vdot(v[:256],v[:256]).real-np.vdot(v[256:],v[256:]).real)
    return dict(projective_state_error=float(np.linalg.norm(a-b/phase)),source_readout=read(a),native_readout=read(b),readout_error=abs(read(a)-read(b)))

def sample(task):
    steps,rate,rare,j=task;h=.04
    key=f'n{steps}_r{rate}_rare{int(rare)}_j{j}'
    path=OUT/'paired_price'/f'{key}.json'
    if path.exists():raise RuntimeError('Duplicate sample '+key)
    seed=int.from_bytes(hashlib.sha256(('TrackC_mixed_20261108:'+key).encode()).digest()[:8],'big')
    start=time.monotonic();tails,p,choices=event_parts(h,steps,rate,rare,seed);families={}
    for name in NAMES:
        parts=circuit_parts(name,h,steps,tails);vectors=[];patterns=[];checks=[]
        for axis in ('X','Y'):
            source=wrapper(parts,axis);native=compile_reference(source)
            vectors.append(metrics(native));patterns.append(topology(native))
            if steps==1 and j==0:checks.append(dict(axis=axis,**check_state(source,native)))
            del source,native
        families[name]=dict(metrics=vectors,native_topologies=patterns,state_checks=checks)
        del parts
    reference=families['refinement_control']
    equality={name:dict(metrics=families[name]['metrics']==reference['metrics'],topology=families[name]['native_topologies']==reference['native_topologies']) for name in NAMES[1:]}
    record=dict(task=task,h=h,seed=seed,probability=p if rare else 1-p,rare_probability=p,event_choices=choices,
        families=families,paired_equality=equality,wall_seconds=time.monotonic()-start,process_peak_RSS_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    save_json(path,record);return record

def run():
    (OUT/'paired_price').mkdir(exist_ok=False)
    cache=Path(os.environ['TRACK_C_H4_COMMON_CACHE']);assert hashlib.sha256(cache.read_bytes()).hexdigest()==CACHE_SHA
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    relative=str(Path(__file__).relative_to(ROOT));assert (ROOT/relative).read_bytes()==subprocess.check_output(['git','show',source+':'+relative],cwd=ROOT)
    tasks=[(steps,rate,rare,j) for steps in (1,2,8) for rate in (1,32) for rare in (False,True) for j in range(8)]
    workers=min(len(tasks),len(os.sched_getaffinity(0)));records=[];start=time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(sample,t) for t in tasks]):records.append(f.result())
    cells=[]
    for steps in (1,2,8):
        for rate in (1,32):
            mean=np.zeros((3,2,4));covariance=np.zeros((24,24));strata=[]
            for rare in (False,True):
                group=[r for r in records if r['task'][:3]==(steps,rate,rare)]
                p=group[0]['probability'];assert all(r['probability']==p for r in group)
                values=np.array([[r['families'][n]['metrics'] for n in NAMES] for r in group],float)
                mean+=p*values.mean(axis=0)
                cov=np.cov(values.reshape(len(group),24),rowvar=False,ddof=1)/len(group)
                covariance+=p*p*cov
                strata.append(dict(rare=rare,probability=p,n=len(group),mean=values.mean(axis=0).tolist()))
            diff=[]
            for k in (1,2):
                diff.append(dict(name=NAMES[k],mean=(mean[k]-mean[0]).tolist(),
                    paired_SE=[[math.sqrt(max(0.,covariance[k*8+i,k*8+i]+covariance[i,i]-2*covariance[k*8+i,i])) for i in range(axis*4,axis*4+4)] for axis in (0,1)]))
            cells.append(dict(steps=steps,h=.04,rate=rate,role='independent length holdout' if steps==8 else 'short anchor',
                mean=mean.tolist(),covariance_of_mean=covariance.tolist(),strata=strata,paired_differences=diff))
    checks=[c for r in records for f in r['families'].values() for c in f['state_checks']]
    assert hashlib.sha256(cache.read_bytes()).hexdigest()==CACHE_SHA
    save_json(OUT/'PAIRED_PRICE_SUMMARY.json',dict(status='LIMITED_PRICE_COMPLETE_NO_LONG_MODEL',source_commit=source,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_cache_sha256=CACHE_SHA,
        compiler=PROFILE,names=NAMES,resource_names=['RZ','CX','depth','size'],axes=['X','Y'],samples=len(records),compiled_wrappers=len(records)*6,workers=workers,
        all_paired_metrics_equal=all(v['metrics'] for r in records for v in r['paired_equality'].values()),
        all_native_topologies_equal=all(v['topology'] for r in records for v in r['paired_equality'].values()),
        maximum_projective_state_error=max(c['projective_state_error'] for c in checks),maximum_readout_error=max(c['readout_error'] for c in checks),state_checks=len(checks),
        peak_worker_RSS_kib=max(r['process_peak_RSS_kib'] for r in records),wall_seconds=time.monotonic()-start,cells=cells,
        qualification='Actual compiled full measured wrappers, controlled before lowering. Short observed equality is not an unqualified m17 absolute price model. Float64 state checks do not certify new-angle 18-round compiler accumulation.'))
    print('PAIRED_PRICE_COMPLETE',len(records),flush=True)

if __name__=='__main__':run()
