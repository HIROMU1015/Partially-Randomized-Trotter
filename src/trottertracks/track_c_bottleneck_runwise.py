"""Singleton-only partial basis, preserving source runs and all RTE draws.

The old always-partial pilot is a negative result. This is the existing WP06
singleton principle implemented with stable partial-column QR. Each stream
also retains its exact full-basis circuit as a compile-time alternative.
"""
from concurrent.futures import ProcessPoolExecutor,as_completed
from functools import lru_cache
from dataclasses import replace
import json
import math
import os
import subprocess
import numpy as np
from qiskit import QuantumCircuit
from .track_c_bottleneck import OUT,ROOT,PRIOR,PUBLIC,saved,components,NAMES
from .track_c_bottleneck_basis import configuration,circuit,profile_rte,basis_price,diagonal_price
from .track_c_d_resources import metrics,split
from .track_c_compiler_a import compile_reference
from .track_c_h4 import common,save_json,finite_rte_distribution,diagonal_primitives
from trotterlib.df_trotter.ops import append_diagonal_primitives
from trotterlib.df_rte_qiskit import QiskitDFRTEEventCircuitBuilder as Builder,_event_phase_angle
from trotterlib.rte import _make_event

@lru_cache(None)
def new_basis_price(name,inverse):
    d=configuration()[0].definition(name)
    return np.array(metrics(compile_reference(circuit(d.runtime_operations,9,inverse))),float)

def events(t,n,choices):
    p=common()['preps'][3];dist=finite_rte_distribution(p.exact_rte_lambda_r*t/n,2)
    return [_make_event(x['indices'],p.rte_preparation.symbolic_tail.components,dist,1 if x['order']==2 else 0) for x in choices]

def build_singletons(t,n,choices):
    ev=events(t,n,choices);flat=[(e,a) for e in ev for a in e.application_sequence];registry,_,lookup,_=configuration()
    q=QuantumCircuit(9);phase=sum(_event_phase_angle(e.phase) for e in ev);basis=np.zeros(4);unctrl=np.zeros(4);ctrl=np.zeros(4);selected=[];exposed=[];cursor=0
    while cursor<len(flat):
        stop=cursor+1;key=flat[cursor][1].basis_id
        while stop<len(flat) and flat[stop][1].basis_id==key:stop+=1
        partial=stop==cursor+1
        if partial:
            spec=lookup[flat[cursor][1].component_id];d=registry.definition(spec.basis_id);getcost=lambda inverse:new_basis_price(d.basis_id,inverse)
        else:
            d=common()['preps'][3].rte_preparation.basis_registry.definition(key);getcost=lambda inverse:basis_price(key,inverse)
        q.compose(circuit(d.runtime_operations,9,True),inplace=True);basis+=getcost(True)
        if cursor==0:exposed.append(getcost(True))
        for e,a in flat[cursor:stop]:
            used=replace(a,diagonal_pauli_support=lookup[a.component_id].diagonal_pauli_support) if partial else a
            phase+=Builder._append_central_application(q,e,used,controlled=True,ancilla_qubit=8)
            angle=2*e.unsigned_rotation_angle
            unctrl+=diagonal_price(e.taylor_order,used.role,used.diagonal_pauli_support,angle,False)
            ctrl+=diagonal_price(e.taylor_order,used.role,used.diagonal_pauli_support,angle,True)
        q.compose(circuit(d.runtime_operations,9),inplace=True);basis+=getcost(False)
        if stop==len(flat):exposed.append(getcost(False))
        selected.append(dict(start=cursor,stop=stop,source_basis=key,partial=partial,implementation_basis=d.basis_id));cursor=stop
    if phase:q.p(phase,8)
    return q,dict(basis=basis.tolist(),uncontrolled_diagonal=unctrl.tolist(),control_increment=(ctrl-unctrl).tolist(),relative_phase=[1,0,1,1] if phase else [0]*4,
        boundary_exposed_basis=np.sum(exposed,axis=0).tolist(),runs=selected)

def price(path):
    raw=json.loads(path.read_text());q,profile=build_singletons(raw['t'],raw['n'],raw['choices']);new=metrics(compile_reference(q));choice='singletons' if new[0]<raw['metrics'][0] else 'full'
    profile['compile_correction']=(np.array(new)-profile['basis']-np.array(profile['uncontrolled_diagonal'])-np.array(profile['control_increment'])-np.array(profile['relative_phase'])).tolist()
    record=dict(t=raw['t'],n=raw['n'],rare=raw['rare'],j=raw['j'],probability=raw['probability'],old=raw['metrics'],proposal=new,selected=new if choice=='singletons' else raw['metrics'],choice=choice,
        old_profile=profile_rte(raw),proposal_profile=profile)
    save_json(OUT/'runwise_prices'/path.name,record);return record

@lru_cache(None)
def d_atom(index,t):
    block=common()['preps'][12].deterministic_blocks[index];basis=sum((np.array(metrics(compile_reference(circuit(block.runtime_basis_operations,9,inverse))),float) for inverse in (False,True)),np.zeros(4))
    p=diagonal_primitives(block,t);prices=[]
    for controlled in (False,True):
        q=QuantumCircuit(9);append_diagonal_primitives(q,p,controlled=controlled,ancilla_qubit=8 if controlled else None);prices.append(np.array(metrics(compile_reference(q)),float))
    return basis,prices[0],prices[1]-prices[0]

def d_profile(record):
    x=record['signal'];_,_,dets,_=saved();raw=dets[(x['name'],x['h'])];groups=split(x['case'],x['h']);n=x['steps'];weights=dict(prefix=1,suffix=1,middle=n-1) if groups['full'] else dict(first=1,last=1,junction=n-1,inside=n)
    totals=np.zeros((3,4));cost=np.zeros(4)
    for role,weight in weights.items():
        for kind,index,t in groups[role]:totals+=weight*np.array(d_atom(index,t))
        cost+=weight*np.array(raw['groups'][role])
    # Source basis exposure at D/R cuts is a subset of basis, not a new cost.
    from .track_c_d_internal import schedule
    seq=schedule(x['case'],x['h']);exposed=np.zeros(4)
    for j,item in enumerate(seq):
        if item[0]=='R':
            for index,inverse in ((seq[j-1][1],False),(seq[j+1][1],True)):
                b=common()['preps'][12].deterministic_blocks[index];exposed+=n*np.array(metrics(compile_reference(circuit(b.runtime_basis_operations,9,inverse))))
    return dict(basis=totals[0].tolist(),uncontrolled_diagonal=totals[1].tolist(),control_increment=totals[2].tolist(),compile_correction=(cost-totals.sum(axis=0)).tolist(),D_R_exposed_basis=exposed.tolist())

def run():
    (OUT/'runwise_prices').mkdir(exist_ok=False);configuration();workers=len(os.sched_getaffinity(0));records=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(price,p) for p in (PRIOR/'rte_price').glob('*.json')]):records.append(f.result())
    policies=saved()[0];profiles=[]
    for name in NAMES:
        for record in policies[name]['selected']:profiles.append(dict(policy=name,m=record['signal']['m'],profile=d_profile(record)))
    save_json(OUT/'D_PROFILES.json',dict(rows=profiles,convention='Isolated native basis + uncontrolled diagonal + control increment + signed compile/context correction = actual group cost. Depth correction also includes DAG scheduling; never double counted.'))
    save_json(OUT/'RUNWISE_SUMMARY.json',dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),samples=len(records),selected_singletons=sum(r['choice']=='singletons' for r in records),
        definition='Per occurrence choose lower actual RZ of baseline full circuit and singleton partial proposal. This is a deterministic compiler policy, not min of estimated cohort means.',
        price_scope='Same saved trajectories; no new RTE distribution or return aggregation. New implementation native precision qualification pending.'))
    print('RUNWISE_COMPLETE',len(records),flush=True)

if __name__=='__main__':run()
