"""Partial Gaussian isometry for unchanged signed RTE component operators.

Only the implementation support is relabelled to leading modes. Component
IDs, probabilities, product/rotation order, signs and event phases stay fixed.
Reuse the production RTE builder; no sampling/return-aggregation replacement.
"""
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import replace
from functools import lru_cache
import json
import math
import os
import subprocess
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RZGate,XXPlusYYGate
from qiskit.quantum_info import Operator
from .track_c_h4 import common,finite_rte_distribution,diagonal_primitives,save_json
from .track_c_compiler_a import compile_reference
from .track_c_d_resources import metrics
from .track_c_bottleneck import OUT,ROOT,PRIOR,saved,components,NAMES
from trotterlib.stable_gaussian_basis import exact_zero_givens
from trotterlib.df_rte_tail import DFBasisRegistry,describe_basis_change_operations
from trotterlib.df_rte_qiskit import QiskitDFRTEEventCircuitBuilder as Builder
from trotterlib.df_rte_circuit import DFRTEEventSequenceCircuitRequest
from trotterlib.df_trotter.ops import append_diagonal_primitives
from trotterlib.rte import _make_event,_basis_reuse_intervals

def partial_ops(u,support):
    """QR only k columns, using k*n-k*(k+1)/2 adjacent Givens at most.

    Maps modes 0..k-1 to the requested columns; unused columns are free. Uses
    stable atan2 and exact-zero skipping, not a small-angle cutoff.
    """
    c=np.array(u[:,support],complex,copy=True);n,k=c.shape;rotations=[]
    for column in range(k):
        for row in range(n-1,column,-1):
            if c[row,column]==0:continue
            g=exact_zero_givens(c[row-1,column],c[row,column]);c[[row-1,row]]=g@c[[row-1,row]]
            rotations.append((g,row-1,row))
    operations=[(RZGate(float(np.angle(c[j,j]))),(j,)) for j in range(k)]
    for g,i,j in reversed(rotations):
        theta=float(np.arctan2(abs(g[0,1]),np.real(g[0,0])));phi=float(np.angle(g[0,1]))
        operations.append((XXPlusYYGate(2*theta,phi-np.pi/2),(j,i)))
    return tuple(operations)

def circuit(ops,n=8,inverse=False):
    q=QuantumCircuit(n)
    for gate,wires in (reversed(ops) if inverse else ops):q.append(gate.inverse() if inverse else gate,wires)
    return q

@lru_cache(None)
def configuration():
    c=common();prep=c['preps'][3];registry=DFBasisRegistry();full={};single={};mapping={};specs=[]
    indices=[1<<i for i in range(8)]
    for name,d in prep.rte_preparation.basis_registry._definitions.items():
        b=Operator(circuit(d.runtime_operations)).data;full[name]=b;single[name]=b[np.ix_(indices,indices)]/b[0,0]
    for spec in prep.rte_preparation.component_specs:
        key=spec.basis_id,spec.diagonal_pauli_support
        if key not in mapping:
            ops=partial_ops(single[spec.basis_id],spec.diagonal_pauli_support)
            d=registry.register(ops,num_system_qubits=8,basis_id='trackC_partial_'+spec.basis_id+'_'+''.join(map(str,spec.diagonal_pauli_support)))
            mapping[key]=d
        d=mapping[key];specs.append(replace(spec,basis_id=d.basis_id,basis_hash=d.basis_hash,diagonal_pauli_support=tuple(range(len(spec.diagonal_pauli_support))),basis_change_operations=d.metadata.operations))
    lookup={s.component_id:s for s in specs}
    return registry,tuple(specs),lookup,full

def remap(event):
    _,_,lookup,_=configuration();apps=[]
    for a in event.application_sequence:
        s=lookup[a.component_id];apps.append(replace(a,basis_id=s.basis_id,basis_hash=s.basis_hash,diagonal_pauli_support=s.diagonal_pauli_support,basis_change_operations=s.basis_change_operations))
    keys={(a.basis_id,a.basis_hash) for a in apps};basis_id,basis_hash=next(iter(keys)) if len(keys)==1 else (None,None)
    new=replace(event,application_sequence=tuple(apps),basis_id=basis_id,basis_hash=basis_hash,basis_reuse_intervals=_basis_reuse_intervals(apps))
    for field in ('event_probability','event_coefficient','event_normalization','selected_component_ids','rotation_angle','phase','taylor_order'):
        assert getattr(new,field)==getattr(event,field)
    return new

def build(t,n,choices):
    registry,specs,_,_=configuration();prep=common()['preps'][3];dist=finite_rte_distribution(prep.exact_rte_lambda_r*t/n,2)
    events=[remap(_make_event(x['indices'],prep.rte_preparation.symbolic_tail.components,dist,1 if x['order']==2 else 0)) for x in choices]
    request=DFRTEEventSequenceCircuitRequest(events=tuple(events),component_specs=specs,controlled=True,ancilla_qubit=8,
        tail_id=prep.rte_preparation.symbolic_tail.tail_id,tail_hash=prep.rte_preparation.symbolic_tail.tail_hash,occurrence_rte_steps=n)
    return Builder(basis_registry=registry).build_sequence(request).circuit

def check_component(index):
    c=common();registry,specs,_,full=configuration();old=c['preps'][3].rte_preparation.component_specs[index];new=specs[index]
    b=full[old.basis_id];d=registry.definition(new.basis_id);v=Operator(circuit(d.runtime_operations)).data
    states=np.arange(256);sign=lambda support:np.prod([1-2*((states>>q)&1) for q in support],axis=0)
    a=(b*sign(old.diagonal_pauli_support))@b.conj().T;z=(v*sign(new.diagonal_pauli_support))@v.conj().T
    error=float(np.linalg.norm(a-z,'fro'));assert error<1e-10
    def rotation(basis,support):
        q=QuantumCircuit(9);q.compose(circuit(basis,9,True),inplace=True)
        if len(support)==1:q.crz(.001,8,support[0])
        else:
            from qiskit.circuit.library import RZZGate
            q.append(RZZGate(.001).control(),[8,*support])
        q.compose(circuit(basis,9),inplace=True);return metrics(compile_reference(q))
    oldd=c['preps'][3].rte_preparation.basis_registry.definition(old.basis_id)
    return dict(component_id=old.component_id,old_support=old.diagonal_pauli_support,new_support=new.diagonal_pauli_support,
        operator_Fro_error=error,full_rotation=rotation(oldd.runtime_operations,old.diagonal_pauli_support),partial_rotation=rotation(d.runtime_operations,new.diagonal_pauli_support),
        source_ops=len(oldd.runtime_operations),partial_ops=len(d.runtime_operations))

@lru_cache(None)
def basis_price(name,inverse):
    d=common()['preps'][3].rte_preparation.basis_registry.definition(name)
    return np.array(metrics(compile_reference(circuit(d.runtime_operations,9,inverse))),float)

@lru_cache(None)
def diagonal_price(order,role,support,angle,controlled):
    q=QuantumCircuit(9);support=tuple(support)
    if role=='product':
        for i in support:
            if controlled:q.cz(8,i)
            else:q.z(i)
    else:
        from qiskit.circuit.library import RZZGate,RZGate
        g=RZGate(angle) if len(support)==1 else RZZGate(angle)
        q.append(g.control() if controlled else g,[8,*support] if controlled else list(support))
    return np.array(metrics(compile_reference(q)),float)

def profile_rte(raw):
    prep=common()['preps'][3];cs=prep.rte_preparation.symbolic_tail.components;dist=finite_rte_distribution(prep.exact_rte_lambda_r*raw['t']/raw['n'],2)
    basis=np.zeros(4);uncontrolled=np.zeros(4);controlled=np.zeros(4);active=None;first=None;last=None
    for choice in raw['choices']:
        event=_make_event(choice['indices'],cs,dist,1 if choice['order']==2 else 0)
        for a in event.application_sequence:
            if active!=a.basis_id:
                if active is not None:basis+=basis_price(active,False)
                basis+=basis_price(a.basis_id,True);active=a.basis_id
            first=a.basis_id if first is None else first;last=a.basis_id
            angle=2*event.unsigned_rotation_angle
            uncontrolled+=diagonal_price(event.taylor_order,a.role,a.diagonal_pauli_support,angle,False)
            controlled+=diagonal_price(event.taylor_order,a.role,a.diagonal_pauli_support,angle,True)
    basis+=basis_price(active,False);phase=np.array([1,0,1,1],float) if raw['rare'] else np.zeros(4)
    cost=np.array(raw['metrics'],float);correction=cost-basis-controlled-phase
    return dict(basis=basis.tolist(),uncontrolled_diagonal=uncontrolled.tolist(),control_increment=(controlled-uncontrolled).tolist(),relative_phase=phase.tolist(),compile_correction=correction.tolist(),
        boundary_exposed_basis=(basis_price(first,True)+basis_price(last,False)).tolist())

def price_saved(path):
    raw=json.loads(path.read_text());q=build(raw['t'],raw['n'],raw['choices']);native=compile_reference(q)
    record=dict(input_file=path.name,t=raw['t'],n=raw['n'],rare=raw['rare'],j=raw['j'],probability=raw['probability'],old=raw['metrics'],partial=metrics(native),profile=profile_rte(raw))
    save_json(OUT/'partial_prices'/path.name,record);return record

def run():
    (OUT/'partial_prices').mkdir(exist_ok=False);configuration();workers=len(os.sched_getaffinity(0));checks=[];records=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(check_component,i) for i in range(len(common()['preps'][3].rte_preparation.component_specs))]):checks.append(f.result())
        # Dominant m16/m17 cells first, paired with already-saved streams.
        paths=[p for p in (PRIOR/'rte_price').glob('*.json') if (lambda r:r['n']==256 and r['t'] in (.16,.32))(json.loads(p.read_text()))]
        for f in as_completed([pool.submit(price_saved,p) for p in paths]):records.append(f.result())
    save_json(OUT/'PARTIAL_BASIS_PILOT.json',dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),checks=checks,samples=len(records),
        operator_error_max=max(x['operator_Fro_error'] for x in checks),
        sampling_law='Unchanged component probabilities, signs, order, angle, scalar normalization and event phase. Only physically equivalent application support/basis representation changes.',
        novelty='Known partial Gaussian/isometry construction, stable adjacent-Givens implementation. No new sampling or return aggregation.'))
    print('PARTIAL_BASIS_PILOT_COMPLETE',len(records),flush=True)

if __name__=='__main__':run()
