"""Dormant typed circuit router and common run-level sequence policy.

Recipes adapted from 2978 DF builder, ae743 arbitrary-Pauli primitive,
WP06-b f29 run policy and its support-completion helper. No legacy source edits.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Any
from .fixture import EvolutionContext, FixtureSpec, digest
from .prototype import MixedAtom, atom_record


def support_completion(full, support, *, atol=1e-10):
    """Narrow port: deterministic twice-reorthogonalized standard-vector fill."""
    import numpy as np
    u=np.asarray(full,dtype=complex)
    n=u.shape[0]
    if u.shape!=(n,n) or not np.isfinite(u).all():
        raise ValueError('Finite square frame required')
    if tuple(sorted(set(support)))!=tuple(support) or not 1<=len(support)<=2:
        raise ValueError('One or two distinct sorted columns required')
    if any(i<0 or i>=n for i in support): raise ValueError('Column out of range')
    if np.max(np.abs(u.conj().T@u-np.eye(n)))>atol:
        raise ValueError('Frame is not unitary')
    result=np.zeros_like(u); filled=[]
    for i in support: result[:,i]=u[:,i]; filled.append(i)
    for target in range(n):
        if target in support: continue
        for seed in range(n):
            v=np.eye(n,dtype=complex)[:,seed].copy()
            for _ in range(2):
                for i in filled: v-=result[:,i]*np.vdot(result[:,i],v)
            norm=float(np.linalg.norm(v))
            if norm>atol:
                result[:,target]=v/norm; filled.append(target); break
        else: raise ValueError('Support completion failed')
    if np.max(np.abs(result.conj().T@result-np.eye(n)))>10*atol:
        raise ValueError('Completed frame is not unitary')
    if np.max(np.abs(result[:,support]-u[:,support]))>atol:
        raise ValueError('Support columns changed')
    return result


def native_gaussian(matrix):
    import numpy as np
    from qiskit import QuantumCircuit
    from trotterlib.df_trotter.ops import U_to_qiskit_ops_jw_givens
    v=np.asarray(matrix,dtype=complex)
    qc=QuantumCircuit(len(v))
    for gate,qubits in U_to_qiskit_ops_jw_givens(v):
        if gate.name=='unitary': raise ValueError('Dense fallback forbidden')
        qc.append(gate,list(qubits))
    # The helper's basis-only global phase cancels in B central B.inverse().
    # Relative event/scalar phases below are retained, never phase-fitted.
    return qc


class BasisProvider:
    def __init__(self, modes, frames):
        self.modes=modes
        self.frames=dict(frames)
        self._cache={}

    def basis(self, atom: MixedAtom, singleton: bool):
        from qiskit import QuantumCircuit
        key=(atom.basis_key,atom.support if singleton and atom.kind=='FRAME_Z' else ())
        if key not in self._cache:
            if atom.kind=='FRAME_Z':
                matrix=self.frames[atom.frame_id]
                if singleton: matrix=support_completion(matrix,atom.support)
                qc=native_gaussian(matrix)
            elif atom.kind=='PHYSICAL_PAULI':
                qc=QuantumCircuit(self.modes)
                for p,ch in enumerate(atom.word):
                    if ch in ('X','Y'): qc.h(p)
                    if ch=='Y': qc.s(p)
            else: raise ValueError('Identity has no basis')
            self._cache[key]=qc
        return self._cache[key].copy()


def append_central(qc, support, ancilla, angle):
    """Unsigned Z-support product or exp(-i angle Z_support)."""
    if not support:
        if angle is not None: qc.p(-angle,ancilla)
        return
    target=support[-1]
    for p in support[:-1]: qc.cx(p,target)
    if angle is None: qc.cz(ancilla,target)
    else:
        qc.rz(angle,target); qc.cx(ancilla,target)
        qc.rz(-angle,target); qc.cx(ancilla,target)
    for p in reversed(support[:-1]): qc.cx(p,target)


def component_probe(atom, provider, *, angle=None):
    from qiskit import QuantumCircuit
    qc=QuantumCircuit(provider.modes+1); anc=provider.modes
    phase=math.pi if angle is None and atom.coefficient<0 else 0.0
    signed_angle=None if angle is None else angle*(1 if atom.coefficient>0 else -1)
    if atom.kind=='IDENTITY': append_central(qc,(),anc,signed_angle)
    else:
        b=provider.basis(atom,True)
        qc.compose(b.inverse(),qubits=range(anc),inplace=True)
        append_central(qc,atom.support,anc,signed_angle)
        qc.compose(b,qubits=range(anc),inplace=True)
    if phase: qc.p(phase,anc)
    return qc


def build_event_sequence(events, atoms, provider, *, optimized=True):
    from qiskit import QuantumCircuit
    qc=QuantumCircuit(provider.modes+1); anc=provider.modes
    by_id={a.atom_id:a for a in atoms}; flattened=[]; phase=0.0
    for event in events:
        if tuple(x.component_id for x in event.application_sequence)!=event.selected_component_ids:
            raise ValueError('Event order metadata mismatch')
        phase+=math.atan2(event.phase.imag,event.phase.real)
        for app in event.application_sequence:
            atom=by_id[app.component_id]
            if app.coefficient_sign!=(1 if atom.coefficient>0 else -1):
                raise ValueError('Stale event coefficient sign')
            if app.is_identity!=(atom.kind=='IDENTITY') or app.basis_id!=atom.basis_key:
                raise ValueError('Stale typed event metadata')
            angle=event.unsigned_rotation_angle if app.role=='rotation' else None
            if app.role=='product' and app.coefficient_sign<0: phase+=math.pi
            flattened.append((atom,angle))
    pos=0
    while pos<len(flattened):
        atom,angle=flattened[pos]
        if atom.kind=='IDENTITY':
            if angle is not None: phase-=angle
            pos+=1; continue
        stop=pos+1
        if optimized:
            while stop<len(flattened):
                other=flattened[stop][0]
                if other.kind=='IDENTITY' or other.basis_key!=atom.basis_key: break
                stop+=1
        b=provider.basis(atom,singleton=optimized and stop-pos==1)
        qc.compose(b.inverse(),qubits=range(anc),inplace=True)
        for a,theta in flattened[pos:stop]: append_central(qc,a.support,anc,theta)
        qc.compose(b,qubits=range(anc),inplace=True)
        pos=stop
    if phase: qc.p(phase,anc)
    return qc


def append_deterministic(qc, coefficients, time, ancilla):
    # exp(-i t h n) = diag(1,exp(-i t h)); cp preserves its scalar exactly.
    from .fixture import fraction
    for p,text in enumerate(coefficients):
        if fraction(text): qc.cp(-float(fraction(text))*time,ancilla,p)


@dataclass(frozen=True)
class MixedEvolutionResult:
    circuit: Any
    modes: int
    ancilla: int
    context: EvolutionContext
    ordered_events: tuple[tuple[dict,...],...]
    input_identity: str
    source_identity: str
    fingerprint: str
    construction_policy: str


def build_evolution(spec: FixtureSpec, context: EvolutionContext, macros, atoms,
                    provider, *, input_identity, source_identity, optimized=True):
    from qiskit import QuantumCircuit
    from .fixture import fraction
    spec.validate(); context.validate()
    macros=tuple(tuple(x) for x in macros)
    if len(macros)!=context.q or any(len(x)!=context.r for x in macros):
        raise ValueError('q/r event shape mismatch')
    qc=QuantumCircuit(spec.modes+1); anc=spec.modes; delta=float(context.delta)
    for j,events in enumerate(macros):
        if not optimized or j==0:
            append_deterministic(qc,spec.deterministic,delta/2,anc)
        qc.compose(build_event_sequence(events,atoms,provider,optimized=optimized),inplace=True)
        # Reverse half D and next forward half D are adjacent, identical block.
        dtime=delta if optimized and j<context.q-1 else delta/2
        append_deterministic(qc,spec.deterministic,dtime,anc)
    qc.p(-float(fraction(spec.scalar))*float(fraction(context.total_time)),anc)
    records=tuple(tuple(e.to_dict() for e in macro) for macro in macros)
    fp=digest(dict(input=input_identity,source=source_identity,context=vars(context),
        macros=records,atoms=[atom_record(a) for a in atoms],optimized=optimized))
    return MixedEvolutionResult(qc,spec.modes,anc,context,records,
        input_identity,source_identity,fp,'common_run_support_full_fused' if optimized else 'raw')


def actual_adjoint(result: MixedEvolutionResult):
    return result.circuit.inverse()


def hadamard_wrapper(result: MixedEvolutionResult, spec: FixtureSpec, axis: str,
                     *, measure=True):
    from qiskit import QuantumCircuit
    if axis not in ('X','Y') or result.ancilla!=spec.modes or result.modes!=spec.modes:
        raise ValueError('Axis/ancilla-last/register mismatch')
    if result.circuit.num_qubits!=spec.modes+1 or result.circuit.num_clbits:
        raise ValueError('Evolution must be full controlled and measurement-free')
    if any(x.operation.name in ('measure','reset') for x in result.circuit.data):
        raise ValueError('Nonunitary evolution')
    result.context.validate()
    if len(result.ordered_events)!=result.context.q or not result.fingerprint:
        raise ValueError('Missing step/time/fingerprint metadata')
    qc=QuantumCircuit(spec.modes+1,1 if measure else 0)
    for p in spec.occupied: qc.x(p)
    qc.h(result.ancilla)
    qc.compose(result.circuit,inplace=True)
    if axis=='Y': qc.sdg(result.ancilla)
    qc.h(result.ancilla)
    if measure: qc.measure(result.ancilla,0)
    qc.metadata=dict(evolution_fingerprint=result.fingerprint,axis=axis,
        wrapper_fingerprint=digest(dict(evolution=result.fingerprint,axis=axis,
            prep=spec.occupied,measure=measure)),state_preparation_included=True,
        total_time=result.context.total_time,delta=str(result.context.delta),
        q=result.context.q,r=result.context.r,ancilla=result.ancilla,
        input_identity=result.input_identity,source_identity=result.source_identity)
    return qc
