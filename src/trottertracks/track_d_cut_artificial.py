"""Two frozen invented fixtures, reusing C7-2/C3 builders and check semantics."""
from dataclasses import asdict
import math
from unittest.mock import patch

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator

from . import track_d_c7_2_toy as toy
from . import track_d_cut_selection as select


def fixture(spec,case,lift,event_features):
    context,lifts,head,reverse,initial=toy.context(spec,lift)
    atoms={a.atom_id:a for a in context.arm.plan.atoms};steps=[]
    words=dict(alternating_plus_minus=["plus","minus"]*8,
               alternating_minus_plus=["minus","plus"]*8,plus_only=["plus"]*16)
    for i,word in enumerate(case["words"]):
        events=tuple(toy.event(atoms[name],math.atan(spec["tau"])) for name in words[word])
        matrix=np.eye(4,dtype=complex)
        for e in events:
            atom=atoms[e.rotation_component_id];u=lifts[int(atom.frame_id)]
            z=np.array([np.prod([1-2*((state>>bit)&1) for bit in atom.support]) for state in range(4)])
            matrix=(u@np.diag(np.exp(-1j*e.unsigned_rotation_angle*z))@u.conj().T)@matrix
        target=np.exp(1j*context.scalar_phase)*reverse@matrix@head
        raw=QuantumCircuit(3);raw.append(UnitaryGate(toy.controlled(target)),range(3))
        steps.append(dict(sample=i,events=events,reference=target,raw=raw))
    decision=select.select_cut([s["events"] for s in steps],event_features)
    with patch.object(toy.c3.base,"N",2):
        center=toy.c3.prepare_steps(context,steps,decision.center,
            ordered_samples=tuple(range(len(steps))),fixture_selection="fixed artificial center",
            cohort="O5_CUT_ARTIFICIAL_ONLY")
        adjusted=select.build_selected(context,steps,decision)
    for bundle in (center,adjusted):
        bundle.identity=dict(artificial=True,modes=2,q=len(steps),
            cut=bundle.identity["cut_after_complete_step"],
            strong=bundle.identity["strong_symmetric"],
            reference=bundle.identity["reference_optimization"],phase_chi=bundle.phase)
    return context,steps,center,adjusted,decision,initial


def eligible_half(circuit,label):
    if circuit.num_qubits!=2 or circuit.num_clbits or circuit.parameters:
        raise ValueError("ARTIFICIAL_HALF_SCOPE")
    if any(i.operation.name not in {"rz","rzz","p","xx_plus_yy","z","x","cx"} for i in circuit.data):
        raise ValueError("UNSUPPORTED_HALF_GATE")
    matrix=Operator(circuit).data
    expected=np.zeros(4,complex);expected[0]=matrix[0,0]
    toy.check(matrix[:,0],expected,label+":vacuum_ray")
    return matrix


def raw_checks(bundle,steps,cut):
    records=[]
    def same(x,y,label):records.append(toy.check(x,y,label))
    A=eligible_half(bundle.later,"A");B=eligible_half(bundle.earlier,"B")
    adj=Operator(bundle.later_adjoint).data;V=bundle.reference
    same(A@B,V,"AB=ordered_V");same(adj,A.conj().T,"actual_A_inverse_including_global_phase")
    same(V.conj().T@V,np.eye(4),"V_unitarity")
    phases=[toy.c3.paired.path_phase(s["events"],-(.17+.04)*.1) for s in steps]
    chi=math.fsum(phases);a=math.fsum(phases[cut:]);b=math.fsum(phases[:cut])
    same(np.exp(1j*bundle.phase),np.exp(1j*chi),"metadata_chi_scalar_identity_once")
    same(A[:,0],np.exp(1j*a)*np.eye(4)[:,0],"A_analytic_vacuum_phase")
    same(B[:,0],np.exp(1j*b)*np.eye(4)[:,0],"B_analytic_vacuum_phase")
    S=Operator(bundle.cores["symmetric_directional"]).data
    same(S[:4,:4].conj().T@S[4:,4:],V,"strong_direct_V_coherence")
    same(S[:4,4:],np.zeros((4,4)),"strong_offdiagonal_control")
    # All four system input columns, including every reference output.
    q=QuantumCircuit(5);q.h(4);q.compose(bundle.cores["reference"],inplace=True)
    columns=Operator(q).data[:,:4].T
    leakage=toy.c3.base.reference_column_check(columns,np.exp(1j*b)*adj/math.sqrt(2),
        np.exp(1j*(chi-a))*B/math.sqrt(2),same,"reference_all_input_columns")
    obs={}
    for method in ("symmetric_directional","reference"):
        obs[method]={}
        for axis in ("X","Y"):
            value=toy.c3.base.artificial.induced_observable(bundle.wrappers[method,axis])
            records.append(toy.check(value,toy.c3.base.artificial.expected_observable(V,axis),
                method+"_"+axis+":all_input_observable",observable=True))
            obs[method][axis]=value
        records.append(toy.check(obs[method]["X"]+1j*obs[method]["Y"],V,
            method+":complex_coherence",observable=True))
    return dict(status="PASS",checks=records,reference_leakage=leakage,
        AB_commutator_frobenius=float(np.linalg.norm(A@B-B@A)),chi=chi,
        fusion=bundle.identity,source_unchanged=True)


def compile_one(raw):
    import time
    from .c1_h4_native_fair import compile_checked
    from .pr_dictionary_validation.fixture import CompilerProfile
    start=time.perf_counter()
    compiled=compile_checked(raw,CompilerProfile())
    clean=lambda c:c.remove_final_measurements(inplace=False)
    check=toy.check(Operator(clean(compiled)).data,Operator(clean(raw)).data,"P0_full_unitary_phase")
    return compiled,dict(status="PASS",phase_check=check,wall_seconds=time.perf_counter()-start)


def post_checks(raw,compiled,V,reference):
    records=[]
    for axis in ("X","Y"):
        if axis not in compiled:continue
        obs=toy.c3.base.artificial.induced_observable(compiled[axis])
        records.append(toy.check(obs,toy.c3.base.artificial.expected_observable(V,axis),
                                 axis+":compiled_all_input_observable",observable=True))
        if reference:
            matrix=Operator(compiled[axis].remove_final_measurements(inplace=False)).data
            rows=[i for i in range(32) if i&12]
            records.append(toy.check(matrix[rows,:4],np.zeros((len(rows),4)),axis+":compiled_reference_leakage"))
    observed={axis:toy.c3.base.artificial.induced_observable(c) for axis,c in compiled.items()}
    records.append(toy.check(observed["X"]+1j*observed["Y"],V,"compiled_complex_coherence",observable=True))
    return dict(status="PASS",checks=records)
