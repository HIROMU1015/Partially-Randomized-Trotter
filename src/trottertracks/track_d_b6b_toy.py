"""Artificial-only measurement theorem probes using the existing gadget."""
from fractions import Fraction
import math
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator
from trotterlib.df_trotter.ops import DiagonalEvolutionPrimitives as D, append_diagonal_primitives
from . import track_d_c7_2_toy as old


def coherence(core):
    n=core.num_qubits//2 if core.num_qubits%2 else None
    # Caller always supplies a two-system-mode reference core or a direct core.
    system=2;d=1<<system;anc=core.num_qubits-1
    c=QuantumCircuit(core.num_qubits);c.h(anc);c.compose(core,inplace=True)
    columns=Operator(c).data[:,:d]
    half=1<<anc;k0,k1=columns[:half],columns[half:]
    return 2*k0.conj().T@k1,columns


def core_from_matrices(a,b,chi):
    qa=QuantumCircuit(2);qa.append(UnitaryGate(a),[0,1])
    qb=QuantumCircuit(2);qb.append(UnitaryGate(b),[0,1])
    return old.c3.base.reference_gadget(qa,qb,chi)


def build_cases(spec,lift):
    context,lifts,head,reverse,initial=old.context(spec,lift)
    def step(frame,sign,phase):
        primitive=D(phase,((0,sign*.29),(1,-sign*.11)),((0,1,sign*.17),))
        g=context.builder._basis(context.arm.model,context.arm.model.blocks[frame])
        c=g.inverse();append_diagonal_primitives(c,primitive);c.compose(g,inplace=True)
        u=lifts[frame];target=u@old.diagonal_matrix(primitive)@u.conj().T
        chi=math.fsum([phase,*[-angle/2 for _,angle in primitive.rz],*[-angle/2 for _,_,angle in primitive.rzz]])
        return c,target,chi
    steps=[step(1,1,-.021),step(2,-1,.043),step(0,1,-.037),step(1,-1,-.019)]
    def combine(items):
        c=QuantumCircuit(2);v=np.eye(4,dtype=complex);phases=[]
        for qc,u,chi in items:c.compose(qc,inplace=True);v=u@v;phases.append(chi)
        return c,v,math.fsum(phases)
    result=[]
    for name,front,back,cut in [('q1_event_cut',steps[:1],steps[1:2],'complete event'),
                              ('q2_step_cut',steps[1:2],steps[2:3],'complete step1'),
                              ('q4_half_cut',steps[:2],steps[2:],'complete step2')]:
        b,mb,cb=combine(front);a,ma,ca=combine(back)
        result.append(dict(name=name,A=a,B=b,V=ma@mb,chi=math.fsum([ca,cb]),
            half_phases=(ca,cb),DF_eligible=True,cut=cut))
    # V=D preserves vacuum, although A and B separately do not. This probes the
    # weaker signal theorem, not a relaxation of the historical zero-leakage checker.
    a=QuantumCircuit(2);a.ry(.43,0);a.global_phase=.13
    primitive=D(-.07,((0,.31),(1,-.11)),((0,1,.19),))
    b=QuantumCircuit(2);append_diagonal_primitives(b,primitive);b.compose(a.inverse(),inplace=True)
    chi=math.fsum([primitive.global_phase,*[-v/2 for _,v in primitive.rz],*[-v/2 for _,_,v in primitive.rzz]])
    result.append(dict(name='V_ray_only_not_DF_eligible',A=a,B=b,V=old.diagonal_matrix(primitive),
        chi=chi,half_phases=None,DF_eligible=False,cut='algebra-only; not a new PR cut'))
    return result,initial


def check_case(case):
    a,b=[Operator(case[n]).data for n in ('A','B')];v=case['V'];d=4;records=[]
    same=lambda x,y,label:records.append(old.check(x,y,label))
    same(a@b,v,'AB=V');same(Operator(case['A'].inverse()).data,a.conj().T,'actual_A_adjoint')
    same(v[:,:1],np.exp(1j*case['chi'])*np.eye(d)[:,:1],'whole_V_vacuum_ray_chi')
    ref=old.c3.base.reference_gadget(case['A'],case['B'],case['chi'])
    direct=QuantumCircuit(3);direct.append(UnitaryGate(old.controlled(v)),range(3))
    cref,columns=coherence(ref);cdir,_=coherence(direct)
    factor=np.exp(1j*case['chi'])*np.conj(v[0,0])
    same(cref,factor*v,'general_complex_coherence_factor');same(cref,v,'Reference_V_coherence')
    same(cdir,v,'direct_V_coherence')
    leakage=float(np.linalg.norm(columns.reshape(2,4,4,4)[:,1:,:,:]))
    if case['DF_eligible']:
        ca,cb=case['half_phases']
        old.c3.base.reference_column_check(columns.T,np.exp(1j*cb)*a.conj().T/math.sqrt(2),
            np.exp(1j*(case['chi']-ca))*b/math.sqrt(2),same,'DF_final_vacuum')
    elif leakage<1e-3:raise ValueError('WEAK_THEOREM_NONVACUUM_CONTROL_NOT_DETECTED')
    wrappers={}
    for method,core in [('direct_controlled',direct),('reference',ref)]:
        for axis in ('X','Y'):
            w=old.c3.base.artificial.hadamard_wrapper(core,axis)
            records.append(old.check(old.c3.base.artificial.induced_observable(w),
                old.c3.base.artificial.expected_observable(v,axis),method+'_'+axis+'_observable',observable=True))
            wrappers[method,axis]=w
    return wrappers,dict(name=case['name'],status='PASS',DF_source_eligible=case['DF_eligible'],
        cut=case['cut'],chi=case['chi'],checks=records,reference_nonvacuum_Frobenius=leakage,
        half_A_vacuum_leakage=float(np.linalg.norm(a[1:,0])),half_B_vacuum_leakage=float(np.linalg.norm(b[1:,0])),
        direct_control_is_cost_baseline=False)


def negative_controls(case):
    v=case['V'];chi=case['chi'];a=case['A'];b=case['B'];rows=[]
    cores=[('omit_chi',old.c3.base.reference_gadget(a,b,0.)),
           ('wrong_chi_sign',old.c3.base.reference_gadget(a,b,-chi)),
           ('forward_A_instead_of_adjoint',old.c3.base.reference_gadget(a.inverse(),b,chi)),
           ('swap_AB_order',old.c3.base.reference_gadget(b,a,chi))]
    for name,core in cores:
        c,_=coherence(core);error=float(np.max(abs(c-v)))
        if error<1e-6:raise ValueError('NEGATIVE_CONTROL_NOT_DETECTED:'+name)
        rows.append(dict(name=name,target_difference=error,detected=True,expected_ineligible=True))
    aa=QuantumCircuit(2);aa.ry(.53,0);bb=QuantumCircuit(2);bb.rz(.2,0)
    va=Operator(aa).data@Operator(bb).data
    c,_=coherence(old.c3.base.reference_gadget(aa,bb,0.))
    factor=np.conj(va[0,0]);old.check(c,factor*va,'invalid_ray_general_factor')
    error=float(np.max(abs(c-va)))
    if error<1e-6:raise ValueError('INVALID_V_RAY_NOT_DETECTED')
    rows.append(dict(name='V_does_not_preserve_reference_ray',detected=True,target_difference=error,
        factor_real=float(factor.real),factor_imag=float(factor.imag),expected_ineligible=True,
        arbitrary_AB_unconditional_equivalence=False))
    return rows


def postcheck(compiled,raw,v,axis,method,DF_eligible):
    matrix=lambda c:Operator(c.remove_final_measurements(inplace=False)).data
    checks=[old.check(matrix(compiled),matrix(raw),'P0_global_phase_full_matrix'),
        old.check(old.c3.base.artificial.induced_observable(compiled),
                  old.c3.base.artificial.expected_observable(v,axis),'P0_XY_observable',observable=True)]
    half=1<<(compiled.num_qubits-1);h=np.array([[1,1],[1,-1]],complex)/math.sqrt(2)
    readout=h if axis=='X' else h@np.diag([1.,-1j])
    before=np.kron(readout.conj().T,np.eye(half))@matrix(compiled)
    c=2*before[:half,:4].conj().T@before[half:,:4]
    checks.append(old.check(c,v,'P0_complex_coherence'))
    leakage=None
    if method=='reference':
        leakage=float(np.linalg.norm(before[:,:4].reshape(2,4,4,4)[:,1:,:,:]))
        if DF_eligible and leakage>old.ATOL:raise ValueError('P0_REFERENCE_LEAKAGE')
    return dict(status='PASS',checks=checks,reference_nonvacuum_Frobenius=leakage,
        final_vacuum_required=DF_eligible,all_system_inputs=4,full_wrapper_operator=True)
