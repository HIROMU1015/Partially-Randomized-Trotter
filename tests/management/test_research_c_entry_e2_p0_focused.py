"""Artificial native circuits/2x2 matrices only; no H4, Arb or saved arrays."""
from fractions import Fraction as F
import math
from unittest.mock import patch
import numpy as np
import pytest
from trottertracks import c_entry_e2_p0_focused as f


@pytest.mark.parametrize('n',[2,7,12,1000,1001])
def test_real_unchanged_P0_nested_CX_deletion_and_guard(n):
    import qiskit
    from qiskit import QuantumCircuit
    from trottertracks.c1_h6_compiler_precision import compiled_wrapper
    from trottertracks.pr_dictionary_validation.fixture import CompilerProfile
    q=QuantumCircuit(3);word=[(0,1) if j%2==0 else (1,2) for j in range(n)]
    for a,b in [*word,*reversed(word)]:q.cx(a,b)
    calls=[];cancellation=[];original=qiskit.transpile
    def callback(**kw):
        if type(kw['pass_']).__name__=='Optimize1qGatesDecomposition':calls.append(kw['dag'].size())
        if type(kw['pass_']).__name__=='InverseCancellation':cancellation.append(kw['dag'].size())
    def observe(*a,**kw):return original(*a,**kw,callback=callback)
    with patch.object(qiskit,'transpile',observe):
        if n==1001:
            with pytest.raises(qiskit.transpiler.TranspilerError,match='Maximum iteration reached'):
                compiled_wrapper(q,CompilerProfile())
            # Last cancellation reaches identity, but no iteration remains to confirm fixed point.
            assert len(calls)==1000 and calls[-1]==2
        else:
            out=compiled_wrapper(q,CompilerProfile())
            assert out.size()==0 and len(calls)==n
    assert cancellation[-1]==0
    # This is a counterexample to a coarse native-basis argument, never an E2 word.
    assert q.num_qubits==3 and q.size()==2*n


@pytest.mark.parametrize('epochs',[0,1,124,125])
def test_epoch_bound_separates_CX_deletion_epochs_from_total_size(epochs):
    r=f.epoch_bound(4752,deleting_iterations=epochs)
    assert r['iterations_upper']==8*epochs+7
    assert r['below_guard_under_conditions']==(epochs<=124)
    assert not r['all_E2_words_return_normally'] and not r['supplied_E2_epoch_bound_verified']
    assert f.epoch_bound(4752)['iterations_upper']==19015


def test_CX_epoch_bound_rejects_impossible_input():
    for args in [(-1,None),(3,2),(4752,-1)]:
        with pytest.raises(ValueError):f.epoch_bound(args[0],deleting_iterations=args[1])


def test_small_entry_phase_is_weighted_without_arg_accuracy():
    eta=F(1,1000000);rho=F(1,100000)
    r=f.weighted_extraction(eta,rho,hypot_error='0',radial_atan2_error='0',
        determinant_error='0',large_arg_phasor_error='0',det_arg_phasor_error='0',parameter_arithmetic_error='0')
    # Invented near-unitary matrix, tiny c. Replace its arg by an arbitrary unit phasor.
    M=np.array([[math.sqrt(1-1e-12),-1e-6],[1e-6,math.sqrt(1-1e-12)]],complex)
    r0=math.hypot(abs(M[0,0]),abs(M[1,0]));A,C=abs(M[0,0])/r0,abs(M[1,0])/r0
    true=M;wrong=np.array([[A,-C/1j],[C*1j,A]],complex)
    assert abs(np.angle(1j)-np.angle(M[1,0]))>1
    assert np.linalg.norm(wrong-true,2)<=float(F(r['omega_extraction']))
    assert r['actual_Rust_omega'] is None and not r['compiler_image_enclosed']


@pytest.mark.parametrize('field',['hypot_error','radial_atan2_error','determinant_error',
    'large_arg_phasor_error','det_arg_phasor_error','parameter_arithmetic_error'])
def test_each_missing_function_contract_stays_unknown(field):
    args=dict(hypot_error='0',radial_atan2_error='0',determinant_error='0',
        large_arg_phasor_error='0',det_arg_phasor_error='0',parameter_arithmetic_error='0')
    args[field]=None;r=f.weighted_extraction('1/100','1/1000',**args)
    assert r['omega_extraction'] is None


def test_bad_domains_and_arg_cut_phasor_contract():
    args=dict(hypot_error='0',radial_atan2_error='0',determinant_error='0',
        large_arg_phasor_error='0',det_arg_phasor_error='0',parameter_arithmetic_error='0')
    with pytest.raises(ValueError):f.weighted_extraction('1/4','1/10',**args)
    with pytest.raises(ValueError):f.weighted_extraction('1/100','1/10',**(args|dict(hypot_error='1')))
    with pytest.raises(ValueError):f.weighted_extraction('1/100','1/10',**(args|dict(determinant_error='1')))
    a,b=math.pi-1e-12,-math.pi+1e-12
    assert abs(a-b)>6 and abs(np.exp(1j*a)-np.exp(1j*b))<3e-12


def test_flip_is_exact_for_all_angles_even_when_predicate_would_be_false():
    rz=lambda x:np.diag([np.exp(-.5j*x),np.exp(.5j*x)])
    ry=lambda x:np.array([[math.cos(x/2),-math.sin(x/2)],[math.sin(x/2),math.cos(x/2)]])
    def semantics(t,p,l,g):return np.exp(1j*(g+(t+p+l)/2))*rz(p)@ry(t)@rz(l)
    for t,p,l,g in [(.31,.71,-.23,.17),(math.pi/2,2.,-1.3,-.6),(.0001,-3.,3.1,0.)]:
        np.testing.assert_allclose(semantics(t,p,l,g),semantics(-t,p+math.pi,l+math.pi,g+t),atol=2e-15,rtol=0)


@pytest.mark.parametrize('theta',[0.,1e-12-1e-16,1e-12+1e-16,
    math.pi/2-1e-12,math.pi/2+1e-12,math.pi-1e-12,math.pi+1e-12])
def test_actual_Rust_branch_output_matches_exact_U1X_semantics_without_phase_fit(theta):
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    from qiskit._accelerate import euler_one_qubit_decomposer as rust
    from qiskit.circuit.quantumcircuitdata import CircuitInstruction
    phi,lam,phase=-math.pi+1e-13,math.pi-1e-13,.19
    seq=rust.generate_circuit(rust.EulerBasis('ZSXX'),theta,phi,lam,phase,True)
    q=QuantumCircuit(1,global_phase=seq.global_phase)
    for name,angles in seq:
        inst=CircuitInstruction.from_standard(name,tuple(q.qubits),angles)
        q.append(inst.operation,[0])
    target=QuantumCircuit(1)
    target.rz(lam,0);target.ry(theta,0);target.rz(phi,0)
    target.global_phase=phase+(theta+phi+lam)/2
    budget=f.branch_budget(predicate_error='0',pi_center_error='0',phase_mod_error='0',generation_arithmetic_error='0')
    # Binary64 residual check only; this finite diagnostic is not a uniform Rust certificate.
    assert np.linalg.norm(Operator(q).data-Operator(target).data,2)<=float(F(budget['omega_generation']))+1e-13
    assert budget['actual_Rust_omega'] is None and not budget['contracts_verified']


def test_modulo_P_semantics_and_control_relative_phase():
    from qiskit.circuit.library import PhaseGate,RZGate
    from qiskit.quantum_info import Operator
    for x in (-math.pi-1e-13,math.pi+1e-13,4*math.pi+1e-13):
        wrapped=(x+math.pi)%(2*math.pi)-math.pi
        assert np.linalg.norm(Operator(PhaseGate(x)).data-Operator(PhaseGate(wrapped)).data,2)<3e-15
    assert np.linalg.norm(Operator(RZGate(.2+2*math.pi)).data-Operator(RZGate(.2)).data,2)>1
    relative=np.diag([1,1,np.exp(.2j),np.exp(.2j)])
    assert np.linalg.norm(relative-np.eye(4),2)>.1


def test_split_and_whole_zeta_cannot_double_count_or_qualify():
    identity=f.previous.binding.target_identity()
    with pytest.raises(ValueError,match='CHOOSE_WHOLE'):f.compose_split(identity,8,whole_zeta='0')
    r=f.compose_split(identity,8)
    assert r['conditional_operator_upper'] is r['error_evidence'] is r['actual_Rust_omega'] is None
    r=f.compose_split(identity,8,node_matrix_error='0',model=f.previous.uniform.ieee_model(),
        omega_extraction='1/1000000',omega_generation='1/1000000',translation='0',phase='0')
    assert r['conditional_operator_upper'] is not None and r['actual_H4_bound'] is None
    assert not r['all_word_normal_return_proven'] and r['RPE_accuracy']=='UNQUALIFIED' and r['mandatory_stop']


def test_missing_branch_contracts_are_not_replaced_by_threshold():
    assert f.branch_budget()['omega_generation'] is None
    assert f.weighted_extraction(None,None)['omega_extraction'] is None
