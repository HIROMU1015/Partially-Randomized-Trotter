"""Invented CAR/2x2 matrices and controller mocks; no H4 or Arb work."""
from fractions import Fraction as F
import math
import numpy as np
import pytest
from trottertracks import c_entry_e2_p0_feasibility as p


@pytest.mark.parametrize('theta',[0.,1e-8,.31,math.pi/2,math.pi-1e-8,math.pi])
@pytest.mark.parametrize('phase',[0.,math.pi/4,math.pi/2,-math.pi/4])
def test_exact_euler_reconstruction_lemma_with_poles_and_cosets(theta,phase):
    # Independent ZYZ formula and nonunitary perturbation; no Rust interval image.
    phi,lam=math.pi-1e-9,-math.pi+1e-9
    rz=lambda a:np.diag([np.exp(-.5j*a),np.exp(.5j*a)])
    ry=lambda a:np.array([[math.cos(a/2),-math.sin(a/2)],[math.sin(a/2),math.cos(a/2)]])
    G=np.exp(1j*phase)*rz(phi)@ry(theta)@rz(lam)
    eta=F(1,1000000);M=G+float(eta)*np.diag([.4+.1j,-.2+.3j])
    det=np.linalg.det(M);ad,ac=np.angle(M[1,1]),np.angle(M[1,0]);dt=np.angle(det)
    t=2*math.atan2(abs(M[1,0]),abs(M[0,0]))
    V=np.exp(.5j*dt)*rz(ad+ac-dt)@ry(t)@rz(ad-ac)
    assert np.linalg.norm(M-G,2)<=float(eta)
    assert np.linalg.norm(V-M,2)<=float(F(p.euler_exact_map_bound(eta)['exact_map_to_input_upper']))
    assert np.linalg.norm(V.conj().T@V-np.eye(2),2)<2e-14
    assert p.euler_exact_map_bound(eta)['Rust_zeta'] is None


def test_exact_poles_zero_entries_and_relative_phase_not_fitted():
    for M in (np.eye(2,dtype=complex),np.array([[0,-1],[1,0]],complex),1j*np.eye(2)):
        a,c,d=M[0,0],M[1,0],M[1,1];r=math.hypot(abs(a),abs(c))
        ph=lambda z:np.exp(1j*np.angle(z))
        t=ph(np.linalg.det(M));V=np.array([[abs(a)/r*t/ph(d),-abs(c)/r*t/ph(c)],
            [abs(c)/r*ph(c),abs(a)/r*ph(d)]])
        np.testing.assert_allclose(V,M,atol=3e-16,rtol=0)
    assert np.linalg.norm(1j*np.eye(2)-np.eye(2),2)>1
    assert p.euler_exact_map_bound('0')['exact_map_to_input_upper']=='0'


@pytest.mark.parametrize('bad',['1/4','1','-1/10',.01])
def test_euler_domain_or_inexact_model_refused(bad):
    with pytest.raises(ValueError):p.euler_exact_map_bound(bad)
    assert p.euler_exact_map_bound(None)['Rust_zeta'] is None


@pytest.mark.parametrize('scale',[1e-4,1e-7])
def test_nonorthogonal_literal_ZZ_is_not_assumed_Hermitian_or_unitary(scale):
    # One-particle sector of two invented CAR modes; full-Fock bound applies to it.
    U=np.array([[1.,scale],[0.,1.]])
    eps=np.linalg.norm(U.T@U-np.eye(2),2)
    model=p.literal_zz_norm_model(F.from_float(float(eps*1.01)))
    Z=[np.eye(2)-2*np.outer(U[:,j],U[:,j]) for j in range(2)]
    P=Z[0]@Z[1]
    assert np.linalg.norm(P-P.T,2)>0 and np.linalg.norm(P.T@P-np.eye(2),2)>0
    assert np.linalg.norm(P-P.T,2)<=float(F(model['P_Hermiticity_defect_upper']))
    assert np.linalg.norm(P.T@P-np.eye(2),2)<=float(F(model['P_adjoint_product_defect_upper']))
    for angle in (0.,.2,math.pi/2):
        E=math.cos(angle)*np.eye(2)-1j*math.sin(angle)*P
        assert np.linalg.norm(E,2)<=float(F(model['event_norm_upper']))
    assert not model['unitarity_inferred']


def test_literal_adjoint_is_not_matrix_inverse_and_bound_two_is_not_universal():
    L=np.diag([1.01,1.]);C=np.eye(2)
    assert np.linalg.norm(L.T-np.linalg.inv(L),2)>.01
    # Compiler C and raw C are exact unitary; literal target L is not.
    assert np.linalg.norm(-C-L,2)>2
    assert p.literal_zz_norm_model(None)['event_norm_upper'] is None
    assert p.literal_zz_norm_model('0')['event_norm_upper']=='1'


def test_IID_categorical_norm_growth_recovers_unitary_case_and_keeps_unknowns():
    for m in (0,4,12):
        r=p.categorical_growth_preview(m,event_TV='1/100',event_norm='1',head_norm_product='1')
        f=r['factored_upper'];assert F(f['prefactor'])==2*min(F(1),16*(1<<m)*F(1,100))
        assert f['event_power']==16*(1<<m) and not r['PCG64_joint_stream_qualified']
        assert r['numerical_bound'] is r['actual_H4_bound'] is None
    assert p.categorical_growth_preview(12,event_TV='0')['factored_upper'] is None


def test_nonunitary_categorical_factor_against_complete_small_toy_law():
    # Two commuting scalar maps: expectation of fresh products is exactly the power.
    r=p.categorical_growth_preview(0,event_TV='1/10',event_norm='11/10',head_norm_product='1')
    f=r['factored_upper'];upper=float(F(f['prefactor'])*F(f['head_base'])**f['head_power']*F(f['event_base'])**f['event_power'])
    old=.4*1.1+.6*(-1.1);new=.5*1.1+.5*(-1.1)
    assert abs(old**16-new**16)<=upper
    assert upper>2 and not r['unitary_distance_two_cap_applied']


def test_amortized_exact_map_uses_existing_visit_sum_without_N_squared_recharge():
    N=16;node='1/1000000';model=p.uniform.ieee_model()
    r=p.amortized_exact_euler_parts(N,node_matrix_error=node,model=model)
    rate=p.uniform.product_rate(node)
    # Invented accepted lengths satisfying both length<=N and total visits<=6N.
    lengths=[16,15,12,8,7,6,5,4,3,2,1]
    exact=F()
    for L in lengths:
        eta=L*rate/(1-L*rate)
        exact+=eta+F(p.euler_exact_map_bound(eta)['exact_map_to_input_upper'])
    assert sum(lengths)<=6*N and exact<=F(r['partial_upper'])
    assert r['whole_P0_bound'] is r['actual_Rust_after_exact_F'] is None
    assert not r['rejected_candidates_charged']


def test_amortized_parts_keep_missing_constants_and_domain_unknown():
    assert p.amortized_exact_euler_parts(16)['partial_upper'] is None
    r=p.amortized_exact_euler_parts(16,node_matrix_error='1/20',model=p.uniform.ieee_model())
    assert r['partial_upper'] is None and r['whole_P0_bound'] is None


def test_guard_and_monotone_termination_are_different_with_actual_controller():
    from qiskit.passmanager.flow_controllers import DoWhileController
    from qiskit.passmanager.compilation_status import PassManagerState,WorkflowStatus,PropertySet
    from qiskit.passmanager.exceptions import PassManagerError
    # No circuit/compiler call: real controller, mock task/state, strictly decreasing integer.
    task=object();state=PassManagerState(WorkflowStatus(),PropertySet(left=1001))
    controller=DoWhileController([task],do_while=lambda s:s['left']>0)
    iterator=controller.iter_tasks(state);count=0
    with pytest.raises(PassManagerError,match='Maximum iteration'):
        next(iterator)
        while True:
            count+=1;state.property_set['left']-=1;iterator.send(state)
    assert count==1000 and state.property_set['left']==1
    r=p.termination_preview(67746)
    assert r['structural']['accepted_gate_visits']==6*67746
    assert not r['count_bound_sufficient_for_guard'] and not r['all_word_normal_return_proven']


@pytest.mark.parametrize('kwargs',[dict(target_pair=('compiled_gate_list_exact_semantics','literal_atom')),
    dict(space='sector72'),dict(coverage='four_fixed_words'),dict(phase_sensitive=False),
    dict(actual_inverse=True),dict(accounting='direct_G_to_V_plus_product_plus_zeta')])
def test_scope_inverse_and_double_count_refused(kwargs):
    with pytest.raises(ValueError):p.compiler_preview(p.binding.target_identity(),8,**kwargs)


def test_existing_composer_reused_with_missing_constants_and_wrong_identity_rejected():
    r=p.compiler_preview(p.binding.target_identity(),67746)
    assert r['conditional_composer']['status']=='UNRESOLVED'
    assert r['conditional_composer']['whole_P0_bound_exact'] is None
    assert r['actual_inverse_error'] is r['raw_to_literal_error'] is r['error_evidence'] is None
    assert r['RPE_accuracy']=='UNQUALIFIED' and not r['scientific_execution_ready'] and r['mandatory_stop']
    identity=p.binding.target_identity();identity['r']=32
    with pytest.raises(ValueError):p.compiler_preview(identity,8)


def test_supplied_artificial_uniform_inputs_never_become_E2_evidence():
    def proof(scope):return dict(kind='uniform_proof_input',scope=scope,norm='operator_2',
        phase_sensitive=True,provenance='ARTIFICIAL_SCALAR_ONLY',upper_exact='0')
    zsxx=proof('M_to_V_all_reachable_D_N');zsxx.update(compiler_interval_image_enclosed=True,covers_D_N=8,all_branches=True)
    r=p.compiler_preview(p.binding.target_identity(),8,node_matrix_error='0',model=p.uniform.ieee_model(),zsxx=zsxx,
        translation=proof('all_initial_translation_paths'),phase=proof('all_phase_accumulation_and_normalizations'))
    assert r['conditional_composer']['whole_P0_bound_exact'] is not None
    assert not r['conditional_composer']['proof_inputs_mathematically_verified']
    assert r['error_evidence'] is None and not r['all_law_qualified']


def test_grammar_ceiling_reuses_actual_symbolic_templates():
    templates=p.structural.symbolic_templates()['templates']
    r=p.grammar_ceiling(templates)
    assert (r['N_upper'],r['one_qubit_upper'],r['CX_upper'])==(67746,62994,4752)
    assert r['actual_H4_nodes'] is None
