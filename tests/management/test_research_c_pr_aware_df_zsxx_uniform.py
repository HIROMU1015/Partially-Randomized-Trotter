"""Artificial uniform families/conditional proofs;no molecular inputs."""
from copy import deepcopy
from fractions import Fraction as F
import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c_pr_aware_df_zsxx_uniform as u
from trottertracks import c_pr_aware_df_zsxx_enclosure as z


@pytest.mark.parametrize('L',[1,4,16,256,1024,2**20])
def test_a_priori_run_bound_includes_unseen_words_and_underflow(L):
    r=u.run_bound(L,node_matrix_error='0',model=u.ieee_model())
    assert r['status']=='CONDITIONAL_ENCLOSURE'
    p=u.product_rate('0');eta=F(r['input_product_error_exact'])
    assert eta==L*p/(1-L*p)>0
    # Exact geometric recurrence for small L;closed rational comparison.
    if L<=16:assert eta>=(1+p)**L-1
    assert F(r['near_unitarity_defect_exact'])==2*eta+eta*eta
    assert r['determinant_centers']==['1','i','-1','-i']
    assert F(r['determinant_distance_upper_exact'])==2*eta+eta*eta
    assert not r['compiler_interval_image_enclosed'] and not r['model_verified']


def test_missing_matrix_eval_or_wrong_hardware_model_never_becomes_zero():
    assert u.run_bound(32,model=u.ieee_model())['status']=='UNRESOLVED'
    assert u.run_bound(32,node_matrix_error='0')['status']=='UNRESOLVED'
    for field,value in [('gradual_underflow',False),('platform_verified',True),('contraction','unverified fastmath')]:
        model=u.ieee_model();model[field]=value
        with pytest.raises(ValueError,match='MODEL'):u.run_bound(8,node_matrix_error='0',model=model)
    assert u.run_bound(2**100,node_matrix_error='0',model=u.ieee_model())['status']=='UNRESOLVED'


@pytest.mark.parametrize('pair',[('0','4'),('-1/1000000000000','1/1000000000000'),
    (str(u.PI/2-u.EPS),str(u.PI/2+u.EPS)),(str(u.PI-u.EPS),str(u.PI+u.EPS))])
def test_interval_predicates_cover_both_sides_and_multiple_branches(pair):
    r=u.theta_branch_cover(*pair,model=u.ieee_model())
    assert r['status']=='CONDITIONAL_BRANCH_SUPERSET' and not r['compiler_interval_image_enclosed']
    lo,hi=map(F,pair)
    for q in [lo,hi,(lo+hi)/2,(3*lo+hi)/4,(lo+3*hi)/4]:
        theta=float(q)
        # Only evaluate floats actually within the original interval.
        if not lo<=F(theta)<=hi:continue
        p=z.captured_angles(theta,.3,-.7,.2)
        names=[x['gate'] for x in p['output']['gates']]
        route=('general_SX_or_X' if 'x' in names or names.count('sx')==2
               else 'one_SX' if names.count('sx')==1 else 'zero_SX')
        assert route in r['routes']
    assert r['uniform_Rust_error'] is None
    if pair==('0','4'):assert set(r['routes'])=={'zero_SX','one_SX','general_SX_or_X'}


def test_ideal_image_is_outward_for_unseen_parameters_but_not_Rust_image():
    from flint import ctx
    image=u.ideal_template_image([('rz',('-1','1')),('sx',None),('ry',('0','4'))],phase_interval=('-1/5','1/5'))
    with z._BALL_LOCK,ctx.workprec(z.PRECISION):
        for a,b,p in [(-.9,.13,.01),(.7,3.7,-.1),(.123,.987,.15)]:
            A=z._product([('rz',a),('sx',None),('ry',b)],p)
            for i in range(2):
                for j in range(2):
                    for n,part in enumerate([A[i,j].real,A[i,j].imag]):
                        lo,hi=map(F,image['entry_real_imag_bounds'][i][j][n])
                        assert lo<=z._fraction(part.lower())<=z._fraction(part.upper())<=hi
    assert not image['compiler_interval_image_enclosed'] and not image['Rust_branch_coverage']


def test_source_order_fold_with_long_cancellation_and_noncommuting_words():
    from flint import ctx
    # A verifier replay of the source fold;not instrumentation of private Rust.
    letters=[('rz',.31),('rz',-.31),('sx',None),('x',None),('rz',.13)]
    with z._BALL_LOCK,ctx.workprec(z.PRECISION):
        matrices=[];a=F()
        for name,t in letters:
            B=z.gate_matrix(name,t)
            chosen=np.array([[complex(float(B[i,j].real.mid()),float(B[i,j].imag.mid())) for j in range(2)] for i in range(2)])
            matrices.append(chosen);a=max(a,F(z._norm(B-z._matrix(chosen))['upper_exact']))
        words=[[0,1]*512,[0,2,4,3]*192]
        for indices in words:
            M=np.eye(2,dtype=complex)
            for k in indices:
                B=matrices[k]
                M=np.array([[B[i,0]*M[0,j]+B[i,1]*M[1,j] for j in range(2)] for i in range(2)])
            G=z._product([letters[k] for k in indices])
            observed=F(z._norm(G-z._matrix(M))['upper_exact'])
            budget=u.run_bound(len(indices),node_matrix_error=a,model=u.ieee_model())
            theoretical=F(budget['input_product_error_exact'])
            assert observed<theoretical
            A=z._matrix(M);defect=z._norm(A.conjugate().transpose()*A-z.gate_matrix('id'))
            assert F(defect['upper_exact'])<2*F(budget['near_unitarity_defect_exact'])
            power=sum(1 if letters[k][0]=='sx' else 2 if letters[k][0]=='x' else 0 for k in indices)%4
            center=[1,1j,-1,-1j][power]
            determinant=A[0,0]*A[1,1]-A[0,1]*A[1,0]
            assert z._fraction((determinant-center).abs_upper())<F(budget['determinant_distance_upper_exact'])
        # Reversing noncommuting letters is a semantic error,not rounding.
        gs=[letters[k] for k in [0,2,4]]
        wrong=z._product(list(reversed(gs)))
        assert F(z._norm(z._product(gs)-wrong)['lower_exact'])>F(1,1000)


@pytest.mark.parametrize('theta',[0.,-0.,math.ulp(0.),3*math.ulp(0.),-3*math.ulp(0.),
    .31,-.71,math.pi,float.fromhex('0x1.fffffffffffffp+1023')])
def test_official_CRZ_uniform_half_error_and_RZZ_copy(theta):
    from qiskit.circuit.library import CRZGate,RZZGate,RZGate
    from trottertracks.c_pr_aware_df_stage_b2_compat import SharedQiskitCalls
    lock=SharedQiskitCalls(lambda k,l,f:f())
    def definitions():return CRZGate(theta).definition,RZZGate(theta).definition,RZGate(theta).inverse()
    crz,rzz,inv=lock('kernel_build','artificial rule family',definitions)
    assert [x.operation.name for x in crz.data]==['rz','cx','rz','cx']
    h=float(crz.data[0].operation.params[0]);minus=float(crz.data[2].operation.params[0])
    assert minus==-h and abs(F(h)-F(theta)/2)<=u.H
    assert [x.operation.name for x in rzz.data]==['cx','rz','cx']
    assert F(float(rzz.data[1].operation.params[0]))==F(theta)
    assert F(float(inv.params[0]))==-F(theta)
    assert u.translation_rule('CRZ_definition')['operator_upper_exact']==str(u.H)
    assert u.translation_rule('RZZ_definition')['operator_upper_exact']=='0'
    assert u.translation_rule('generic_controlled_XX')['status']=='UNRESOLVED'


PHASE=dict(kind='conditional_phase_arithmetic',accumulator_correctly_rounded=True,
    no_accumulator_underflow_overflow=True,euclidean_quotient_relation=True,
    verified_for_fixed_wrapper=False,absolute_remainder_error='1/1000000000000000000000000000000')


def test_phase_budget_keeps_true_period_correction_cast_and_unknown():
    r=u.phase_bound(1000,'100',remainder_model=PHASE)
    assert r['status']=='CONDITIONAL_ENCLOSURE'
    assert all(F(r[k])>0 for k in ['accumulation_exact','final_cast_exact','remainder_exact','true_period_correction_exact'])
    assert F(1,10**16)<u.period_gap()<F(3,10**16)
    assert u.phase_bound(1000,'100')['status']=='UNRESOLVED'
    assert u.phase_bound(0,str(2**1024),remainder_model=PHASE)['status']=='UNRESOLVED'
    bad=dict(PHASE,verified_for_fixed_wrapper=True)
    with pytest.raises(ValueError,match='PHASE_MODEL'):u.phase_bound(1,'1',remainder_model=bad)


def test_actual_negative_subnormal_phase_wrap_needs_true_pi_correction():
    from qiskit import QuantumCircuit
    from flint import acb,ctx
    q=QuantumCircuit(1);p=-math.ulp(0.);q.global_phase=p;r=float(q.global_phase)
    with z._BALL_LOCK,ctx.workprec(z.PRECISION):
        D=(acb(0,1)*z._real(p)).exp()-(acb(0,1)*z._real(r)).exp()
        lower=z._fraction(D.abs_lower());upper=z._fraction(D.abs_upper())
    assert r==2*math.pi and F(2,10**16)<lower<=upper<F(3,10**16)
    # This is a captured setter example,not a uniform libm accuracy proof.


def proof(scope,eps='1/1000000000000',**kw):
    return dict(kind='uniform_proof_input',scope=scope,norm='operator_2',phase_sensitive=True,
                provenance='hypothetical artificial theorem input',upper_exact=eps,**kw)


def test_whole_P0_uses_length_amortization_not_sixN_times_samplemax():
    N=100;args=dict(conditions=u.CONDITIONS,node_matrix_error='0',model=u.ieee_model(),exact_unitary_targets=True,
        zsxx=proof('M_to_V_all_reachable_D_N',compiler_interval_image_enclosed=True,covers_D_N=N,all_branches=True),
        translation=proof('all_initial_translation_paths'),phase=proof('all_phase_accumulation_and_normalizations'))
    r=u.uniform_whole_bound(N,**args);p=u.product_rate('0')
    assert F(r['run_product_sum_upper_exact'])==6*N*p/(1-N*p)
    assert F(r['ZSXX_sum_upper_exact'])==N*F(args['zsxx']['upper_exact'])
    assert not r['rejected_rewrites_charged'] and not r['direct_bound_also_charged']
    assert r['whole_law_bound'] is r['actual_H4_bound'] is None
    args['phase']=None
    assert u.uniform_whole_bound(N,**args)['missing']==['phase']
    args['conditions']=dict(u.CONDITIONS,target_none=False)
    assert u.uniform_whole_bound(N,**args)['status']=='UNRESOLVED'


def test_captured_pair_reference_box_cannot_masquerade_as_compiler_image():
    captured=z.captured_matrix(np.eye(2),reference_radius='1/1000000000000000')
    bad=proof('M_to_V_all_reachable_D_N',compiler_interval_image_enclosed=False,covers_D_N=10,all_branches=True)
    bad['upper_exact']=captured['region_distance']['upper_exact']
    with pytest.raises(ValueError,match='COMPILER_IMAGE'):
        u.uniform_whole_bound(10,conditions=u.CONDITIONS,node_matrix_error='0',model=u.ieee_model(),zsxx=bad,exact_unitary_targets=True)


def test_nonunitary_wrong_norm_or_phase_fit_cannot_enter_telescope():
    assert u.uniform_whole_bound(10,conditions=u.CONDITIONS)['missing']==['exact_unitary_targets']
    for change in [dict(norm='max_entry'),dict(phase_sensitive=False),dict(provenance='')]:
        row=proof('all_initial_translation_paths');row.update(change)
        with pytest.raises(ValueError,match='UNIFORM_PROOF_SCOPE'):
            u.uniform_whole_bound(10,conditions=u.CONDITIONS,translation=row,exact_unitary_targets=True)


def test_passive_import_has_no_numeric_or_science_side_effect():
    import subprocess
    subprocess.run([sys.executable,'-B','-c',"import sys;sys.path.insert(0,'src');from trottertracks import c_pr_aware_df_zsxx_uniform;assert not any(k in sys.modules for k in ('numpy','qiskit','flint'))"],cwd=ROOT,check=True)
    result=subprocess.run([sys.executable,'-B','scripts/check_research_c_pr_aware_df_zsxx_uniform.py','--contract','unused','--execute'],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==2 and 'unrecognized arguments' in result.stderr


def test_source_topology_ceiling_and_actual_symbolic_U_witness():
    from trottertracks.c_pr_aware_df_c2a import symbolic_templates,P0
    from trottertracks.c_pr_aware_df_stage_b2_compat import SharedQiskitCalls
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import BasisTranslator
    templates=symbolic_templates()['templates']
    bound=u.grammar_node_bound(official_topology_coverage=True)
    cost=lambda c:5*c.get('u',0)+sum(v for k,v in c.items() if k!='u')
    assert cost(templates['XX']['lowered_counts'])==bound['translated_weights_upper']['XX']==44
    assert cost(templates['controlled_RZZ']['lowered_counts'])==bound['translated_weights_upper']['CRZZ']==106
    assert bound['initial_nodes_upper']==67731 and bound['actual_H4_node_count'] is None
    assert u.grammar_node_bound()['status']=='UNRESOLVED'
    lock=SharedQiskitCalls(lambda k,l,f:f())
    def translate():
        q=QuantumCircuit(1);q.u(Parameter('a'),Parameter('b'),Parameter('c'),0)
        return PassManager([BasisTranslator(SessionEquivalenceLibrary,P0['basis_gates'])]).run(q)
    translated=lock('kernel_build','artificial symbolic U witness',translate)
    assert translated.size()<=5 and set(translated.count_ops())<=set(P0['basis_gates'])


def test_actual_MPFR_marker_cancellation_is_instance_only():
    import symengine
    from qiskit.circuit import Parameter,ParameterExpression
    marker=Parameter('artificial_phase_accumulator')
    e=ParameterExpression(marker._parameter_symbols,
        marker._symbol_expr+symengine.RealMPFR('1.0',160))
    terms=[1e16,.125,-1e16]*128
    for value in terms:e=e+value
    expected=sum(map(F,terms),F())
    assert F(float(e.bind({marker:-1.})))==expected==16
    r=u.phase_bound(len(terms)+2,sum(map(lambda x:abs(F(x)),terms),F())+2,
                    remainder_model=u.conditional_phase_model())
    assert r['status']=='CONDITIONAL_ENCLOSURE' and not r['model_verified']


def test_preflight_preserves_scope_and_rejects_source_input_mismatch(tmp_path,monkeypatch):
    from trottertracks.c1_stage1_execution import save,sha
    from trottertracks.c_pr_aware_df_c2a import P0
    manifest=tmp_path/'SOURCE.json';save(manifest,dict(files={}))
    saved=tmp_path/'saved.npz';saved.write_bytes(b'artificial identity bytes')
    c=dict(scope=u.SCOPE,implementation=u.VERSION,executable=False,status='BLOCKED_UNIFORM_RUST_IMAGE',
        maximum_logical_units=None,precision_bits=z.PRECISION,compiler=P0,retry=0,resume=0,
        source_manifest='SOURCE.json',source_manifest_sha256=sha(manifest),saved_inputs={'saved.npz':sha(saved)},
        packages={},external_python_source_hashes={})
    path=tmp_path/'contract.json';save(path,c)
    monkeypatch.setattr(np,'load',lambda *a,**kw:pytest.fail('no array decode'))
    assert u.preflight(tmp_path,path)['executable'] is False
    saved.write_bytes(b'changed')
    with pytest.raises(ValueError,match='SOURCE_OR_INPUT'):u.preflight(tmp_path,path)
    saved.write_bytes(b'artificial identity bytes');manifest.write_bytes(b'changed')
    with pytest.raises(ValueError,match='SOURCE_MANIFEST'):u.preflight(tmp_path,path)


def test_artificial_E2E_separates_validated_conditional_captured_and_unknown():
    from trottertracks.c_pr_aware_df_zsxx_uniform_artificial import evidence
    r=evidence()
    assert r['ideal_template_image']['status']=='VALIDATED_IDEAL_TEMPLATE_IMAGE'
    assert all(x['status']=='CONDITIONAL_ENCLOSURE' for x in r['run_models'])
    assert r['unknown_actual_whole_P0']['whole_P0_bound_exact'] is None
    assert r['actual_H4_bound'] is r['whole_law_bound'] is None
    assert r['new_H4_scientific_calls']==0 and not r['template_image_is_Rust_image']
