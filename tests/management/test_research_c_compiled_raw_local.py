"""Invented native records/circuits only; never saved molecular records."""
from copy import deepcopy
from fractions import Fraction as F
import math

import pytest
from trottertracks import c_compiled_raw_local as local
from trottertracks import c_pr_aware_df_fixed_certificate as fixed
from trottertracks import c_arb_column_batch as old


def gate(name, qs, *values):
    return dict(name=name, qubits=qs, params_hex=[float(v).hex() for v in values])


def record(n, gates=(), phase=0.):
    return dict(fixed.identity(n), gates=list(gates), phase_hex=float(phase).hex())


def analytic_phase(t):
    from flint import ctx, acb
    from trottertracks import c_pr_aware_df_zsxx_enclosure as z
    with ctx.workprec(384): return z._fraction(((acb(0,1)*z._real(F(t))).exp()-1).abs_upper())


@pytest.mark.parametrize('n', [1,3,9])
@pytest.mark.parametrize('phase', [.125,-.375])
def test_phase_operator_full_space_analytic_containment(n, phase):
    a,b=record(n),record(n,phase=phase);r=local.compare(a,b,parallel=False)
    assert F(r['upper_exact']) >= analytic_phase(phase)
    assert r['dimension']==1<<n and r['full_operator_coverage'] and r['columns_evaluated']==0
    assert r['phase_fit'] is False and local.check_binding(r,a,b)


@pytest.mark.parametrize('name', ['rz','p','ry'])
def test_local_small_rotation_telescope(name):
    a,b=record(2),record(2,[gate(name,[1],.0625)])
    r=local.compare(a,b,parallel=False)
    actual=analytic_phase(.0625 if name=='p' else .03125)
    assert F(r['upper_exact']) >= actual
    assert float(F(r['upper_exact']))<.07


@pytest.mark.parametrize('gates', [[gate('sx',[0])]*3, [gate('s',[0])]*3])
def test_discrete_SU2_sign_preserves_compensating_phase(gates):
    inverse='sxdg' if gates[0]['name']=='sx' else 'sdg'
    a,b=record(1,gates),record(1,[gate(inverse,[0])])
    r=local.compare(a,b,parallel=False)
    assert float(F(r['upper_exact']))<1e-50
    assert any(w['sign']==-1 for w in r['block_witnesses'].values())
    assert F(r['global_delta_pi_exact'])%2==0


def test_order_difference_does_not_become_equality():
    a=record(2,[gate('sx',[0]),gate('rz',[0],.5)])
    b=record(2,list(reversed(a['gates'])))
    r=local.compare(a,b,parallel=False);previous=old.compare(a,b,parallel=False)
    assert F(r['upper_exact'])>F('1/10')
    assert F(r['upper_exact'])>=F(previous['lower_exact'])


def test_disjoint_CX_ordering_and_inverse_cancellations():
    a=record(4,[gate('cx',[0,1]),gate('rz',[2],.25),gate('cx',[0,1]),gate('cx',[2,3])])
    b=record(4,[gate('rz',[2],.25),gate('cx',[2,3])])
    r=local.compare(a,b,parallel=False)
    assert float(F(r['upper_exact']))<1e-50
    assert local.plan(a,b)['skeletons'][0]['cancelled_inverse_pair_count']==1
    c=record(4,[gate('cx',[0,1]),gate('cx',[2,3])])
    d=record(4,list(reversed(c['gates'])))
    assert float(F(local.compare(c,d,parallel=False)['upper_exact']))<1e-50


def test_known_commutations_are_explicit_opt_in():
    a=record(2,[gate('cx',[0,1]),gate('rz',[0],.25),gate('cx',[0,1])])
    b=record(2,[gate('rz',[0],.25)])
    assert local.compare(a,b,parallel=False)['upper_exact'] is None
    r=local.compare(a,b,parallel=False,commute_CX=True)
    assert float(F(r['upper_exact']))<1e-50


def test_unmatched_noncommuting_barrier_no_dense_fallback():
    r=local.compare(record(2,[gate('cx',[0,1])]),record(2,[gate('cx',[1,0])]),parallel=False)
    assert r['status']=='NOT_APPLICABLE_NO_SHARED_CX_SKELETON'
    assert r['upper_exact'] is None and not r['full_operator_coverage'] and not r['dense_fallback_used']


@pytest.mark.parametrize('n', [2,3])
def test_existing_P0_MPFR160_definitions_and_old_Arb_agree(n):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import XXPlusYYGate,RZZGate
    from trottertracks.c1_h4_native_fair import compile_checked,CompilerProfile
    q=QuantumCircuit(n);q.append(XXPlusYYGate(.375,.25),[0,1]);q.crz(.125,1,0);q.rz(.2,0)
    if n==3:q.append(RZZGate(.25).control(1,annotated=False),[2,0,1])
    a,b=fixed.record(compile_checked(q,CompilerProfile())),fixed.record(q)
    r=local.compare(a,b,parallel=False);previous=old.compare(a,b,parallel=False)
    assert F(r['upper_exact'])>=F(previous['lower_exact'])
    assert float(F(r['upper_exact']))<1e-12
    assert r['unique_definitions']>=2 and r['translation_upper_exact']!='0'
    assert local.check_binding(r,a,b)


def test_larger_local_accumulation_remains_an_upper_bound():
    n=60;a=record(1,[gate('rz',[0],.125)]*n)
    b=record(1,[gate('rz',[0],7.5+2**-40)])
    r=local.compare(a,b,parallel=False)
    assert F(r['upper_exact'])>=analytic_phase(F(1,2**41))
    assert float(F(r['upper_exact']))<1e-11


def test_controlled_relative_phase_not_scalar_fit():
    a,b=record(3),record(3,[gate('p',[2],.25)])
    r=local.compare(a,b,parallel=False)
    assert F(r['upper_exact'])>=analytic_phase(.25)
    assert F(r['global_phase_upper_exact'])>0 and F(r['blocks_upper_exact'])>0


@pytest.mark.parametrize('fault', ['literal','phase','angle','register','gate'])
def test_invalid_inputs_fail_explicitly(fault):
    a=record(2,[gate('rz',[0],.125)])
    if fault=='literal':a['target']='literal_saved_atom'
    if fault=='phase':a['phase_hex']='inf'
    if fault=='angle':a['gates'][0]['params_hex']=['nan']
    if fault=='register':a['qubits']=True
    if fault=='gate':a['gates'][0]['name']='unknown'
    with pytest.raises((ValueError,KeyError)):local.compare(a,record(2),parallel=False)


def test_nonfinite_Arb_matrix_fails(monkeypatch):
    from flint import acb_mat
    monkeypatch.setattr(fixed,'_local',lambda *a:acb_mat([['nan',0],[0,1]]))
    with pytest.raises(ValueError):local.compare(record(1,[gate('sx',[0])]),record(1),parallel=False)


def test_source_record_and_plan_identity_no_mutation():
    a,b=record(3,[gate('p',[1],.1)]),record(3);original=deepcopy(a)
    r=local.compare(a,b,parallel=False);assert a==original
    altered=deepcopy(a);altered['phase_hex']=.125.hex()
    with pytest.raises(ValueError):local.check_binding(r,altered,b)
    p=local.plan(a,b);p['block_pairs'][0]['qubit']=2
    with pytest.raises(ValueError,match='PLAN_IDENTITY'):local.evaluate(p,parallel=False)


def test_local_proof_cannot_impersonate_full_column_certificate():
    a,b=record(3),record(3);r=local.compare(a,b,parallel=False)
    forged=deepcopy(r);forged['columns_evaluated']=8
    with pytest.raises(ValueError):local.check_binding(forged,a,b)
    lifted=deepcopy(r);lifted['dimension']=512
    with pytest.raises(ValueError):local.check_binding(lifted,a,b)
    assert r['calibration_ready'] is r['Stage_C_ready'] is False


@pytest.mark.parametrize('fault',['register','partition','source'])
def test_structural_witness_is_not_just_a_self_hash(fault):
    p=local.plan(record(3,[gate('rz',[0],.1)]),record(3))
    if fault=='register':p['qubits']=9
    if fault=='partition':p['block_pairs']=[]
    if fault=='source':p['left_id']='0'*64
    p['plan_id']=local._digest({k:v for k,v in p.items() if k!='plan_id'})
    with pytest.raises(ValueError):local.evaluate(p,parallel=False)


def test_parallel_local_blocks_same_bound_as_serial():
    a,b=record(3),record(3)
    for j in range(70):
        a['gates'] += [gate('ry',[0],.1+j*.001),gate('cx',[0,1])]
        b['gates'] += [gate('ry',[0],.1+j*.001+2**-45),gate('cx',[0,1])]
    serial=local.compare(a,b,parallel=False);parallel=local.compare(a,b,parallel=True)
    assert serial['upper_exact']==parallel['upper_exact']
    assert serial['block_witnesses']==parallel['block_witnesses']


def test_existing_column_only_integration_does_not_promote_local_proof():
    from flint import acb_mat
    from trottertracks import c_fresh_iid_calibration as fresh
    from trottertracks import c_iid_calibration_integration as integration
    from trottertracks.c_b1_b2_scope_integration import identity
    from trottertracks.c_pr_aware_df_stage_c2 import choice_grid
    from trottertracks.c_pr_aware_df_fixed_car import pack
    c=fresh.make_context(source_id='3'*64,identity=identity(),atom_ids=['INVENTED:atom'],
        table=choice_grid([1.]),event_angle_bits={'INVENTED:atom':0.0.hex()})
    s=fresh.sample_from_integers(c,dataset_id='INVENTED-local-integration',sample_index=0,
        integers=[[0]*64],unit='kernel',axis='kernel')
    a,b=record(3),record(3)
    raw=integration.bind_operator(b,s,c,0,role='raw')
    compiled=integration.bind_operator(a,s,c,0,role='compiled',raw_id=raw['record_id'])
    target=dict(target=integration.population.TARGET,candidate=c['identity']['candidate'],invented_dimension=4)
    reference=dict(status='VALIDATED_LITERAL_REFERENCE_BLOCKS',target=integration.population.TARGET,
        binding=dict(target=target,target_id=fresh.digest(target),word=s['kernels'][0]['word'],word_id=s['kernels'][0]['word_id']),
        system_dimension=4,precision_bits=192,blocks=[dict(indices=[i],entries=pack(acb_mat([[1]]))) for i in range(4)])
    oldcert=integration.evaluate_word(raw,compiled,reference,sample=s,context=c,occurrence=0,target=target,parallel=False)
    oldcert['compiled_raw']=local.compare(a,b,parallel=False)
    with pytest.raises(ValueError,match='NUMERICAL_QUALIFICATION'):
        integration.check_certificate(oldcert,raw,compiled,sample=s,context=c,occurrence=0,target=target)
