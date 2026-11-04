"""Artificial native pairs;no saved molecular operator is evaluated."""
from copy import deepcopy
from fractions import Fraction as F
import pytest
from trottertracks import c_pauli_frame_enclosure as f,c_cx_structural_blocks as blocks
from trottertracks import c_arb_column_batch as columns
from trottertracks import c_pr_aware_df_fixed_certificate as fixed
from test_research_c_relative_pauli import g,r,dense


def checked(a,b,**kwargs):
    value=f.compare(a,b,parallel=False,**kwargs);assert f.check_binding(value,a,b)
    old=columns.compare(a,b,parallel=False)
    assert F(value['upper_exact'])>=F(old['lower_exact'])
    return value


@pytest.mark.parametrize('c,t',[(0,1),(1,0),(2,0)])
@pytest.mark.parametrize('px',range(4))
@pytest.mark.parametrize('pz',range(4))
def test_signed_CX_conjugation_exact_tableau(c,t,px,pz):
    import numpy as np
    from qiskit.quantum_info import Pauli
    from trottertracks.c_relative_pauli_enclosure import _cliffords
    other=({0,1,2}-{c,t}).pop();x=(((px&1)<<c)|(((px>>1)&1)<<t))|(1<<other)
    zz=(((pz&1)<<c)|(((pz>>1)&1)<<t))|(1<<other)
    P=Pauli((np.array([bool(zz>>i&1) for i in range(3)]),np.array([bool(x>>i&1) for i in range(3)])))
    Q=P.evolve(_cliffords()['cx'],qargs=[c,t],frame='s');a,b,sign=f._cx(x,zz,c,t)
    assert a==sum(int(v)<<i for i,v in enumerate(Q.x)) and b==sum(int(v)<<i for i,v in enumerate(Q.z))
    assert Q.phase==2*sign


@pytest.mark.parametrize('n',[2,3,9])
def test_Pauli_cross_boundary_cancellation_full_register(n):
    a=r(n,[g('z',[0]),g('cx',[0,1]),g('z',[0])]);b=r(n,[g('cx',[0,1])])
    coarse=blocks.compare(a,b,parallel=False);new=checked(a,b)
    assert float(F(coarse['upper_exact']))>1
    assert float(F(new['upper_exact']))<1e-50
    assert new['final_frame']==dict(x=0,z=0) and new['columns_evaluated']==0 and new['dimension']==1<<n


def test_many_large_local_errors_cancel_with_signed_propagation():
    a,b=r(3),r(3)
    for j in range(12):
        a['gates'] += [g('z',[0]),g('cx',[0,1]),g('z',[0]),g('x',[2]),g('cx',[1,2]),g('x',[2])]
        b['gates'] += [g('cx',[0,1]),g('cx',[1,2])]
    assert float(F(checked(a,b)['upper_exact']))<1e-50
    assert float(F(blocks.compare(a,b,parallel=False)['upper_exact']))>1


def test_internal_block_different_CX_skeleton():
    hh=[g('h',[0]),g('h',[1])]
    a,b=r(2,hh+[g('cx',[0,1])]+hh),r(2,[g('cx',[1,0])])
    assert float(F(checked(a,b)['upper_exact']))<1e-50


def test_global_SU2_sign_and_controlled_relative_phase():
    assert checked(r(2),r(2,phase=.125))['endpoint_upper_exact']!='0'
    assert float(F(checked(r(3),r(3,[g('p',[2],.25)]))['upper_exact']))>.24
    assert checked(r(2,[g('x',[0]),g('z',[0]),g('x',[0]),g('z',[0])]),r(2))['upper_exact']=='2'
    assert float(F(checked(r(1,[g('sx',[0])]*3),r(1,[g('sxdg',[0])]))['upper_exact']))<1e-50


def test_nontrivial_final_frame_is_retained_not_dropped():
    result=checked(r(9,[g('x',[8])]),r(9))
    assert result['final_frame']!=dict(x=0,z=0) and result['endpoint_upper_exact']=='2'
    assert result['upper_exact']=='2'


def test_float_selector_only_selects_and_cannot_lower_rigorous_bound(monkeypatch):
    monkeypatch.setattr(f,'_choose',lambda *args:(1,0))
    result=checked(r(2,[g('rz',[0],.125)]),r(2))
    assert F(result['upper_exact'])>F('0.06')
    assert result['frame_witnesses'] and all(v['selector_is_not_a_bound'] for v in result['frame_witnesses'].values())


def test_artificial_raw_primitive_existing_P0():
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import XXPlusYYGate,RZZGate
    from trottertracks.c1_h4_native_fair import compile_checked,CompilerProfile
    q=QuantumCircuit(3);q.append(XXPlusYYGate(.375,.25),[0,1]);q.crz(.125,1,0);q.append(RZZGate(.25).control(1,annotated=False),[2,0,1])
    a,b=fixed.record(compile_checked(q,CompilerProfile())),fixed.record(q)
    value=checked(a,b)
    # Always retain a valid enclosure, including an honestly coarse result.
    assert F(value['upper_exact'])>=0 and F(value['translation_upper_exact'])>0
    assert value['full_operator_coverage']


def test_continuous_cross_factor_cancellation_can_still_be_lost():
    a,b=r(2,[g('rz',[0],.25),g('cx',[0,1]),g('rz',[0],-.25)]),r(2,[g('cx',[0,1])])
    value=checked(a,b)
    assert float(F(value['upper_exact']))>.2


def test_support_growth_and_unqualified_certificate_refused():
    a,b=r(9,[g('cx',[q,q+1]) for q in range(4)]),r(9)
    value=f.compare(a,b,parallel=False)
    assert value['upper_exact'] is None and not value['full_operator_coverage']
    with pytest.raises(ValueError):f.check_binding(value,a,b)


@pytest.mark.parametrize('fault',['nonunitary','literal','phase','angle','gate'])
def test_bad_input_explicit_FAIL(fault):
    a=r(2,[g('rz',[0],.125)])
    if fault=='nonunitary':a['unitary']=False
    if fault=='literal':a['target']='literal_saved_atom'
    if fault=='phase':a['phase_hex']='inf'
    if fault=='angle':a['gates'][0]['params_hex']=['nan']
    if fault=='gate':a['gates'][0]['name']='unknown'
    with pytest.raises((ValueError,KeyError)):f.compare(a,r(2),parallel=False)


@pytest.mark.parametrize('fault',['source','plan','dimension','columns','phase_fit','calibration'])
def test_identity_and_qualification(fault):
    a,b=r(3,[g('rz',[0],.125)]),r(3)
    saved,plan=blocks.make_plan(a,b)
    if fault=='source':
        a['phase_hex']=.125.hex()
        with pytest.raises(ValueError):f.evaluate(a,b,saved=saved,plan=plan,parallel=False)
    elif fault=='plan':
        plan['factors']=[]
        with pytest.raises(ValueError):f.evaluate(a,b,saved=saved,plan=plan,parallel=False)
    else:
        value=f.evaluate(a,b,saved=saved,plan=plan,parallel=False)
        key={'dimension':'dimension','columns':'columns_evaluated','phase_fit':'phase_fit','calibration':'calibration_ready'}[fault]
        value[key]=512 if fault in ('dimension','columns') else True
        with pytest.raises(ValueError):f.check_binding(value,a,b)


def test_local_and_full_operator_dense_sanity():
    import numpy as np
    for n in (2,3):
        a=r(n,[g('rz',[0],.125),g('cx',[0,1]),g('ry',[1],.25)])
        b=r(n,[g('ry',[1],.25),g('cx',[0,1]),g('rz',[0],.125)])
        value=checked(a,b)
        assert float(F(value['upper_exact']))>=np.linalg.norm(dense(a)-dense(b),2)-1e-13


def test_parallel_frame_tables_equal_serial():
    a,b=r(3),r(3)
    for j in range(24):
        a['gates'] += [g('ry',[0],.1+j*.001),g('cx',[0,1])]
        b['gates'] += [g('ry',[0],.1+j*.001+2**-45),g('cx',[0,1])]
    serial=f.compare(a,b,parallel=False);parallel=f.compare(a,b,parallel=True)
    assert serial['upper_exact']==parallel['upper_exact'] and serial['frame_witnesses']==parallel['frame_witnesses']


def test_nonfinite_local_arithmetic_FAIL(monkeypatch):
    from flint import acb_mat
    monkeypatch.setattr(fixed,'_local',lambda *args:acb_mat([['nan',0],[0,1]]))
    with pytest.raises(ValueError):f.compare(r(2,[g('ry',[0],.125)]),r(2),parallel=False)


def test_old_column_only_calibration_admission_refuses_frame_proof(monkeypatch):
    from trottertracks import c_compiled_raw_local as old
    from test_research_c_compiled_raw_local import test_existing_column_only_integration_does_not_promote_local_proof
    monkeypatch.setattr(old,'compare',f.compare)
    test_existing_column_only_integration_does_not_promote_local_proof()
