"""Artificial matrix fixtures only; no molecular inputs, circuits or RNG."""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.sparse import csr_matrix

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import run_h4_p3_stage2 as runner
K=runner.modules()
from test_matched_partition_stage1 import toy
from qiskit import QuantumCircuit
from openfermion.circuits import low_rank

X=np.array([[0.,1.],[1.,0.]])
Z=np.diag([1.,-1.]); I=np.eye(2)


def dictionary(coefficients,ops):
    return SimpleNamespace(coefficients=coefficients,ops=[csr_matrix(a) for a in ops],
        matrix=sum((c*a for c,a in zip(coefficients,ops)),np.zeros_like(ops[0])))


def test_distinct_dictionaries_same_polynomial_different_normalization():
    arms=[dictionary([.3,.2],[X,Z]),dictionary([.5,-.2,.2],[X,X,Z])]
    bundles=[K.normalization(a.coefficients) for a in arms]
    assert bundles[0]['lambda_R']!=bundles[1]['lambda_R']
    assert bundles[0]['B']!=bundles[1]['B']
    for arm,b in zip(arms,bundles):
        raw,error,info=K.raw_event_mean(arm,b)
        expected=K.finite_taylor_operator(arm.matrix,float(K.H),4)
        np.testing.assert_allclose(b['beta']*raw,expected,rtol=0,atol=2e-15)
        assert error>0 and info['finite_bit_law_evaluated'] is False
        assert abs(float(b['decimal50']['B'])-b['B'])<=b['B_uncertainty']


def test_raw_moment_matches_explicit_small_event_sum_with_signed_order():
    arm=dictionary([.3,-.2],[X,Z]);b=K.normalization(arm.coefficients)
    actual,_,_=K.raw_event_mean(arm,b);explicit=np.zeros_like(actual)
    prob=[abs(c)/b['lambda_R'] for c in arm.coefficients]
    signed=[np.sign(c)*p.toarray() for c,p in zip(arm.coefficients,arm.ops)]
    for k,pi in zip((0,2,4),b['order_probabilities']):
        angle=np.arctan(b['dimensionless_step_time']/(k+1))
        for picks in product(range(2),repeat=k+1):
            products=I.copy()
            for index in picks[:-1]:products=signed[index]@products
            rotation=np.cos(angle)*I-1j*np.sin(angle)*signed[picks[-1]]
            probability=pi*np.prod([prob[i] for i in picks])
            explicit+=probability*((-1)**(k//2))*rotation@products
    np.testing.assert_allclose(actual,explicit,atol=2e-15,rtol=0)


def test_noncommuting_s2_scalar_nu_and_plus_im_sign():
    t=.4*Z;v=.3*X;et=expm(-1j*float(K.DELTA)/2*t);tail=expm(-1j*float(K.DELTA)*v)
    phi=np.array([2.,1.j]);nu=F(5);phase=np.exp(-.37j)
    row=K.signal_propagation(et,tail,phi,nu,phase)
    expected=np.vdot(phi,phase*np.linalg.matrix_power(et@tail@et,8)@phi)/5
    np.testing.assert_allclose(K.value(row),expected,atol=2e-15,rtol=0)
    assert row['imag']==pytest.approx(expected.imag) and abs(row['imag'])>.01
    assert row['norm_divisions']==row['scalar_phase_applications']==1


def test_raw_reference_division_exactly_once_and_exact_norm():
    state=np.array([2.,1.j]);nu=K.exact_norm(state);assert nu==5
    raw={'real':{'lower':'-2','upper':'-1'},'imag':{'lower':'1','upper':'3'}}
    interval,display=K.physical_reference(raw,nu)
    assert interval['real']==dict(lower='-2/5',upper='-1/5')
    assert interval['imag']==dict(lower='1/5',upper='3/5')
    assert display['imag']==pytest.approx(.4)


@pytest.mark.parametrize('weight',[.21,-.17])
def test_artificial_independent_build_and_six_signal_pipeline(weight):
    data=toy(weight,[[.7,.19],[.19,-.4]])
    native=K.native_context(4);t,v=K.canonical(data,native);vbar=float(np.trace(v).real/len(v))
    with patch.object(K,'VBAR',vbar), \
         patch.object(QuantumCircuit,'__init__',side_effect=AssertionError('CIRCUIT')), \
         patch.object(low_rank,'low_rank_two_body_decomposition',side_effect=AssertionError('FACTOR')):
        t,v0,arms,checks=K.build(data,workers=2)
        assert all(x['status']=='PASS' for x in checks.values())
        phi=np.array([complex((i%3)-1,(i%5)-2) for i in range(len(t))])/8
        nu=K.exact_norm(phi);full=t+v0+(data['constant']+vbar)*np.eye(len(t))
        ref=np.vdot(phi,expm(-1j*float(K.T)*full)@phi)
        raw={a:dict(lower=str(F(float(x))),upper=str(F(float(x))))
             for a,x in [('real',ref.real),('imag',ref.imag)]}
        calls=[];original=K.paired_polynomial_action
        def paired(*args,**kwargs):
            calls.append((args[2:5],kwargs));return original(*args,**kwargs)
        K.OPERATIONS.clear()
        with patch.object(K,'paired_polynomial_action',side_effect=paired):
            result=K.evaluate(t,v0,arms,phi,nu,data['constant'],full,raw,F(0))
    assert len(result['signals'])==6 and len(result['normalization'])==2
    assert result['numerical_status']=='PASS'
    assert all(x['status']=='PASS' for x in result['closures'].values())
    assert len(calls)==8 and all(c[0]==(float(K.DELTA),4,4) and c[1]['normalization']==1 for c in calls)
    assert K.OPERATIONS['physical_complex_signals']==6
    assert K.OPERATIONS['common_expm']==K.OPERATIONS['common_spectral_eigendecompositions']==2
    assert K.OPERATIONS['paired_polynomial_matvecs']==160
    np.testing.assert_allclose(K.value(result['signals']['nom_P']),K.value(result['signals']['nom_DF']),atol=1e-13)
    assert 'NO_AUTOMATIC_FOLLOWUP' in result['stop'] or result['stop']=='STOP_STAGE2_UNRESOLVED_COMPONENTS'


def test_nonhermitian_factor_rejected_and_scalar_defect_detected():
    data=toy();bad=dict(data,g_matrices=data['g_matrices'].copy())
    bad['g_matrices'][0,0,1]+=.01j
    with pytest.raises(ValueError,match='HERMITICITY'):K.build(bad,workers=1)
    assert K.check(X,X+.01*I)['status']=='FAIL'


def test_uncertainty_classification_and_bias_components():
    a=dict(real=0.,imag=0.,u_num=.1);b=dict(real=.1,imag=0.,u_num=.1)
    assert K.closure(a,b)['status']=='PASS'
    assert K.closure(a,dict(b,real=.3))['status']=='UNRESOLVED'
    assert K.closure(a,dict(b,real=1.))['status']=='FAIL'
    row=K.bias(a,b,bind=.03,ref_width={'real':.04,'imag':.05},transfer=.02)
    assert row['real']['u_num']==pytest.approx(.2)
    assert row['real']['u_bind']==.03 and row['real']['u_ref_transfer']==pytest.approx(.06)
    assert row['real']['status']=='UNRESOLVED'


def test_nominal_nonpass_stops_before_raw_moments():
    data=toy();native=K.native_context(2);t,v=K.canonical(data,native)
    vbar=float(np.trace(v).real/4)
    with patch.object(K,'VBAR',vbar):
        t,v0,arms,_=K.build(data,workers=1)
        phi=np.array([1.,0.,1.j,0.]);raw={a:dict(lower='0',upper='0') for a in ('real','imag')}
        with patch.object(K,'closure',return_value={'status':'FAIL'}), \
             patch.object(K,'raw_event_mean',side_effect=AssertionError('RAW_STARTED')):
            with pytest.raises(ValueError,match='NOMINAL_COMPARISON_NONPASS'):
                K.evaluate(t,v0,arms,phi,F(2),data['constant'],t+v,raw,F(0))
