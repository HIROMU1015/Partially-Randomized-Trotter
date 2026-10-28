"""Artificial masses/bases only; no saved molecule/trajectory decoding."""
from pathlib import Path
import sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c_df_basis_switch as s


def tensors(a=1., b=1.):
    J=np.array([np.zeros((2,2)),a*np.array([[1.,-.4],[-.4,-.2]]),b*np.array([[1.,-.4],[-.4,-.2]])])
    return J


def test_zero_tail_is_not_epsilon_law():
    spec=s.Partition(('a','b'),ld=1)
    J=np.zeros((3,2,2))
    exact=s.exact_law(J,spec); smooth=s.smooth_terms(J,spec,np)
    assert exact['lambda_R']==0 and exact['q_basis']=={} and exact['P_same'] is None
    assert exact['expected_switches']==0
    assert smooth['mass']>0 and smooth['expected_switches']==pytest.approx(63.5)


def test_same_group_and_r_one():
    for spec in [s.Partition(('a','a'),ld=1),s.Partition(('a','b'),ld=1,r=1)]:
        assert s.exact_law(tensors(),spec)['expected_switches']==0
        assert s.smooth_terms(tensors(),spec,np)['expected_switches']==0


@pytest.mark.parametrize('scale',[1.,1e-240,-1e-240,-3.])
def test_stable_general_nonPSD_signed_mass(scale):
    spec=s.Partition(('a','b'),ld=1,epsilon=1e-250)
    z=s.exact_law(tensors(scale,3*scale),spec)
    assert z['lambda_R']>0 and np.isfinite(z['lambda_R'])
    assert z['q_basis']==pytest.approx({'a':.25,'b':.75})
    assert z['expected_switches']==pytest.approx(127*.375)
    soft=s.smooth_terms(tensors(scale,3*scale),spec,np)
    assert np.isfinite(soft['mass']) and soft['expected_switches']==pytest.approx(z['expected_switches'],rel=1e-8)


def test_mass_switch_tradeoff_and_nonconvexity():
    spec=s.Partition(('a','b'),ld=1)
    x=s.exact_law(tensors(1,1),spec); y=s.exact_law(tensors(1,9),spec)
    assert y['lambda_R']>x['lambda_R'] and y['expected_switches']<x['expected_switches']
    # Same lambda, different distribution; not a reweighted lambda-only term.
    lo=s.exact_law(tensors(0,2),spec); mid=s.exact_law(tensors(1,1),spec)
    hi=s.exact_law(tensors(2,0),spec)
    assert mid['lambda_R']==pytest.approx(lo['lambda_R'])
    assert mid['expected_switches']>.5*(lo['expected_switches']+hi['expected_switches'])


def test_classical_coefficient_gradient():
    c=np.array([[.1,-.7,.3],[.9,-.2,.4]])
    keys=('a','b');grad=s.exact_coefficient_gradient(c,keys)
    def f(v):
        mass=abs(v).sum(axis=1);q=mass/mass.sum();return 127*(1-np.sum(q*q))
    for ix in np.ndindex(c.shape):
        h=1e-6;d=np.zeros_like(c);d[ix]=h
        assert grad[ix]==pytest.approx((f(c+d)-f(c-d))/(2*h),rel=1e-7,abs=1e-7)
    with pytest.raises(ValueError,match='NONZERO'):s.exact_coefficient_gradient(np.zeros((2,3)),keys)


def test_gauge_and_near_U_are_not_exact_sharing():
    from trottertracks.c1_h6_initialization import reconstruct
    J=tensors();U=np.repeat(np.eye(2)[None],3,axis=0)
    orig=reconstruct(J,U);keys=s.exact_keys(U,1)
    assert keys[0]==keys[1]
    U[1,:,0]*=-1
    np.testing.assert_allclose(reconstruct(J,U),orig,atol=0,rtol=0)
    assert len(set(s.exact_keys(U,1)))==2
    U=np.repeat(np.eye(2)[None],3,axis=0);a=1e-9
    U[1]=[[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]]
    assert len(set(s.exact_keys(U,1)))==2
    assert s.exact_law(J,s.Partition(s.exact_keys(U,1),ld=1))['expected_switches']>60
    # A joint column permutation preserves the tensor but is not key identity.
    U=np.repeat(np.eye(2)[None],3,axis=0);P=np.array([[0.,1.],[1.,0.]])
    U[1]=U[1]@P;JJ=J.copy();JJ[1]=P.T@J[1]@P
    np.testing.assert_allclose(reconstruct(JJ,U),orig,atol=1e-15)
    assert len(set(s.exact_keys(U,1)))==2


def test_fixed_zero_epsilon_bias_and_validation():
    spec=s.Partition(('a','b'),ld=1,epsilon=.01)
    J=tensors(1,0);exact=s.exact_law(J,spec);soft=s.smooth_terms(J,spec,np)
    assert exact['q_basis']['b']==0 and soft['q_basis'][1]>0
    for bad in [J.astype(np.float32),J+np.eye(2)[None]*np.nan]:
        with pytest.raises(ValueError):s.exact_law(bad,spec)
    with pytest.raises(ValueError):s.Partition(('a',),r=0)


def test_existing_law_coefficients_and_exact_basis_producer():
    from trottertracks import c_pr_aware_df_stage_b as b
    from trottertracks.c_df_basis_surrogate_v2 import law_features
    rng=np.random.default_rng(2907);J=rng.normal(size=(8,4,4));J=(J+J.transpose(0,2,1))/2
    U=np.repeat(np.eye(4)[None],8,axis=0)
    for i in [3,5,7]:U[i]=np.linalg.qr(rng.normal(size=(4,4)))[0]
    arm=b.prepare_arm(np.eye(4),np.zeros((4,)*4),.12,J,U,role='baseline',source_identity='SYNTHETIC',input_identity='SYNTHETIC',ld=2)
    a=law_features(arm.plan.atoms);v=s.exact_law(J,s.Partition(s.exact_keys(U,2)))
    assert v['lambda_R']==pytest.approx(a['lambda_R'])
    assert v['q_basis']==pytest.approx(a['q_basis'])
    assert v['expected_switches']==pytest.approx(a['expected_switches'])
