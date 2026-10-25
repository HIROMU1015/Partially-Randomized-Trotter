"""Small independent dense oracles, leakage/invariance, no saved molecules."""
from pathlib import Path
import sys,math
import numpy as np
import pytest
from scipy.linalg import expm
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c1_subspace_synthesis_bound as b


@pytest.mark.parametrize('dimension',[2,5,9])
def test_complete_columns_upper_bound_independent_dense_spectral_norm(dimension):
    rng=np.random.default_rng(dimension);x=rng.normal(size=(dimension,dimension))+1j*rng.normal(size=(dimension,dimension))
    v=expm(.02j*(x+x.conj().T));u=np.eye(dimension,dtype=complex);indices=list(range(0,dimension,2))
    stream=b.ColumnStream(len(indices),dimension)
    for j in reversed(range(len(indices))):stream.add([j],v[:,indices[j]][None,:],u[:,indices[j]][None,:])
    r=stream.finish();spectral=np.linalg.norm((v-u)[:,indices],2)
    assert r['restricted_operator_bound_pilot']>=spectral*(1-5e-15)
    assert r['evidence_level']==b.LEVEL and r['rigorous_operator_bound'] is None
    np.testing.assert_allclose(r['represented_difference_frobenius'],np.linalg.norm((v-u)[:,indices],'fro'),rtol=3e-15)


def test_partial_basis_and_unseen_error_never_qualify():
    stream=b.ColumnStream(2,3);stream.add([0],np.array([[1,0,0]],complex),np.array([[1,0,0]],complex))
    assert stream.finish()['restricted_operator_bound_pilot'] is None
    assert stream.finish()['evidence_level']=='A_PARTIAL_COLUMNS_ONLY'
    with pytest.raises(ValueError,match='COLUMN_IDENTITY'):stream.add([0],np.array([[1,0,0]],complex),np.array([[1,0,0]],complex))


def test_full_output_leakage_and_control_relative_phase_are_retained():
    theta=.02;v=np.eye(8,dtype=complex)
    v[np.ix_([0,3],[0,3])]=[[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]]
    v[4,4]=np.exp(.03j);idx=[0,4];u=np.eye(8,dtype=complex)
    stream=b.ColumnStream(2,8);stream.add([0,1],v[:,idx].T,u[:,idx].T);r=stream.finish()
    assert r['restricted_operator_bound_pilot']>=np.linalg.norm((v-u)[:,idx],2)
    assert r['columns'][0]['L2']>.019 and r['columns'][1]['L2']>.029


@pytest.mark.parametrize('n',[1,2,17,16384])
def test_telescoping_invariant_target_allows_actual_leakage(n):
    u=np.diag([np.exp(.03j),np.exp(-.04j),np.exp(.1j)]);h=np.array([[0,0,1],[0,1,0],[1,0,0]],complex)
    v=expm(.0001j*h)@u;idx=[0,1];eps=float(np.linalg.norm((v-u)[:,idx],'fro'))
    bound=b.repeat_bound(eps,n,reference_invariant=True)['operator_bound_pilot']
    actual=np.linalg.norm((np.linalg.matrix_power(v,n)-np.linalg.matrix_power(u,n))[:,idx],2)
    assert bound>=actual*(1-3e-12)


def test_noninvariant_target_cannot_use_restricted_one_step_error():
    u=np.array([[0,1],[1,0]],complex);v=np.array([[0,-1],[1,0]],complex)
    assert np.linalg.norm((v-u)[:,[0]])==0
    assert np.linalg.norm((v@v-u@u)[:,[0]])==2
    assert b.repeat_bound(0.,2,reference_invariant=False)['operator_bound_pilot'] is None


def test_reference_rounding_defect_is_retained_in_growth_guard():
    w=np.diag([1+1e-12,1.]).astype(complex);r=b.reference_info(w)
    assert r['reference_norm_upper_pilot']>=np.linalg.norm(w,2)
    repeated=b.repeat_bound(1e-12,16384,reference_invariant=True,reference_norm_upper=r['reference_norm_upper_pilot'])
    assert repeated['telescoping_factor']>16384
    assert r['reference_exactly_unitary_certified'] is False and r['rigorous_enclosure'] is None
    with pytest.raises(ValueError,match='NONUNITARY'):b.reference_info(np.diag([1.01,1.]).astype(complex))


def test_basis_order_control_and_no_output_projection():
    w=np.array([[0,1j],[1,0]],complex);initial,target=b.basis_batch(np.array([3,1]),4,w,[0,1,2,3])
    np.testing.assert_array_equal(np.argmax(abs(initial),axis=1),[3,1,7,5])
    np.testing.assert_array_equal(target[2,4+np.array([3,1])],w[:,0]);assert target.shape==(4,8)
    with pytest.raises(ValueError):b.basis_batch(np.array([1,1]),4,w,[0])


@pytest.mark.parametrize('bad',['nonfinite','shape','dtype','column'])
def test_invalid_stream_inputs_stop(bad):
    actual=np.array([[1,0]],complex);expected=actual.copy();ids=[0]
    if bad=='nonfinite':actual[0,0]=np.nan
    if bad=='shape':actual=actual[:,:1]
    if bad=='dtype':actual=actual.astype(np.complex64)
    if bad=='column':ids=[2]
    with pytest.raises(ValueError):b.ColumnStream(1,2).add(ids,actual,expected)


def test_round_bound_and_missing_evidence_never_become_CA_or_certificate():
    k=dict(evidence_level=b.LEVEL,restricted_operator_bound_pilot=1e-12,reference_norm_upper_pilot=1.)
    shell={name:1e-14 for name in ['head','X','Y']}
    small=b.round_bound(k,shell,s=4,m=12,saved_pf_error=.12)
    assert small['occurrences']==16384 and small['status']=='POTENTIALLY_PHASE_ELIGIBLE_NUMERICAL'
    assert small['rigorous_signal_error_bound'] is small['CA_achieved'] is small['shots'] is small['total_cost'] is None
    assert small['operator_bound_pilot']>=16384e-12
    assert b.round_bound(k,shell,s=1,m=0,saved_pf_error=.3)['status']=='INFEASIBLE_REGISTERED_BOUND'
    unknown=dict(k,evidence_level='A_FINITE_PROBES')
    assert b.round_bound(unknown,shell,s=1,m=0,saved_pf_error=0.)['complex_signal_bound_pilot'] is None
    with pytest.raises(ValueError):b.round_bound(k,dict(shell,X=None),s=1,m=0,saved_pf_error=0.)


def test_tiny_differences_not_silently_underflowed_to_zero():
    stream=b.ColumnStream(1,2);stream.add([0],np.array([[1e-200,0]],complex),np.zeros((1,2),complex))
    assert stream.finish()['restricted_operator_bound_pilot']>=1e-200


@pytest.mark.parametrize('m',[0,4,12])
def test_complete_shell_and_repeated_complex_signal_against_dense_oracle(m):
    # Full output has one extra system state. V leaks into it; U preserves S.
    # Numeric target W is slightly nonunitary, so use the alpha guard as well.
    w=np.diag([np.exp(.03j)*(1+1e-13),np.exp(-.02j)]).astype(complex)
    u=np.eye(6,dtype=complex);u[np.ix_([3,4],[3,4])]=w
    h=np.zeros((6,6),complex);h[3,5]=h[5,3]=1.
    v=expm(1e-8j*h)@np.diag([1,1,1,np.exp(.03j),np.exp(-.02j),1])
    ids=[0,1,3,4];stream=b.ColumnStream(4,6)
    stream.add(list(range(4)),v[:,ids].T,u[:,ids].T)
    kernel=dict(stream.finish(),**b.reference_info(w))
    head=np.kron(np.array([[1,1],[1,-1]])/math.sqrt(2),np.eye(3))
    x=head;y=head@np.kron(np.diag([1,-1j]),np.eye(3))
    perturb=expm(2e-9j*np.kron(np.array([[0,1],[1,0]]),np.eye(3)))
    actual_head=perturb@head;tails=dict(X=perturb@x,Y=perturb@y)
    shells=dict(head=float(np.linalg.norm(actual_head-head,2)),
        X=float(np.linalg.norm(tails['X']-x,2)),Y=float(np.linalg.norm(tails['Y']-y,2)))
    r=b.round_bound(kernel,shells,s=4,m=m,saved_pf_error=0.)
    initial=np.array([1,1j,0,0,0,0],complex)/math.sqrt(2)
    n=4*(1<<m);z=np.diag([1,1,1,-1,-1,-1])
    differences=[]
    for axis,tail in [('X',x),('Y',y)]:
        a=tails[axis]@np.linalg.matrix_power(v,n)@actual_head@initial
        t=tail@np.linalg.matrix_power(u,n)@head@initial
        assert np.linalg.norm(a-t)<=r['synthesis_axis_state_bound_pilot'][axis]*(1+1e-7)
        differences.append(float(abs(np.vdot(a,z@a)-np.vdot(t,z@t))))
    assert math.hypot(*differences)<=r['complex_signal_bound_pilot']*(1+1e-7)
