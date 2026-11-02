"""Invented J/U only: merge, raw CAR, PF/law and exact native dispatch.

No NPZ/QPY reads, optimizer, molecular sampling, ground state or real circuit
build/compile. The two-dimensional Pauli counterexample is an artificial PF.
"""
from pathlib import Path
from fractions import Fraction
from types import SimpleNamespace
import sys
import numpy as np
import pytest
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks import c_df_shared_basis as shared
from trottertracks import c_pr_aware_df_stage_b as bridge
from trottertracks import c1_stage2_native as native
from trottertracks.c1_h6_initialization import reconstruct


def invented(n=2, rank=4):
    rng = np.random.default_rng(2910+n)
    a = rng.normal(size=(rank,n,n))
    J = (a+a.transpose(0,2,1))/16
    U = np.array([np.linalg.qr(rng.normal(size=(n,n)))[0] for _ in J])
    if rank==4:
        U[2] = U[1].copy()
    return J, U


def arm(J, U, *, ld=1, g=None):
    n = J.shape[-1]
    if g is None:
        g = reconstruct(J, U)
    return bridge.prepare_arm(np.diag(np.linspace(-.3,.2,n)),g,.17,J,U,
        role='baseline',source_identity='ARTIFICIAL',input_identity='ARTIFICIAL',ld=ld)


@pytest.mark.parametrize('n,rank,ld,pair',[(2,4,1,(1,2)),(4,8,2,(2,3))])
def test_tensor_merge_general_J_rank8_to7_and_surjection(n,rank,ld,pair):
    J,U = invented(n,rank);U[pair[1]] = U[pair[0]].copy()
    groups = tuple((i,i+1) if i==pair[0] else (i,) for i in range(rank) if i!=pair[1])
    A,V,info = shared.merge(J,U,groups,ld=ld)
    np.testing.assert_allclose(reconstruct(J,U),reconstruct(A,V),atol=2e-14,rtol=0)
    assert len(A)==rank-1 and info['ld']==ld
    assert info['effective_J_null_dimensions']==n*(n+1)//2
    assert info['minimal_outer_rank'] is info['finite_PR_law_preserved'] is None
    # Any reduced general-J factor can be lifted by an arbitrary symmetric split.
    split=J.copy();split[pair[0]]=.25*A[pair[0]];split[pair[1]]=.75*A[pair[0]]
    np.testing.assert_allclose(reconstruct(split,U),reconstruct(A,V),atol=2e-14,rtol=0)


def test_indefinite_J_null_direction_and_group_count_not_minimal_rank():
    J,U=invented();J[1]=[[1.,2.],[2.,-1.]];J[2]=-J[1]
    assert np.linalg.eigvalsh(J[1])[0]<0<np.linalg.eigvalsh(J[1])[-1]
    A,V,info=shared.merge(J,U,((0,),(1,2),(3,)),ld=1)
    np.testing.assert_array_equal(A[1],np.zeros((2,2))) # no automatic pruning
    d=np.array([[.7,-.2],[-.2,.1]])
    plus=J.copy();minus=J.copy()
    plus[1]+=d;plus[2]-=d;minus[1]-=d;minus[2]+=d
    np.testing.assert_allclose(reconstruct(plus,U),reconstruct(minus,U),atol=1e-14)
    assert info['group_count']==3 # one group is zero, so not a minimal-rank claim


def test_spin_scalar_original_g_kappa_and_independent_full_CAR():
    J,U=invented();g=reconstruct(J,U)
    A,V,_=shared.merge(J,U,((0,),(1,2),(3,)),ld=1)
    before,after=arm(J,U,g=g),arm(A,V,g=g)
    np.testing.assert_array_equal(before.general.one_body_tensor,after.general.one_body_tensor)
    assert before.general.constant==after.general.constant==.17
    car=bridge.stage1.car_bilinears(4)
    F=[bridge.stage1.raw_frame(j,u,car)[0] for j,u in zip(J,U)]
    G=[bridge.stage1.raw_frame(j,u,car)[0] for j,u in zip(A,V)]
    for a in [before,after]:
        for j,u,frame in zip(a.J,a.U,a.general.frames):
            np.testing.assert_allclose(bridge.stage1.raw_frame(j,u,car)[0],
                bridge.stage1.candidate_frame(frame)[0],atol=1e-13)
        np.testing.assert_allclose(bridge.stage1.raw_base(bridge.stage1.corrected_kappa(
            np.diag([-.3,.2]),g),.17,car),bridge.stage1.candidate_base(a.general),atol=1e-13)
    np.testing.assert_allclose(sum(F),sum(G),atol=1e-13)
    assert before.general.frames[0].spin_order=='interleaved_2i_plus_sigma'
    assert before.plan.extracted_identity==pytest.approx(after.plan.extracted_identity)


def test_exact_keys_sign_zero_gauge_transpose_and_near_U():
    from trottertracks.c_df_basis_switch import exact_keys
    J,U=invented();U[:]=np.eye(2)
    assert len(set(exact_keys(U,0)))==1
    for changed in [np.array([[1.,-0.],[0.,1.]]),np.diag([-1.,1.]),
                    np.array([[0.,1.],[1.,0.]]),
                    np.array([[np.cos(1e-9),-np.sin(1e-9)],[np.sin(1e-9),np.cos(1e-9)]])]:
        X=U.copy();X[2]=changed
        assert exact_keys(X,0)[1]!=exact_keys(X,0)[2]
        with pytest.raises(ValueError,match='EXACT_NATIVE_BASIS'):
            shared.merge(J,X,((0,),(1,2),(3,)),ld=1)
    J,U=invented();t=.41;U[1]=[[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]];U[2]=U[1].T
    with pytest.raises(ValueError,match='EXACT_NATIVE_BASIS'):
        shared.merge(J,U,((0,),(1,2),(3,)),ld=1)
    # Native deterministic tuple equality can ignore signed zero; tail repr keys do not.
    from trottertracks.c1_general_k_adapter import _matrix,_spin_lift
    positive=_spin_lift(_matrix(np.eye(2),'U'));negative=_spin_lift(_matrix([[1.,-0.],[0.,1.]],'U'))
    assert positive==negative and repr(positive)!=repr(negative)


def test_adjacent_run_dispatch_and_central_sum_without_Gaussian_build():
    from trotterlib.rte import _basis_reuse_intervals
    J,U=invented();a=arm(J,U)
    schedule=((2,Fraction(1,2)),(3,Fraction(1,2)),(4,Fraction(1,2)))
    groups=native.runs(a.model,schedule,optimized=True)
    assert [len(g) for g in groups]==[2,1]
    combined=native.combined_diagonal(a.model,groups[0])
    A,V,_=shared.merge(J,U,((0,),(1,2),(3,)),ld=1);b=arm(A,V)
    expected=native.combined_diagonal(b.model,((2,Fraction(1,2)),))
    assert combined.global_phase==pytest.approx(expected.global_phase)
    assert dict(combined.rz)==pytest.approx(dict(expected.rz))
    assert {(p,q):x for p,q,x in combined.rzz}==pytest.approx({(p,q):x for p,q,x in expected.rzz})
    atoms=[next(x for x in a.plan.atoms if x.frame_id.endswith(':F'+str(i))) for i in [1,2,3]]
    intervals=_basis_reuse_intervals([SimpleNamespace(basis_id=x.basis_key,basis_hash=x.basis_key) for x in atoms])
    assert [(x.start,x.stop) for x in intervals]==[(0,2)]


@pytest.mark.parametrize('cancel',[False,True])
def test_tail_mass_law_and_fixed_corrected_RTE_with_same_D_R(cancel):
    from trottertracks.c_df_basis_surrogate import law_features
    J,U=invented();J[2]=(-.5 if cancel else .5)*J[1]
    g=reconstruct(J,U);A,V,_=shared.merge(J,U,((0,),(1,2),(3,)),ld=1)
    a,b=arm(J,U,g=g),arm(A,V,g=g)
    x,y=law_features(a.plan.atoms),law_features(b.plan.atoms)
    assert (y['lambda_R']<x['lambda_R']) if cancel else y['lambda_R']==pytest.approx(x['lambda_R'])
    # Labels/count change even in the same-sign case; physical pushforward may agree.
    assert len(a.plan.atoms)>len(b.plan.atoms)
    assert x['raw_central_RZ']==y['raw_central_RZ'] # fixed event count, not static atom count
    if not cancel:
        assert x['q_basis']==pytest.approx(y['q_basis'])
        assert x['expected_switches']==pytest.approx(y['expected_switches'])
    car=bridge.stage1.car_bilinears(4)
    def generators(z):
        return [bridge.stage1.raw_base(bridge.stage1.corrected_kappa(np.diag([-.3,.2]),g),.17,car)]+[
            bridge.stage1.raw_frame(j,u,car)[0] for j,u in zip(z.J,z.U)]
    for K in [0,2]:
        ta,ca,_=bridge.configure(a,Fraction(1,10),4,K);tb,cb,_=bridge.configure(b,Fraction(1,10),4,K)
        ka=bridge.kernels(a,generators(a),Fraction(1,10),ca);kb=bridge.kernels(b,generators(b),Fraction(1,10),cb)
        for k in ['pf','corrected']:np.testing.assert_allclose(ka[k],kb[k],atol=2e-13)
        from trotterlib.rte import finite_rte_attenuation
        aa=finite_rte_attenuation(ca,tail_evolutions=2);ab=finite_rte_attenuation(cb,tail_evolutions=2)
        assert ab>aa if cancel else ab==pytest.approx(aa)


def test_mass_cancellation_can_increase_basis_switches():
    from trottertracks.c_df_basis_surrogate import law_features
    J,U=invented();template=np.array([[1.,.2],[.2,-.3]])
    J[1]=3.5*template;J[2]=-2.5*template;J[3]=template
    A,V,_=shared.merge(J,U,((0,),(1,2),(3,)),ld=1)
    x=law_features(arm(J,U).plan.atoms);y=law_features(arm(A,V).plan.atoms)
    assert y['lambda_R']<x['lambda_R']
    assert y['expected_switches']>x['expected_switches']


def test_noncontiguous_only_Hamiltonian_merge_not_ordered_PF():
    J,U=invented();U[3]=U[1].copy();groups=((0,),(1,3),(2,))
    with pytest.raises(ValueError,match='NONCONTIGUOUS'):
        shared.merge(J,U,groups,ld=1)
    A,V,info=shared.merge(J,U,groups,ld=1,mode='representation')
    assert not info['contiguous_groups']
    np.testing.assert_allclose(reconstruct(J,U),reconstruct(A,V),atol=2e-14)
    # Same-basis Z fragments separated by X cannot generally be moved through X.
    Z=np.diag([1.,-1.]);X=np.array([[0.,1.],[1.,0.]])
    old=[Z,X,2*Z];new=[3*Z,X]
    f=lambda G:bridge.reference._ordered_exponentials(G,native.ordered_schedule(len(G),'1/3'))
    assert np.linalg.norm(f(old)-f(new))>1e-3
    # Adjacent same-basis factors *can* be combined in this ordered symmetric schedule.
    np.testing.assert_allclose(f([Z,2*Z,X]),f([3*Z,X]),atol=1e-14)


@pytest.mark.parametrize('groups,ld,mode,message',[
    (((0,),(1,),(1,2)),1,'adjacent','PARTITION'),
    (((0,),(2,),(1,3)),1,'representation','PARTITION'),
    (((0,1),(2,),(3,)),1,'adjacent','HEAD_TAIL'),
    (((0,),(1,),(2,)),1,'adjacent','PARTITION'),
    (((0,),(1,2),(3,)),1,'silent','MERGE_MODE')])
def test_invalid_partition_and_split_stop(groups,ld,mode,message):
    J,U=invented();U[:]=U[0]
    with pytest.raises(ValueError,match=message):shared.merge(J,U,groups,ld=ld,mode=mode)


@pytest.mark.parametrize('bad',['nonsymmetric','nonfinite','complex','nonorthogonal','float32'])
def test_bad_arrays_stop(bad):
    J,U=invented()
    if bad=='nonsymmetric':J[0,0,1]+=.1
    if bad=='nonfinite':J[0,0,0]=np.nan
    if bad=='complex':U=U.astype(complex)
    if bad=='nonorthogonal':U[0,0,0]+=.1
    if bad=='float32':J=J.astype(np.float32)
    with pytest.raises(ValueError):shared.merge(J,U,((0,),(1,2),(3,)),ld=1)
