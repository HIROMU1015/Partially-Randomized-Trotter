"""Artificial 2/6-dimensional matrices only; no saved molecular NPZ decode."""
from pathlib import Path
import sys
from unittest.mock import patch,Mock
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c1_real_orthogonal_codec as c


def rotation(angle):
    return np.array([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]])


@pytest.mark.parametrize('n',[2,6])
@pytest.mark.parametrize('angle',[0.,.4,np.pi,np.pi-1e-7,np.pi+1e-7,np.pi-1e-12,np.pi+1e-12,np.pi-1e-14,np.pi+1e-14])
@pytest.mark.parametrize('improper',[False,True])
def test_schur_exact_and_near_pi_with_both_determinants(n,angle,improper):
    from scipy.linalg import block_diag,expm
    rng=np.random.default_rng(401+n);Q=np.linalg.qr(rng.normal(size=(n,n)))[0]
    T=rotation(angle) if n==2 else block_diag(rotation(angle),rotation(-.731),rotation(np.pi))
    U=Q@T@Q.T
    if improper:U[:,0]*=-1
    initial=U.copy();A,V,signs,report=c.real_skew_log(U)
    np.testing.assert_allclose(expm(A),V,atol=2e-13,rtol=0)
    np.testing.assert_allclose(A+A.T,0,atol=1e-14,rtol=0)
    assert np.linalg.det(V)>0 and report['representative_exponential_maxabs']<1e-12
    np.testing.assert_array_equal(U,initial)
    np.testing.assert_allclose(np.einsum('pi,qi->ipq',U,U),np.einsum('pi,qi->ipq',V,V),atol=0,rtol=0)


@pytest.mark.parametrize('n',[2,6])
def test_exact_minus_identity_pairs_single_real_minus_one_blocks(n):
    from scipy.linalg import expm
    A,V,signs,report=c.real_skew_log(-np.eye(n))
    assert len(report['minus_one_pairs'])==n//2
    np.testing.assert_allclose(expm(A),-np.eye(n),atol=1e-14,rtol=0)


def test_old_positive_det_pi_counterexample_is_repaired_without_logm_or_fitter():
    from trottertracks import c1_h6_initialization as old
    import scipy.linalg,scipy.optimize
    from ffsim.linalg import util
    U=np.eye(6);U[:3,:3]=[[0.,1.,0.],[1.,0.,0.],[0.,0.,-1.]]
    J=np.diag([1.,.2,.3,.4,.5,.6])[None];U=U[None]
    assert old.codec(J,U)[3]['tensor_roundtrip_frobenius']>.1
    helper=util.unitaries_to_parameters;forbidden=Mock(side_effect=AssertionError('No logm/fitter/optimizer'))
    with patch.object(scipy.linalg,'logm',forbidden),patch.object(scipy.optimize,'minimize',forbidden),patch.object(old,'initial_decomposition',forbidden):
        a,b,x,report,raw=c.roundtrip(J,U)
    np.testing.assert_allclose(old.reconstruct(a,b),old.reconstruct(J,U),atol=1e-12,rtol=0)
    assert report['tensor_frobenius']<1e-12 and util.unitaries_to_parameters is helper
    forbidden.assert_not_called()


@pytest.mark.parametrize('n',[2,6])
def test_multiple_frames_general_J_and_numpy_JAX_parameter_compatibility(n):
    from scipy.linalg import expm
    from ffsim.linalg import util
    rng=np.random.default_rng(203+n);A=rng.normal(size=(5,n,n));A=.4*(A-A.transpose(0,2,1))
    U=expm(A);U[0,:,0]*=-1;U[2,:,-1]*=-1
    j=rng.normal(size=(5,n,n));J=j+j.transpose(0,2,1)
    J0,U0=J.copy(),U.copy()
    a,b,x,report,raw=c.roundtrip(J,U)
    assert x.dtype==np.float64 and x.shape==(5*n*n,)
    np.testing.assert_array_equal(a,J)
    np.testing.assert_array_equal(J,J0);np.testing.assert_array_equal(U,U0)
    aa,bb=util.df_tensors_from_params_jax(x,5,n,None,real=True)
    np.testing.assert_array_equal(np.asarray(aa),J)
    np.testing.assert_allclose(np.asarray(bb),b,atol=1e-12,rtol=0)
    np.testing.assert_allclose(util.antihermitians_from_parameters(x[:5*n*(n-1)//2],n,5,real=True),raw['generators'],atol=0,rtol=0)
    assert all(f['density_projector_maxabs']<1e-12 for f in report['frames'])


@pytest.mark.parametrize('bad',['nonorthogonal','nonfinite','complex','wrong_shape','nonsymmetric_J','wrong_dtype'])
def test_invalid_input_is_rejected_without_orthogonalizing_or_real_casting(bad):
    J=np.eye(2)[None];U=np.eye(2)[None]
    if bad=='nonorthogonal':U[0,0,0]=1.01
    if bad=='nonfinite':U[0,0,0]=np.nan
    if bad=='complex':U=U.astype(complex)
    if bad=='wrong_shape':U=np.zeros((1,2,3))
    if bad=='nonsymmetric_J':J[0,0,1]=1e-14
    if bad=='wrong_dtype':U=U.astype(np.float32)
    with pytest.raises(ValueError):c.roundtrip(J,U)


def test_schur_or_ffsim_corruption_causes_explicit_failure():
    import scipy.linalg
    from ffsim.linalg import util
    with patch.object(scipy.linalg,'schur',return_value=(np.diag([-1.,1.]),np.eye(2))):
        with pytest.raises(ValueError,match='UNPAIRED_MINUS_ONE'):c.real_skew_log(np.eye(2))
    J=np.diag([1.,.2])[None];U=rotation(.4)[None]
    with patch.object(util,'df_tensors_from_params',return_value=(J,np.eye(2)[None])):
        with pytest.raises(ValueError,match='FRAME_PROJECTOR'):c.roundtrip(J,U)


@pytest.mark.parametrize('bad_identity',[False,True])
def test_artificial_worker_saved_input_reuse_no_initializer_and_partial_STOP(tmp_path,bad_identity):
    from trottertracks import c1_h6_codec_roundtrip as m
    root=tmp_path/'root';root.mkdir();J=np.zeros((18,2,2));J[0]=[[.2,.03],[.03,.1]]
    U=np.repeat(rotation(.4)[None],18,axis=0);U[0,:,0]*=-1;U[2]=-np.eye(2)
    g=m.old.reconstruct(J,U);np.savez(root/'input.npz',g=g);np.savez(root/'full.npz',J=J,U=U)
    arrays={}
    for r in (14,18):
        initial=U[:r].copy()
        if bad_identity and r==18:initial[0,:,0]*=-1
        np.savez(root/f'initial{r}.npz',J=J[:r],U=initial);arrays[str(r)]=f'initial{r}.npz'
    contract=dict(original_input='input.npz',XDF_full='full.npz',initial_arrays=arrays)
    forbidden=Mock(side_effect=AssertionError('No new initializer or optimizer'))
    with patch.object(m.old,'initial_decomposition',forbidden),patch.object(m.old,'objective_gradient',forbidden):
        if bad_identity:
            with pytest.raises(ValueError,match='IDENTITY_NOT_ESTABLISHED'):m.worker(root,root/'out',contract)
        else:m.worker(root,root/'out',contract)
    stop=m.io.read(root/'out/STOP.json');assert stop['failed']==bad_identity
    assert (root/'out/R14_codec.json').exists();forbidden.assert_not_called()
    if not bad_identity:assert stop['scientific_units']==9
    else:assert (root/'out/failure.json').exists() and stop['scientific_units']<9
