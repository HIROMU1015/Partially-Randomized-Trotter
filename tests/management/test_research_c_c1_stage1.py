"""Artificial data only; molecular files and real ffsim fitting are forbidden."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c1_stage1 as s


def fixture(n=2, complex_rotation=False):
    h=np.diag(np.linspace(-.3,.2,n)); h[0,1]=h[1,0]=.125
    g=np.zeros((n,)*4)
    U=np.eye(n,dtype=complex if complex_rotation else float)
    c,t=3/5,4/5
    U[:2,:2]=[[c,1j*t],[1j*t,c]] if complex_rotation else [[c,t],[-t,c]]
    J=np.diag(np.linspace(-.4,.7,n));J[0,1]=J[1,0]=-.27
    return h,g,.375,h.copy(),J[None],U[None]


def evaluate_fixture(n=2,complex_rotation=False):
    h,g,c,k,J,U=fixture(n,complex_rotation)
    return s.evaluate(h,g,c,k,J,U,np.kron(h,np.eye(2)),np.array([-.2]),
                      np.eye(2*n)[None]*.3,nalpha=1 if n==2 else 2,
                      nbeta=1 if n==2 else 2,
                      source_identity='ARTIFICIAL_SOURCE',input_identity='ARTIFICIAL_INPUT')


@pytest.mark.parametrize('complex_rotation',[False,True])
def test_independent_density_and_K_routes_and_call_inventory(complex_rotation):
    result=evaluate_fixture(complex_rotation=complex_rotation)
    assert result['status']=='STAGE1_REPRESENTATION_PASS'
    assert result['calls']==s.expected_evaluation_calls(1,4)
    for row in result['records']:
        if row['gate']:
            assert row['passed'], row
    errors={r['id']:r for r in result['records']}
    assert errors['sector_DF_CDF']['error']>1e-3
    assert errors['sector_DF_CDF']['passed'] is None
    assert errors['sector_DF_XDF']['passed'] is None


def test_artificial_eight_mode_full256_and_sector36():
    result=evaluate_fixture(n=4)
    assert result['status']=='STAGE1_REPRESENTATION_PASS'
    assert result['full_dimension']==256 and result['sector_dimension']==36
    assert result['calls']['xdf_matvec']==36


def test_two_frames_keep_order_and_formula():
    h,g,c,k,J,U=fixture()
    result=s.evaluate(h,g,c,k,np.concatenate((J,-.5*J)),np.concatenate((U,U)),
                      np.kron(h,np.eye(2)),np.array([.2]),np.eye(4)[None],
                      nalpha=1,nbeta=1,source_identity='artificial',input_identity='artificial')
    assert result['status']=='STAGE1_REPRESENTATION_PASS'
    assert result['calls']==s.expected_evaluation_calls(2,4)


def test_K_error_is_detected_without_changing_physical_DF_gate():
    original=s.adapter.general_k_coefficients
    def wrong(K):
        rows=list(original(K)); support,value=rows[0]; rows[0]=(support,value+.01)
        return tuple(rows)
    with patch.object(s.adapter,'general_k_coefficients',wrong):
        result=evaluate_fixture()
    assert result['status']=='SCIENTIFIC_STOP_EQUALITY'
    assert any(r['passed'] is False and r['id']=='full_frame_0' for r in result['records'])


def test_raw_reference_does_not_use_adapter_coefficients():
    h,g,c,k,J,U=fixture()
    with patch.object(s.adapter,'general_k_coefficients',side_effect=AssertionError('candidate only')):
        H,B=s.raw_frame(J[0],U[0],s.car_bilinears(4))
    assert H.shape==(16,16) and B>0


def test_offdiagonal_base_sign_constant_and_endianness():
    from scipy.sparse import csr_matrix
    h,g,c,k,J,U=fixture()
    a=s.car_bilinears(4)
    raw=s.raw_base(k,c,a)
    physical,_=s.physical_sector(h,g,c,np.arange(16))
    assert np.max(np.abs(raw-physical))<1e-14
    assert raw[0,0]==c
    # mode0 -> mode2 excitation, parity is defined by the occupied mode1.
    assert a[2,0,6,3]==-1
    assert csr_matrix(a[2,0]).nnz==4


def test_tiny_coefficients_survive_openfermion_conversion():
    tiny=2.0**-60
    H=s.unthresholded_sparse([(((0,1),(1,0)),tiny)],2).toarray()
    assert np.count_nonzero(H)==1 and np.max(np.abs(H))==tiny
    h=np.array([[tiny,0.],[0.,-tiny]]); g=np.zeros((2,)*4)
    full,_=s.physical_sector(h,g,0.,np.arange(16))
    assert np.max(np.abs(full))==2*tiny


def test_fixed_N_can_hide_a_wrong_base():
    idx=s.sector_indices(2,1,1)
    shift=np.diag([sum((i>>p)&1 for p in range(4))-2 for i in range(16)])
    assert np.count_nonzero(shift[np.ix_(idx,idx)])==0
    assert np.max(np.abs(shift))==2


@pytest.mark.parametrize('defect',['kappa','unitarity','saved_kappa'])
def test_structure_fail_stops_before_operator_build(defect):
    h,g,c,k,J,U=fixture(); saved=np.kron(h,np.eye(2))
    if defect=='kappa':k[0,1]+=.01
    if defect=='unitarity':U[0,0,0]+=.01
    if defect=='saved_kappa':saved[0,0]+=.01
    with patch.object(s,'physical_sector',side_effect=AssertionError('must stop')):
        result=s.evaluate(h,g,c,k,J,U,saved,np.array([1.]),np.eye(4)[None],
                          nalpha=1,nbeta=1,source_identity='artificial',input_identity='artificial')
    assert result['status']=='SCIENTIFIC_STOP_STRUCTURE'


def test_nonzero_g_correction_and_export_with_one_mock_fit():
    h,g,c,k,J,U=fixture(); g[0,1,1,0]=.25
    result=SimpleNamespace(success=True,status=0,message='mock',x=np.zeros(3),
                           jac=np.zeros(3),fun=.1,nit=2,nfev=3,njev=3)
    mock=Mock(return_value=(J,U,result)); calls=s.Calls()
    arrays,meta=s.fit_once(h,g,c,calls,fit_function=mock)
    mock.assert_called_once()
    assert mock.call_args.kwargs==s.FIT_SETTINGS
    assert arrays['kappa'][0,0]==h[0,0]-.125
    assert arrays['constant']==c and meta['one_body_role']=='ffsim_corrected_kappa'
    assert calls.counts['fit']==1 and meta['optimizer']['success']


def test_nonzero_artificial_g_physical_CAR_and_signed_square_agree():
    h,g,c,k,J,U=fixture()
    U=U[0]; weights=np.array([.6,-.25]); G=U@np.diag(weights)@U.T
    lam=-.7
    g=lam*np.einsum('pq,rs->pqrs',G,G)
    J=lam*np.outer(weights,weights)
    k=s.corrected_kappa(h,g)
    indices=s.sector_indices(2,1,1)
    raw=s.raw_base(k,c,s.car_bilinears(4))+s.raw_frame(J,U,s.car_bilinears(4))[0]
    physical,_=s.physical_sector(h,g,c,indices)
    saved,_=s.saved_xdf_sector(np.kron(k,np.eye(2)),np.array([lam/2]),
           np.kron(G,np.eye(2))[None],c,1,1,indices,s.Calls())
    # Same original-g correction and explicit spatial prefactor 1/2.
    assert np.max(np.abs(physical-raw[np.ix_(indices,indices)]))<1e-14
    assert np.max(np.abs(physical-saved))<1e-14


def test_unsuccessful_fit_is_partial_not_automatic_restart():
    h,g,c,k,J,U=fixture()
    result=SimpleNamespace(success=False,status=1,message='mock nonconvergence',
                           x=np.zeros(1),jac=np.zeros(1),fun=.3,nit=1,nfev=2,njev=2)
    mock=Mock(return_value=(J,U,result))
    _,meta=s.fit_once(h,g,c,s.Calls(),fit_function=mock)
    assert not meta['optimizer']['success'] and mock.call_count==1


def test_allowance_is_frozen_scale_dependent_and_detects_small_perturbation():
    assert s.ROUNDING_GUARD==8
    result=evaluate_fixture()
    row=next(r for r in result['records'] if r['id']=='full_frame_0')
    assert row['allowance']>0 and row['error']<row['allowance']
    a=s.allowance(2,3,100,200,16,np.zeros((16,16)))
    b=s.allowance(4,6,100,200,16,np.zeros((16,16)))
    assert b==2*a
    with pytest.raises(ValueError):s.gamma(2**54)


def test_nonfinite_fit_preserves_raw_before_stopping():
    h,g,c,k,J,U=fixture();J[0,0,0]=np.nan
    result=SimpleNamespace(success=True,status=0,message='mock',x=np.zeros(1),jac=np.zeros(1),fun=.1)
    raw=Mock()
    with pytest.raises(ValueError,match='NONFINITE'):
        s.fit_once(h,g,c,s.Calls(),fit_function=Mock(return_value=(J,U,result)),raw_callback=raw)
    raw.assert_called_once()
