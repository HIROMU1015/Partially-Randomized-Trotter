"""Artificial-only codec tests; fixed H6 files are never numerically decoded."""
from pathlib import Path
import importlib,json,sys
from unittest.mock import patch,Mock
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c1_h6_initialization as m


def test_reflection_counterexample_and_column_sign_gauge_proposal():
    U=np.array([[[1.,1.],[1.,-1.]]])/np.sqrt(2)
    J=np.array([[[1.,.2],[.2,.4]]])
    A,B,x,report=m.codec(J,U)
    assert x.dtype==np.float64 and np.linalg.det(U[0])<0
    assert report['tensor_roundtrip_frobenius']>.1
    assert report['frames'][0]['complex_log_exponential_residual']<1e-12
    assert report['frames'][0]['log_imaginary_frobenius']>3.
    A,B,x,report=m.column_sign_proposal(J,U)
    np.testing.assert_allclose(m.reconstruct(A,B),m.reconstruct(J,U),atol=1e-12,rtol=0)
    assert np.linalg.det(B[0])>0


def test_positive_determinant_does_not_resolve_all_pi_branches():
    U=np.array([[[0.,1.,0.],[1.,0.,0.],[0.,0.,-1.]]])
    J=np.array([np.diag([1.,0.,0.])])
    assert np.linalg.det(U[0])==1
    assert m.codec(J,U)[3]['tensor_roundtrip_frobenius']>1
    with pytest.raises(ValueError,match='BRANCH_UNRESOLVED'):m.column_sign_proposal(J,U)


@pytest.mark.parametrize('n',[2,6])
def test_batch_order_general_J_dtype_and_column_sign_tensor_invariance(n):
    from scipy.linalg import expm
    rng=np.random.default_rng(n);a=rng.normal(size=(3,n,n))/6;a=a-a.transpose(0,2,1)
    U=expm(a);j=rng.normal(size=(3,n,n));J=j+j.transpose(0,2,1)
    fixed=U.copy();fixed[0,:,-1]*=-1;fixed[2,:,0]*=-1
    np.testing.assert_allclose(m.reconstruct(J,fixed),m.reconstruct(J,U),atol=1e-12,rtol=0)
    A,B,x,r=m.codec(J,U);np.testing.assert_allclose(A,J,atol=0,rtol=0)
    np.testing.assert_allclose(B,U,atol=1e-12,rtol=0)
    assert x.dtype==np.float64 and len(r['frames'])==3
    A,B,_,_=m.column_sign_proposal(J,fixed)
    np.testing.assert_allclose(m.reconstruct(A,B),m.reconstruct(J,U),atol=1e-11,rtol=0)


def test_real_ffsim_initialization_path_matches_direct_helper_with_optimizer_blocked():
    import ffsim,scipy.optimize
    from trottertracks import c1_ffsim_compat as bridge
    from ffsim.linalg import util
    d=importlib.import_module('ffsim.linalg.double_factorized_decomposition')
    h=np.array([[.2,.03],[.03,.1]]);g=np.einsum('pq,rs->pqrs',h,h)
    expected=m.initial_decomposition(g,2);seen={}
    original=d.df_tensors_to_params
    class Blocked(Exception):pass
    def capture(J,U,*args,**kw):seen.update(J=J.copy(),U=U.copy());return original(J,U,*args,**kw)
    old=util.unitaries_to_parameters
    with patch.object(d,'df_tensors_to_params',side_effect=capture),patch.object(scipy.optimize,'minimize',side_effect=Blocked) as optimizer:
        with pytest.raises(Blocked):bridge.double_factorized(g,tol=1e-8,max_vecs=2,optimize=True,
            method='L-BFGS-B',return_optimize_result=True,cholesky=True)
    for a,b in zip(expected,(seen['J'],seen['U']),strict=True):np.testing.assert_array_equal(a,b)
    assert optimizer.call_count==1 and optimizer.call_args.kwargs==dict(method='L-BFGS-B',jac=True,callback=None,options=None)
    np.testing.assert_array_equal(optimizer.call_args.args[1],m.codec(*expected)[2])
    assert util.unitaries_to_parameters is old


def test_existing_JAX_objective_gradient_matches_tensor_and_finite_difference():
    U=np.array([[[.8,-.6],[.6,.8]]]);J=np.array([[[.3,.04],[.04,.1]]])
    g=m.reconstruct(J,U)*1.01
    _,_,x,_=m.codec(J,U);value,grad=m.objective_gradient(g,x,1)
    assert value['objective']==pytest.approx(m.tensor_metrics(g,J,U)['objective_from_tensor'],abs=1e-14)
    from ffsim.linalg.util import df_tensors_from_params
    def loss(p):return m.tensor_metrics(g,*df_tensors_from_params(p,1,2,None,real=True))['objective_from_tensor']
    step=np.zeros_like(x);step[0]=1e-6
    assert grad[0]==pytest.approx((loss(x+step)-loss(x-step))/2e-6,abs=1e-9)
    assert value['gradient_infinity']>0


def test_initial_rotation_sign_identity_and_objective_mismatch_fail():
    J=np.ones((1,2,2));U=np.eye(2)[None]
    assert m.initialization_identity(J,U,J,U)['status']=='PASS'
    changed=U.copy();changed[:,:,0]*=-1
    with pytest.raises(ValueError,match='IDENTITY_NOT_ESTABLISHED'):m.initialization_identity(J,changed,J,U)
    with pytest.raises(ValueError,match='OBJECTIVE_IDENTITY'):m.check_objective(.1,.2)


@pytest.mark.parametrize('bad_identity',[False,True])
def test_fit_phase_artificial_end_to_end_and_partial_STOP(tmp_path,bad_identity):
    import ffsim,scipy.optimize
    root=tmp_path/'root';saved=root/'saved';saved.mkdir(parents=True)
    U=np.repeat(np.array([[[.8,-.6],[.6,.8]]]),18,axis=0)
    J=np.repeat(np.array([[[.2,.03],[.03,.1]]]),18,axis=0)/18
    g=m.reconstruct(J,U);np.savez(saved/'input.npz',g=g)
    for name,rank in [('XDF_full',18),('XDF_selected',14),('CDF_14',14),('CDF_18',18)]:
        np.savez(saved/(name+'.npz'),J=J[:rank],U=U[:rank])
        if name.startswith('CDF'):m.io.save(saved/(name+'.json'),dict(fun=m.tensor_metrics(g,J[:rank],U[:rank])['objective_from_tensor']))
    old=m.io.sha(saved/'XDF_full.npz');original=scipy.optimize.minimize
    def init(g,rank):
        value=U[:rank].copy()
        if bad_identity and rank==18:value[:,:,0]*=-1
        return J[:rank],value
    out=root/'output'
    with patch.object(m,'initial_decomposition',side_effect=init),patch.object(m,'column_sign_proposal',side_effect=AssertionError('Artificial repair not part of diagnosis')):
        if bad_identity:
            with pytest.raises(ValueError,match='IDENTITY_NOT_ESTABLISHED'):m.fit_diagnosis(root,out,dict(saved_run='saved'))
        else:m.fit_diagnosis(root,out,dict(saved_run='saved'))
    stop=m.io.read(out/'STOP.json')
    assert stop['failed']==bad_identity and stop['retry']==stop['resume']==0
    assert (out/'R14_result.json').exists() and m.io.sha(saved/'XDF_full.npz')==old
    assert scipy.optimize.minimize is original
    if not bad_identity:
        assert stop['scientific_units']==21
        result=m.io.read(out/'result.json');assert result['new_fit']==result['optimizer_iterations']==0
        assert result['ranks']['18']['original_optimizer_jac'] is None
    else:
        assert stop['scientific_units']<21 and (out/'failure.json').exists()


def test_parent_artificial_end_to_end_fresh_output_and_no_PF_native(tmp_path):
    from trottertracks import c1_molecule_extension as old
    root=tmp_path/'root';root.mkdir();saved=root/'saved';saved.mkdir()
    packet=dict(h=np.diag([-.2,.1]),constant=np.asarray(.125))
    U=np.array([[[.8,-.6],[.6,.8]]]);J=np.array([[[.2,.03],[.03,.1]]]);packet['g']=m.reconstruct(J,U)
    np.savez(saved/'input.npz',**packet)
    ctx=old.SectorCAR(2);physical=ctx.physical(**{k:packet[k] for k in ('h','g','constant')})
    reference=old.rank_assessment(ctx,packet,J,U,physical,14)[0]
    for name in ('XDF_rank_14.json','CDF_rank_14.json','CDF_rank_18.json'):m.io.save(saved/name,reference)
    contract=dict(saved_run='saved',entry='UNUSED',DF_reserve_Ha=.00159/8)
    out=root/'output'
    def worker(*args,**kw):
        p=out/'codec';p.mkdir();ranks={}
        for rank in (14,18):
            for state in ('before','after','final'):np.savez(p/f'R{rank}_{state}.npz',J=J,U=U)
            t=m.tensor_metrics(packet['g'],J,U)
            ranks[str(rank)]=dict(tensor={state:t for state in ('before','after','final')},codec=dict(tensor_roundtrip_frobenius=0.))
        m.io.save(p/'result.json',dict(ranks=ranks));m.io.save(p/'STOP.json',dict(call_counts=m.FIT_LIMITS))
    forbidden=Mock(side_effect=AssertionError('No new fit/PF/native'))
    with patch.object(m,'preflight',return_value=contract),patch.object(m.subprocess,'run',side_effect=worker),patch.object(old,'factor',forbidden),patch.object(old,'PF_rounds',forbidden),patch.object(old,'native_H2',forbidden):
        m.run(root,root/'contract.json',out,Path('/path/python'))
    result=m.io.read(out/'result.json');stop=m.io.read(out/'STOP.json')
    assert all(v['classification']=='INITIAL_CONVERSION_PRESERVED' for v in result['conclusions'].values())
    assert not stop['failed'] and stop['scientific_units']==35 and result['new_fit']==0
    assert not (out/'circuits').exists();forbidden.assert_not_called()
    before=(out/'STOP.json').read_bytes()
    with patch.object(m,'preflight',return_value=contract),pytest.raises(FileExistsError):m.run(root,root/'contract.json',out,'UNUSED')
    assert (out/'STOP.json').read_bytes()==before


def test_byte_preflight_rejects_changed_source_without_loading_npz(tmp_path):
    source=tmp_path/'source';source.write_text('artificial')
    manifest=tmp_path/'SOURCE.json';m.io.save(manifest,dict(files={'source':m.io.sha(source)}))
    contract=dict(scope=m.SCOPE,ranks=list(m.RANKS),fit_calls=m.FIT_LIMITS,sector_calls=m.SECTOR_LIMITS,
        maximum_logical_units=35,source_manifest='SOURCE.json',source_manifest_sha256=m.io.sha(manifest),
        saved_inputs={},evaluation_packages={})
    path=tmp_path/'contract.json';m.io.save(path,contract)
    assert m.preflight(tmp_path,path)==contract
    source.write_text('changed')
    with patch.object(np,'load',side_effect=AssertionError('No numeric decode')),pytest.raises(ValueError,match='SOURCE:'):
        m.preflight(tmp_path,path)


def test_scientific_limits_are_two_ranks_six_sector_states_zero_fits():
    assert sum(m.FIT_LIMITS.values())==21 and sum(m.SECTOR_LIMITS.values())==14
    assert m.RANKS==(14,18) and m.SECTOR_LIMITS['sector_assessment']==6
    assert not set(m.FIT_LIMITS|m.SECTOR_LIMITS)&{'fit','optimizer','PF','compile','shots'}
