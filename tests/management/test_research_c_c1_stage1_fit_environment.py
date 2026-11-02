"""Fit-venv technical checks; the actual ffsim fitter is always mocked."""
import hashlib
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock,patch

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c1_stage1 as s
from trottertracks import c1_stage1_execution as run


def test_locked_fit_runtime_and_102_source_files():
    import ffsim
    runtime=run.runtime_identity(ROOT,'fit')
    assert runtime['packages']['ffsim']=='0.0.83'
    source=json.loads((ROOT/run.ENV_DIR/'ffsim-python-source.json').read_text())
    assert source['commit']==s.FFSIM_COMMIT and len(source['files'])==102
    installed=Path(ffsim.__file__).parent
    for name,want in source['files'].items():
        assert hashlib.sha256((installed/name).read_bytes()).hexdigest()==want


def test_real_API_signature_and_single_mocked_call():
    import ffsim
    assert 'return_optimize_result' in inspect.signature(ffsim.linalg.double_factorized).parameters
    result=SimpleNamespace(success=True,status=0,message='mock',x=np.zeros(1),jac=np.zeros(1),fun=.5,nit=1,nfev=1,njev=1)
    mock=Mock(return_value=(np.eye(2)[None],np.eye(2)[None],result))
    with patch.object(ffsim.linalg,'double_factorized',mock):
        arrays,metadata=s.fit_once(np.eye(2),np.zeros((2,)*4),.25,s.Calls())
    assert mock.call_count==1 and mock.call_args.kwargs==s.FIT_SETTINGS
    assert arrays['kappa'].dtype==np.float64 and metadata['optimizer']['success']


def test_JAX_x64_JIT_value_and_gradient_on_artificial_scalars():
    import ffsim
    import jax
    import jax.numpy as jnp
    assert jax.config.jax_enable_x64
    f=jax.jit(jax.value_and_grad(lambda x: .5*jnp.sum(x*x)))
    value,gradient=f(jnp.array([1.,2.],dtype=jnp.float64))
    assert value.dtype==jnp.float64 and gradient.dtype==jnp.float64
    assert float(value)==2.5 and np.array_equal(gradient,np.array([1.,2.]))


def test_claim_refuses_second_fit_before_import_or_API_call(tmp_path):
    import ffsim
    (tmp_path/'fit.claim').write_text('already attempted')
    with patch.object(ffsim.linalg,'double_factorized') as mock:
        with pytest.raises(FileExistsError):run.worker_fit({},tmp_path)
    mock.assert_not_called()


def test_worker_serializes_real_ffsim_interface_under_mock(tmp_path):
    import ffsim
    out=tmp_path/'output';out.mkdir();input_path=tmp_path/'ARTIFICIAL.npz'
    np.savez(input_path,one_mo=np.eye(4),eri_chemist=np.zeros((4,)*4),
             nuclear_repulsion=np.array(.25),mo_coeff=np.eye(4))
    source_path=tmp_path/'SOURCE.json';run.save(source_path,{'artificial':True})
    refs=json.loads((ROOT/run.ENV_DIR/'ffsim-python-source.json').read_text())
    v=dict(root=ROOT,input=input_path,source_path=source_path,source={'ffsim_python_files':refs['files']},
           contract={'call_limits':{'fit':dict(input_load=1,fit=1,fit_qualification=1,tensor_reconstruct=1,tensor_maxabs=1,tensor_frobenius=1)}})
    result=SimpleNamespace(success=True,status=0,message='mock',x=np.zeros(1),jac=np.zeros(1),fun=.5,nit=1,nfev=1,njev=1)
    with patch.object(ffsim.linalg,'double_factorized',return_value=(np.eye(4)[None],np.eye(4)[None],result)) as mock:
        run.worker_fit(v,out)
    assert mock.call_count==1
    arrays,metadata=run.load_export(out,v)
    assert metadata['actual_rank']==1 and arrays['constant']==.25
    assert (out/'raw_fit.npz').exists() and (out/'raw_status.json').exists()
