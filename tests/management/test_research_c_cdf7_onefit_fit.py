"""Actual rank7 ffsim initializer/loss/codec; optimizer always mocked."""
from pathlib import Path
import sys,importlib
from unittest.mock import patch
import numpy as np
import scipy.optimize
import pytest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c_cdf7_onefit as m


def test_actual_initializer_and_qualified_x0_one_mock_optimizer(tmp_path):
    rng=np.random.default_rng(301007)
    a=rng.normal(size=(10,4,4));a=(a+a.transpose(0,2,1))*.03
    g=np.einsum('lpq,lrs->pqrs',a,a)
    np.savez(tmp_path/'input.npz',one_mo=np.diag([-.3,-.1,.1,.3]),eri_chemist=g,
        nuclear_repulsion=np.array(.5),mo_coeff=np.eye(4))
    d=importlib.import_module('ffsim.linalg.double_factorized_decomposition')
    init=d._double_factorized_explicit_cholesky
    with patch.object(scipy.optimize,'minimize',side_effect=AssertionError('NO_OPTIMIZER_IN_PREPARE')),patch.object(
            d,'_double_factorized_explicit_cholesky',wraps=init) as initialization:
        m.prepare_worker(tmp_path,tmp_path/'prepare',dict(original_input='input.npz'))
        initialization.assert_called_once()
        assert initialization.call_args.kwargs==dict(tol=1e-8,max_vecs=7)
    p=m.pilot.old_input(tmp_path/'prepare/initial.npz')
    assert p['J'].shape==p['U'].shape==(7,4,4) and p['x0'].shape==(112,)
    m.io.save(tmp_path/'initial_gate.json',dict(status='PASS',initial_budget_is_admission=False,
        initial_budget_diagnostic=dict(before=False,initial=False)))
    expected=m.objective.make_value_and_grad(m.objective.Objective(**m.io.read(tmp_path/'prepare/objective.json')['spec']))
    v0,g0=expected(p['x0'],g)
    def minimize(fun,x0,**kwargs):
        np.testing.assert_array_equal(x0,p['x0'])
        assert kwargs==dict(method='L-BFGS-B',jac=True,callback=None,options=None)
        v,dg=fun(x0);np.testing.assert_allclose(v,v0,atol=1e-12);np.testing.assert_allclose(dg,g0,atol=1e-12)
        return scipy.optimize.OptimizeResult(x=x0.copy(),jac=dg,fun=v,success=True,status=0,
            message='ARTIFICIAL_NO_ITERATIONS',nit=0,nfev=1,njev=1)
    with patch.object(scipy.optimize,'minimize',side_effect=minimize) as optimize,patch.object(
            m.objective.corrected.codec,'roundtrip',side_effect=AssertionError('NO_SECOND_CODEC')):
        m.fit_worker(tmp_path,tmp_path/'fit',{})
        optimize.assert_called_once()
    assert m.io.read(tmp_path/'prepare/STOP.json')['scientific_units']==6
    assert m.io.read(tmp_path/'fit/STOP.json')['scientific_units']==5
    assert m.io.read(tmp_path/'fit/optimizer.json')['nit']==0
    np.testing.assert_array_equal(m.pilot.old_input(tmp_path/'fit/optimizer_parameters.npz')['x'],p['x0'])
    diagnostic=m.io.read(tmp_path/'fit/initial_diagnostic.json')
    assert diagnostic['normalized_objective']==pytest.approx(1.)
    assert diagnostic['raw_tensor_objective']==pytest.approx(m.io.read(tmp_path/'prepare/initial_tensor.json')['objective_from_tensor'])
