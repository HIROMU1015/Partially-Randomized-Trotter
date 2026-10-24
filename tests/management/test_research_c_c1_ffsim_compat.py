"""Artificial-only regression through real ffsim initialization; no optimizer."""
import importlib
import json
from pathlib import Path
import runpy
import signal
import subprocess
import sys
from unittest.mock import Mock, patch

import numpy as np
import pytest
import scipy.linalg

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks import c1_ffsim_compat as bridge
from trottertracks import c1_stage1 as stage
from trottertracks import c1_stage1_execution as execution


def rotations():
    angles = [.17, -.43, .81]
    return np.array([[[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]
                     for t in angles])


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.complex64, np.complex128])
def test_batch_order_shape_dtype_matches_locked_2d_logm(dtype):
    mats = rotations().astype(dtype)[::-1]  # non-contiguous, deliberately reordered
    expected = np.stack([scipy.linalg.logm(mat) for mat in mats])
    with patch.object(scipy.linalg, 'logm', wraps=scipy.linalg.logm) as logm:
        actual = bridge.batch_logm(mats)
    assert actual.shape == mats.shape and actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)
    assert logm.call_count == len(mats)
    for call, mat in zip(logm.call_args_list, mats):
        np.testing.assert_array_equal(call.args[0], mat)
        assert call.args[0].ndim == 2 and call.kwargs == {}


@pytest.mark.parametrize('shape', [(2, 2), (2, 2, 3), (0, 2, 2), (2, 0, 0)])
def test_unsupported_shapes_rejected_before_logm(shape):
    with patch.object(scipy.linalg, 'logm') as logm:
        with pytest.raises(ValueError, match='SQUARE_BATCH'):
            bridge.batch_logm(np.zeros(shape))
    logm.assert_not_called()


def test_original_failure_reproduced_and_context_restored_after_exception():
    import ffsim.linalg.util as util
    original, logm = util.unitaries_to_parameters, scipy.linalg.logm
    with pytest.raises(ValueError, match='square array_like'):
        original(rotations(), real=True)
    with pytest.raises(RuntimeError, match='artificial interruption'):
        with bridge.per_matrix_logm():
            assert scipy.linalg.logm is logm
            np.testing.assert_allclose(util.unitaries_to_parameters(rotations(), real=True),
                                       -np.array([.17, -.43, .81]), atol=1e-14)
            raise RuntimeError('artificial interruption')
    assert util.unitaries_to_parameters is original and scipy.linalg.logm is logm


@pytest.mark.parametrize('real', [False, True])
def test_original_parameter_packing_and_real_discard(real):
    import ffsim.linalg.util as util
    mats = rotations().astype(complex)
    mats[1] = np.diag(np.exp(1j*np.array([.2, -.7])))
    logs = np.stack([scipy.linalg.logm(mat) for mat in mats])
    expected = util.antihermitians_to_parameters(logs, real=real)
    with bridge.per_matrix_logm():
        actual = util.unitaries_to_parameters(mats, real=real)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float64
    if real:
        assert actual.shape == (3,) and actual[1] == 0.
    else:
        assert actual.shape == (12,) and np.any(actual != 0.)


@pytest.mark.parametrize('norb', [2, 4])
def test_real_ffsim_initialization_reaches_mocked_optimizer_only(norb):
    import ffsim.linalg.util as util
    decomp = importlib.import_module('ffsim.linalg.double_factorized_decomposition')
    # Deterministic artificial PSD chemist tensor, unrelated to P-H4.
    a = np.arange(1., norb*norb+1).reshape(norb, norb)/100
    a = a+a.T
    b = np.diag(np.arange(1., norb+1)/10)
    g = np.einsum('pq,rs->pqrs', a, a)+np.einsum('pq,rs->pqrs', b, b)
    captured = {}
    original_pack = decomp.df_tensors_to_params
    original_helper = util.unitaries_to_parameters
    original_logm = scipy.linalg.logm

    def pack(J, U, indices, real):
        assert real is True and indices is None
        captured.update(J=J.copy(), U=U.copy())
        return original_pack(J, U, indices, real=real)

    class OptimizerBlocked(Exception):
        pass

    calls = stage.Calls()
    with patch.object(decomp, 'df_tensors_to_params', side_effect=pack) as packing, \
         patch.object(scipy.optimize, 'minimize', side_effect=OptimizerBlocked) as optimizer, \
         patch.object(bridge, 'batch_logm', wraps=bridge.batch_logm) as batch, \
         patch.object(decomp, '_double_factorized_explicit_cholesky',
                      wraps=decomp._double_factorized_explicit_cholesky) as initialization:
        with pytest.raises(OptimizerBlocked):
            stage.fit_once(np.eye(norb), g, .25, calls)
    initialization.assert_called_once_with(g, tol=1e-8, max_vecs=8)
    assert packing.call_count == batch.call_count == optimizer.call_count == 1
    assert dict(calls.counts) == {'fit': 1}  # artificial attempt, no fit export/residual
    assert optimizer.call_args.kwargs == dict(method='L-BFGS-B', jac=True,
                                             callback=None, options={})
    assert callable(optimizer.call_args.args[0])  # never invoked by the mock
    J, U = captured['J'], captured['U']
    logs = np.stack([original_logm(mat) for mat in U])
    upper = np.triu_indices(norb, 1)
    diagonal_upper = np.triu_indices(norb)
    expected = np.concatenate([logs[:, upper[0], upper[1]].real.ravel(),
                               J[:, diagonal_upper[0], diagonal_upper[1]].ravel()])
    x0 = optimizer.call_args.args[1]
    assert x0.dtype == np.float64 and np.isfinite(x0).all()
    np.testing.assert_array_equal(x0, expected)
    assert util.unitaries_to_parameters is original_helper
    assert scipy.linalg.logm is original_logm


def test_cli_keeps_venv_python_and_real_subprocess_can_import_numpy():
    python = ROOT/'.venv-fit/bin/python'
    assert python.is_symlink() and python.resolve() != python.absolute()
    argv = ['runner', '--execute', '--contract', 'ARTIFICIAL', '--source', 'ARTIFICIAL',
            '--input', 'ARTIFICIAL', '--approval', 'ARTIFICIAL', '--output', 'ARTIFICIAL',
            '--fit-python', str(python)]
    previous = signal.getsignal(signal.SIGTERM)
    try:
        with patch.object(sys, 'argv', argv), \
             patch.object(execution, 'preflight', return_value={}), \
             patch.object(execution, 'execute', return_value={'status': 'MOCK'}) as execute:
            runpy.run_path(str(ROOT/'scripts/run_research_c_c1_stage1.py'), run_name='__main__')
    finally:
        signal.signal(signal.SIGTERM, previous)
    forwarded = execute.call_args.args[3]
    assert forwarded == python.absolute()
    probe = subprocess.run([str(forwarded), '-c',
        'import sys,json,numpy;print(json.dumps(dict(prefix=sys.prefix,numpy=numpy.__version__)))'],
        check=True, capture_output=True, text=True)
    identity = json.loads(probe.stdout)
    assert Path(identity['prefix']) == ROOT/'.venv-fit'
    assert identity['numpy'] == '1.26.4'
