import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))


def test_repaired_coefficient_solver_validates_pre_step_jac_at_its_point():
    from dataclasses import replace
    from trottertracks import c_separated_j_u_consistent as core
    spec=core.pr.Objective(2,1,1,1.,1.,0.)
    p=core.Coefficients(np.array([[[[.9]]]]),np.ones((2,1,1)),spec)
    J,info=core.solve_j(p,np.array([.1,.2])[:,None,None],replace(core.DEFAULT,j_maxiter=1))
    assert not info['success'] and info['status']==1
    assert info['consistency']['last_step_start']['PASS']
    assert not info['consistency']['raw_jac_at_endpoint']['PASS']
    assert info['stationarity']['status']=='MET'


def test_terminal_native_comparison_uses_decoded_point_and_preserves_original(monkeypatch):
    import jax;jax.config.update('jax_enable_x64',True)
    from trottertracks import c_separated_j_u_consistent_pilot as p
    core=p.algorithm
    rng=np.random.default_rng(42);U=np.array([np.linalg.qr(rng.normal(size=(2,2)))[0] for _ in range(3)])
    J=rng.normal(size=(3,2,2))*.1;J=J+J.transpose(0,2,1)
    g=core.linear.reconstruct(J,U)
    s=core.pr.Objective(3,2,1,1e-9,1.,.1)
    before=U.copy();perturbed=core.retract(U,np.array([2e-12,-1e-12,1e-12]))
    original_codec=core.pr.corrected.codec.roundtrip
    fixed=original_codec(J,perturbed)
    monkeypatch.setattr(core.pr.corrected.codec,'roundtrip',lambda *a:fixed)
    _,_,info=p.native_diagnostic(g,J,U,core.pr,s,{},core.DEFAULT)
    assert info['status']=='PASS' and info['checks']['J_gradient_same_point']['PASS']
    assert not info['checks']['original_point_gradient_observation']['PASS']
    np.testing.assert_array_equal(U,before)
    assert not info['thresholds_changed']
