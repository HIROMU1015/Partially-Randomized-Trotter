"""Invented arrays and mocked optimizer only; never loads molecular artifacts."""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))
from trottertracks import c_integrated_robustness as m
from trottertracks import c_near_transfer as transfer


def invented():
    from trottertracks.c1_h6_initialization import reconstruct
    system = transfer.System(2, 3, 1, 1., 'a'*64)
    rng = np.random.default_rng(440)
    j = rng.normal(size=(3, 2, 2))
    j = np.asarray((j+j.transpose(0, 2, 1))/2, dtype=np.float64)
    u = np.tile(np.eye(2), (3, 1, 1))
    g = reconstruct(j, u)+np.full((2,)*4, .001)
    return g, j, u, system


def test_fixed_plan_and_arithmetic():
    assert len(m.cells()) == 18
    assert len(m.cells(transfer_strengths={'PR': .03, 'ENTRY': .3})) == 6
    assert sum(m.call_limits().values()) == 3213
    assert m.call_limits()['inner_J_solve'] == 1539
    for wrong in ({'PR': .1}, {'PR': .1, 'ENTRY': .2}, {'PR': .1, 'ENTRY': .1, 'other': .1}):
        with pytest.raises((ValueError, KeyError)):
            m.cells(transfer_strengths=wrong)


def test_rotation_is_fixed_and_does_not_mutate_inputs():
    _, j, u, _ = invented()
    a, b, report = m.perturb_initial(j, u, m.SEEDS[0])
    a2, b2, _ = m.perturb_initial(j, u, m.SEEDS[0])
    np.testing.assert_array_equal(a, j)
    np.testing.assert_array_equal(a, a2)
    np.testing.assert_array_equal(b, b2)
    np.testing.assert_array_equal(u, np.tile(np.eye(2), (3, 1, 1)))
    np.testing.assert_allclose(b.transpose(0, 2, 1) @ b, u, atol=1e-14)
    assert report['aggregate_generator_frobenius'] == pytest.approx(.05)
    assert not np.array_equal(b, m.perturb_initial(j, u, m.SEEDS[1])[1])
    with pytest.raises(ValueError):
        m.perturb_initial(j, u, 17)
    with pytest.raises(ValueError):
        m.perturb_initial(j, u, m.SEEDS[0], amplitude=.1)


def test_objectives_match_initial_contribution_but_not_raw_mass():
    from trottertracks import c_near_stable as existing
    g, j, u, system = invented()
    specs = m.paired_specs(g, j, u, system, strength=.03, loss_scale=2e-9)
    assert specs['PR'].loss_scale == specs['ENTRY'].loss_scale == 2e-9
    pr, entry = specs['PR'], specs['ENTRY']
    assert pr.gamma*(existing.near.pr.smooth_mass(j, pr, np)/pr.mass_scale)**2 == pytest.approx(.03)
    assert entry.strength*(existing.near.mass(j, entry, np)/entry.mass_scale)**2 == pytest.approx(.03)
    assert pr.mass_scale != entry.mass_scale
    for bad in (.2, 0.):
        with pytest.raises(ValueError):
            m.paired_specs(g, j, u, system, strength=bad, loss_scale=2e-9)


@pytest.mark.parametrize('physical_ok', [True, False])
def test_tensor_only_restore_delegates_one_fitter_without_fallback(physical_ok):
    g, j, u, system = invented()
    seen = []
    raw = SimpleNamespace(success=False, status=1, message='limit', nit=15, nfev=20, njev=20,
        x=np.zeros(system.rank*system.norb**2), jac=np.zeros(system.rank*system.norb**2), fun=.05)
    def codec(a, b):
        seen.append('codec')
        return a.copy(), b.copy(), np.zeros(system.rank*system.norb**2), {'version': 'mock'}, None
    def fitter(module, target, a, b, x, spec, **kwargs):
        seen.append('fit')
        assert module.__name__.endswith('c_pr_aware_df') and spec.gamma == 0.
        assert kwargs['options']['maxiter'] == kwargs['options']['maxfun'] == 15000
        assert kwargs['options']['ftol'] == 1e-15
        kwargs['after_optimizer'](lambda x: (raw.fun, raw.jac), raw)
        return a, b, raw
    cold, report = m.neutral_restore(g, j, u, system, loss_scale=2e-9,
        physical=lambda *a: {'budget_PASS': physical_ok}, before_optimizer=lambda *a: None,
        after_optimizer=lambda *a: None, codec=codec, fitter=fitter)
    assert seen == ['codec', 'fit']
    assert report['eligible'] is physical_ok
    assert report['optimizer']['success'] is False and report['replacement_seed'] is False
    np.testing.assert_array_equal(cold['J'], j)


@pytest.mark.parametrize('updates,terminal_ok', [(8, True), (64, True), (64, False)])
def test_real_arm_wrapper_receives_task_policy_and_preserves_prefix(monkeypatch, updates, terminal_ok):
    from trottertracks import c_near_practical as practical
    g, j, u, system = invented()
    spec = m.paired_specs(g, j, u, system, strength=.1, loss_scale=2e-9)['PR']
    captured = []
    monkeypatch.setattr(practical.algorithm.control.small, 'assess',
        lambda *a, **k: {'J_numerically_qualified': terminal_ok})
    def trajectory(target, jcold, ustart, spec, physical, *, step, policy,
                   maximum_total_points, call, record):
        assert step.max_updates == policy.outer_maxiter == 64
        assert maximum_total_points == 513
        point = dict(J=jcold.copy(), U=ustart.copy(), state={'first_order_met': False})
        for count in range(1, updates+1):
            call('inner_J_solve', str(count), lambda: None)
            record('inner_raw', str(count), dict(inner_summary_record=True, status=0, nfev=2, njev=2,
                nit=1, value_gradient_calls_in_block=3, hessian_calls_in_block=1), None)
            point['J'] = jcold+count*.001
            row = physical(str(count), point['J'], point['U'])
            record('physical_candidate', str(count), dict(update=count, state=point['state'], physical=row), None)
            record('accepted', str(count), dict(physically_qualified=True),
                dict(J=point['J'], U=point['U']))
        return dict(terminal=point, status='REGISTERED_OBJECTIVE_STALL' if updates < 64 else 'MAXIMUM_ACCEPTED_UPDATES',
            accepted_U_updates=updates)
    def call(kind, label, fn):
        captured.append(kind)
        return fn()
    events = []
    result = m.recipe_cell('PR', g, dict(J=j, U=u), spec, system,
        lambda *a: {'budget_PASS': True}, call, lambda *a: events.append(a), trajectory=trajectory)
    points = result['checkpoints']
    assert set(points) == ({16, 64} if terminal_ok else {16})
    np.testing.assert_array_equal(points[16]['J'], j+min(updates, 16)*.001)
    if terminal_ok:
        np.testing.assert_array_equal(points[64]['J'], j+updates*.001)
    assert points[16]['solver_metrics']['nfev'] == 2*min(updates, 16)
    if terminal_ok:
        assert points[64]['solver_metrics']['nfev'] == 2*updates
    assert captured.count('arm_export') == len(points)
    assert len([event for event in events if event[0] == 'checkpoint_export']) == len(points)
    assert 'mass_diagnostics' in points[16]
    assert practical.STEP.max_updates == 16 and practical.MAX_POINTS == 129
    assert practical.ARM_LIMITS['inner_J_solve'] == 387


def test_real_ffsim_tensor_restore_route_with_optimizer_mocked(monkeypatch):
    """Exercise actual codec, fitter and final-hop options, zero iterations."""
    import scipy.optimize
    from trottertracks import c_common_convergence as common
    g, j, u, system = invented()
    observed = []
    original = scipy.optimize.minimize
    def optimizer(fun, x, **options):
        observed.append(options)
        value, gradient = fun(x)
        return scipy.optimize.OptimizeResult(x=x.copy(), fun=value, jac=gradient,
            success=False, status=1, message='ARTIFICIAL_ZERO_ITERATIONS', nit=0, nfev=1, njev=1)
    monkeypatch.setattr(scipy.optimize, 'minimize', optimizer)
    cold, report = m.neutral_restore(g, j, u, system, loss_scale=2e-9,
        physical=lambda *args: {'budget_PASS': True}, before_optimizer=lambda *args: None,
        after_optimizer=lambda *args: None)
    assert scipy.optimize.minimize is optimizer
    assert observed == [dict(method='L-BFGS-B', jac=True, callback=None, options=common.OPTIONS)]
    assert report['eligible'] is True and report['optimizer']['nit'] == 0
    assert report['independent_terminal']['independent_value_abs'] == 0.
    np.testing.assert_array_equal(cold['J'], j)
    monkeypatch.setattr(scipy.optimize, 'minimize', original)
