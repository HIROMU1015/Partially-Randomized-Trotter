"""Fit-environment synthetic objectives/rotations only; no molecular arrays."""
from collections import Counter
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pytest
import jax
jax.config.update('jax_enable_x64', True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks import c_separated_j_u as m
from trottertracks import c_separated_j_u_pilot as pilot


def invented(seed=41, n=2, rank=3, ld=1, role='pr', scale=1.):
    rng = np.random.default_rng(seed)
    U = np.array([np.linalg.qr(rng.normal(size=(n, n)))[0] for _ in range(rank)])
    a = rng.normal(size=(rank, n, n))*.1
    J = a+a.transpose(0, 2, 1)
    V = np.array([np.linalg.qr(rng.normal(size=(n, n)))[0] for _ in range(rank)])
    g = m.linear.reconstruct(J, V)
    spec = (m.pr.Objective(rank, n, ld, scale, 1., .1) if role == 'pr' else
            m.rc2.Objective(rank, n, ld, scale, 1., .1))
    return g, J, U, spec


def test_passive_import_no_scientific_modules():
    code = "import sys;sys.path.insert(0,'src');from trottertracks import c_separated_j_u_pilot as m;assert not any(x in sys.modules for x in ('numpy','scipy','jax','ffsim','qiskit'));print(m.MAX_CALLS)"
    r = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT,
                       text=True, capture_output=True, check=True)
    assert int(r.stdout) == 417


@pytest.mark.parametrize('role', ['pr', 'rc2'])
def test_isometric_full_J_linear_adjoint_and_existing_joint_objective(role):
    g, J, U, spec = invented(role=role)
    p = m.Coefficients(g, U, spec)
    x = p.pack(J)
    np.testing.assert_allclose(p.unpack(x), J, atol=1e-16, rtol=0)
    assert np.dot(x, x) == pytest.approx(np.sum(J*J), abs=1e-15)
    np.testing.assert_allclose(p.A@x, m.linear.reconstruct(J, U).ravel(), atol=1e-15)
    v, grad = p.vg(x)
    vi, gi = p.vg(x, independent=True)
    np.testing.assert_allclose(grad, gi, atol=1e-13)
    assert v == pytest.approx(vi, abs=1e-14)
    module = m.pr if role == 'pr' else m.rc2
    _, _, native, _, _ = m.pr.corrected.codec.roundtrip(J, U)
    value, raw = module.make_value_and_grad(spec)(native, g)
    split = spec.rank*spec.norb*(spec.norb-1)//2
    assert v == pytest.approx(float(value), abs=3e-13)
    np.testing.assert_allclose(grad, np.asarray(raw)[split:]/p.w, atol=3e-12)
    direction = np.linspace(-.3, .4, len(x))
    h = 1e-6
    fd = (p.vg(x+h*direction)[0]-p.vg(x-h*direction)[0])/(2*h)
    assert fd == pytest.approx(grad@direction, abs=2e-9)
    hv = (p.vg(x+h*direction)[1]-p.vg(x-h*direction)[1])/(2*h)
    np.testing.assert_allclose(hv, p.hess(x)@direction, atol=2e-8)
    assert np.linalg.eigvalsh(p.hess(x)).min() >= -1e-11
    # Convexity along an arbitrary segment; this is not a molecular certificate.
    y = x+direction
    assert p.vg(.3*x+.7*y)[0] <= .3*p.vg(x)[0]+.7*p.vg(y)[0]+1e-12


def test_PR_penalty_excludes_head_but_head_is_optimized_no_tensor_equality():
    g, J, U, spec = invented()
    p = m.Coefficients(g, U, spec)
    baseline = m.Coefficients(g, U, replace(spec, gamma=0.))
    x = p.pack(J)
    penalty_grad = p.vg(x)[1]-baseline.vg(x)[1]
    np.testing.assert_array_equal(penalty_grad[:3], 0.)
    result, info = m.solve_j(p, J)
    assert info['stationarity']['status'] == 'MET'
    assert not np.allclose(result[:spec.ld], J[:spec.ld])
    assert np.max(abs(p.mapping.apply(m.linear.pack(result))-p.mapping.apply(m.linear.pack(J)))) > 1e-4
    assert info['progress']['after'] <= info['progress']['before']+1e-12


@pytest.mark.parametrize('strength', [.1, 0.])
def test_RC2_augmented_independent_solution_and_offdiagonal_metric(strength):
    spec = m.rc2.Objective(1, 1, 0, .4, 2., strength)
    p = m.Coefficients(np.array([[[[.7]]]]), np.ones((1, 1, 1)), spec)
    J, report = m.solve_j(p, np.array([[[.2]]]))
    assert J[0, 0, 0] == pytest.approx(.7/(1+2*strength*.4/2), abs=1e-14)
    assert report['stationarity']['status'] == 'MET' and not report['normal_equations']
    g, J, U, spec = invented(role='rc2')
    p = m.Coefficients(g, U, spec)
    result, _ = m.solve_j(p, J)
    # Normal equations are an independent SMALL synthetic reference only.
    expected = np.linalg.solve(p.hess(p.pack(J)), p.A.T@g.ravel()/spec.loss_scale)
    np.testing.assert_allclose(p.pack(result), expected, atol=3e-13)
    assert m.rc2.full_l2(result, np) == pytest.approx(np.dot(expected, expected))


def test_flat_nonunique_PR_problem_keeps_original_objective_without_ridge():
    spec = m.pr.Objective(3, 1, 2, 1., 1., .1)
    p = m.Coefficients(np.array([[[[.8]]]]), np.ones((3, 1, 1)), spec)
    endpoints = []
    for values in ([1., -.3, .2], [3., -2.3, .2]):
        J, report = m.solve_j(p, np.array(values)[:, None, None])
        assert report['stationarity']['status'] == 'MET'
        endpoints.append(J)
    assert not np.allclose(endpoints[0], endpoints[1])
    a, b = [p.pack(J) for J in endpoints]
    assert p.vg(a)[0] == pytest.approx(p.vg(b)[0], abs=1e-14)
    np.testing.assert_allclose(p.hess(a)@np.array([1., -1., 0.]), 0., atol=1e-14)
    assert p.vg(np.array([.4, .4, 0.]))[0] > 0  # epsilon floor is not subtracted.


def test_wide_metric_preserves_every_flat_direction():
    spec = m.pr.Objective(8, 1, 7, 1., 1., .1)
    p = m.Coefficients(np.zeros((1,)*4), np.ones((8, 1, 1)), spec)
    raw = []
    _, report = m.solve_j(p, np.zeros((8, 1, 1)), record=lambda *args: raw.append(args))
    H = raw[0][1]['initial_inverse_metric']
    assert H.shape == (8, 8) and np.linalg.eigvalsh(H).min() > 0
    assert report['metric_directions_dropped'] == 0
    assert report['stationarity']['status'] == 'MET'


@pytest.mark.parametrize('n', [2, 6])
def test_local_U_gradient_det_sign_pi_and_independent_finite_difference(n):
    g, J, U, spec = invented(n=n, rank=2, ld=1)
    U[0] = np.eye(n)
    U[0, :2, :2] = -np.eye(2)  # pi branch
    U[1, :, 0] *= -1
    p = m.Coefficients(g, U, spec)
    penalty = p.parts(p.pack(J))['penalty']
    zero = np.zeros(spec.rank*n*(n-1)//2)
    value, grad = m.u_value_gradient(zero, g, J, U, spec, penalty)
    direction = np.linspace(-.4, .6, len(zero))
    def independent(eta):
        V = m.retract(U, eta)
        delta = g-m.linear.reconstruct(J, V)
        return .5*np.sum(delta*delta)/spec.loss_scale+penalty
    assert value == pytest.approx(independent(zero), abs=2e-13)
    h = 1e-6
    fd = (independent(h*direction)-independent(-h*direction))/(2*h)
    assert grad@direction == pytest.approx(fd, abs=3e-8)
    V, info = m.solve_u(g, J, U, spec, replace(m.DEFAULT, u_maxiter=3))
    np.testing.assert_allclose(V.transpose(0, 2, 1)@V, np.broadcast_to(np.eye(n), V.shape), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(V), np.linalg.det(U), atol=1e-12)
    assert info['progress']['after'] <= info['progress']['before']+1e-10
    assert info['final_U_tangent_l2_at_fixed_J'] >= 0


@pytest.mark.parametrize('role,cycles', [('pr', 16), ('rc2', 39)])
def test_alternation_progress_and_physical_stationarity(role, cycles, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('old LP/one-fit runner must not be invoked')
    monkeypatch.setattr(m.linear, 'solve_once', forbidden)
    monkeypatch.setattr(m.pr, 'fit', forbidden)
    monkeypatch.setattr(m.rc2, 'fit', forbidden)
    g, J, U, spec = invented(role=role)
    events = []
    a, u, report = m.optimize(g, J, U, spec,
        record=lambda kind, label, data, arrays: events.append((kind, label)))
    assert report['status'] == 'FIRST_ORDER_MET' and report['first_order_qualified']
    # The reported counts are local implementation observations, not H4 speedups.
    assert report['completed_cycles'] <= m.DEFAULT.max_cycles
    assert report['terminal']['components']['objective'] < report['states'][0]['components']['objective']
    assert report['terminal']['J_stationarity']['status'] == report['terminal']['U_stationarity']['status'] == 'MET'
    for i, (kind, label) in enumerate(events):
        if kind == 'U':
            assert ('J', label) in events[i+1:]  # mandatory post-U J refit
    assert a.shape == J.shape and u.shape == U.shape
    assert not report['rigorous_stationarity']


def test_near_singular_scaled_RC2_and_unresolved_PR_do_not_fake_convergence():
    g, J, U, spec = invented(role='rc2', scale=1e-9)
    U[1] = U[0]
    p = m.Coefficients(g, U, spec)
    a, info = m.solve_j(p, J)
    assert info['rank'] == len(p.pack(J)) and np.isfinite(a).all()
    assert info['stationarity']['status'] in ('MET', 'NOT_MET', 'UNRESOLVED_MARGIN')
    g, J, U, spec = invented()
    a, u, report = m.optimize(g, J, U, spec, replace(m.DEFAULT, j_maxiter=1))
    assert report['status'] == 'J_STATIONARITY_UNQUALIFIED'
    assert not report['first_order_qualified'] and report['completed_cycles'] == 0
    assert len(report['blocks']) == 1


def test_false_solver_success_nonfinite_and_increase_STOP(monkeypatch):
    g, J, U, spec = invented()
    p = m.Coefficients(g, U, spec)
    raw = []
    def unchanged(fun, x, **kwargs):
        value, grad = fun(x)
        return SimpleNamespace(x=x, jac=grad, fun=value, success=True,
                               status=0, message='invented false success')
    _, info = m.solve_j(p, J, solver=unchanged)
    assert info['success'] and info['stationarity']['status'] == 'NOT_MET'
    def bad(fun, x, **kwargs):
        y = x*100
        value, grad = fun(y)
        return SimpleNamespace(x=y, jac=grad, fun=value, success=True,
                               status=0, message='invented increase')
    with pytest.raises(ValueError, match='OBJECTIVE_INCREASE'):
        m.solve_j(p, J, solver=bad, record=lambda *args: raw.append(args))
    assert len(raw) == 1
    def nan(fun, x, **kwargs):
        return SimpleNamespace(x=np.full_like(x, np.nan), jac=np.zeros_like(x),
            fun=np.nan, success=False, status=2, message='invented nonfinite')
    with pytest.raises(ValueError, match='NONFINITE_OR_MALFORMED'):
        m.solve_j(p, J, solver=nan, record=lambda *args: raw.append(args))
    assert len(raw) == 2
    monkeypatch.setattr(m, 'retract', lambda U, eta: U*np.array([1.01, 1.]))
    with pytest.raises(ValueError, match='ORTHOGONALITY'):
        m.solve_u(g, J, U, spec)


def test_small_gradient_with_solver_warning_is_recorded_separately():
    spec = m.pr.Objective(2, 1, 1, 1., 1., .1)
    p = m.Coefficients(np.zeros((1,)*4), np.ones((2, 1, 1)), spec)
    def warning(fun, x, **kwargs):
        value, grad = fun(x)
        return SimpleNamespace(x=x, jac=grad, fun=value, success=False,
                               status=2, message='invented precision-loss stop at stationary point')
    _, info = m.solve_j(p, np.zeros((2, 1, 1)), solver=warning)
    assert not info['success'] and info['stationarity']['status'] == 'MET'
    assert info['metric_directions_dropped'] == 0 and not info['objective_regularization_added']


@pytest.mark.parametrize('mode,cycles', [('limit', 64), ('stall', 3), ('U_stop', 1)])
def test_explicit_cycle_call_budget_and_unqualified_stops(monkeypatch, mode, cycles):
    g, J, U, spec = invented()
    count = Counter()
    index = []
    def state(*args):
        index.append(1)
        value = 1000. if mode == 'stall' else 1000.-len(index)
        return dict(components=dict(objective=value), J_stationarity=dict(status='MET'),
            U_stationarity=dict(status='NOT_MET'), first_order_met=False), None, None
    monkeypatch.setattr(m, 'state', state)
    monkeypatch.setattr(m, 'solve_j', lambda p, J, *args, **kw:
        (J, dict(success=True, status=0)))
    monkeypatch.setattr(m, 'solve_u', lambda g, J, U, *args, **kw:
        (U, dict(success=mode != 'U_stop', status=2 if mode == 'U_stop' else 0)))
    def call(kind, label, function):
        count[kind] += 1
        assert count[kind] <= pilot.WORKER[kind]
        return function()
    _, _, r = m.optimize(g, J, U, spec, call=call)
    assert r['completed_cycles'] == cycles and not r['first_order_qualified']
    assert count == dict(J_solve=cycles+1, U_solve=cycles, state_diagnostic=cycles+2)
    assert r['status'] == dict(limit='OUTER_LIMIT_UNQUALIFIED',
        stall='STALLED_NOT_STATIONARY', U_stop='U_SOLVER_STOP_UNQUALIFIED')[mode]
    assert sum(pilot.WORKER.values()) == 3*64+11
    assert pilot.MAX_CALLS == 2*sum(pilot.WORKER.values())+sum(pilot.PARENT.values()) == 417


def test_terminal_existing_ffsim_reencoding_and_no_fit(monkeypatch):
    g, J, U, spec = invented()
    terminal, _, _ = m.state(g, J, U, spec)
    monkeypatch.setattr(m.pr, 'fit', lambda *a, **k: pytest.fail('joint fitter forbidden'))
    x, gradient, report = pilot.native_diagnostic(g, J, U, m.pr, spec,
                                                 terminal, m.DEFAULT)
    assert report['status'] == 'PASS' and report['J_gradient_agreement_maxabs'] < 1e-12
    assert report['native_x_is_diagnostic_not_one_global_optimizer']
    assert len(x) == len(gradient) == spec.rank*spec.norb**2


def artificial_study():
    """Reproducible invented examples, never molecular evidence or speedups."""
    from scipy.optimize import minimize
    rows = []
    for role in ('pr', 'rc2'):
        g, J, U, spec = invented(role=role)
        _, _, report = m.optimize(g, J, U, spec)
        module = m.pr if role == 'pr' else m.rc2
        _, _, x0, _, _ = m.pr.corrected.codec.roundtrip(J, U)
        vg = module.make_value_and_grad(spec)
        def fun(x):
            value, gradient = vg(x, g)
            return float(value), np.asarray(gradient)
        reference = minimize(fun, x0, jac=True, method='L-BFGS-B',
            options=dict(gtol=1e-7, ftol=1e-15, maxiter=1000, maxfun=5000))
        rows.append(dict(role=role, invented_seed=41, norb=2, rank=3, ld=1,
            initial=report['states'][0], terminal=report['terminal'], status=report['status'],
            completed_cycles=report['completed_cycles'],
            block_counts=dict(Counter(b['block'] for b in report['blocks'])),
            block_solver_evaluations={kind:sum(b.get('nfev', 0) for b in report['blocks'] if b['block']==kind)
                                     for kind in ('J', 'U')},
            joint_artificial_reference=dict(method='L-BFGS-B', success=bool(reference.success),
                status=int(reference.status), objective=float(reference.fun),
                native_gradient_infinity=float(np.max(abs(reference.jac))),
                nfev=int(reference.nfev), nit=int(reference.nit)),
            interpretation='Within-objective technical control only; no universal ordering or H4 performance conclusion'))
    return dict(schema='C_SEPARATED_J_U_ARTIFICIAL_STUDY_V1', rows=rows,
        actual_molecular_arrays_loaded=0, actual_scientific_calls=0,
        strict_guarantee=False, cross_objective_raw_F_ranking=None)


def test_joint_artificial_control_does_not_assume_separated_superiority():
    result = artificial_study()
    for row in result['rows']:
        assert row['terminal']['components']['objective'] <= row['initial']['components']['objective']
        assert row['joint_artificial_reference']['objective'] <= row['initial']['components']['objective']
        assert row['terminal']['first_order_met']
    assert result['actual_scientific_calls'] == 0
    assert result['cross_objective_raw_F_ranking'] is None


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser(description='Invented DF example only')
    parser.add_argument('--synthetic-report', type=Path, required=True)
    args = parser.parse_args()
    with args.synthetic_report.open('x') as f:
        json.dump(artificial_study(), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
