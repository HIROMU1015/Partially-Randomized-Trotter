"""Invented DF regressions and mock fresh-run/physical gates only."""
from dataclasses import asdict, replace
from pathlib import Path
import json
import subprocess
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks import c_u_step_control as a
from trottertracks import c_u_step_control_pilot as m
from test_research_c_separated_J_U import invented


def physical(label, J, U):
    return dict(budget_PASS=True, exact_tail_lambda_Ha=float(np.sum(abs(a.stable.pr.coefficients(J[1:], np)))))


def test_invented_multistep_preserves_cold_J_refinements_recenter_and_qualification(monkeypatch):
    g, J, U, spec = invented()
    cold, bases, calls, events = [], [], {}, []
    real = a.small.Oracle.__init__
    def initialize(self, g, j, u, *args, **kwargs):
        cold.append(j.copy()); bases.append(u.copy())
        real(self, g, j, u, *args, **kwargs)
    monkeypatch.setattr(a.small.Oracle, '__init__', initialize)
    def call(kind, label, fn):
        calls[kind] = calls.get(kind, 0) + 1
        return fn()
    r = a.trajectory(g, J, U, spec, physical, step=replace(a.STEP, max_updates=3),
                     call=call, record=lambda *x: events.append(x))
    assert r['accepted_U_updates'] >= 2
    assert all(np.array_equal(x, J) for x in cold)
    assert not np.array_equal(bases[1], bases[2])
    assert calls['inner_J_solve'] == 3 * r['point_evaluations']
    assert calls['precision_state'] == 2 * r['point_evaluations']
    assert r['refinement_solves'] == 2 * r['point_evaluations']
    assert calls['independent_terminal_assessment'] == r['accepted_U_updates']
    assert all(all(row['qualification']['admission_gates'].values()) for row in r['history'])
    assert all(row['physical']['budget_PASS'] for row in r['history'])
    values = [row['state']['components']['objective'] for row in r['history']]
    assert all(y < x for x, y in zip(values, values[1:]))
    np.testing.assert_allclose(r['terminal']['U'].transpose(0, 2, 1) @ r['terminal']['U'],
                               np.broadcast_to(np.eye(2), U.shape), atol=1e-12)
    assert r['certified_descent'] is r['true_reduced_gradient_qualification'] is None
    kinds = [event[0] for event in events]
    assert kinds.index('independent_terminal_assessment') < kinds.index('accepted')


@pytest.mark.parametrize('mode', ['post_J', 'terminal_mismatch', 'DF', 'initial_DF', 'physical_correctness'])
def test_post_trial_gates_never_export_a_failed_candidate(monkeypatch, mode):
    g, J, U, spec = invented()
    events = []
    real = a.small.assess
    def assess(*args, **kwargs):
        q = real(*args, **kwargs)
        if mode == 'terminal_mismatch':
            raise ValueError('INDEPENDENT_TERMINAL_STATE_MISMATCH')
        if mode == 'post_J':
            return dict(q, J_numerically_qualified=False, status='POST_UPDATE_J_UNQUALIFIED')
        return q
    monkeypatch.setattr(a.small, 'assess', assess)
    def assessment(label, j, u):
        if mode == 'physical_correctness':
            raise ValueError('ARTIFICIAL_GENERAL_K_CAR_MISMATCH')
        return dict(physical(label, j, u), budget_PASS=mode != 'initial_DF' and (mode != 'DF' or label == 'initial'))
    if mode in ('terminal_mismatch', 'physical_correctness'):
        with pytest.raises(ValueError):
            a.trajectory(g, J, U, spec, assessment, step=replace(a.STEP, max_updates=2),
                         record=lambda *x: events.append(x))
    else:
        r = a.trajectory(g, J, U, spec, assessment, step=replace(a.STEP, max_updates=2),
                         record=lambda *x: events.append(x))
        assert r['accepted_U_updates'] == 0
        if mode == 'DF':
            assert r['optimization_accepted_updates'] == 1
            assert r['status'] == 'PROVISIONAL_CANDIDATE_DF_BUDGET_NOT_MET'
        if mode == 'initial_DF':
            assert r['terminal'] is None
        else:
            np.testing.assert_array_equal(r['terminal']['U'], U)
    assert not any(event[0] == 'accepted' for event in events)


def fixture(root, monkeypatch):
    from trottertracks.c_pr_integrated_evaluation import arrays
    g, J, U, spec = invented()
    paths = dict(input='input.npz', initial='initial.npz', selected='selected.npz',
                 scales='scales.json', result='old_result.json', STOP='old_STOP.json')
    arrays(root / 'input.npz', g=g)
    arrays(root / 'initial.npz', J=J, U=U)
    arrays(root / 'selected.npz', J=J, U=U)
    (root / 'scales.json').write_text(json.dumps(dict(initial_tensor_objective=spec.loss_scale,
                                                   initial_smooth_tail_mass=spec.mass_scale)))
    candidate = dict(id='invented', sha256=m.io.sha(root / 'selected.npz'))
    (root / 'old_result.json').write_text(json.dumps(dict(selected=dict(id=candidate['id'],
        physical=dict(candidate_sha256=candidate['sha256'], budget_PASS=True)))))
    (root / 'old_STOP.json').write_text(json.dumps(dict(failed=False, mandatory_stop=True)))
    entry = root / m.ENTRY
    entry.parent.mkdir(); entry.write_text('# invented entry\n')
    (root / 'inherited.json').write_text(json.dumps(dict(files={})))
    source = dict(files={m.ENTRY: m.io.sha(entry)}, inherited_manifest=dict(
        path='inherited.json', sha256=m.io.sha(root / 'inherited.json')))
    (root / 'SOURCE.json').write_text(json.dumps(source))
    c = dict(base_commit=m.BASE, scope=m.SCOPE, entry=m.ENTRY, objective=asdict(spec),
             policy=asdict(a.DEFAULT), step_policy=asdict(a.STEP), DF_budget_Ha=.00019875,
             fresh_output=m.FRESH, call_limits=m.LIMITS, maximum_logical_units=m.MAX_CALLS,
             maximum_U_points=m.POINTS, retry=0, resume=0, mandatory_stop=True,
             source_manifest='SOURCE.json', source_manifest_sha256=m.io.sha(root / 'SOURCE.json'),
             input_paths=paths, initial_candidate=candidate,
             saved_inputs={p: m.io.sha(root / p) for p in paths.values()},
             fit_packages={}, evaluation_packages={}, ffsim_sources={})
    (root / 'contract.json').write_text(json.dumps(c))
    monkeypatch.setattr(m, 'OBJECTIVE', asdict(spec))
    return c


class Service:
    calls = []
    mode = 'pass'
    def __init__(self, *args):
        self.__class__.calls = []
    def request(self, operation, **kwargs):
        self.calls.append(operation)
        if self.mode == 'correctness':
            raise ValueError('ARTIFICIAL_PHYSICAL_MISMATCH')
        if operation == 'assess':
            return dict(budget_PASS=self.mode != 'DFfail', exact_tail_lambda_Ha=1.)
        return dict(status='READY')
    def close(self):
        pass


def test_passive_does_not_import_science_or_decode_npz(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(np, 'load', lambda *args, **kwargs: pytest.fail('passive decode'))
    m.preflight(tmp_path, tmp_path / 'contract.json')
    assert m.POINTS == 2049 and m.MAX_CALLS == sum(m.LIMITS.values())
    code = "import sys;sys.path.insert(0,'src');from trottertracks import c_u_step_control_pilot;assert not any(k in sys.modules for k in ('numpy','scipy','jax','ffsim','qiskit'))"
    subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, check=True)


@pytest.mark.parametrize('mode', ['pass', 'DFfail', 'correctness'])
def test_mock_fresh_run_no_downstream_and_STOP(tmp_path, monkeypatch, mode):
    c = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(Service, 'mode', mode)
    def trajectory(g, j, u, spec, physical, **kwargs):
        physical('initial', j, u)
        return dict(status='ARTIFICIAL_LIMIT', terminal=dict(J=j, U=u, state={'first_order_met': False}),
                    history=[], trials=[], accepted_U_updates=0)
    monkeypatch.setattr(a, 'trajectory', trajectory)
    if mode == 'correctness':
        with pytest.raises(ValueError, match='ARTIFICIAL_PHYSICAL_MISMATCH'):
            m.run(tmp_path, tmp_path / 'contract.json', service_factory=Service)
    else:
        result = m.run(tmp_path, tmp_path / 'contract.json', service_factory=Service)
        assert result['native'] is result['PF_RTE'] is result['shots'] is None
        assert set(Service.calls) <= {'initialize', 'assess'}
    stop = m.io.read(tmp_path / m.FRESH / 'STOP.json')
    assert stop['failed'] == (mode == 'correctness') and stop['scientific_units'] <= m.MAX_CALLS
    assert stop['retry'] == stop['resume'] == 0 and stop['mandatory_stop']
    assert all(m.io.sha(tmp_path / p) == h for p, h in c['saved_inputs'].items())
    with pytest.raises(ValueError, match='FRESH_U_STEP_CONTROL'):
        m.run(tmp_path, tmp_path / 'contract.json', service_factory=Service)


def test_source_input_and_scope_mismatches_fail_before_decode(tmp_path, monkeypatch):
    c = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(np, 'load', lambda *args, **kwargs: pytest.fail('unexpected NPZ load'))
    (tmp_path / 'selected.npz').write_bytes(b'invented mismatch')
    with pytest.raises(ValueError, match='SOURCE_INPUT'):
        m.preflight(tmp_path, tmp_path / 'contract.json')
    assert not (tmp_path / m.FRESH).exists()
    c['step_policy']['max_trials'] = 33
    (tmp_path / 'contract.json').write_text(json.dumps(c))
    with pytest.raises(ValueError, match='FIXED_U_STEP_CONTROL'):
        m.preflight(tmp_path, tmp_path / 'contract.json')


def test_evaluation_worker_preflight_allows_parent_created_fresh_directory(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)
    (tmp_path / m.FRESH).mkdir(parents=True)
    m.preflight(tmp_path, tmp_path / 'contract.json', evaluation=True)
    with pytest.raises(ValueError, match='FRESH_U_STEP_CONTROL'):
        m.preflight(tmp_path, tmp_path / 'contract.json')


def test_worker_cannot_enter_performance_stage(monkeypatch):
    import io
    monkeypatch.setattr(sys, 'stdin', io.StringIO('{"operation":"performance"}\n'))
    with pytest.raises(ValueError, match='OPTIMIZATION_ONLY_EVALUATION'):
        m.evaluation_worker(ROOT, {}, ROOT)


def test_initial_inner_qualification_failure_prevents_physical_and_outer_use(monkeypatch):
    g, J, U, spec = invented()
    real = a.small.Oracle.evaluate
    def evaluate(self, eta):
        real(self, eta)
        raise a.stable.Unqualified('J_UNQUALIFIED')
    monkeypatch.setattr(a.small.Oracle, 'evaluate', evaluate)
    def never(*args):
        pytest.fail('unqualified initial used for physical assessment')
    r = a.trajectory(g, J, U, spec, never)
    assert r['status'] == 'J_UNQUALIFIED' and r['terminal'] is None
    assert r['accepted_U_updates'] == 0 and not r['trials']


def test_raw_precision_loss_flag_does_not_replace_numerical_admission(monkeypatch):
    g, J, U, spec = invented()
    real = a.small.Oracle.evaluate
    def evaluate(self, eta):
        p = real(self, eta)
        p['qualification']['raw_solver_flags'][0].update(success=False, status=2,
                                                       message='invented precision loss observation')
        return p
    monkeypatch.setattr(a.small.Oracle, 'evaluate', evaluate)
    r = a.trajectory(g, J, U, spec, physical, step=replace(a.STEP, max_updates=2))
    assert r['accepted_U_updates'] >= 1
    assert all(row['qualification']['raw_solver_flags'][0]['status'] == 2 for row in r['history'])
    assert all(all(row['qualification']['admission_gates'].values()) for row in r['history'])


def test_worker_close_failure_sets_failed_STOP(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)
    class BadClose(Service):
        mode = 'DFfail'
        def close(self):
            raise RuntimeError('ARTIFICIAL_WORKER_NONZERO_EXIT')
    with pytest.raises(RuntimeError, match='ARTIFICIAL_WORKER_NONZERO_EXIT'):
        m.run(tmp_path, tmp_path / 'contract.json', service_factory=BadClose)
    stop = m.io.read(tmp_path / m.FRESH / 'STOP.json')
    assert stop['failed'] and stop['stop_reason'] == 'CORRECTNESS_OR_EXECUTION_FAILURE'
