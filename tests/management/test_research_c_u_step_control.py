"""Scalar/analytic and fault-injection tests, no saved molecules or JAX."""
from dataclasses import replace
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from trottertracks import c_u_step_control as m


def point(x, *, c=1., L=1.):
    value = c + L * float(x[0]) ** 2 / 2
    return dict(eta=x.copy(), gradient=L * x, value=value,
                state=dict(first_order_met=False, value_same_array={'maxabs': 0.},
                           binary64_AD_value_same_array={'maxabs': 0.}),
                qualification=dict(status='NUMERICAL_PILOT_ADMITTED',
                                   admission_gates={key: True for key in m.GATES},
                                   objective_refinement=dict(before=value, after=value)))


def test_positive_offset_legacy_three_steps_fail_but_standard_step_decreases():
    x = np.array([1e-3])
    p0 = point(x)
    for q in m.previous.FRACTIONS:
        candidate, decision = m.small.propose(point, p0, m.small.StepPolicy(q))
        alpha = q * p0['value'] / float(x @ x)
        assert decision['alpha'] == pytest.approx(alpha)
        assert alpha > 2.  # Exact quadratic decrease needs 0 < alpha*L < 2.
        assert candidate is p0 and decision['accepted_U_updates'] == 0
    candidate, decision = m.search(point, p0)
    assert decision['alpha'] == 1. and decision['trials'] == 1
    assert candidate['value'] < p0['value']
    assert decision['status'] == 'PROVISIONAL_ARMIJO_PASS'
    assert decision['accepted_U_updates'] == 0 and decision['certified_descent'] is None


def test_step_generation_is_offset_independent_and_shrinks_with_gradient():
    displacement = []
    for x in (1e-2, 1e-3):
        first = []
        for c in (1., 100.):
            p0 = point(np.array([x]), c=c)
            events = []
            _, result = m.search(lambda eta: point(eta, c=c), p0,
                                 record=lambda *event: events.append(event))
            first.append(events[0][3]['eta'])
            assert result['alpha0'] == 1.
        np.testing.assert_array_equal(first[0], first[1])
        displacement.append(abs(float(first[0][0]) - x))
    assert displacement[1] < displacement[0]


def test_backtracking_finds_decrease_beyond_three_trials_and_stops_at_limit():
    p0 = point(np.array([.01]), L=16.)
    evaluate = lambda eta: point(eta, L=16.)
    p, r = m.search(evaluate, p0)
    assert r['trials'] == 5 and r['alpha'] == 1 / 16
    assert p['value'] < p0['value']
    p, r = m.search(evaluate, p0, replace(m.STEP, max_trials=4))
    assert p is p0 and r['trials'] == 4 and r['accepted_U_updates'] == 0


def test_decrease_below_value_resolution_is_not_accepted():
    p0 = point(np.array([1e-9]), L=1e8)
    p, r = m.search(lambda eta: point(eta, L=1e8), p0)
    assert p is p0 and r['trials'] == 32
    assert r['status'] == 'VALUE_RESOLUTION_UNQUALIFIED'


@pytest.mark.parametrize('status', ['J_UNQUALIFIED', 'REDUCED_GRADIENT_UNQUALIFIED'])
def test_unqualified_trial_is_discarded_without_value_or_gradient_use(status):
    events, queries = [], []
    def evaluate(eta):
        queries.append(eta.copy())
        if len(queries) == 1:
            raise m.stable.Unqualified(status)
        return point(eta)
    _, result = m.search(evaluate, point(np.array([.01])), record=lambda *x: events.append(x))
    assert result['trials'] == 2
    failed = next(x[2] for x in events if x[0] == 'backtracking_trial')
    assert failed['objective'] is None and not failed['gradient_used']
    np.testing.assert_allclose(queries[1], np.array([.005]))


def test_all_unqualified_trials_have_a_finite_stop():
    queries = []
    def evaluate(eta):
        queries.append(eta)
        raise m.stable.Unqualified('REDUCED_GRADIENT_UNQUALIFIED')
    _, r = m.search(evaluate, point(np.array([.01])))
    assert r['status'] == 'BACKTRACKING_NUMERICAL_QUALIFICATION_EXHAUSTED'
    assert len(queries) == 32 and r['accepted_U_updates'] == 0


@pytest.mark.parametrize('mode', ['correctness', 'other_stop', 'eta', 'value', 'gradient', 'profile'])
def test_correctness_and_unknown_stops_are_never_halved_away(mode):
    queries = []
    def evaluate(eta):
        queries.append(eta)
        if mode == 'correctness':
            raise ValueError('ARTIFICIAL_CORRECTNESS_FAIL')
        if mode == 'other_stop':
            raise m.stable.Unqualified('OUTER_LIMIT_UNQUALIFIED')
        p = point(eta)
        if mode == 'eta':
            p['eta'] = eta + 1
        if mode == 'value':
            p['value'] = np.inf
        if mode == 'gradient':
            p['gradient'][:] = np.nan
        if mode == 'profile':
            p['state']['value_same_array']['maxabs'] = np.nan
        return p
    with pytest.raises((ValueError, m.stable.Unqualified)):
        m.search(evaluate, point(np.array([.01])))
    assert len(queries) == 1


@pytest.mark.parametrize('gate', m.GATES)
def test_false_admission_flag_cannot_be_accepted(gate):
    def evaluate(eta):
        p = point(eta)
        p['qualification']['admission_gates'][gate] = False
        return p
    with pytest.raises(ValueError, match='FAILED_ADMISSION_GATE'):
        m.search(evaluate, point(np.array([.01])))


def test_zero_direction_rounding_stall_and_nonfinite_slope_do_not_query():
    def never(eta):
        pytest.fail('unexpected trial')
    _, r = m.search(never, point(np.array([0.])))
    assert r['status'] == 'NO_QUALIFIED_DESCENT_DIRECTION'
    p = point(np.array([1e16]))
    p['gradient'][:] = 1.
    _, r = m.search(never, p)
    assert r['status'] == 'STEP_ROUNDING_STALLED' and r['trials'] == 0
    p = point(np.array([.01]))
    p['gradient'][:] = 1e308
    with pytest.raises(ValueError, match='NONFINITE_DESCENT_DIRECTION'):
        m.search(never, p)


def test_only_previous_accepted_alpha_controls_next_initial_step():
    _, r = m.search(point, point(np.array([.01])), previous_alpha=.01)
    assert r['alpha0'] == .02 and r['alpha'] == .02
    for alpha in (0., np.inf, np.nan):
        with pytest.raises(ValueError, match='PREVIOUS_ACCEPTED_ALPHA'):
            m.search(point, point(np.array([.01])), previous_alpha=alpha)


@pytest.mark.parametrize('change', [dict(max_trials=0), dict(max_updates=0),
                                   dict(max_trials=True), dict(contraction=1.),
                                   dict(c1=0.), dict(alpha_max=np.nan),
                                   dict(accepted_alpha_growth=.5)])
def test_malformed_policies_rejected(change):
    with pytest.raises(ValueError):
        replace(m.STEP, **change)


def test_real_invented_evaluation_worker_pipe(tmp_path):
    import json
    import subprocess
    from test_research_c_pr_integrated_pipeline import synthetic_packet
    from trottertracks.c_pr_integrated_evaluation import arrays
    packet, J, U = synthetic_packet()
    arrays(tmp_path / 'input.npz', **packet)
    arrays(tmp_path / 'candidate.npz', J=J, U=U)
    out = tmp_path / 'out'
    out.mkdir()
    c = dict(input_paths={'input': 'input.npz'}, DF_budget_Ha=.00019875)
    code = ("import sys,json;from pathlib import Path;sys.path.insert(0,sys.argv[1]);"
            "from trottertracks.c_u_step_control_pilot import evaluation_worker;"
            "evaluation_worker(Path(sys.argv[2]),json.loads(sys.argv[3]),Path(sys.argv[4]))")
    requests = [dict(operation='initialize'), dict(operation='assess', label='invented',
                path=str(tmp_path / 'candidate.npz')), dict(operation='close')]
    result = subprocess.run([sys.executable, '-B', '-c', code, str(Path(__file__).resolve().parents[2] / 'src'),
                             str(tmp_path), json.dumps(c), str(out)],
                            input=''.join(json.dumps(x) + '\n' for x in requests),
                            text=True, capture_output=True, check=True)
    replies = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(replies) == 2 and all(row['ok'] for row in replies)
    physical = replies[1]['value']
    assert physical['budget_PASS'] and physical['sector_error'] < 1e-10
    assert physical['representation']['max_entry'] < 1e-10
    assert physical['kappa'] == 'original_g_corrected' and physical['rigorous_enclosure'] is None
