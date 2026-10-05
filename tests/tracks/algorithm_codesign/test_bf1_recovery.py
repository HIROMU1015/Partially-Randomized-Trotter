"""Recovery checks use invented scalar cells, never molecular/science data."""
import copy
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import mpmath as mp
import pytest

from trottertracks.algorithm_codesign import pilot
from trottertracks.algorithm_codesign.adapter import allocate, stationary_template, tail_statistics
from trottertracks.algorithm_codesign.cross_objectives import cross_score, attribute_F_winner
from trottertracks.algorithm_codesign.domain import DPS, Domain, fixed_references
from trottertracks.algorithm_codesign.freeze import canonical
from trottertracks.algorithm_codesign.recovery import (
    ReadOnlyEvaluator, RecoveryFailure, forbid_physical_evaluation, parse_cells, replay)


class NoLimits:
    def check(self):
        pass


class InventedCells(pilot.Evaluator):
    """Populate artificial records with formula metadata, no operator/signal."""
    def __init__(self, lambda_r=1e-4, ideal_bias=1e-4):
        self.ideal, self.finite, self.coefficients = {}, {}, {}
        self.task = SimpleNamespace(lambda_r=lambda_r)
        self.deterministic_count, self.limits, self._cache_only = 4, NoLimits(), False
        self.ideal_bias = ideal_bias

    def ideal_cell(self, point, q):
        key = point.identity, q
        if key not in self.ideal:
            one, full = stationary_template(point, 4, q)
            self.ideal[key] = dict(signal=[0., 0.], bias=[self.ideal_bias]*2, u_signal=1e-10,
                deterministic_actions=sum(f.generator != 'R' for f in full),
                ideal_actions=len(full), tail_count=sum(f.generator == 'R' for f in one))
        return self.ideal[key]

    def finite_cell(self, point, q, budget, K):
        key = point.identity, q, budget, K
        if key not in self.finite:
            one, _ = stationary_template(point, 4, q)
            tails = [f.time for f in one if f.generator == 'R']
            allocation = allocate(point, tails, budget)
            with mp.workdps(DPS):
                times = [float(t.value(point.basis)*mp.mpf(4)/(5*q)) for t in tails]
            self.finite[key] = dict(valid=True, coefficient=point.identity, q=q, R_bud=budget, K=K,
                allocation=list(allocation), tail_times_per_step=times,
                Gamma=q/.8*sum(abs(t) for t in times),
                **tail_statistics(times, allocation, self.task.lambda_r, q, K),
                **dict(self.ideal_cell(point, q), bias=[1e-4]*2))
        return self.finite[key]


@pytest.fixture(scope='module')
def artificial_completed_run():
    domain = Domain()
    _, initial = domain.initial_points()
    fixed = fixed_references(domain)
    evaluator = InventedCells()
    points, searches = {}, {}
    for arm in ('O', 'L', 'F'):
        points[arm], searches[arm] = pilot.search(domain, initial, evaluator, arm)
    reference_points = {p.identity:p for p in points['O']+points['L']+fixed}
    F_points = {p.identity:p for p in points['F']}
    reference = evaluator.rescore(list(reference_points.values()), .01)
    f_rows = evaluator.rescore(list(F_points.values()), .01)
    leading = [r for r in reference if r['coefficient'] in {p.identity for p in points['L']}]
    decision = pilot.classify(f_rows, reference, leading)
    cross = cross_score(evaluator, points, fixed)
    union = {p.identity:p for p in list(reference_points.values())+list(F_points.values())}
    bridge = evaluator.rescore(list(union.values()), .05)
    frozen = dict(domain.manifest(), fixed_references=[p.record() for p in fixed])
    original = dict(status='synthetic_incomplete', outcome='INCONCLUSIVE',
                    input_audit=dict(lambda_r=evaluator.task.lambda_r, native_generator_count=4))
    expected = dict(searches=searches, primary_decision=decision, primary_F=f_rows,
        primary_reference=reference, secondary_frontier=pilot.secondary_frontier(f_rows+reference),
        cross_objectives=cross, objective_attribution=attribute_F_winner(cross, decision),
        bridge=bridge)
    return original, frozen, evaluator.ideal, evaluator.finite, expected


def test_replay_matches_original_rules_and_never_calls_physical_providers(artificial_completed_run):
    original, frozen, ideal, finite, expected = copy.deepcopy(artificial_completed_run)
    before = copy.deepcopy((ideal, finite))
    result = replay(original, frozen, ideal, finite, NoLimits())
    assert result['status'] == 'BF1_READ_ONLY_RECOVERY_COMPLETE', result.get('failure')
    for key in ('searches', 'primary_decision', 'primary_F', 'primary_reference',
                'secondary_frontier', 'cross_objectives', 'objective_attribution'):
        assert canonical(result[key]) == canonical(expected[key])
    assert canonical(result['bridge']['rows']) == canonical(expected['bridge'])
    assert result['cache_values_unchanged'] and (ideal, finite) == before
    assert result['original_cell_counts'] == result['final_cell_counts']
    assert all(result[key] == 0 for key in ('added_ideal_cells','added_finite_cells',
                                         'added_science_signals','added_science_candidates'))
    assert result['mandatory_stop'] and not result['science_retry_authorized']
    assert json.loads(canonical(result))['attribution_scope'] == 'POSTHOC_RECOVERED_OBJECTIVE_ATTRIBUTION'


@pytest.mark.parametrize('kind', ['ideal', 'finite'])
def test_missing_original_cell_stops_without_filling_cache(artificial_completed_run, kind):
    original, frozen, ideal, finite, _ = copy.deepcopy(artificial_completed_run)
    point = frozen['initial_points'][0]['identity']
    if kind == 'ideal':
        ideal.pop((point, 1))
    else:
        finite.pop((point, 1, 5, 2))
    before = copy.deepcopy((ideal, finite))
    result = replay(original, frozen, ideal, finite, NoLimits())
    assert result['status'] == 'BF1_READ_ONLY_RECOVERY_INCOMPLETE'
    assert result['failure']['reason'] == 'RESCUE_INCOMPLETE_MISSING_ORIGINAL_CELL'
    assert result['failure']['kind'] == kind and not result['primary_recovered']
    assert (ideal, finite) == before and result['cache_values_unchanged']


def test_read_only_cache_rejects_mutation_and_unknown_coefficient(artificial_completed_run):
    original, _, ideal, finite, _ = artificial_completed_run
    evaluator = ReadOnlyEvaluator(ideal, finite, original['input_audit']['lambda_r'], 4, NoLimits())
    key = next(iter(ideal))
    with pytest.raises(TypeError):
        evaluator.ideal[key]['bias'] = [0., 0.]
    with pytest.raises(RecoveryFailure, match='MISSING_ORIGINAL_CELL'):
        evaluator.ideal_cell(SimpleNamespace(identity='f'*64), 1)


def test_bridge_trigger_cannot_acquire_a_missing_K4_cell():
    domain = Domain()
    point = domain.suzuki()
    synthetic = InventedCells(lambda_r=2., ideal_bias=.004)
    # At epsilon=.01 the ideal-bias trigger excludes K4 for this point.
    synthetic.rescore([point], .01)
    assert not any(k[3] == 4 for k in synthetic.finite)
    readonly = ReadOnlyEvaluator(synthetic.ideal, synthetic.finite, 2., 4, NoLimits())
    before = readonly.snapshot()
    with pytest.raises(RecoveryFailure, match='MISSING_ORIGINAL_CELL'):
        readonly.rescore([point], .05)
    assert readonly.snapshot() == before


@pytest.mark.parametrize('problem', ['duplicate', 'unknown_kind', 'unsupported_q'])
def test_original_cache_parser_rejects_ambiguous_or_unsupported_records(problem):
    row = dict(kind='ideal', coefficient='a'*64, q=1, bias=[0., 0.], u_signal=0.)
    if problem == 'unknown_kind':
        row['kind'] = 'cross_objective'
    if problem == 'unsupported_q':
        row['q'] = 16
    text = json.dumps(row)+'\n'
    if problem == 'duplicate':
        text += text
    with pytest.raises(RecoveryFailure):
        parse_cells(text)


def test_physical_evaluation_guard_rejects_constructor_loader_and_import():
    with forbid_physical_evaluation():
        with pytest.raises(RecoveryFailure):
            pilot.Evaluator(None)
        with pytest.raises(RecoveryFailure):
            pilot.np.load('synthetic_should_never_be_opened')
        with pytest.raises(RecoveryFailure):
            __import__('trottertracks.algorithm_codesign.science_input')


def test_missing_optional_bridge_preserves_complete_primary(artificial_completed_run, monkeypatch):
    original, frozen, ideal, finite, _ = artificial_completed_run
    rescore = ReadOnlyEvaluator.rescore
    def missing_bridge(self, points, epsilon):
        if epsilon == .05:
            raise RecoveryFailure('RESCUE_INCOMPLETE_MISSING_ORIGINAL_CELL', kind='finite', key=['synthetic',1,5,4])
        return rescore(self, points, epsilon)
    monkeypatch.setattr(ReadOnlyEvaluator, 'rescore', missing_bridge)
    result = replay(original, frozen, ideal, finite, NoLimits())
    assert result['status'] == 'BF1_READ_ONLY_RECOVERY_COMPLETE'
    assert result['primary_recovered'] and result['cross_objectives_recovered']
    assert result['bridge']['status'] == 'MISSING_FROM_ORIGINAL_RUN'


@pytest.fixture
def runner_namespace():
    path = Path(__file__).absolute().parents[3]/'scripts/tracks/algorithm_codesign/run_bf1_read_only_recovery.py'
    return runpy.run_path(str(path))


@pytest.mark.parametrize('path', ['synthetic.npz', '../synthetic.json', '/synthetic.json'])
def test_runner_rejects_nontext_or_escaping_inputs_without_opening_them(runner_namespace, path):
    with pytest.raises(PermissionError):
        runner_namespace['checked_text'](path)


def test_draft_recovery_rejects_before_git_or_original_cache_IO(runner_namespace, tmp_path, monkeypatch):
    main = runner_namespace['main']
    monkeypatch.setitem(main.__globals__, 'ROOT', tmp_path)
    monkeypatch.setitem(main.__globals__, 'CONTRACT', 'draft.json')
    (tmp_path/'draft.json').write_text(json.dumps(dict(read_only_recovery_authorized=False,
                                                     science_execution_authorized=False)))
    monkeypatch.setattr(sys, 'argv', ['recovery'])
    def never_git(*args):
        raise AssertionError('git/cache phase must not be reached')
    monkeypatch.setitem(main.__globals__, 'git', never_git)
    with pytest.raises(PermissionError, match='draft recovery'):
        main()
