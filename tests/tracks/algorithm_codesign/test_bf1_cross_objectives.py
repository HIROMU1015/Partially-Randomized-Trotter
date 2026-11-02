"""Synthetic post-search attribution; no Hamiltonian or science inputs."""
from contextlib import contextmanager
from dataclasses import dataclass
import copy
import json
import numpy as np
import pytest

from trottertracks.algorithm_codesign.cross_objectives import cross_score, attribute_F_winner
from trottertracks.algorithm_codesign.freeze import canonical
from trottertracks.algorithm_codesign.pilot import Evaluator, parameter_score


@dataclass(frozen=True)
class SyntheticPoint:
    identity: str
    label: str = 'synthetic'


class SyntheticScorer:
    def __init__(self):
        self.ideal, self.finite = {'existing': {}}, {'existing': {}}
        self.calls = []
        self.cached = False
        self.limits = self
    def check(self):
        pass
    @contextmanager
    def cached_diagnostics(self):
        self.cached = True
        try:
            yield
        finally:
            self.cached = False
    def objective_record(self, point, arm):
        assert self.cached
        self.calls.append((point.identity, arm))
        return dict(arm=arm, feasible=True, value=100.+int(point.identity),
                    best_cell=dict(q=1, R_bud=None if arm == 'O' else 5, K=None if arm == 'O' else 2))


def completed_sets():
    return {arm: [SyntheticPoint(str(i)) for i in range(32)] for arm in ('O', 'L', 'F')}


def test_existing_union_is_cross_scored_once_and_does_not_affect_search():
    points = completed_sets()
    fixed = [SyntheticPoint(str(i)) for i in range(30, 34)]
    before = copy.deepcopy(points)
    evaluator = SyntheticScorer()
    recorded = []
    result = cross_score(evaluator, points, fixed, on_row=recorded.append)
    assert points == before
    assert len(result['rows']) == 34 and len(evaluator.calls) == 3*34
    assert len(set(evaluator.calls)) == len(evaluator.calls)
    assert result['added_search_evaluations'] == result['added_ideal_cells'] == result['added_finite_cells'] == 0
    assert result['cell_counts_before'] == result['cell_counts_after']
    assert len(recorded) == 34 and not result['primary_route_changed']
    row = next(r for r in result['rows'] if r['coefficient'] == '30')
    assert row['origins'] == ['F', 'L', 'O', 'fixed']
    assert set(row['scores']) == {'O', 'L', 'F'}
    assert row['scores']['L']['union_rank'] == 31


def test_cross_scoring_cannot_start_with_an_unfinished_search():
    points = completed_sets()
    points['F'].pop()
    with pytest.raises(RuntimeError, match='COMPLETED_32_POINT'):
        cross_score(SyntheticScorer(), points, [SyntheticPoint(str(i)) for i in range(4)])


def test_cached_diagnostic_phase_never_acquires_a_missing_signal():
    # A bare Evaluator suffices to test the guard: no target/spectrum is built.
    evaluator = Evaluator.__new__(Evaluator)
    evaluator.ideal, evaluator.finite = {}, {}
    evaluator._cache_only = False
    point = SyntheticPoint('one')
    with evaluator.cached_diagnostics():
        with pytest.raises(RuntimeError, match='MISSING_POSTSEARCH_IDEAL_CELL'):
            evaluator.ideal_cell(point, 1)
        with pytest.raises(RuntimeError, match='MISSING_POSTSEARCH_FINITE_CELL'):
            evaluator.finite_cell(point, 1, 5, 2)
    assert not evaluator._cache_only and not evaluator.ideal and not evaluator.finite


class ParameterScoreScorer(SyntheticScorer):
    """Real score scalar types, with synthetic costs and no signal acquisition."""
    def __init__(self, case, native_floats=False):
        super().__init__()
        self.case, self.native_floats = case, native_floats

    def objective_record(self, point, arm):
        assert self.cached
        self.calls.append((point.identity, arm))
        index = int(point.identity)
        ordinal = index//2 if self.case == 'ties' else index
        bias = [.1, .1] if self.case == 'infeasible' and index == 3 else [.001, .002]
        score = parameter_score(bias, 1e-8, 0., 100+ordinal, 5, .01)
        if score['feasible']:
            assert isinstance(score['value'], np.float64)
            if self.native_floats:
                score = dict(score, value=float(score['value']), lower=float(score['lower']))
        cell = dict(q=1, R_bud=None if arm == 'O' else 5,
                    K=None if arm == 'O' else 2, **score) if score['feasible'] else None
        return dict(arm=arm, feasible=score['feasible'], value=score['value'], best_cell=cell)


@pytest.mark.parametrize('case', ['ordered', 'ties', 'infeasible'])
def test_parameter_score_cross_rows_and_result_are_json_serializable(case):
    points = completed_sets()
    before = copy.deepcopy(points)
    fixed = [SyntheticPoint(str(i)) for i in range(30, 34)]
    scorer = ParameterScoreScorer(case)
    emitted = []
    # Use the runner's actual encoder at the callback boundary, then on the
    # complete payload. Python-float-only mocks did not exercise this boundary.
    result = cross_score(scorer, points, fixed, on_row=lambda row: emitted.append(canonical(row)))
    decoded = json.loads(canonical(result))
    native = cross_score(ParameterScoreScorer(case, native_floats=True), points, fixed)
    assert decoded == json.loads(canonical(native))
    assert [json.loads(row) for row in emitted] == [dict(kind='cross_objective', **row) for row in decoded['rows']]
    assert len(emitted) == 34 and len(scorer.calls) == len(set(scorer.calls)) == 102
    assert points == before and result['cell_counts_before'] == result['cell_counts_after']
    assert result['added_search_evaluations'] == result['added_ideal_cells'] == result['added_finite_cells'] == 0
    assert not result['primary_route_changed'] and not scorer.cached
    for row in result['rows']:
        index = int(row['coefficient'])
        expected = 2*(index//2)+1 if case == 'ties' else index+1
        if case == 'infeasible':
            expected = None if index == 3 else expected-int(index > 3)
        for score in row['scores'].values():
            assert score['union_rank'] == expected
            if expected is None:
                assert score['relative_regret_to_union_minimum'] is None
            else:
                assert type(score['union_rank']) is int


@pytest.mark.parametrize('L_value, expected', [
    (90., 'SEARCH_REACHABILITY_OR_BUDGET_EXPLANATION_NOT_EXCLUDED'),
    (100., 'SEARCH_REACHABILITY_OR_BUDGET_EXPLANATION_NOT_EXCLUDED'),
    (110., 'L_DID_NOT_PREFER_F_WINNER_POINT_ESTIMATE')])
def test_attribution_distinguishes_objective_ordering_from_search_reachability(L_value, expected):
    cross = dict(rows=[dict(coefficient='F_winner', scores={
        'O': dict(feasible=True, value=150.), 'L': dict(feasible=True, value=L_value),
        'F': dict(feasible=True, value=80.)})], own_search_minima={'L': {'value': 100.}})
    decision = dict(outcome='BF-C', best_F=dict(coefficient='F_winner'))
    before = copy.deepcopy(decision)
    result = attribute_F_winner(cross, decision)
    assert result['interpretation'] == expected and decision == before
    assert not result['affects_primary_classification'] and not result['establishes_design_principle']
