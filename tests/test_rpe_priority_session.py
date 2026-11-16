"""Artificial equivalence of the resumable priority menu to fixed e839b0d3f.

No real signal or scientific Q3 is evaluated.  The oracle is loaded from the
fixed Git object, not from another import of the edited implementation.
"""
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import subprocess
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from trottertracks import rpe_method_optimizer as O
from trottertracks import rpe_priority_search as P
import check_rpe_priority_search as CP

FIXED = types.ModuleType('_fixed_priority_e839b0d3f')
FIXED.__file__ = str(ROOT / 'src/trottertracks/rpe_priority_search.py')
exec(compile(subprocess.check_output(['git', 'show', 'e839b0d3f:src/trottertracks/rpe_priority_search.py'], cwd=ROOT),
             'e839b0d3f:src/trottertracks/rpe_priority_search.py', 'exec'), FIXED.__dict__)


def normalized(result):
    value = deepcopy(result)
    value.pop('selection_timing', None)  # timers are measured separately on each invocation
    return value


def fixtures():
    cands = [CP.config('A', 5, '2', 1), CP.config('B', 5, '1', Fraction(11, 10)), CP.config('C', 3, '1', 2)]
    base = sum(w / n for w, n in zip(CP.W, CP.CAPS))
    function = CP.mse_fn({'A': Fraction(1, 2) / base, 'B': Fraction(3, 5) / base, 'C': Fraction(7, 10) / base})
    return cands, function, CP.seeded(cands, function)


def drive_default(session):
    try:
        frame = next(session)
        while True:
            frame = session.send(frame['default_choice'])
    except StopIteration as finished:
        return finished.value


class PrioritySessionTests(unittest.TestCase):
    def compare(self, cands, function, knowledge, search, budget=None, on_missing='STOP', backend_factory=None):
        previews = [[], [], []]
        results = []
        for i, implementation in enumerate((FIXED.run_priority, P.run_priority, P.priority_session)):
            backend = O.FixtureBackend(function, CP.TAU) if backend_factory is None else backend_factory()
            result = implementation('PR_PAULI', deepcopy(cands), deepcopy(knowledge), backend, dict(search), CP.TAU,
                CP.none_generator, budget, baseline=dict(formula='2nd', delta_over_h='1', kappa_multiplier='1'),
                prefetch=lambda ps, p=previews[i]: p.append(deepcopy(ps)), on_missing=on_missing)
            results.append(drive_default(result) if i == 2 else result)
        self.assertEqual(normalized(results[0]), normalized(results[1]))
        self.assertEqual(normalized(results[0]), normalized(results[2]))
        self.assertEqual(previews[0], previews[1])
        self.assertEqual(previews[0], previews[2])
        return results[0]

    def test_default_wrapper_and_session_match_fixed_multistep(self):
        cands, function, knowledge = fixtures()
        result = self.compare(cands, function, knowledge,
                              dict(CP.SEARCH, max_steps_per_method=120, lookahead=6, lookahead_mode='HYBRID'))
        self.assertGreater(len(result['trace']), 12)
        self.assertIn('EXPLORE', {r['cls'] for r in result['trace']})
        self.assertIn(O.FAIL, {r['status'] for r in result['trace']})

    def test_fine_and_finite_budget_match_fixed(self):
        cands, function, knowledge = fixtures()
        result = self.compare(cands, function, knowledge,
                              dict(CP.SEARCH, max_steps_per_method=400, coarse_gain_fraction='1/5',
                                   lookahead=6, lookahead_mode='HYBRID'))
        self.assertIn('FINE', {r['cls'] for r in result['trace']})
        self.compare(cands, function, knowledge, dict(CP.SEARCH, max_steps_per_method=30, lookahead=6,
                                                     lookahead_mode='HYBRID'), budget=3)

    def test_nonmonotone_deferred_success_matches_fixed(self):
        cands = [CP.config('A', 5, '2', 1)]
        base = tuple(n for n in CP.CAPS for axis in O.AXES)
        failed = list(base);failed[22] -= 1;failed[23] -= 1;failed = tuple(failed)
        target = list(base);target[22] -= 2;target[23] -= 2;target = tuple(target)
        def truth(cid, vector):
            x = Fraction(1, 2) if vector in (base, target) else Fraction(2)
            return x, x
        knowledge = {('A', base): dict(status=O.PASS, MSE_lower=Fraction(1, 2), MSE_upper=Fraction(1, 2), source='SAVED', calls=0),
                     ('A', failed): dict(status=O.FAIL, MSE_lower=Fraction(2), MSE_upper=Fraction(2), source='SAVED', calls=0)}
        result = self.compare(cands, truth, knowledge, dict(CP.SEARCH, max_steps_per_method=400))
        self.assertEqual(result['incumbent']['vector'], target)
        self.assertTrue(any(r['cls'] == 'DEFERRED' and r['status'] == O.PASS for r in result['trace']))

    def test_missing_answers_skip_matches_fixed(self):
        cands, function, knowledge = fixtures()
        class Answers:
            def answer(self, proposal):
                if sum(proposal['vector']) % 3 == 0:
                    return None
                lo, up = function(proposal['configuration_id'], proposal['vector'])
                return dict(status=O.PASS if up <= CP.TAU else O.FAIL, MSE_lower=lo, MSE_upper=up,
                            source='ARTIFICIAL_LOG', calls=1)
        result = self.compare(cands, function, knowledge, dict(CP.SEARCH, max_steps_per_method=30),
                              on_missing='SKIP', backend_factory=Answers)
        self.assertTrue(result['unevaluated'])

    def test_refresh_activates_static_configuration_without_consuming(self):
        cands, function, knowledge = fixtures();cands = cands[:2]
        cands[1]['prices'] = [p / 10 for p in cands[1]['prices']]
        active = {'A'};answered = []
        class Backend:
            def answer(self, proposal):
                answered.append(proposal['configuration_id'])
                return O.FixtureBackend(function, CP.TAU).answer(proposal)
        session = P.priority_session('PR_PAULI', cands, knowledge, Backend(),
                    dict(CP.SEARCH, max_steps_per_method=5), CP.TAU, CP.none_generator, None,
                    active_configurations=lambda: active)
        first = next(session)
        self.assertEqual(first['active_configurations'], ['A'])
        self.assertEqual({p['configuration_id'] for p in first['eligible']}, {'A'})
        active.add('B');second = session.send({'refresh': True})
        self.assertEqual(first['step'], second['step'])
        self.assertEqual(first['since_explore'], second['since_explore'])
        self.assertEqual(first['new_calls'], second['new_calls'])
        self.assertLess(second['method_min'], first['method_min'])
        self.assertEqual(second['default_choice']['configuration_id'], 'B')
        self.assertEqual(answered, [])
        session.send(second['default_choice']);self.assertEqual(answered, ['B']);session.close()

    def test_refresh_preserves_exploration_counter(self):
        cands, function, knowledge = fixtures()
        session = P.priority_session('PR_PAULI', cands, knowledge, O.FixtureBackend(function, CP.TAU),
                    dict(CP.SEARCH, max_steps_per_method=30), CP.TAU, CP.none_generator, None)
        frame = next(session)
        for _ in range(12):
            if frame['since_explore'] >= 3 and frame['default_choice']['cls'] == 'EXPLORE':
                break
            frame = session.send(frame['default_choice'])
        self.assertGreaterEqual(frame['since_explore'], 3)
        refreshed = session.send({'refresh': True})
        self.assertEqual(refreshed['step'], frame['step'])
        self.assertEqual(refreshed['since_explore'], frame['since_explore'])
        self.assertEqual(refreshed['default_choice'], frame['default_choice']);session.close()

    def test_refresh_new_known_neighbour_updates_transfer_seeds(self):
        cands, function, unused = fixtures();cands = cands[:2]
        knowledge = {};active = {'A'}
        session = P.priority_session('PR_PAULI', cands, knowledge, O.FixtureBackend(function, CP.TAU),
                    dict(CP.SEARCH, max_steps_per_method=5), CP.TAU, CP.none_generator, None,
                    active_configurations=active,
                    baseline=dict(formula='2nd', delta_over_h='1', kappa_multiplier='1'))
        first = next(session)
        self.assertFalse(any(p['origin']['kind'] == 'TRANSFER_FROM_NEAREST_KAPPA' for p in first['eligible']))
        neighbour = [n for n in CP.CAPS for axis in O.AXES];neighbour[0] -= 1;neighbour[1] -= 1;neighbour = tuple(neighbour)
        knowledge[('B', neighbour)] = dict(status=O.PASS, MSE_lower=Fraction(1, 2), MSE_upper=Fraction(1, 2),
                                          source='ARTIFICIAL_VISIBLE_COMPLETION', calls=0)
        active.add('B');second = session.send({'refresh': True})
        transfers = [p for p in second['eligible'] if p['configuration_id'] == 'A'
                     and p['origin']['kind'] == 'TRANSFER_FROM_NEAREST_KAPPA']
        self.assertTrue(any(p['vector'] == neighbour for p in transfers))
        self.assertEqual(second['step'], 0);session.close()

    def test_small_and_deferred_stop_remains_a_stop(self):
        cands, function, knowledge = fixtures()
        session = P.priority_session('PR_PAULI', cands, knowledge, O.FixtureBackend(function, CP.TAU),
                    dict(CP.SEARCH, max_steps_per_method=10, small_gain_fraction='1', small_and_deferred='STOP'),
                    CP.TAU, CP.none_generator, None)
        with self.assertRaises(StopIteration) as ended:
            next(session)
        self.assertEqual(ended.exception.value['stop_reason'], 'ONLY_SMALL_OR_DEFERRED_LEFT')
        self.assertEqual(ended.exception.value['new_calls'], 0)

    def test_b_selects_only_exposed_eligible_proposal(self):
        cands, function, knowledge = fixtures()
        session = P.priority_session('PR_PAULI', cands, knowledge, O.FixtureBackend(function, CP.TAU),
                    dict(CP.SEARCH, max_steps_per_method=1, small_and_deferred='STOP'),
                    CP.TAU, CP.none_generator, None)
        frame = next(session)
        self.assertFalse(any(p['cls'] in ('SMALL', 'DEFERRED') for p in frame['eligible']))
        choice = frame['eligible'][-1]
        with self.assertRaises(StopIteration) as ended:
            session.send(choice)
        self.assertEqual(tuple(ended.exception.value['trace'][0]['vector']), choice['vector'])
        session = P.priority_session('PR_PAULI', cands, knowledge, O.FixtureBackend(function, CP.TAU),
                    dict(CP.SEARCH, max_steps_per_method=1), CP.TAU, CP.none_generator, None)
        frame = next(session);wrong = dict(frame['default_choice'], vector=(1,) * 26)
        with self.assertRaisesRegex(ValueError, 'eligible pool'):
            session.send(wrong)


if __name__ == '__main__':
    unittest.main(verbosity=2)
