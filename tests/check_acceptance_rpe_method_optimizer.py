"""The acceptance predicates detect the failures they are meant to detect (synthetic results only)."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
import acceptance_rpe_method_optimizer as AC  # noqa: E402


def result(trace, stop='NO_IMPROVING_PROPOSAL', pending=None):
    return dict(trace=trace, stop_reason=stop, pending_request=pending)


def step(vector, cost, improved):
    return dict(vector=vector, cost=cost, improved=improved)


class Log:
    def __init__(self, vectors):
        self.queries = [dict(vector=v) for v in vectors]


V1, V2, V3 = [1] * 26, [2] * 26, [3] * 26
PENDING = dict(configuration_id='C', vector=tuple(V3), cost=5, route=dict(caller='X'), origin={})


class Predicates(unittest.TestCase):
    def test_reveal_after_selection(self):
        r = result([step(V1, 10, True)], 'NEW_Q3_REQUIRED_OUTSIDE_APPROVED_BUDGET', PENDING)
        self.assertEqual(AC.reveal_after_selection(r, Log([V1, V3]))['status'], 'PASS')
        self.assertEqual(AC.reveal_after_selection(r, Log([V2, V1, V3]))['status'], 'FAIL')   # an unselected peek

    def test_no_user_choice(self):
        self.assertEqual(AC.no_user_choice(result([], 'NEW_Q3_REQUIRED_OUTSIDE_APPROVED_BUDGET', PENDING))['status'], 'PASS')
        self.assertEqual(AC.no_user_choice(result([], 'ASK_USER_WHICH_ROUND', PENDING))['status'], 'FAIL')
        bad = dict(PENDING, route=None)
        self.assertEqual(AC.no_user_choice(result([], 'Q3_BUDGET_EXHAUSTED', bad))['status'], 'FAIL')

    def test_continues_after_improvement(self):
        self.assertEqual(AC.continues_after_improvement(result([step(V1, 10, True), step(V2, 8, False)]))['status'], 'PASS')
        stuck = result([step(V1, 10, True)], 'Q3_BUDGET_EXHAUSTED', None)
        self.assertEqual(AC.continues_after_improvement(stuck)['status'], 'FAIL')
        self.assertEqual(AC.continues_after_improvement(result([step(V1, 10, False)]))['status'], 'FAIL')

    def test_search_range(self):
        ctx = dict(candidates=[dict(method='M', name='in', key=dict(formula='2nd', delta_over_h='16/7')),
                               dict(method='M', name='out', key=dict(formula='4th', delta_over_h=None))])
        self.assertEqual(AC.candidates_in_range(ctx, 'M', AC.P7_S2)['outside_candidates'], ['out'])
        ctx['candidates'].pop()
        self.assertEqual(AC.candidates_in_range(ctx, 'M', AC.P7_S2)['status'], 'PASS')


if __name__ == '__main__':
    unittest.main(verbosity=2)
