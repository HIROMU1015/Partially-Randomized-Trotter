"""The Skill-route acceptance predicates detect the failures they are meant to detect (synthetic outputs only)."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts')]
import acceptance_rpe_cost_compare_skill as SK  # noqa: E402

IN = 'TRUNCATED_PF:formula=2nd,delta_over_h=16/7:a6e55777'
OUT = 'TRUNCATED_PF:formula=4th,delta_over_h=32/5:dd8bfc24'
V = [4] * 26


def output(trace=(), evaluated=(), pending=None, stop='NEW_Q3_REQUIRED_OUTSIDE_APPROVED_BUDGET', needs=(IN,), best=None):
    method = dict(trace=list(trace), evaluated=list(evaluated), pending_request=pending, stop_reason=stop, best=best,
                  search_range=dict(candidates=list(needs)))
    return dict(method=method, needs=dict(candidates=[dict(name=n, method=SK.METHOD) for n in needs]))


def step(vector, cost, before, cid='A'):
    return dict(candidate=IN, configuration_id=cid, vector=vector, cost=cost, incumbent_before=before, improved=False,
                answer_source='SAVED', step=1)


def saved(label, vector, cost, cid='A', status='MODEL_BOUND_MEETS_ADOPTED_THRESHOLD'):
    return dict(candidate=IN, configuration_id=cid, vector=vector, label=label, cost=cost, status=status, source='SAVED')


class Predicates(unittest.TestCase):
    def test_range(self):
        self.assertEqual(SK.range_check(output(), SK.P7_S2)['status'], 'PASS')
        self.assertEqual(SK.range_check(output(needs=(IN, OUT)), SK.P7_S2)['status'], 'FAIL')
        pend = dict(candidate=OUT, queue=[], vector=V, cost=1, configuration_id='B', route={})
        self.assertEqual(SK.range_check(output(pending=pend), SK.P7_S2)['outside'], [OUT])

    def test_start_and_reveal(self):
        o = output(trace=[step([3] * 26, 9, 10)], evaluated=[saved('SEED', V, 10), saved('OTHER', [3] * 26, 9)])
        self.assertEqual(SK.start_check(o, ['SEED'])['status'], 'PASS')
        self.assertEqual(SK.reveal_check(o, ['SEED'])['status'], 'PASS')
        peek = output(trace=[step([3] * 26, 9, 10)], evaluated=[saved('SEED', V, 10), saved('PEEK', [2] * 26, 8)])
        self.assertEqual(SK.reveal_check(peek, ['SEED'])['status'], 'FAIL')        # known but never selected
        wrong = output(trace=[step([3] * 26, 9, 12)], evaluated=[saved('SEED', V, 10)])
        self.assertEqual(SK.start_check(wrong, ['SEED'])['status'], 'FAIL')        # started from another point

    def test_next_action(self):
        pend = dict(candidate=IN, configuration_id='A', vector=V, cost=5, route=dict(kind='pf_b_pf6'))
        self.assertEqual(SK.next_action_check(output(pending=pend))['status'], 'PASS')
        self.assertEqual(SK.next_action_check(output(pending=dict(pend, route=None)))['status'], 'FAIL')
        self.assertEqual(SK.next_action_check(output(pending=pend, stop='ASK_USER'))['status'], 'FAIL')
        self.assertEqual(SK.next_action_check(output(pending=dict(pend, vector=V[:13])))['status'], 'FAIL')


if __name__ == '__main__':
    unittest.main(verbosity=2)
