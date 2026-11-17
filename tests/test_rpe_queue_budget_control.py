"""Queue budget responses are control flow, never numerical Q3 outcomes.

Only existing artificial workers/providers are used, including the normal width
CLI and same-directory resume. No H4 signal, circuit or scientific Q3 is called.
"""
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts'), str(ROOT / 'tests')]
from trottertracks import rpe_method_optimizer as O
from trottertracks import rpe_optimizer_concurrent as C
import test_rpe_optimizer_concurrent as F
import test_pr_width_overlap as W


def reserve_other_jobs(count, prices=None):
    """Reproduce a finite batch filled by independent speculative jobs."""
    original = C.ConcurrentExecutor.prefetch
    def prefetch(view, proposals, scope='CONFIGURATION'):
        seen = getattr(view.state, 'artificial_budget_reservations', set())
        view.state.artificial_budget_reservations = seen
        if proposals and view.method not in seen:
            seen.add(view.method)
            extras = [dict(proposals[0], vector=tuple([3 + i] * 26)) for i in range(count)]
            # Reservation happens after the selector's spent/free snapshot, as
            # with production lookahead. These are actual artificial children.
            for extra in extras:
                own = (prices or {}).get(extra['configuration_id'])
                extra['cost'] = O.cost(extra['vector'], own) if own else (
                    proposals[0]['cost'] * Fraction(sum(extra['vector']), sum(proposals[0]['vector'])))
                original(view, [extra], scope=scope)
            # Both fake hosts can reach their reservation before selection
            # resumes; make cancellation/dispatch timing irrelevant to counts.
            deadline = time.monotonic() + 8
            while any(view.state.tasks[view.job_dir(p)].status == 'QUEUED' for p in extras):
                if time.monotonic() > deadline:
                    raise AssertionError('Artificial reserved workers never started')
                time.sleep(.005)
        return original(view, proposals, scope=scope)
    return prefetch


class BudgetControlTests(unittest.TestCase):
    def test_refused_reservation_has_zero_calls_and_no_adoption(self):
        problem, context, routes = F.fixture(['PR_DF'], 1)
        with tempfile.TemporaryDirectory() as tmp:
            executor = C.ConcurrentExecutor(ROOT, tmp, routes, {}, problem,
                                            budgets={'PR_DF': 0}, max_concurrent=1)
            try:
                proposal = dict(configuration_id=context['candidates'][0]['configuration_id'],
                                vector=tuple([1] * 26), cost=Fraction(1), origin={})
                answer = executor.for_method('PR_DF').answer(proposal)
                self.assertEqual(answer['status'], 'BUDGET_EXHAUSTED')
                self.assertEqual(answer['calls'], 0)
                self.assertEqual(answer['scientific_evaluation_calls'], 0)
                self.assertEqual(executor.metrics()['adopted'], 0)
                self.assertEqual(executor.launched('PR_DF'), 0)
            finally:
                executor.finish()

    def test_v2_budget_control_can_continue_next_batch_with_same_queue(self):
        problem, context, routes = F.fixture(['PR_DF'], 1, lookahead=3, steps=3)
        problem['search'].update(rule='METHOD_MINIMUM_PRIORITY_V2', kappa_neighbourhood={'outer': 'ARTIFICIAL'})
        budgets = {'PR_DF': 1}
        with tempfile.TemporaryDirectory() as tmp:
            executor = C.ConcurrentExecutor(ROOT, tmp, routes, {}, problem, budgets=budgets, max_concurrent=2)
            try:
                with patch.object(C.ConcurrentExecutor, 'prefetch', reserve_other_jobs(1, {c['configuration_id']:c['prices'] for c in context['candidates']})):
                    first = C.optimize(ROOT, problem, context=deepcopy(context), executor=executor,
                                       methods=['PR_DF'], max_new=budgets, close_executor=False)['PR_DF']
                    self.assertEqual(first['stop_reason'], 'Q3_BUDGET_EXHAUSTED')
                    self.assertEqual(first['trace'], [])
                    self.assertEqual(first['held'], [])
                    self.assertEqual(first['adopted_new_calls'], 0)  # selected answer not evaluated
                    self.assertEqual(first['new_calls'], 1)  # speculative launch is charged
                    self.assertEqual(executor.launched('PR_DF'), 1)  # speculation still consumes the cap
                    budgets['PR_DF'] = 2
                    second = C.optimize(ROOT, problem, context=deepcopy(context), executor=executor,
                                        methods=['PR_DF'], max_new=budgets, close_executor=False)['PR_DF']
                self.assertEqual(executor.launched('PR_DF'), 2)
                self.assertEqual(second['adopted_new_calls'], 1)
                self.assertEqual(second['new_calls'], 2)  # same queue's cumulative launches
                self.assertTrue(all(row['status'] != 'BUDGET_EXHAUSTED' for row in second['trace']))
                self.assertTrue(all(row['calls'] in (0, 1) for row in second['trace']))
            finally:
                executor.finish()

    def test_legacy_selector_does_not_absorb_budget_control(self):
        problem, context, _ = F.fixture(['PR_DF'], 1)
        class BudgetBackend:
            def answer(self, proposal):
                return dict(status='BUDGET_EXHAUSTED', calls=0, source='EXECUTOR_BUDGET',
                            scientific_evaluation_calls=0, job=None)
        known = O.initial_knowledge(context['saved'], context['candidates'])
        before = deepcopy(known)
        result = O.run_method('PR_DF', context['candidates'], known, BudgetBackend(), problem['search'],
                              context['tau'], context['generate'], 1)
        self.assertEqual(result['stop_reason'], 'Q3_BUDGET_EXHAUSTED')
        self.assertEqual(result['new_calls'], 0)
        self.assertEqual(result['trace'], [])
        self.assertEqual(known, before)

    def test_normal_width_entry_and_same_run_resume_do_not_relaunch(self):
        with tempfile.TemporaryDirectory(prefix='queue-budget-width-') as tmp:
            args, Driver, _ = W.setup(tmp, True)
            specfile = Path(tmp) / 'ROUND.json'
            spec = json.loads(specfile.read_text())
            spec['search_overrides'].update(lookahead=3, rule='METHOD_MINIMUM_PRIORITY_V2',
                                            kappa_neighbourhood={'outer': 'ARTIFICIAL'})
            specfile.write_text(json.dumps(spec))
            with patch.object(C.ConcurrentExecutor, 'prefetch', reserve_other_jobs(2)):
                driver = W.call(args, Driver)
            self.assertNotEqual(driver.state.get('stop', {}).get('reason'), 'RUNTIME_EXCEPTION')
            markers = {str(p): p.read_bytes() for p in driver.run.glob('jobs/*/*/LAUNCH.json')}
            self.assertEqual(len(markers), 4)
            result = json.loads((driver.run / 'ROUND_RESULT.json').read_text())
            for row in result['methods'].values():
                self.assertNotIn(row['stop_reason'], ('UNKNOWN_RESULT', 'RUNTIME_EXCEPTION'))
                self.assertTrue(all(t['status'] != 'BUDGET_EXHAUSTED' for t in row['trace']))
                self.assertEqual(row['executor_metrics']['new_real_route_launches'], 0)
            events = W.records(driver.run / 'PROGRESS.jsonl')
            self.assertTrue(any(e['event'] == 'BUDGET_WAIT' for e in events))
            W.call(args, Driver)
            self.assertEqual(markers, {str(p): p.read_bytes() for p in driver.run.glob('jobs/*/*/LAUNCH.json')})
            evidence = ROOT / 'docs/research/pr_r08_budget_fix_20261118/checks/NORMAL_ENTRY.json'
            evidence.write_text(json.dumps(dict(artificial=True, entry='scripts/run_rpe_schedule_search.py:main',
                queue='concurrent', methods=list(result['methods']), artificial_jobs=4,
                stop={m:r['stop_reason'] for m,r in result['methods'].items()},
                new_scientific_calls=0, resume_additional_launches=0), indent=2) + '\n')


if __name__ == '__main__':
    unittest.main()
