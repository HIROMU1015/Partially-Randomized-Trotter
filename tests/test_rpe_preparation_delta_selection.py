"""Artificial saved-evidence addition through HistoryCache -> normal optimizer selection.

Every REQUEST/OUTCOME below is a labelled artificial fixture in a temporary
directory. No scientific leaf, process launch, collect or production run is used.
"""
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks import rpe_method_optimizer as O
from trottertracks.rpe_plan_history_cache import HistoryCache
from test_rpe_optimizer_concurrent import fixture


class PreparationDeltaSelectionTests(unittest.TestCase):
    def test_added_saved_pass_changes_next_pending_through_optimizer_with_zero_new_calls(self):
        method = 'FULL_PF'
        problem, context, routes = fixture(methods=[method], configurations=1)
        problem['search'].update(rule='METHOD_MINIMUM_PRIORITY_V2',
                                 kappa_neighbourhood={'outer_planner': 'ARTIFICIAL_METADATA_ONLY'})
        context['saved'].table.clear()  # The starting incumbent also comes through identity-checked prior evidence.
        candidate = context['candidates'][0]
        cid = candidate['configuration_id']
        expected = {cid: dict(total_configuration_id='ARTIFICIAL_TOTAL', binding_id='ARTIFICIAL_BINDING')}
        routes[method][0]['view_commit'] = 'ARTIFICIAL_SAVED_SOURCE'
        with tempfile.TemporaryDirectory(prefix='rpe-artificial-preparation-delta-') as temporary:
            root = Path(temporary);prior = root / 'artificial-prior';run = root / 'artificial-plan'
            run.mkdir()
            executor = O.Executor(root, run, routes, expected, problem, optimizer_source='ARTIFICIAL_ONLY',
                                  prior_run_dirs=[prior], launch_disabled=True)
            reader = executor.for_method(method)

            def save_fixture(proposal, bound):
                job = prior / 'jobs' / method / reader.job_dir(proposal).name
                job.mkdir(parents=True)
                nx, ny = O.split(proposal['vector'])
                request = reader.request(proposal, job, routes[method][0])
                outcome = dict(artificial=True, fixture='ARTIFICIAL_PREPARATION_DELTA', status=O.PASS,
                    MSE_lower=O.enc(bound), MSE_upper=O.enc(bound),
                    identity=dict(configuration_id=cid, total_configuration_id='ARTIFICIAL_TOTAL',
                        binding_id='ARTIFICIAL_BINDING', contract_matches=True,
                        allocation=[dict(m=m, N_X=x, N_Y=y) for m, (x, y) in enumerate(zip(nx, ny))]))
                for name, value in [('REQUEST.json', request), ('OUTCOME.json', outcome),
                    ('LAUNCH.json', dict(artificial=True, fixture_only=True, pid=99999999, started=0)),
                    ('PROCESS_RETURN.json', dict(artificial=True, returncode=0, signal=None))]:
                    (job / name).write_text(json.dumps(value))
                return job

            seed = dict(configuration_id=cid, vector=tuple([2] * 26),
                        cost=O.cost([2] * 26, candidate['prices']), origin={'kind': 'ARTIFICIAL_SAVED_SEED'})
            save_fixture(seed, Fraction(13, 2))
            with HistoryCache(root / 'artificial-cache.sqlite3') as cache:
                executor.preparation_history_cache = cache
                def plan():
                    return O.optimize(root, problem, context=context, methods=[method], initial='all',
                                      max_new={method: 0}, executor=executor, science_approved=False)[method]
                with patch.object(O.Executor, 'launch', side_effect=AssertionError('new execution is forbidden')), \
                     patch.object(O.Executor, 'run_adapter', side_effect=AssertionError('execute/collect is forbidden')):
                    before = plan()
                    first = deepcopy(before['pending_request'])
                    self.assertIsNotNone(first)
                    self.assertEqual(before['prior_rounds']['loaded'], 1)
                    self.assertEqual(cache.stats_by_method[method]['validated_jobs'], 1)
                    self.assertEqual(cache.stats_by_method[method]['scientific_calls'], 0)
                    added = save_fixture(dict(first, origin={'kind': 'ARTIFICIAL_SAVED_PENDING_PASS'}), Fraction(3))
                    after = plan()  # Same cache: its marker scan discovers only the newly saved job.

            self.assertNotEqual(tuple(after['pending_request']['vector']), tuple(first['vector']))
            self.assertEqual(after['best']['N_X'], O.split(first['vector'])[0])
            self.assertEqual(after['best']['N_Y'], O.split(first['vector'])[1])
            self.assertEqual(after['best']['cost'], first['cost'])
            self.assertLess(after['best']['cost'], before['best']['cost'])
            self.assertEqual(after['prior_rounds']['loaded'], 2)
            self.assertEqual((cache.stats_by_method[method]['cache_hits'], cache.stats_by_method[method]['parsed_jobs'],
                              cache.stats_by_method[method]['validated_jobs']), (1, 1, 1))
            self.assertEqual(cache.stats_by_method[method]['collection_calls'], 0)
            self.assertEqual(cache.stats_by_method[method]['scientific_calls'], 0)
            self.assertEqual(after['evaluated_count'], before['evaluated_count'] + 1)
            for result in (before, after):
                self.assertEqual(result['new_calls'], 0)
                self.assertFalse(result['budget']['science_approved'])
                self.assertEqual(result['budget']['max_new_q3'], 0)
                self.assertEqual(result['stop_reason'], 'NEW_SCIENCE_NOT_APPROVED')
            self.assertEqual(executor.launched(method), 0)
            self.assertFalse((run / 'jobs').exists())
            saved = json.loads((added / 'OUTCOME.json').read_text())
            self.assertTrue(saved['artificial'])


if __name__ == '__main__':
    unittest.main()
