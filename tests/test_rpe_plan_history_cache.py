"""Small synthetic saved-evidence delta checks; no scientific adapter runs."""
from fractions import Fraction
import json
from pathlib import Path
import sys
import tempfile
import unittest
import pickle
import zlib
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks import rpe_method_optimizer as O
from trottertracks.rpe_plan_history_cache import HistoryCache, _unpack


class SavedHistoryDeltas(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.prior = self.root / 'prior'
        self.problem = dict(accuracy=dict(epsilon_E_decimal='1'), target=dict(energy_ha=0), rpe=dict(M=0))
        self.rule = dict(kind='artificial', interpreter=sys.executable, view_commit='SYNTHETIC',
                         configuration_ids='*')
        self.expected = {'cfg': dict(total_configuration_id='total', binding_id='binding')}
        self.executor = O.Executor(self.root, self.root / 'current', {'FULL_PF': [self.rule]},
                                   self.expected, self.problem, prior_run_dirs=[self.prior], launch_disabled=True)
        self.executor = self.executor.for_method('FULL_PF')
        self.candidates = [dict(configuration_id='cfg', ready=True, prices=[Fraction(1), Fraction(1)])]
        self.cachefile = self.root / 'cache/history.sqlite3'
        self.cache = HistoryCache(self.cachefile)
        self.addCleanup(lambda: self.cache.close())

    def evidence(self, n, status=O.PASS, outcome=True):
        proposal = dict(configuration_id='cfg', vector=(n, n), cost=Fraction(2*n), origin=dict(kind='SYNTHETIC'))
        job = self.prior / 'jobs/FULL_PF' / self.executor.job_dir(proposal).name
        job.mkdir(parents=True)
        request = self.executor.request(proposal, job, self.rule)
        (job / 'REQUEST.json').write_text(json.dumps(request))
        (job / 'LAUNCH.json').write_text(json.dumps(dict(pid=99999999, started=0)))
        if outcome:
            self.write_outcome(job, n, status)
        return job

    def write_outcome(self, job, n, status):
        bound = Fraction(10**6000-1, 10**6001+7) if status == O.PASS else Fraction(2)
        result = dict(status=status, MSE_lower=O.enc(bound), MSE_upper=O.enc(bound),
                      artificial=True, identity=dict(configuration_id='cfg', total_configuration_id='total',
                      binding_id='binding', allocation=[dict(m=0, N_X=n, N_Y=n)], contract_matches=True))
        (job / 'OUTCOME.json').write_text(json.dumps(result))
        (job / 'PROCESS_RETURN.json').write_text(json.dumps(dict(returncode=0, signal=None)))

    def test_persistent_warm_results_exact_and_current_prices(self):
        self.evidence(4)
        self.evidence(5, O.FAIL)
        baseline = self.executor.prior_records(self.candidates)
        first = self.cache.prior_records(self.executor, self.candidates)
        self.assertEqual(first[:2], baseline)
        self.assertEqual(first[2]['validated_jobs'], 2)
        self.cache.close()
        self.cache = HistoryCache(self.cachefile)
        with patch.object(self.executor, 'ingest', wraps=self.executor.ingest) as ingest:
            warm = self.cache.prior_records(self.executor, self.candidates)
        self.assertEqual(warm[:2], baseline)
        self.assertEqual(warm[2]['parsed_jobs'], 0)
        self.assertEqual(warm[2]['cache_hits'], 2)
        self.assertEqual(ingest.call_count, 0)
        self.candidates[0]['prices'] = [Fraction(2), Fraction(3)]
        repriced = self.cache.prior_records(self.executor, self.candidates)
        self.assertEqual(repriced[0][('cfg', (4, 4))]['cost'], 20)
        self.assertEqual(repriced[2]['validated_jobs'], 0)

    def test_only_new_changed_and_identity_invalidated_evidence_is_revalidated(self):
        job = self.evidence(4)
        self.cache.prior_records(self.executor, self.candidates)
        self.evidence(5, O.FAIL)
        added = self.cache.prior_records(self.executor, self.candidates)
        self.assertEqual((added[2]['cache_hits'], added[2]['parsed_jobs'], added[2]['validated_jobs']), (1, 1, 1))
        self.write_outcome(job, 4, O.FAIL)
        changed = self.cache.prior_records(self.executor, self.candidates)
        self.assertEqual(changed[0][('cfg', (4, 4))]['status'], O.FAIL)
        self.assertEqual(changed[2]['validated_jobs'], 1)
        data = json.loads((job / 'OUTCOME.json').read_text())
        data['identity']['binding_id'] = 'wrong'
        (job / 'OUTCOME.json').write_text(json.dumps(data))
        invalid = self.cache.prior_records(self.executor, self.candidates)
        self.assertNotIn(('cfg', (4, 4)), invalid[0])
        self.assertIn('IDENTITY_MISMATCH', invalid[1][0]['reason'])

    def test_unknown_markers_refresh_and_late_valid_completion_is_ingested(self):
        job = self.evidence(4, outcome=False)
        old = self.cache.prior_records(self.executor, self.candidates)
        self.assertEqual(old[0], {})
        self.assertEqual(old[2]['current_unresolved'][0]['state'], 'LAUNCHED_STATE_UNKNOWN')
        (job / 'LAUNCH.json').write_text(json.dumps(dict(pid=99999998, started=1)))
        current = self.cache.prior_records(self.executor, self.candidates)
        self.assertEqual(current[2]['current_unresolved'][0]['pid'], 99999998)
        self.write_outcome(job, 4, O.PASS)
        finished = self.cache.prior_records(self.executor, self.candidates)
        self.assertEqual(len(finished[0]), 1)
        self.assertEqual(finished[2]['validated_jobs'], 1)
        self.assertEqual(finished[2]['current_unresolved'], [])
        # An optional integration hook shares the same local cache per method.
        self.executor.preparation_history_cache = self.cache
        known, refused = self.executor.prior_records(self.candidates)
        self.assertEqual(known, finished[0])
        self.assertEqual(refused, [])
        self.assertEqual(self.cache.stats_by_method['FULL_PF']['cache_hits'], 1)

    def test_route_change_invalidates_warm_record_and_unavailable_unknown_still_refreshes(self):
        job = self.evidence(4)
        self.cache.prior_records(self.executor, self.candidates)
        self.rule['view_commit'] = 'CHANGED_SYNTHETIC_ROUTE'
        changed = self.cache.prior_records(self.executor, self.candidates)
        self.assertEqual(changed[0], {})
        self.assertEqual(changed[1][0]['reason'], 'DIFFERENT_JOB_KEY_OR_ROUTE')
        (job / 'OUTCOME.json').write_text(json.dumps(dict(status='UNKNOWN', reason='SYNTHETIC_NO_FINAL_EVIDENCE')))
        self.candidates[0]['ready'] = False
        first = self.cache.prior_records(self.executor, self.candidates)
        self.assertEqual(first[2]['current_unresolved'][0]['state'], 'UNKNOWN_OUTCOME')
        (job / 'LAUNCH.json').write_text(json.dumps(dict(pid=99999996, started=2)))
        refreshed = self.cache.prior_records(self.executor, self.candidates)
        self.assertEqual(refreshed[2]['current_unresolved'][0]['pid'], 99999996)

    def test_old_local_fraction_payload_decodes_exactly_without_repeated_gcd(self):
        original = [Fraction(10**6000-1, 10**6001+7), Fraction(-7, 13), Fraction(0)]
        old_payload = zlib.compress(pickle.dumps(dict(bounds=original), protocol=5), 1)
        with patch('math.gcd', side_effect=AssertionError('cached Fraction was already normalized')):
            restored = _unpack(old_payload)
        self.assertEqual(restored['bounds'], original)
        self.assertTrue(all(type(x) is Fraction for x in restored['bounds']))
        self.assertTrue(all(x.denominator > 0 for x in restored['bounds']))


if __name__ == '__main__':
    unittest.main()
