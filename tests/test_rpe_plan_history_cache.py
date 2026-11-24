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
from trottertracks.rpe_optimizer_concurrent import ConcurrentExecutor, DistributedExecutor
from trottertracks.rpe_plan_history_cache import HistoryCache, HistoryInventory, _unpack, _validator


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
        job = self.prior / 'jobs' / self.executor.method / self.executor.job_dir(proposal).name
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

    def test_outcome_reads_signed_exact_json_integers(self):
        job = self.root / 'large'
        job.mkdir()
        (job / 'OUTCOME.json').write_text('{"lower": -' + '9' * 20000 + ', "upper": ' + '9' * 20001 + '}')
        outcome, _ = O.job_outcome(job)
        self.assertEqual(outcome, dict(lower=1 - 10**20000, upper=10**20001 - 1))

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

    def test_stop_commits_verified_prefix_and_resume_checks_the_rest(self):
        self.evidence(4)
        self.evidence(5)
        with patch.object(self.executor, '_user_stop_requested', create=True, side_effect=[False, True]):
            known, _, part = self.cache.prior_records(self.executor, self.candidates, details=False)
        self.assertTrue(part['interrupted'])
        self.assertEqual(len(known), 1)
        self.assertEqual(part['validated_jobs'], 1)
        with patch.object(self.executor, '_user_stop_requested', create=True, return_value=False):
            known, _, resumed = self.cache.prior_records(self.executor, self.candidates, details=False)
        self.assertFalse(resumed['interrupted'])
        self.assertEqual(len(known), 2)
        self.assertEqual((resumed['cache_hits'], resumed['validated_jobs']), (1,1))

    def test_projected_route_cache_tracks_only_its_actual_source_binding(self):
        self.rule['horizon_projection'] = dict(source_configuration_id='source', source_M=1, M=0)
        self.rule['configurations'] = {'source': {'signal_packet': 'fixed'}, 'other': {'signal_packet': 'unrelated'}}
        self.evidence(4)
        first = self.cache.prior_records(self.executor, self.candidates)
        self.assertEqual(first[2]['validated_jobs'], 1)
        self.rule['configurations']['other']['signal_packet'] = 'replaced unrelated'
        self.assertEqual(self.cache.prior_records(self.executor, self.candidates)[2]['cache_hits'], 1)
        self.rule['configurations']['source']['signal_packet'] = 'replaced actual input'
        known, refused, stats = self.cache.prior_records(self.executor, self.candidates)
        self.assertFalse(known)
        self.assertEqual(refused[0]['reason'], 'DIFFERENT_JOB_KEY_OR_ROUTE')
        self.assertEqual(stats['context_changes'], {'entry': 1})

    def test_validator_relocation_preserves_hits_and_changed_bytes_revalidate(self):
        self.evidence(4)
        selected = 'src/trottertracks/rpe_method_optimizer.py'
        with patch('trottertracks.rpe_plan_history_cache._validator', return_value={'/first/'+selected: 'a'*64}):
            cold = self.cache.prior_records(self.executor, self.candidates)
        with patch('trottertracks.rpe_plan_history_cache._validator', return_value={'/second/'+selected: 'a'*64}):
            relocated = self.cache.prior_records(self.executor, self.candidates)
        self.assertEqual(relocated[:2], cold[:2])
        self.assertEqual((relocated[2]['cache_hits'], relocated[2]['validated_jobs']), (1, 0))
        with patch('trottertracks.rpe_plan_history_cache._validator', return_value={'/second/'+selected: 'b'*64}):
            changed = self.cache.prior_records(self.executor, self.candidates)
        self.assertEqual(changed[:2], cold[:2])
        self.assertEqual((changed[2]['cache_hits'], changed[2]['validated_jobs']), (0, 1))
        self.assertEqual(changed[2]['context_changes'], {'validator': 1})

    def test_pure_dispatch_source_changes_keep_the_validated_record(self):
        concurrent = ConcurrentExecutor.__new__(ConcurrentExecutor)
        distributed = DistributedExecutor.__new__(DistributedExecutor)
        transport = ROOT/'src'/'trottertracks'/'rpe_optimizer_concurrent.py'
        self.assertIn(str(transport), _validator(concurrent))
        self.assertEqual(_validator(distributed), _validator(concurrent))
        self.evidence(4)
        self.cache.prior_records(self.executor, self.candidates, details=False)

        relay = ROOT/'scripts'/'run_rpe_q3_relay.py'
        original_read = Path.read_bytes

        def read_changed(path):
            data = original_read(path)
            return data + b'\n# dispatch edit\n' if path == relay else data

        with patch.object(Path, 'read_bytes', read_changed):
            _, _, stats = self.cache.prior_records(self.executor, self.candidates, details=False)
        self.assertEqual((stats['cache_hits'], stats['validated_jobs']), (1, 0))

    def test_new_validation_hook_or_wrapper_restores_full_dispatch_hash(self):
        distributed = DistributedExecutor.__new__(DistributedExecutor)

        def changed(self, *args, **kwargs):
            raise AssertionError('Only the source should be inspected')

        hash_path = str(ROOT/'src'/'trottertracks'/'rpe_optimizer_concurrent.py')
        with patch.object(ConcurrentExecutor, 'same_route', changed):
            self.assertIn(hash_path, _validator(distributed))
        with patch.object(DistributedExecutor, 'ingest', changed):
            self.assertIn(hash_path, _validator(distributed))
        with patch.object(DistributedExecutor, 'scope_unknown', changed):
            self.assertIn(hash_path, _validator(distributed))

    def test_late_ready_subset_skips_unrelated_exact_payloads_until_request_changes(self):
        job = self.evidence(4)
        self.cache.prior_records(self.executor, self.candidates, details=False)
        late = [dict(self.candidates[0], configuration_id='late')]
        with patch('trottertracks.rpe_plan_history_cache._unpack',
                   side_effect=AssertionError('unrelated exact record was decoded')):
            known, refused, stats = self.cache.prior_records(self.executor, late, details=False)
        self.assertFalse(known)
        self.assertEqual(stats['unrelated_request_hits'], 1)
        self.assertEqual(refused[0]['reason'], 'CONFIGURATION_NOT_IN_THIS_ROUND')
        path = job/'REQUEST.json'
        request = json.loads(path.read_bytes())
        request['configuration_id'] = 'late'
        path.write_text(json.dumps(request))
        known, refused, stats = self.cache.prior_records(self.executor, late, details=False)
        self.assertFalse(known)
        self.assertEqual(stats['unrelated_request_hits'], 0)
        self.assertEqual(refused[0]['reason'], 'DIFFERENT_JOB_KEY_OR_ROUTE')

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

    def test_index_uses_candidate_metadata_and_warm_exact_values(self):
        job = self.evidence(4)
        unrelated = self.evidence(5)
        request_path = unrelated/'REQUEST.json'
        request = json.loads(request_path.read_bytes())
        request_path.write_text(json.dumps(dict(request, configuration_id='other')))
        self.executor.prior_history_inventory = HistoryInventory()
        initial, _, stats = self.cache.prior_records(self.executor, self.candidates, details=False)
        self.assertEqual((stats['scanned_jobs'], stats['indexed_out_of_scope']), (1, 1))
        self.assertEqual(len(initial), 1)
        self.cache.close()
        self.cache = HistoryCache(self.cachefile)
        self.executor.prior_history_inventory = HistoryInventory()
        with patch('trottertracks.rpe_plan_history_cache.os.scandir',
                   side_effect=AssertionError('unchanged indexed directory was walked')):
            known, _, _ = self.cache.prior_records(self.executor, self.candidates, details=False)
            self.assertEqual(known, initial)
            with patch('trottertracks.rpe_plan_history_cache.zlib.decompress',
                       side_effect=AssertionError('verified value was decoded again')):
                self.candidates[0]['prices'] = [Fraction(2), Fraction(3)]
                known, _, stats = self.cache.prior_records(self.executor, self.candidates, details=False)
                self.assertEqual(known[('cfg', (4, 4))]['cost'], 20)
                self.assertEqual(stats['memory_hits'], 1)
        self.write_outcome(job, 4, O.FAIL)
        known, _, stats = self.cache.prior_records(self.executor, self.candidates, details=False)
        self.assertEqual(known[('cfg', (4, 4))]['status'], O.FAIL)
        self.assertEqual(stats['validated_jobs'], 1)

    def test_index_finds_new_job_and_existing_unresolved_job_keeps_ownership(self):
        executor = ConcurrentExecutor.__new__(ConcurrentExecutor)
        executor.__dict__.update(self.executor.__dict__)
        executor.prior_history_inventory = HistoryInventory()
        pending = self.evidence(4, outcome=False)
        path = pending/'REQUEST.json'
        request = path.read_bytes()
        path.unlink()
        self.assertFalse(self.cache.prior_records(executor, self.candidates, details=False)[0])
        proposal = dict(configuration_id='cfg', vector=(4, 4))
        self.assertEqual(executor.prior_job(proposal), pending)
        path.write_bytes(request)
        self.write_outcome(pending, 4, O.PASS)
        known, _, _ = self.cache.prior_records(executor, self.candidates, details=False)
        self.assertIn(('cfg', (4, 4)), known)
        new_job = self.evidence(5)
        self.assertEqual(executor.prior_job(dict(configuration_id='cfg', vector=(5, 5))), new_job)
        known, _, _ = self.cache.prior_records(executor, self.candidates, details=False)
        self.assertEqual(set(known), {('cfg', (4, 4)), ('cfg', (5, 5))})
        executor.prior_q3_holds = [dict(parent_directory=str(self.prior), method='FULL_PF', job_id=new_job.name)]
        known, refused, _ = self.cache.prior_records(executor, self.candidates, details=False)
        self.assertEqual(set(known), {('cfg', (4, 4))})
        self.assertEqual(refused[0]['reason'], 'PRIOR_Q3_PERSISTENCE_HOLD')

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

    def test_normal_factory_reuses_prior_results_and_honors_holds_for_all_methods(self):
        from run_rpe_method_optimizer import make_executor

        methods = ('FULL_PF', 'TRUNCATED_PF', 'PR_PAULI', 'PR_DF')
        spec = dict(max_new_q3_per_method=dict.fromkeys(methods, 0),
                    executor=dict(prior_run_dirs=[str(self.prior)]))
        executor, _ = make_executor(ROOT, self.root / 'current',
                                    dict.fromkeys(methods, [self.rule]), self.expected, self.problem, spec)
        for method in methods:
            with self.subTest(method=method):
                self.executor = executor.for_method(method)
                accepted = self.evidence(4)
                self.evidence(5, O.FAIL)
                self.evidence(6, outcome=False)
                with patch.object(self.executor, 'optimizer_history_directory', None):
                    baseline = self.executor.prior_records(self.candidates)
                self.assertEqual(len(baseline[0]), 2)
                self.assertEqual(self.executor.prior_records(self.candidates), baseline)
                with patch.object(self.executor, 'ingest', wraps=self.executor.ingest) as ingest:
                    self.assertEqual(executor.for_method(method).prior_records(self.candidates), baseline)
                    self.assertEqual(self.executor.prior_records(self.candidates), baseline)
                self.assertEqual(ingest.call_count, 0)
                self.executor.prior_q3_holds = [dict(parent_directory=str(self.prior), method=method, job_id=accepted.name)]
                known, refused = self.executor.prior_records(self.candidates)
                self.assertNotIn(('cfg', (4, 4)), known)
                self.assertEqual(refused, [dict(job=str(accepted), reason='PRIOR_Q3_PERSISTENCE_HOLD')])


if __name__ == '__main__':
    unittest.main()
