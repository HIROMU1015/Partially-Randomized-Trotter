"""Small request/case identity and dry-cache freshness regressions; no Q3."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from trottertracks import rpe_plan_readiness as R

FIXED = Path('/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/rpe-distributed-q3-source-28d976b99')


class ReadinessTests(unittest.TestCase):
    def test_queue_state_snapshot_is_read_only_and_unknown_schema_is_explicit(self):
        with tempfile.TemporaryDirectory(prefix='rpe-plan-queue-') as temporary:
            root = Path(temporary)
            queue = root / 'distributed/QUEUE.sqlite3'
            queue.parent.mkdir()
            with sqlite3.connect(queue) as database:
                database.execute('CREATE TABLE jobs(job_id TEXT,method TEXT,host_id TEXT,attempt_id TEXT,state TEXT)')
                database.executemany('INSERT INTO jobs VALUES(?,?,?,?,?)',
                    [('j1', 'PR_DF', 'server1', 'a1', 'RUNNING'),
                     ('j2', 'FULL_PF', 'server2', 'a2', 'ACCEPTED')])
            before = queue.read_bytes(), queue.stat().st_mtime_ns
            snapshot = R.current_queue_states([root, root])
            self.assertEqual([row['state'] for row in snapshot], ['RUNNING', 'ACCEPTED'])
            self.assertTrue(all(row['status'] == 'READ_ONLY_QUEUE_STATE' for row in snapshot))
            self.assertEqual((queue.read_bytes(), queue.stat().st_mtime_ns), before)
            with sqlite3.connect(queue) as database:
                database.execute('DROP TABLE jobs')
                database.execute('CREATE TABLE jobs(unrecognized TEXT)')
            before = queue.read_bytes(), queue.stat().st_mtime_ns
            error = R.current_queue_states([root])
            self.assertEqual(error[0]['status'], 'QUEUE_SCHEMA_OR_READ_UNCONFIRMED')
            self.assertFalse(error[0]['dispatchable'])
            self.assertEqual((queue.read_bytes(), queue.stat().st_mtime_ns), before)
            self.assertEqual(R.current_queue_states([root / 'not-a-run']), [])

    def test_baseline_requests_assignments_and_original_case_inputs_are_complete_matches(self):
        if not FIXED.is_dir():
            self.skipTest('Fixed computation view is not available on this host')
        def read(name):
            return json.loads(R._blob(FIXED, R.BASELINE, R.BASE + '/' + name))
        problem = read('inputs/PROBLEM.json')
        expected, routes, methods = {}, {}, {}
        for method in R.METHODS:
            request = read('requests/' + method + '.json')
            expected[request['configuration_id']] = request['expected']
            routes[method] = [request['route']]
            methods[method] = dict(pending_request=dict(configuration_id=request['configuration_id'],
                vector=[v for pair in zip(request['N_X'], request['N_Y']) for v in pair],
                cost=request['cost'], origin=request['origin']))
        with tempfile.TemporaryDirectory(prefix='rpe-plan-readiness-') as temporary:
            prepared = R.build_readiness(FIXED, dict(expected=expected, routes=routes), problem,
                                         {}, dict(methods=methods), Path(temporary) / 'new')
            self.assertTrue(prepared['old_remote_delivery_unchanged'])
            spec = importlib.util.spec_from_file_location('_readiness_fixed_adapter', FIXED / 'scripts/run_rpe_optimizer_q3_job.py')
            adapter = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(adapter)
            for method in R.METHODS:
                for folder in ('requests', 'assignments'):
                    relative = folder + '/' + method + '.json'
                    self.assertEqual((Path(prepared['output']) / relative).read_bytes(),
                                     R._blob(FIXED, R.BASELINE, R.BASE + '/' + relative))
                old = read('requests/' + method + '.json')
                new = R._read(Path(prepared['output']) / 'requests' / (method + '.json'))
                self.assertEqual(adapter.build(Path(old['route']['view']), old),
                                 adapter.build(Path(new['route']['view']), new))
            for name in ('LOCAL0_DRY_CHECK.md', 'LOCAL0_DRY_CHECK.zip'):
                self.assertEqual((Path(prepared['output']) / name).read_bytes(),
                                 R._blob(FIXED, R.BASELINE, R.BASE + '/' + name))

    def test_cache_reuses_only_matching_dimensions_and_keeps_failure(self):
        with tempfile.TemporaryDirectory(prefix='rpe-plan-cache-') as temporary:
            base = Path(temporary)
            assignment = base / 'assignment.json'
            profile = base / 'profile.json'
            R._write(assignment, dict(request={}, job_id='synthetic-no-execution'))
            R._write(profile, {})
            count = []
            def fake_checker(command, **kwargs):
                target = Path(command[command.index('--output') + 1])
                count.append(str(target))
                R._write(target / 'READINESS.json', dict(status='REAL_HOST_DRY_CHECK_MISSING_OR_INVALID_INPUT',
                    scientific_evaluation_calls=0, error='missing synthetic route'))
                return subprocess.CompletedProcess(command, 2)
            with patch.object(R, 'dry_check_fingerprint', return_value=dict(source='fixed', runtime='old')), \
                 patch.object(R.subprocess, 'run', side_effect=fake_checker):
                first = R.run_server2_dry_checks(base, {'M': str(assignment)}, profile, base / 'one')
                second = R.run_server2_dry_checks(base, {'M': str(assignment)}, profile, base / 'two', cache=base / 'one')
                self.assertEqual(first['live_checks'], 1)
                self.assertEqual(second['cached_checks'], 1)
                self.assertEqual(second['methods']['M']['original_status'], 'REAL_HOST_DRY_CHECK_MISSING_OR_INVALID_INPUT')
                self.assertEqual(len(count), 1)
            with patch.object(R, 'dry_check_fingerprint', return_value=dict(source='fixed', runtime='changed')), \
                 patch.object(R.subprocess, 'run', side_effect=fake_checker):
                third = R.run_server2_dry_checks(base, {'M': str(assignment)}, profile, base / 'three', cache=base / 'one')
                self.assertEqual(third['live_checks'], 1)
                self.assertEqual(third['cached_checks'], 0)
                self.assertEqual(len(count), 2)


if __name__ == '__main__':
    unittest.main()
