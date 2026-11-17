"""Verified real envelopes, artificial queues and invocation recording only.

Each host is tested in its own temporary four-job queue. Remote filesystem,
Git/runtime and adapter metadata come from verified evidence; every supervisor
and adapter Popen is mocked. No OUTCOME, completion or adoption is fabricated.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks import rpe_q3_coordinator as Q
from trottertracks import rpe_q3_worker as W
from trottertracks.rpe_host_profiles import profile_core
from trottertracks.rpe_optimizer_concurrent import DistributedExecutor

DOC = ROOT / 'docs/research/rpe_q3_host_eligibility'
BASE = ROOT / 'docs/research/rpe_distributed_h4_plan_20261117'


def read(path):
    return json.loads(Path(path).read_text())


class TwelveHostAssignmentTests(unittest.TestCase):
    def test_verified_twelve_cells_preserve_profiles_through_worker_invocation(self):
        policy_path = DOC / 'HOST_ELIGIBILITY.json'
        policy = read(policy_path)
        candidates = [read(p) for p in sorted((BASE / 'assignments').glob('*.json'))]
        candidates.append(read(DOC / 'pauli_descriptor_route/ASSIGNMENT.json'))
        assignments = {}
        for job, decision in policy['jobs'].items():
            matches = [a for a in candidates if a['job_id'] == job and
                       Q.digest(a['request']) == decision['request_sha256']]
            self.assertTrue(matches, 'No public matching scientific REQUEST: ' + decision['method'])
            assignments[job] = matches[0]
            self.assertEqual(decision['source_manifest'], matches[0]['source_manifest'])
            self.assertEqual(set(decision['eligible_host_ids']), {'server1', 'server2', 'local'})
            self.assertTrue(all(c['status'] == 'PASS' for c in decision['hosts'].values()))
        self.assertEqual(len(assignments), 4)
        first = next(iter(policy['jobs'].values()))
        computation = Path(os.environ.get('RPE_TWELVE_COMPUTATION_ROOT',
                           first['hosts']['server2']['execution_profile']['root']))
        if not (computation / '.git').exists():
            self.skipTest('This evidence integration needs the existing fixed computation checkout')
        manifest = first['source_manifest'];source = manifest['commit']
        self.assertEqual(Q.source_manifest(computation, source), manifest)
        original_bytes = {name: (computation / name).read_bytes() for name in manifest['files']}
        input_blobs = {}
        for assignment in assignments.values():
            for locator in assignment['input_hashes']:
                key = locator['commit'] + ':' + locator['path']
                if key not in input_blobs:
                    input_blobs[key] = subprocess.check_output(['git', 'show', key], cwd=computation)
                self.assertEqual(hashlib.sha256(input_blobs[key]).hexdigest(), locator['sha256'])
        problem = read(BASE / 'inputs/PROBLEM.json')
        routes = {a['method']: [a['request']['route']] for a in assignments.values()}
        expected = {a['request']['configuration_id']: a['request']['expected'] for a in assignments.values()}
        controller_source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        all_rows = [];host_counts = {};recorded_commands = []
        with tempfile.TemporaryDirectory(prefix='rpe-artificial-twelve-cells-') as temporary:
            temporary = Path(temporary)
            for host in ['server2', 'server1', 'local']:
                with self.subTest(host=host):
                    run = temporary / host / 'ARTIFICIAL-controller'
                    saved_policy = run.parent / 'VERIFIED_POLICY_COPY.json'
                    saved_policy.parent.mkdir(parents=True);saved_policy.write_text(json.dumps(policy))
                    executor = DistributedExecutor(ROOT, run, routes, expected, problem,
                        optimizer_source=controller_source, max_concurrent=4,
                        budgets={method: 1 for method in routes}, science_approved=True,
                        distributed=dict(bind='127.0.0.1', port=0, affinity_wait_seconds=0,
                            computation_root=str(computation), computation_source=source,
                            host_eligibility=str(saved_policy)))
                    def release():
                        # Release this test's in-memory waiter states only. No fake completion.
                        with executor.state.cv:
                            for task in executor.state.tasks.values():
                                task.status = 'DONE';task.error = 'ARTIFICIAL_TEARDOWN_NO_OUTCOME';task.event.set()
                            executor.state.active = 0;executor.state.cv.notify_all()
                        executor.finish()
                    try:
                        for job, assignment in assignments.items():
                            decision = executor._eligibility_for(assignment['request'], job)
                            self.assertEqual(decision['eligible_host_ids'], policy['jobs'][job]['eligible_host_ids'])
                            self.assertTrue(executor.store.reserve(assignment['method'], job, assignment['request'],
                                1, True, contract={'fixture': 'ARTIFICIAL_METADATA_QUEUE_ONLY'}, eligibility=decision))
                            with executor.state.cv:
                                executor._restore_task(executor.store.row(job), demand=True)
                        facts = {job: d['hosts'][host]['current_host_fingerprint'] for job, d in policy['jobs'].items()}
                        poll = dict(host_id=host, slots=4, source_sha=source, cached_prefixes=[], host_fingerprints=facts)
                        claimed = executor.rpc('/v1/poll', poll)['assignments']
                        self.assertEqual(len(claimed), 4)
                        self.assertEqual({a['job_id'] for a in claimed}, set(assignments))
                        self.assertEqual(len({a['attempt_id'] for a in claimed}), 4)
                        self.assertEqual(executor.rpc('/v1/poll', poll)['assignments'], claimed)
                        for other in {'server2', 'server1', 'local'} - {host}:
                            elsewhere = dict(poll, host_id=other, host_fingerprints={
                                job: d['hosts'][other]['current_host_fingerprint'] for job, d in policy['jobs'].items()})
                            self.assertEqual(executor.rpc('/v1/poll', elsewhere)['assignments'], [])
                            # Also exercise ownership without the controller's full
                            # capacity masking a duplicate-claim regression.
                            self.assertEqual(executor.store.claim(other, 4, allowed_jobs=list(assignments),
                                host_fingerprints=elsewhere['host_fingerprints']), [])
                        cells = {a['job_id']: policy['jobs'][a['job_id']]['hosts'][host] for a in claimed}
                        core = deepcopy(cells[claimed[0]['job_id']]['execution_profile'])
                        base = dict(core, work_dir=str(run.parent / 'ARTIFICIAL-spool'), slots=4)
                        recipes = {}
                        for assignment in claimed:
                            r = assignment['request']['route'];key = (r['kind'], r['view_commit'])
                            chosen = cells[assignment['job_id']]['execution_profile']
                            self.assertEqual(recipes.setdefault(key, chosen), chosen)
                        base['route_profiles'] = [dict(route_kind=k[0], view_commit=k[1], profile=p)
                                                  for k, p in recipes.items()]
                        heads = {};filesystem = {}
                        for cell in cells.values():
                            p = cell['execution_profile'];heads[str(Path(p['root']).resolve())] = source
                            for commit, view in p['route_views'].items():heads[str(Path(view).resolve())] = commit
                            for name, raw in original_bytes.items():filesystem[str(Path(p['root']).resolve() / name)] = raw
                        original_read = Path.read_bytes;original_file = Path.is_file
                        def metadata_head(path):return heads[str(Path(path).resolve())]
                        def metadata_bytes(path):
                            key = str(path)
                            return filesystem[key] if key in filesystem else original_read(path)
                        def metadata_file(path):return str(path) in filesystem or original_file(path)
                        def metadata_git(command, **kwargs):
                            self.assertEqual(command[:2], ['git', 'show'])
                            return input_blobs[command[2]]
                        def find_cell(p):
                            found = [c for c in cells.values() if c['execution_profile'] == profile_core(p)]
                            self.assertTrue(found, 'Execution profile differs from every verified PASS cell')
                            return found[0]
                        runtime_calls = []
                        def metadata_runtime(p):
                            cell = find_cell(p);fp = cell['current_host_fingerprint'];runtime_calls.append(profile_core(p))
                            self.assertTrue(p.get('runtime_checks'))
                            return dict(status='PASS', runtime=fp['runtime'], checks=fp['runtime_checks'])
                        def metadata_adapter(p):
                            find_cell(p)
                            return dict(root=p['adapter_root'], **p['adapter_manifest']) if p.get('adapter_root') else None
                        commands = []
                        class FakeProcess:
                            def __init__(self, command, **kwargs):
                                self.pid = 99000000 + len(recorded_commands)
                                commands.append((list(command), kwargs));recorded_commands.append(list(command))
                            def wait(self, *args, **kwargs):return 0
                            def poll(self):return 0
                        def fake_popen(command, **kwargs):
                            self.assertIn(Path(command[2]).name, ['run_rpe_q3_worker.py', 'run_rpe_optimizer_q3_job.py'])
                            return FakeProcess(command, **kwargs)
                        with patch.object(W, 'head', side_effect=metadata_head), \
                             patch.object(Path, 'read_bytes', metadata_bytes), patch.object(Path, 'is_file', metadata_file), \
                             patch.object(W.subprocess, 'check_output', side_effect=metadata_git), \
                             patch.object(W.subprocess, 'Popen', side_effect=fake_popen), \
                             patch.object(W, 'validate_adapter_profile', side_effect=metadata_adapter), \
                             patch.object(W, 'check_runtime_profile', side_effect=metadata_runtime):
                            worker = W.Worker(base, executor.controller_url)
                            try:
                                for assignment in claimed:
                                    job = assignment['job_id'];cell = cells[job];want = cell['execution_profile']
                                    self.assertEqual(assignment['request'], assignments[job]['request'])
                                    self.assertEqual(assignment['request_sha256'], policy['jobs'][job]['request_sha256'])
                                    self.assertEqual(assignment['source_manifest'], manifest)
                                    self.assertEqual(assignment['host_execution_profile'], want)
                                    self.assertEqual(assignment['host_execution_profile_sha256'], Q.digest(want))
                                    evidence = cell['evidence'][0];readiness = read(evidence['readiness'])
                                    self.assertEqual(hashlib.sha256(original_read(Path(evidence['readiness']))).hexdigest(), evidence['sha256'])
                                    self.assertEqual(readiness['request_sha256'], assignment['request_sha256'])
                                    worker._acquire(assignment)
                                    attempt = worker.work / job / assignment['attempt_id']
                                    stored = W.read(attempt / 'WORKER_PROFILE.json')
                                    self.assertEqual(profile_core(stored), want)
                                    resolved = W.validate_assignment(stored, assignment)
                                    self.assertEqual(resolved['route']['interpreter'], want['interpreter'])
                                    self.assertEqual(resolved['route']['view'], want['route_views'][resolved['route']['view_commit']])
                                    W.atomic(attempt / 'RESOLVED_REQUEST.json', resolved)
                                    worker._launch(attempt)
                                    self.assertEqual(commands[-1][0][0], want['interpreter'])
                                    self.assertEqual(W._run_adapter(stored, attempt, 'execute', attempt), 0)
                                    command, options = commands[-1]
                                    self.assertEqual(command[0], want['interpreter'])
                                    self.assertEqual(command[2], str(Path(want.get('adapter_root', want['root'])) / 'scripts/run_rpe_optimizer_q3_job.py'))
                                    self.assertEqual(options['cwd'], want.get('adapter_root', want['root']))
                                    self.assertEqual(command[command.index('--mode') + 1], 'execute')
                                    if want.get('adapter_manifest'):
                                        self.assertEqual(W.read(attempt / 'ROUTE_VIEWS.json'), want['route_views'])
                                        self.assertEqual(W.read(attempt / 'ADAPTER_SOURCE.json'), dict(root=want['adapter_root'], **want['adapter_manifest']))
                                        self.assertEqual(command[command.index('--route-views') + 1], str(attempt / 'ROUTE_VIEWS.json'))
                                    else:self.assertNotIn('--route-views', command)
                                    if want.get('runtime_checks'):
                                        self.assertEqual(readiness['runtime_precheck']['checks'], cell['current_host_fingerprint']['runtime_checks'])
                                        self.assertIn(want, runtime_calls)
                                    count = len(commands);worker._acquire(assignment)
                                    self.assertEqual(len(commands), count)
                                    self.assertEqual(W.read(attempt / 'WORKER_PROFILE.json'), stored)
                                    self.assertFalse((attempt / 'OUTCOME.json').exists())
                                    all_rows.append(dict(host_id=host, method=assignment['method'], job_id=job,
                                        request_sha256=assignment['request_sha256'], profile_sha256=Q.digest(want),
                                        interpreter=want['interpreter'], route_view=resolved['route']['view'],
                                        adapter_root=want.get('adapter_root'), adapter_manifest=want.get('adapter_manifest'),
                                        runtime_checks=want.get('runtime_checks', []), simulated_claim=True,
                                        simulated_supervisor=True, simulated_adapter=True, actual_processes=0))
                            finally:worker.close()
                            resumed = W.Worker(dict(base, route_profiles=[]), executor.controller_url)
                            try:
                                count = len(commands)
                                for assignment in claimed:
                                    resumed._acquire(assignment)
                                    attempt = resumed.work / assignment['job_id'] / assignment['attempt_id']
                                    saved = W.read(attempt / 'WORKER_PROFILE.json')
                                    self.assertEqual(profile_core(saved), cells[assignment['job_id']]['execution_profile'])
                                    self.assertEqual(W._adapter_command(saved, attempt, 'execute', attempt)[0], saved['interpreter'])
                                    W.save_state(attempt, 'UNKNOWN', fixture='ARTIFICIAL_TEARDOWN_NO_EXECUTION')
                                self.assertEqual(len(commands), count)
                            finally:resumed.close()
                        self.assertEqual(len(commands), 8)
                        for assignment in claimed:
                            executor.rpc('/v1/state', dict(host_id=host, job_id=assignment['job_id'],
                                attempt_id=assignment['attempt_id'], state='UNKNOWN', detail={'fixture': 'ARTIFICIAL_TEARDOWN_ONLY'}))
                        self.assertFalse(any(e['event'] in ('ACCEPT', 'ADOPT') for e in executor.store.events()))
                        self.assertFalse(executor.state.adopted_jobs)
                        self.assertFalse(list(run.rglob('OUTCOME.json')))
                        host_counts[host] = dict(independent_queue=True, claimed=4, unique_jobs=4,
                            mock_supervisor_invocations=4, mock_adapter_invocations=4,
                            other_host_reclaims=0, restart_extra_launches=0, outcomes=0, adoptions=0)
                    finally:release()
        for name, raw in original_bytes.items():self.assertEqual((computation / name).read_bytes(), raw)
        result = dict(schema='ARTIFICIAL_TWELVE_CELL_ASSIGNMENT_V1', status='PASS',
            scope='REAL_VERIFIED_ENVELOPES_ARTIFICIAL_CLAIMS_AND_INVOCATIONS_ONLY',
            scientific_source=source, policy_sha256=hashlib.sha256(policy_path.read_bytes()).hexdigest(),
            hosts=host_counts, rows=all_rows, simulated_claims=12, mock_supervisor_invocations=12,
            mock_adapter_invocations=12, actual_worker_or_adapter_processes=0, outcomes=0, adoptions=0,
            new_scientific_evaluations=0, production_queue_operations=0,
            metadata_stubs=['Remote HEAD/files mirror fixed SOURCE; Git blobs read from fixed local SOURCE',
                            'Adapter manifest and runtime AST result replay verified evidence; no fresh runtime probe'],
            fixed_source_files_unchanged=len(original_bytes))
        output = os.environ.get('RPE_TWELVE_ASSIGNMENT_ACCEPTANCE')
        if output:
            path = Path(output);path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    unittest.main()
