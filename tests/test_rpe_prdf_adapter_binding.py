"""Artificial assignment/transport checks of separately fixed PR-DF adapters.

Published host recipes and adapter bytes are data. Supervisor and adapter Popen
calls are mocked; only read-only Git metadata commands run. No Q3 or dry-check
adapter is invoked, and the scientific assignment retains SOURCE 28d976b99.
Runtime probes are mocked here; separate runtime checks validate the real ABI.
"""
from copy import deepcopy
from fractions import Fraction
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
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests')]
from trottertracks import rpe_q3_coordinator as Q
from trottertracks import rpe_q3_worker as W
from trottertracks.rpe_host_profiles import profile_core, select_host_profile, validate_adapter_profile
from trottertracks.rpe_optimizer_concurrent import DistributedExecutor
from test_rpe_optimizer_concurrent import fixture

SOURCE = '28d976b99cdf8329adcc1b79327e329e947ef774'
IMPLEMENTATION = 'e37f6544dea3e021d3694264f7dae87add7ad30e'
ADAPTER_COMMIT = '09efb068f242d5d6ff88296df3e50283b2f951bf'
OUTER = '23160cc876807c0300be3de811f767b5810640a7'
INNER = '27e70efe533862f38f4080aabb9cdd3ad1749fe9'
ADAPTER_FILES = {
    'scripts/run_rpe_optimizer_q3_job.py': '63cf463024241af20e9c6b3ca55eaf0ce2ebf4a4a7b8c9ca91a8ea74f6cb7fb6',
    'src/trottertracks/rpe_dp_prefix_reuse.py': '554460755ab6fcd38ad5b9f1a680b543aed7e26906579b26470d73fa659148f2',
}
POLICY = ROOT / 'docs/research/rpe_q3_host_eligibility/HOST_ELIGIBILITY.json'


def git(*args, root=ROOT):
    return subprocess.check_output(['git', *args], cwd=root)


def published_profiles(host):
    policy = json.loads(POLICY.read_text())
    selected = {record['method']: record['hosts'][host] for record in policy['jobs'].values()}
    for method in ('FULL_PF', 'PR_DF'):
        if selected[method]['status'] != 'PASS':
            raise AssertionError('Published ' + host + '/' + method + ' profile must be PASS')
    return deepcopy(selected['FULL_PF']['execution_profile']), deepcopy(selected['PR_DF']['execution_profile'])


def private_view(path, commit, names):
    """Own two/eight-file fixture with real saved blobs and read-only objects."""
    path.mkdir()
    subprocess.run(['git', 'init', '-q', str(path)], check=True)
    common = Path(git('rev-parse', '--git-common-dir').decode().strip())
    if not common.is_absolute():
        common = ROOT / common
    (path / '.git/objects/info/alternates').write_text(str(common.resolve() / 'objects') + '\n')
    (path / '.git/HEAD').write_text(commit + '\n')
    for name in names:
        destination = path / name;destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(git('show', commit + ':' + name))
    return path


class AdapterBindingTests(unittest.TestCase):
    def test_two_file_adapter_manifest_is_published_byte_identical(self):
        for name, expected in ADAPTER_FILES.items():
            with self.subTest(path=name):
                implementation = git('show', IMPLEMENTATION + ':' + name)
                official = git('show', ADAPTER_COMMIT + ':' + name)
                self.assertEqual(implementation, official)
                self.assertEqual(hashlib.sha256(official).hexdigest(), expected)
                self.assertEqual((ROOT / name).read_bytes(), official)
        self.assertEqual(git('show', SOURCE + ':src/trottertracks/rpe_dp_prefix_reuse.py'),
                         git('show', IMPLEMENTATION + ':src/trottertracks/rpe_dp_prefix_reuse.py'))

    def test_controller_mailbox_and_resume_bind_server1_and_local_profiles(self):
        outcomes = []
        for host in ('server1', 'local'):
            with self.subTest(host=host):
                outcomes.append(self._exercise_host(host))
        result = dict(schema='ARTIFICIAL_PRDF_ADAPTER_BINDING_ACCEPTANCE_V1', status='PASS',
            scope='ARTIFICIAL_REQUESTS_AND_MOCKED_WORKER_SUPERVISOR_ADAPTER_COMMANDS',
            scientific_source=SOURCE, adapter_commit=ADAPTER_COMMIT, implementation_commit=IMPLEMENTATION,
            adapter_files=ADAPTER_FILES, implementation_and_official_bytes_equal=True,
            prefix_helper_unchanged_from_scientific_source=True, hosts=outcomes,
            published_profile_locator=str(POLICY.relative_to(ROOT)),
            recorded_supervisor_commands=sum(x['recorded_supervisor_commands'] for x in outcomes),
            recorded_adapter_commands=sum(x['recorded_adapter_commands'] for x in outcomes),
            actual_worker_Popen_calls=0, real_Q3_calls=0, real_dry_check_calls=0,
            runtime_probe='Mocked metadata result; actual ABI checks belong to separate real-host evidence.',
            Popen_count_scope='Supervisor/adapter calls only; read-only Git metadata is executed.',
            same_scientific_job_assigned_to_two_hosts=False,
            fixture='Each host uses a separate private controller/run and host-specific artificial configuration IDs.')
        output = os.environ.get('RPE_Q3_ADAPTER_BINDING_ACCEPTANCE')
        if output:
            path = Path(output);path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2) + '\n')

    def _exercise_host(self, host):
        methods = ['FULL_PF', 'TRUNCATED_PF', 'PR_DF']
        problem, context, routes = fixture(methods=methods, configurations=1)
        controller_source = git('rev-parse', 'HEAD').decode().strip()
        for candidate in context['candidates']:
            candidate['configuration_id'] += '_' + host
        for method in methods:
            routes[method][0]['view_commit'] = OUTER if method == 'PR_DF' else 'ARTIFICIAL_PF_ROUTE'
        with tempfile.TemporaryDirectory(prefix='rpe-artificial-adapter-' + host + '-') as temporary:
            base = Path(temporary);spool = base / 'worker';run = base / 'controller'
            source_names = Q.source_manifest(ROOT, controller_source)['files'].keys()
            scientific = private_view(base / 'scientific-source', SOURCE, source_names)
            common, special = published_profiles(host)
            published_special = deepcopy(special)
            # Relocate only the synthetic host's scientific root to the exact saved eight-file fixture.
            common['root'] = str(scientific);special['root'] = str(scientific)
            common.update(work_dir=str(spool), slots=3)
            if host == 'local':
                self.assertEqual(special['adapter_manifest']['files'], ADAPTER_FILES)
                self.assertEqual(special['adapter_manifest']['commit'], ADAPTER_COMMIT)
                self.assertEqual(special['adapter_manifest']['implementation_commit'], IMPLEMENTATION)
                self.assertIn(INNER, special['route_views'])
                self.assertTrue(special['interpreter'].endswith('/.venv311/bin/python'))
                adapter = private_view(base / 'published-adapter', ADAPTER_COMMIT, ADAPTER_FILES)
                special['adapter_root'] = str(adapter)
            else:
                self.assertIn('runtime-acceptance/bin/python3', special['interpreter'])
                self.assertNotIn('adapter_manifest', special)
            # PF and DF may correctly share the published 3.11 launcher. Bind
            # the complete selected recipe (views/cache/runtime checks), not
            # an incidental difference between interpreter path strings.
            self.assertNotEqual(profile_core(special), profile_core(common))
            common['route_profiles'] = [dict(route_kind='artificial', view_commit=OUTER, profile=special)]
            policy_file = base / 'ARTIFICIAL_POLICY.json'
            executor = DistributedExecutor(ROOT, run, routes, context['expected'], problem,
                optimizer_source=controller_source, max_concurrent=3, budgets={m: 1 for m in methods}, science_approved=True,
                distributed=dict(bind='127.0.0.1', port=0, affinity_wait_seconds=0,
                    computation_root=str(scientific), computation_source=SOURCE, host_eligibility=str(policy_file)))
            try:
                proposals, facts = {}, {}
                policy = dict(schema='RPE_Q3_HOST_ELIGIBILITY_V1', fixture='ARTIFICIAL_ONLY', jobs={})
                for candidate in context['candidates']:
                    method = candidate['method'];view = executor.for_method(method)
                    proposal = dict(configuration_id=candidate['configuration_id'], vector=tuple([1] * 26),
                        cost=Fraction(sum(candidate['prices'])), origin={'kind': 'ARTIFICIAL_ADAPTER_BINDING'})
                    proposals[method] = proposal
                    job = view.job_dir(proposal);request = view.request(proposal, job, view.route(proposal['configuration_id']))
                    core = profile_core(select_host_profile(common, dict(request=request)))
                    fact = dict(fixture='ARTIFICIAL_ONLY', host_id=host, profile=core)
                    facts[job.name] = fact
                    record = dict(method=method, request_sha256=Q.digest(request), source_manifest=executor.source_manifest,
                        source_manifest_sha256=Q.digest(executor.source_manifest), input_hashes=[], eligible_host_ids=[host],
                        hosts={host: dict(status='PASS', current_host_fingerprint=fact,
                            current_host_fingerprint_sha256=Q.digest(fact), execution_profile=core)})
                    record['decision_sha256'] = Q.digest(record);policy['jobs'][job.name] = record
                policy_file.write_text(json.dumps(policy))
                for method in methods:
                    self.assertTrue(executor.for_method(method)._enqueue(proposals[method], demand=True))
                polled = executor.rpc('/v1/poll', dict(host_id=host, slots=3, source_sha=SOURCE,
                    cached_prefixes=[], host_fingerprints=facts))
                self.assertEqual(len(polled['assignments']), 3)
                assignments = {assignment['method']: assignment for assignment in polled['assignments']}
                for method, assignment in assignments.items():
                    self.assertEqual(assignment['source_manifest']['commit'], SOURCE)
                    self.assertEqual(assignment['request']['optimizer_source'], SOURCE)
                    self.assertEqual(assignment['request']['route']['kind'], 'artificial')
                    want = special if method == 'PR_DF' else common
                    self.assertEqual(assignment['host_execution_profile'], profile_core(want))
                    self.assertEqual(assignment['host_execution_profile_sha256'], Q.digest(profile_core(want)))
                df_assignment = assignments['PR_DF']
                df = df_assignment['job_id']
                changed = deepcopy(policy)
                changed['jobs'][df]['hosts'][host]['execution_profile'] = profile_core(common)
                policy_file.write_text(json.dumps(changed))
                replay = executor.rpc('/v1/poll', dict(host_id=host, slots=3, source_sha=SOURCE,
                    cached_prefixes=[], host_fingerprints=facts))
                self.assertEqual({a['method']: a for a in replay['assignments']}, assignments)
                other = 'local' if host == 'server1' else 'server1'
                self.assertEqual(executor.rpc('/v1/poll', dict(host_id=other, slots=3, source_sha=SOURCE,
                    cached_prefixes=[], host_fingerprints=facts))['assignments'], [])

                commands, attempts = [], {}
                actual_popen = subprocess.Popen
                class FakeProcess:
                    def __init__(self, command, options):
                        commands.append(dict(command=list(command), cwd=str(options.get('cwd'))))
                        self.pid = 99900000 + len(commands)
                    def wait(self, *args, **kwargs): return 0
                    def poll(self): return 0
                def popen(command, *args, **options):
                    if str(command[0]) == 'git':
                        return actual_popen(command, *args, **options)
                    return FakeProcess(command, options)
                with patch.object(W, 'head', return_value=SOURCE), \
                     patch.object(W.subprocess, 'Popen', side_effect=popen), \
                     patch.object(W, 'check_runtime_profile', return_value={'status': 'PASS', 'fixture': 'ARTIFICIAL_ONLY'}) as runtime_probe:
                    for method, assignment in assignments.items():
                        self.assertEqual(W.mailbox(common, 'submit', assignment), dict(ok=True, duplicate=False))
                        attempt = spool / assignment['job_id'] / assignment['attempt_id']
                        stored = W.read(attempt / 'WORKER_PROFILE.json');attempts[method] = (attempt, stored)
                        want = special if method == 'PR_DF' else common
                        self.assertEqual(profile_core(stored), profile_core(want))
                        self.assertEqual(stored['work_dir'], str(spool));self.assertEqual(stored['slots'], 3)
                        supervisor = commands[-1]['command']
                        self.assertEqual(supervisor[0], want['interpreter'])
                        self.assertEqual(supervisor[2], str(ROOT / 'scripts/run_rpe_q3_worker.py'))
                        self.assertNotEqual(supervisor[2], str(Path(want['root']) / 'scripts/run_rpe_q3_worker.py'))
                        # Only construction/transport is under test; no source-bound adapter is executed.
                        resolved = deepcopy(assignment['request']);resolved['route']['interpreter'] = want['interpreter']
                        W.atomic(attempt / 'RESOLVED_REQUEST.json', resolved)
                        self.assertEqual(W._run_adapter(stored, attempt, 'execute', attempt), 0)
                        if stored.get('runtime_checks'):
                            self.assertEqual(runtime_probe.call_args.args, (stored,))
                        command = commands[-1]['command']
                        self.assertEqual(command[0], want['interpreter'])
                        adapter_root = want.get('adapter_root', want['root'])
                        self.assertEqual(command[2], str(Path(adapter_root) / 'scripts/run_rpe_optimizer_q3_job.py'))
                        self.assertEqual(commands[-1]['cwd'], adapter_root)
                        if method == 'PR_DF' and host == 'local':
                            self.assertEqual(command[-2:], ['--route-views', str(attempt / 'ROUTE_VIEWS.json')])
                            self.assertEqual(W.read(attempt / 'ROUTE_VIEWS.json'), special['route_views'])
                            self.assertIn(INNER, W.read(attempt / 'ROUTE_VIEWS.json'))
                            adapter_source = W.read(attempt / 'ADAPTER_SOURCE.json')
                            self.assertEqual(adapter_source, dict(root=str(adapter), **special['adapter_manifest']))
                            self.assertNotEqual(adapter_source['commit'], assignment['source_manifest']['commit'])
                        else:
                            self.assertNotIn('--route-views', command)
                            self.assertFalse((attempt / 'ADAPTER_SOURCE.json').exists())
                        W.save_state(attempt, 'COMPLETED_PENDING_TRANSFER', fixture='ARTIFICIAL_ONLY')
                        before = len(commands)
                        self.assertEqual(W.mailbox(common, 'submit', assignment), dict(ok=True, duplicate=True))
                        self.assertEqual(len(commands), before)
                        self.assertEqual(W.read(attempt / 'WORKER_PROFILE.json'), stored)
                    wrong_common = dict(common);wrong_common.pop('route_profiles')
                    before = len(commands)
                    with self.assertRaisesRegex(ValueError, 'common fallback is forbidden'):
                        W.mailbox(dict(wrong_common, work_dir=str(base / 'wrong-spool')), 'submit', df_assignment)
                    self.assertEqual(len(commands), before)
                    resumed = W.Worker(wrong_common, 'http://127.0.0.1:1', transport=lambda *args: None)
                    try:
                        for method, assignment in assignments.items():
                            resumed._acquire(assignment)
                            attempt, stored = attempts[method]
                            self.assertEqual(W.read(attempt / 'WORKER_PROFILE.json'), stored)
                            self.assertEqual(W._adapter_command(stored, attempt, 'execute', attempt)[0], stored['interpreter'])
                        self.assertEqual(len(commands), before)
                    finally:
                        resumed.close()
                    if host == 'local':
                        attempt, stored = attempts['PR_DF']
                        # Both exact adapter files, its checkout HEAD, and the attempt's route map are guarded.
                        for name in ADAPTER_FILES:
                            target = adapter / name;original = target.read_bytes();target.write_bytes(original + b'\n# synthetic change\n')
                            try:
                                with self.assertRaisesRegex(ValueError, 'Adapter SOURCE hash differs'):
                                    W._adapter_command(stored, attempt, 'execute', attempt)
                            finally:
                                target.write_bytes(original)
                        head = adapter / '.git/HEAD';original = head.read_text();head.write_text(IMPLEMENTATION + '\n')
                        try:
                            with self.assertRaisesRegex(ValueError, 'Adapter HEAD differs'):
                                W._adapter_command(stored, attempt, 'execute', attempt)
                        finally:
                            head.write_text(original)
                        W.atomic(attempt / 'ROUTE_VIEWS.json', {})
                        with self.assertRaisesRegex(ValueError, 'Assigned adapter route views changed'):
                            W._adapter_command(stored, attempt, 'execute', attempt)
                        W.atomic(attempt / 'ROUTE_VIEWS.json', stored['route_views'])
                        record = W.read(attempt / 'ADAPTER_SOURCE.json')
                        W.atomic(attempt / 'ADAPTER_SOURCE.json', dict(record, implementation_commit='CHANGED'))
                        with self.assertRaisesRegex(ValueError, 'Assigned adapter manifest changed'):
                            W._adapter_command(stored, attempt, 'execute', attempt)
                        W.atomic(attempt / 'ADAPTER_SOURCE.json', record)
                        self.assertEqual(len(commands), before)
                        self.assertEqual(validate_adapter_profile(stored)['files'], ADAPTER_FILES)
                return dict(host_id=host, methods=methods, assignments=3,
                    scientific_source=SOURCE, selected_prdf_interpreter=published_special['interpreter'],
                    published_adapter_root=published_special.get('adapter_root'),
                    adapter_fixture='Temporary two-file Git view; official HEAD and bytes checked, no remote access.' if host == 'local' else None,
                    adapter_source_separate=host == 'local', inner_route_preserved=host == 'local',
                    policy_change_does_not_rewrite_assigned_attempt=True, duplicate_and_resume_profile_preserved=True,
                    common_fallback_rejected=True, other_host_cannot_claim_assigned_job=True,
                    source_and_manifest_mutation_rejected=host == 'local', recorded_supervisor_commands=3,
                    recorded_adapter_commands=3)
            finally:
                with executor.state.cv:
                    for task in executor.state.tasks.values():
                        if task.status == 'RUNNING':
                            task.status = 'DONE';task.error = 'ARTIFICIAL_TEST_CLEANUP';task.event.set()
                    executor.state.active = 0;executor.state.cv.notify_all()
                executor.finish()


if __name__ == '__main__':
    unittest.main()
