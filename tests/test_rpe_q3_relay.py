"""Simulated SSH relay with real controller RPC/mailbox/artificial adapters.

The fake qg1 SSH command executes locally. external_tunnel=True substitutes an
already-reachable loopback endpoint; these tests establish no actual host link.
Windows branches below use stubs on Linux; no native Windows machine was tested.
"""
from contextlib import redirect_stdout
from fractions import Fraction
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks import rpe_optimizer_concurrent as C
from trottertracks import rpe_q3_worker as W
from trottertracks import rpe_q3_coordinator as Q

spec = importlib.util.spec_from_file_location('_relay_under_test', ROOT/'scripts/run_rpe_q3_relay.py')
R = importlib.util.module_from_spec(spec);spec.loader.exec_module(R)


def write(path, value):
    Path(path).write_text(json.dumps(value))


def wait_until(predicate, seconds=8):
    deadline = time.monotonic()+seconds
    while time.monotonic() < deadline:
        value = predicate()
        if value:return value
        time.sleep(.025)
    raise AssertionError('Artificial relay condition did not complete before deadline')


class Harness:
    def __init__(self, directory, sleep_seconds=.12):
        self.directory = Path(directory)
        problem = dict(accuracy=dict(fixture='ARTIFICIAL'), target=dict(energy_ha=0), rpe=dict(fixture=True))
        rule = dict(kind='artificial', interpreter=sys.executable, configuration_ids='*', view_commit=None,
            artificial=dict(weights=[1]*13,scale='1',tau='7',sleep_seconds=sleep_seconds))
        self.executor = C.DistributedExecutor(ROOT, self.directory/'controller', {'PR_DF':[rule]}, {}, problem,
            optimizer_source=W.head(ROOT), max_concurrent=1, budgets={'PR_DF':4}, science_approved=True,
            wait_for_running=True, distributed=dict(port=0, affinity_wait_seconds=0))
        self.view = self.executor.for_method('PR_DF')
        self.profile = dict(host_id='server1-simulated',root=str(ROOT),work_dir=str(self.directory/'server1-spool'),
            interpreter=sys.executable,route_views={},prefix_cache_directory=str(self.directory/'prefix'),slots=1)
        profile = self.directory/'profile with spaces.json';write(profile,self.profile)
        fake = self.directory/'fake_ssh.py'
        fake.write_text('''import os, shlex, sys
if len(sys.argv)<3 or sys.argv[-2]!='qg1':
    raise SystemExit('Only artificial qg1 SSH mailbox calls are supported')
command=shlex.split(sys.argv[-1])
os.execv(command[0], command)
''')
        self.configuration = self.directory/'relay.json'
        worker = dict(host_id=self.profile['host_id'],ssh_alias='qg1',python=sys.executable,
            root=str(ROOT),profile=str(profile))
        config = dict(state_directory=str(self.directory/'relay-state'),external_tunnel=True,
            ssh_command=[sys.executable,str(fake)],controller=dict(ssh_alias='qg2',local_port=self.executor.rpc_server.server_address[1]),
            workers=[worker],poll_seconds=.025,request_timeout_seconds=5)
        write(self.configuration, config)
        self.relay = R.Relay(self.configuration);self.worker = worker
        self.daemon_pid = None

    def propose(self, shot):
        proposal = dict(configuration_id='ARTIFICIAL_RELAY', vector=tuple([shot]*26),cost=Fraction(26*shot),origin=dict(kind='ARTIFICIAL_RELAY'))
        task = self.view._enqueue(proposal,demand=True)
        return proposal,task

    def tick(self):
        self.relay.tick(self.worker)

    def cli(self, command):
        return subprocess.check_output([sys.executable,'-B',str(ROOT/'scripts/run_rpe_q3_relay.py'),
            '--config',str(self.configuration),command],cwd=ROOT,text=True)

    def status(self):
        return json.loads(self.cli('--status'))

    def stop_daemon(self):
        self.cli('--stop')
        status = self.directory/'relay-state/STATUS.json'
        return wait_until(lambda: W.read(status) if status.exists() and W.read(status).get('stopped') else None)

    def close(self):
        if self.daemon_pid:
            self.stop_daemon()
        # Only owned artificial work is reconciled here. Never stop another run.
        for _ in range(100):
            if not any(t.status=='RUNNING' for t in self.executor.state.tasks.values()):break
            try:self.tick()
            except Exception:pass
            time.sleep(.02)
        for row in self.executor.store.rows():
            if row['host_id'] and row['state'] not in ('ACCEPTED','UNKNOWN','CANCELLED','QUEUED'):
                self.executor.rpc('/v1/state',dict(host_id=row['host_id'],job_id=row['job_id'],attempt_id=row['attempt_id'],
                    state='UNKNOWN',detail=dict(reason='ARTIFICIAL_TEST_CLEANUP')))
        self.executor.finish()


class RelayTests(unittest.TestCase):
    def test_discovery_returns_one_ready_request_before_other_bundle_delivery(self):
        with tempfile.TemporaryDirectory(prefix='relay-single-readiness-') as temporary:
            configuration = Path(temporary)/'relay.json'
            worker = dict(host_id='ARTIFICIAL_WSL')
            write(configuration, dict(state_directory=str(Path(temporary)/'relay-state'),
                controller=dict(ssh_alias='qg2'), workers=[worker]))
            relay = R.Relay(configuration)
            challenges = [dict(job_id=name, request=dict(route=dict(kind='preparation')))
                          for name in ('FIRST', 'SECOND')]
            status = dict(host_id=worker['host_id'], attempts=[], slots=1,
                source_sha='ARTIFICIAL', cached_prefixes=[], metadata_only=True)
            calls = []
            def remote(_, action, value=None):
                calls.append(action)
                if action == 'status': return status, .01
                self.assertEqual(action, 'readiness')
                self.assertEqual(value['challenges'], challenges[:1])
                return dict(host_id=worker['host_id'], host_fingerprints={'FIRST':{}}, readiness_errors={}), .01
            def deliver(_, assignments):
                calls.append('bundle')
                self.assertEqual(assignments, challenges[:1])
            with patch.object(relay, 'remote', side_effect=remote), \
                    patch.object(relay, '_deliver_challenge_artifacts', side_effect=deliver), \
                    patch.object(relay, 'post', return_value=dict(assignments=[], eligibility_challenges=challenges)) as post:
                observed = relay._discover(worker)
                self.assertEqual(calls, ['status', 'bundle', 'readiness'])
                self.assertTrue(all(call.args[1]['observe_only'] for call in post.call_args_list))
                self.assertEqual(observed[3]['host_fingerprints'], {'FIRST':{}})
                # Even when the first probe HOLDs and the claim still has no
                # assignment, cycle's delivery step must not start a second one.
                relay.tick(worker, observed)
                self.assertEqual(calls, ['status', 'bundle', 'readiness'])
                self.assertNotIn('observe_only', post.call_args_list[-1].args[1])

    def test_readiness_fairness_survives_repeated_holds_and_rotating_batches(self):
        with tempfile.TemporaryDirectory(prefix='relay-readiness-fairness-') as temporary:
            configuration = Path(temporary)/'relay.json'
            write(configuration, dict(state_directory=str(Path(temporary)/'relay-state'),
                controller=dict(ssh_alias='qg2'), workers=[]))
            relay = R.Relay(configuration)
            first = [dict(job_id=name, request=dict(route=dict(kind='preparation'))) for name in 'ABCD']
            second = [dict(job_id=name, request=dict(route=dict(kind='preparation'))) for name in 'EFGH']
            worker = dict(host_id='ARTIFICIAL_WSL')
            selected = [relay._readiness_batch(worker, batch)[0]['job_id']
                        for batch in [first, second]*4]
            self.assertEqual(selected, list('AEBFCGDH'))
            self.assertEqual(relay._readiness_batch(worker, first)[0]['job_id'], 'A')
            # Another host's first tie follows its own original batch order.
            self.assertEqual(relay._readiness_batch(dict(host_id='OTHER'), first)[0]['job_id'], 'A')
            q3 = dict(job_id='Q3', request=dict(route=dict(kind='prdf_d2_split')))
            other = dict(job_id='OTHER_ROUTE')
            mixed = [q3, *first, other]
            batch = relay._readiness_batch(dict(host_id='MIXED'), mixed)
            self.assertEqual(batch, [q3, first[0], other])
            self.assertEqual(relay._readiness_batch(worker, [q3, other]), [q3, other])

    def test_wsl_discovery_checks_cached_input_without_delivery_roundtrip(self):
        with tempfile.TemporaryDirectory() as temporary:
            configuration = Path(temporary)/'relay.json'
            wsl = dict(host_id='ARTIFICIAL_WSL', transport='ssh-wsl')
            write(configuration, dict(state_directory=str(Path(temporary)/'state'),
                                      controller=dict(ssh_alias='qg2'), workers=[wsl]))
            relay = R.Relay(configuration)
            requests = [dict(job_id=name, request=dict(route=dict(kind='preparation'),
                        preparation=dict(input_git_bundles=[dict(sha256=name)]))) for name in ['MISS', 'HIT']]
            status = dict(host_id=wsl['host_id'], slots=1, source_sha='ARTIFICIAL',
                          cached_artifact_sha256=['HIT'], cached_prefixes=[])
            facts = dict(host_id=wsl['host_id'], host_fingerprints={'HIT': {}})
            with patch.object(relay, 'post', return_value=dict(eligibility_challenges=requests)), \
                    patch.object(relay, 'remote', side_effect=[(status, .01), (facts, .01)]) as remote, \
                    patch.object(relay, '_deliver_challenge_artifacts') as deliver:
                relay._discover(wsl)
            deliver.assert_not_called()
            self.assertEqual(remote.call_args.args[2]['challenges'], [requests[1]])
            self.assertEqual(relay._readiness_batch(wsl, requests, {'HIT'}), [requests[0]])

    def test_mailbox_json_error_is_bounded_and_excludes_unrelated_payload(self):
        with tempfile.TemporaryDirectory(prefix='relay-mailbox-json-error-') as temporary:
            configuration = Path(temporary)/'relay.json'
            write(configuration, dict(state_directory=str(Path(temporary)/'relay-state'),
                controller=dict(ssh_alias='qg2'), workers=[]))
            relay = R.Relay(configuration)
            worker = dict(ssh_alias='ARTIFICIAL_HOST')
            for error in ['BlockingIOError: worker lock\n\x00'+'x'*2000, {'unexpected':'PAYLOAD'}]:
                with self.subTest(error_type=type(error).__name__):
                    response = SimpleNamespace(returncode=2, stderr='',
                        stdout=json.dumps(dict(error=error, payload='DO_NOT_LOG')))
                    with patch.object(R.subprocess, 'run', return_value=response), \
                            self.assertRaises(RuntimeError) as raised:
                        relay.remote_command(worker, ['ARTIFICIAL_COMMAND'], 'status')
                    detail = str(raised.exception).split('failed (2): ', 1)[1]
                    self.assertLessEqual(len(detail), 1500)
                    self.assertNotIn('DO_NOT_LOG', detail)
                    self.assertNotIn('\n', detail)
                    self.assertNotIn('\x00', detail)
                    self.assertEqual(detail.startswith('BlockingIOError:'), isinstance(error, str))

    def test_mailbox_stderr_remains_primary_error(self):
        with tempfile.TemporaryDirectory(prefix='relay-mailbox-stderr-') as temporary:
            configuration = Path(temporary)/'relay.json'
            write(configuration, dict(state_directory=str(Path(temporary)/'relay-state'),
                controller=dict(ssh_alias='qg2'), workers=[]))
            relay = R.Relay(configuration)
            response = SimpleNamespace(returncode=255, stderr='ARTIFICIAL SSH failure',
                stdout=json.dumps(dict(error='SECONDARY_ERROR')))
            with patch.object(R.subprocess, 'run', return_value=response), \
                    self.assertRaisesRegex(RuntimeError, r'failed \(255\): ARTIFICIAL SSH failure$'):
                relay.remote_command(dict(ssh_alias='ARTIFICIAL_HOST'), ['ARTIFICIAL_COMMAND'], 'status')

    def test_worker_connection_failures_back_off_without_slowing_other_workers(self):
        with tempfile.TemporaryDirectory(prefix='relay-backoff-clock-') as temporary:
            configuration = Path(temporary)/'relay.json'
            workers = [dict(host_id=name, ssh_alias=name, root='/ARTIFICIAL/root', profile='/ARTIFICIAL/profile')
                       for name in ('failing', 'healthy')]
            write(configuration, dict(state_directory=str(Path(temporary)/'relay-state'), external_tunnel=True,
                controller=dict(ssh_alias='qg2'), workers=workers, poll_seconds=1))
            now, recover = [1000.], [False]
            clock = SimpleNamespace(time=lambda:now[0], monotonic=lambda:now[0])
            calls = {name:[] for name in ('failing', 'healthy')}
            def ssh(command, **options):
                host = command[-2]; calls[host].append(now[0])
                if host == 'failing' and not recover[0]:
                    return SimpleNamespace(returncode=255, stdout='', stderr='ARTIFICIAL connection unavailable')
                return SimpleNamespace(returncode=0, stderr='', stdout=json.dumps(dict(host_id=host,
                    attempts=[], slots=1, source_sha='ARTIFICIAL_SOURCE', cached_prefixes=[])))
            with patch.object(R, 'time', clock), patch.object(R.subprocess, 'run', side_effect=ssh):
                relay = R.Relay(configuration)
                with patch.object(relay, 'post', return_value=dict(assignments=[])):
                    for second in range(92):
                        now[0] = 1000.+second
                        relay.cycle()
                    self.assertEqual(calls['failing'], [1000.+x for x in (0, 1, 3, 7, 15, 31, 61, 91)])
                    self.assertEqual(len(calls['healthy']), 92)
                    status = json.loads((relay.directory/'STATUS.json').read_text())
                    self.assertEqual(status['heartbeat_epoch'], now[0])
                    self.assertEqual(status['workers']['failing']['failure_count'], 8)
                    self.assertEqual(status['workers']['failing']['next_retry_epoch'], 1121.)
                    events = [json.loads(line) for line in (relay.directory/'EVENTS.jsonl').read_text().splitlines()]
                    pending = [e for e in events if e['event'] == 'TRANSPORT_PENDING']
                    self.assertEqual(len(pending), 8)
                    self.assertEqual([e['retry_seconds'] for e in pending], [1, 2, 4, 8, 16, 30, 30, 30])
                    now[0] = 1121.; recover[0] = True
                    relay.cycle()
                    self.assertEqual(relay.state['workers']['failing']['failure_count'], 0)
                    self.assertEqual(relay.state['workers']['failing']['next_retry_epoch'], 0)
                    now[0] = 1122.; recover[0] = False
                    relay.cycle()
                    self.assertEqual(relay.state['workers']['failing']['failure_count'], 1)
                    self.assertEqual(relay.state['workers']['failing']['next_retry_epoch'], 1123.)

    def test_immediately_exiting_or_unstartable_tunnel_does_not_respawn_each_poll(self):
        with tempfile.TemporaryDirectory(prefix='relay-tunnel-backoff-') as temporary:
            configuration = Path(temporary)/'relay.json'
            write(configuration, dict(state_directory=str(Path(temporary)/'relay-state'),
                controller=dict(ssh_alias='qg2'), workers=[], poll_seconds=1))
            now, alive = [1000.], [False]
            clock = SimpleNamespace(time=lambda:now[0], monotonic=lambda:now[0])
            starts = []
            def launch(*args, **options):
                starts.append(now[0])
                return SimpleNamespace(pid=40000+len(starts), poll=lambda:None if alive[0] else 255)
            with patch.object(R, 'time', clock), patch.object(R.subprocess, 'Popen', side_effect=launch):
                relay = R.Relay(configuration)
                for second in range(40):
                    now[0] = 1000.+second
                    relay.cycle()
                self.assertEqual(starts, [1000.+x for x in (0, 2, 5, 10, 19, 36)])
                self.assertEqual(relay.state['tunnel']['failure_count'], 6)
                self.assertEqual(relay.state['tunnel']['next_retry_epoch'], 1067.)
                alive[0] = True; now[0] = 1067.
                relay.cycle()
                now[0] = 1072.
                relay.cycle()
                self.assertEqual(relay.state['tunnel']['failure_count'], 0)
                self.assertEqual(len(starts), 7)
            now[0] = 2000.
            with patch.object(R, 'time', clock), \
                    patch.object(R.subprocess, 'Popen', side_effect=OSError('ARTIFICIAL SSH executable unavailable')) as failed:
                relay = R.Relay(configuration)
                for second in range(8):
                    now[0] = 2000.+second
                    relay.cycle()
                self.assertEqual(failed.call_count, 4)
                self.assertEqual(relay.state['tunnel']['next_retry_epoch'], 2015.)

    def test_recovery_log_survives_optional_metric_failure_and_later_poll_failure(self):
        with tempfile.TemporaryDirectory(prefix='relay-complete-metric-') as temporary:
            configuration = Path(temporary)/'relay.json'
            worker = dict(host_id='ARTIFICIAL_SERVER1')
            write(configuration, dict(state_directory=str(Path(temporary)/'relay-state'), external_tunnel=True,
                controller=dict(ssh_alias='qg2'), workers=[worker]))
            relay = R.Relay(configuration)
            transferred, poll_failure = [False], [False]
            def remote(_, action, value=None):
                if action == 'ack':
                    transferred[0] = True
                    return dict(ok=True), .01
                self.assertEqual(action, 'status')
                attempt = dict(job_id='ARTIFICIAL_JOB', attempt_id='ARTIFICIAL_ATTEMPT',
                    state='TRANSFERRED' if transferred[0] else 'COMPLETED_PENDING_TRANSFER',
                    request_sha256='ARTIFICIAL_HASH', files={})
                return dict(host_id=worker['host_id'], attempts=[attempt], slots=1,
                    source_sha='ARTIFICIAL_SOURCE', cached_prefixes=[]), .02
            def post(path, value):
                if path == '/v1/transfer-metric': raise OSError('ARTIFICIAL optional metric unavailable')
                if path == '/v1/poll':
                    if poll_failure[0]: raise OSError('ARTIFICIAL controller closed after completion')
                    return dict(assignments=[])
                return dict(ok=True, duplicate=False)
            with patch.object(relay, 'remote', side_effect=remote), patch.object(relay, 'post', side_effect=post):
                relay.cycle()
                self.assertEqual(relay.state['workers'][worker['host_id']]['failure_count'], 0)
                events = [json.loads(line) for line in (relay.directory/'EVENTS.jsonl').read_text().splitlines()]
                self.assertEqual([e['event'] for e in events if e['event']!='MAILBOX_STATUS_RETURN'], ['RESULT_RECOVERED', 'TELEMETRY_PENDING'])
                self.assertTrue(transferred[0])
                poll_failure[0] = True
                relay.cycle()
                events = [json.loads(line) for line in (relay.directory/'EVENTS.jsonl').read_text().splitlines()]
                self.assertEqual([e['event'] for e in events if e['event']!='MAILBOX_STATUS_RETURN'], ['RESULT_RECOVERED', 'TELEMETRY_PENDING', 'TRANSPORT_PENDING'])
                self.assertEqual(events[-1]['phase'], 'MAILBOX_DISCOVERY')
                self.assertEqual(next(e for e in events if e['event']=='RESULT_RECOVERED')['job_id'], 'ARTIFICIAL_JOB')
                self.assertEqual(relay.state['workers'][worker['host_id']]['failure_count'], 1)

    def test_startup_status_replaces_old_stopped_state_before_first_blocked_tick(self):
        with tempfile.TemporaryDirectory(prefix='relay-startup-status-') as temporary:
            directory = Path(temporary)/'relay-state'; directory.mkdir()
            configuration = Path(temporary)/'relay.json'
            worker = dict(host_id='ARTIFICIAL_BLOCKED_WORKER')
            write(configuration, dict(state_directory=str(directory), external_tunnel=True,
                controller=dict(ssh_alias='qg2'), workers=[worker], poll_seconds=.01))
            write(directory/'STATUS.json', dict(pid=-12345, stopped=True, stopped_epoch=1))
            relay = R.Relay(configuration)
            entered, release = threading.Event(), threading.Event()
            def blocking_discovery(_):
                entered.set()
                release.wait(5)
                status=dict(host_id=worker['host_id'], attempts=[], slots=0, source_sha='ARTIFICIAL',cached_prefixes=[])
                return status,0,{},status,{}
            with patch.object(relay, 'ensure_tunnel'), patch.object(relay, '_discover', side_effect=blocking_discovery):
                process = threading.Thread(target=relay.run, name='artificial-blocked-relay')
                process.start()
                try:
                    self.assertTrue(entered.wait(3), 'Fixture did not reach its blocked first tick')
                    observed = json.loads((directory/'STATUS.json').read_text())
                    self.assertEqual(observed['pid'], os.getpid())
                    self.assertIs(observed['stopped'], False)
                    self.assertNotIn('stopped_epoch', observed)
                    self.assertGreater(observed['started_epoch'], 1)
                finally:
                    (directory/'STOP').touch()
                    release.set()
                    process.join(timeout=3)
            self.assertFalse(process.is_alive())
            self.assertIs(json.loads((directory/'STATUS.json').read_text())['stopped'], True)

    def test_real_mailbox_rpc_next_assignment_and_lost_ack_duplicate(self):
        with tempfile.TemporaryDirectory(prefix='q3-relay-artificial-') as temporary:
            harness = Harness(temporary)
            try:
                first, task = harness.propose(2)
                harness.tick()
                assignment = harness.executor.store.row(task.job.name)
                self.assertEqual(assignment['host_id'],harness.profile['host_id'])
                attempt = Path(harness.profile['work_dir'])/assignment['job_id']/assignment['attempt_id']
                wait_until(lambda:(attempt/'PROCESS_RETURN.json').exists())
                original_remote = harness.relay.remote
                lost_ack = [True]
                def flaky_remote(worker, action, value=None):
                    if action=='ack' and lost_ack[0]:
                        lost_ack[0]=False
                        raise OSError('ARTIFICIAL relay disconnect after controller accepted completion')
                    return original_remote(worker,action,value)
                harness.relay.remote=flaky_remote
                with self.assertRaisesRegex(OSError,'disconnect'):
                    harness.tick()
                self.assertEqual(harness.executor.store.row(task.job.name)['state'],'ACCEPTED')
                self.assertEqual(W.read(attempt/'WORKER_STATE.json')['state'],'COMPLETED_PENDING_TRANSFER')
                # A fresh local relay recovers the same bytes and attempt.
                harness.relay=R.Relay(harness.configuration)
                harness.tick()
                self.assertEqual(W.read(attempt/'WORKER_STATE.json')['state'],'TRANSFERRED')
                answer = harness.view.answer(first,allow_new=False)
                self.assertEqual(answer['MSE_lower'],Fraction(13,2))
                self.assertEqual(answer['MSE_upper'],Fraction(13,2))
                self.assertEqual(answer['calls'],1)
                self.assertEqual(harness.view.answer(first,allow_new=False)['calls'],0)
                self.assertEqual(harness.executor.store.counts('PR_DF'),1)
                second,next_task = harness.propose(1)
                harness.tick()
                wait_until(lambda: (harness.tick() or next_task.event.is_set()))
                second_answer = harness.view.answer(second,allow_new=False)
                self.assertEqual(second_answer['MSE_upper'],Fraction(13))
                self.assertNotEqual(answer['status'],second_answer['status'])
                self.assertEqual(harness.executor.store.counts('PR_DF'),2)
                self.assertEqual(sum(e['event']=='ASSIGN' for e in harness.executor.store.events()),2)
                self.assertEqual(sum(e['event']=='ACCEPT' for e in harness.executor.store.events()),2)
                events=[json.loads(line)for line in (harness.relay.directory/'EVENTS.jsonl').read_text().splitlines()]
                self.assertTrue(any(e['event']=='RESULT_RECOVERED' and e.get('duplicate') is True for e in events))
                for path in Path(harness.profile['work_dir']).glob('*/*/JOB_STAGES.jsonl'):
                    self.assertEqual(path.read_text().count('ARTIFICIAL_EVALUATION_START'),1)
                metrics=harness.executor.metrics()
                self.assertEqual(metrics['by_host'][harness.profile['host_id']]['completed'],2)
            finally:
                harness.close()

    def test_foreign_mailbox_attempts_do_not_block_claim_or_recovery(self):
        for local_ledger in (True, False):
            with self.subTest(local_ledger=local_ledger), tempfile.TemporaryDirectory() as temporary:
                harness = Harness(temporary)
                try:
                    if local_ledger:
                        config = W.read(harness.configuration)
                        config['state_directory'] = str(harness.executor.run_dir/'distributed'/'relay')
                        write(harness.configuration, config)
                        harness.relay = R.Relay(harness.configuration)
                    proposal, task = harness.propose(2)
                    stale = [
                        dict(job_id='old-job', attempt_id='old-attempt', state='UNKNOWN'),
                        dict(job_id=task.job.name, attempt_id='old-owned-job-attempt',
                             state='COMPLETED_PENDING_TRANSFER'),
                    ]
                    remote = harness.relay.remote
                    def mailbox(worker, action, value=None):
                        result, seconds = remote(worker, action, value)
                        if action == 'status':
                            result['attempts'] = stale + result['attempts']
                        return result, seconds
                    harness.relay.remote = mailbox
                    with patch.object(harness.relay, 'post', wraps=harness.relay.post) as post:
                        harness.tick()
                        wait_until(lambda: harness.tick() or task.event.is_set())
                        ignored = {row['attempt_id'] for row in stale}
                        reported = {call.args[1].get('attempt_id') for call in post.call_args_list
                                    if call.args[0] == '/v1/state'}
                        self.assertEqual(reported & ignored, set() if local_ledger else ignored)
                        self.assertFalse(any(call.args[1].get('attempt_id') in ignored
                                             for call in post.call_args_list if call.args[0] == '/v1/complete'))
                    self.assertEqual(harness.view.answer(proposal, allow_new=False)['MSE_upper'], Fraction(13, 2))
                    self.assertEqual(sum(e['event'] == 'ASSIGN' for e in harness.executor.store.events()), 1)
                    self.assertEqual(sum(e['event'] == 'ACCEPT' for e in harness.executor.store.events()), 1)
                    events = (harness.relay.directory/'EVENTS.jsonl').read_text()
                    self.assertEqual(events.count('MAILBOX_ATTEMPT_OUTSIDE_RUN'), 2)
                finally:
                    harness.close()

    def test_cli_stop_leaves_received_job_running_then_start_recovers(self):
        with tempfile.TemporaryDirectory(prefix='q3-relay-cli-artificial-') as temporary:
            harness = Harness(temporary,sleep_seconds=1.)
            try:
                self.assertEqual(harness.status()['status'],'NOT_STARTED')
                proposal,task=harness.propose(2)
                started=json.loads(harness.cli('--start'));harness.daemon_pid=started['pid']
                launch = wait_until(lambda: next(Path(harness.profile['work_dir']).glob('*/*/WORKER_LAUNCH.json'),None))
                self.assertTrue(harness.status()['heartbeat_age_seconds']<5)
                stopped=harness.stop_daemon();harness.daemon_pid=None
                self.assertTrue(stopped['stopped'])
                attempt=launch.parent
                supervisor=W.read(attempt/'SUPERVISOR_STARTED.json')['pid']
                self.assertTrue(W.process_alive(supervisor,attempt))
                wait_until(lambda:(attempt/'PROCESS_RETURN.json').exists())
                # Relay outage leaves completed artifacts on the worker.
                self.assertNotEqual(harness.executor.store.row(task.job.name)['state'],'ACCEPTED')
                resumed=json.loads(harness.cli('--start'));harness.daemon_pid=resumed['pid']
                wait_until(lambda:task.event.is_set())
                self.assertEqual(harness.view.answer(proposal,allow_new=False)['status'],'MODEL_BOUND_MEETS_ADOPTED_THRESHOLD')
                harness.stop_daemon();harness.daemon_pid=None
                self.assertTrue(harness.status()['stopped'])
                self.assertEqual((attempt/'JOB_STAGES.jsonl').read_text().count('ARTIFICIAL_EVALUATION_START'),1)
                self.assertEqual(sum(e['event']=='ASSIGN' for e in harness.executor.store.events()),1)
                self.assertEqual(sum(e['event']=='ACCEPT' for e in harness.executor.store.events()),1)
            finally:
                harness.close()


class PortableRelayBranches(unittest.TestCase):
    """Artificial Windows decisions on Linux, not native Windows validation."""

    def test_background_platform_options_and_windows_start_arguments(self):
        self.assertEqual(R.background_options('posix'), dict(start_new_session=True))
        windows = R.background_options('nt')
        expected = getattr(subprocess, 'DETACHED_PROCESS', 0x8) | getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0x200)
        self.assertEqual(windows, dict(creationflags=expected))
        # Also prove --start passes the selected Windows options to Popen,
        # rather than only checking a helper unused by the actual entry point.
        with tempfile.TemporaryDirectory(prefix='relay-portable-') as temporary:
            directory = Path(temporary) / 'ユーザー relay state'
            configuration = Path(temporary) / 'relay 設定.json'
            value = dict(state_directory=str(directory), external_tunnel=True,
                controller=dict(ssh_alias='qg2', local_port=18763), workers=[])
            configuration.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8-sig')
            original_options = R.background_options
            output = io.StringIO()
            with patch.object(R, 'background_options', side_effect=lambda: original_options('nt')), \
                    patch.object(R.subprocess, 'Popen', return_value=SimpleNamespace(pid=43210)) as launch, \
                    redirect_stdout(output):
                R.main(['--config', str(configuration), '--start'])
            command = launch.call_args.args[0]
            options = launch.call_args.kwargs
            self.assertEqual(command[-3:], ['--config', str(configuration.resolve()), '--run'])
            self.assertEqual(options['creationflags'], expected)
            self.assertNotIn('start_new_session', options)
            self.assertEqual(options['stdin'], subprocess.DEVNULL)
            self.assertEqual(Path(options['stdout'].name), directory/'RELAY.log')
            self.assertEqual(json.loads(output.getvalue())['pid'], 43210)

    def test_windows_mailbox_and_tunnel_children_suppress_console_creation(self):
        self.assertEqual(R.child_options('posix'), {})
        expected = getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000)
        self.assertEqual(R.child_options('nt'), dict(creationflags=expected))
        with tempfile.TemporaryDirectory(prefix='relay-hidden-children-') as temporary:
            configuration = Path(temporary)/'relay.json'
            write(configuration, dict(state_directory=str(Path(temporary)/'relay-state'),
                controller=dict(ssh_alias='qg2'), workers=[]))
            relay = R.Relay(configuration)
            worker = dict(ssh_alias='qg1', root='/ARTIFICIAL/worker root', profile='/ARTIFICIAL/worker profile.json')
            original_options = R.child_options
            with patch.object(R, 'child_options', side_effect=lambda: original_options('nt')), \
                    patch.object(R.subprocess, 'run', return_value=SimpleNamespace(returncode=0,
                        stdout='{"host_id":"ARTIFICIAL_SERVER1"}', stderr='')) as mailbox, \
                    patch.object(R.subprocess, 'Popen', return_value=SimpleNamespace(pid=42420, poll=lambda:None)) as tunnel:
                observed, _ = relay.remote(worker, 'status')
                self.assertEqual(observed, dict(host_id='ARTIFICIAL_SERVER1'))
                self.assertEqual(mailbox.call_args.kwargs['creationflags'], expected)
                self.assertNotIn('start_new_session', mailbox.call_args.kwargs)
                self.assertEqual(mailbox.call_args.args[0][-2], 'qg1')
                relay.ensure_tunnel()
                self.assertEqual(tunnel.call_args.kwargs['creationflags'], expected)
                self.assertNotIn('start_new_session', tunnel.call_args.kwargs)
                self.assertEqual(tunnel.call_args.kwargs['stdin'], subprocess.DEVNULL)
                self.assertEqual(tunnel.call_args.args[0][-1], 'qg2')
                relay.ensure_tunnel()
                self.assertEqual(tunnel.call_count, 1, 'A live tunnel must not spawn another child each poll')

    def test_msvcrt_lock_fallback_uses_first_byte_and_normalizes_busy_error(self):
        with tempfile.TemporaryDirectory(prefix='relay-windows-lock-stub-') as temporary:
            path = Path(temporary)/'RELAY.lock'
            with path.open('a+') as stream:
                positions = []
                def locking(descriptor, mode, length):
                    positions.append((descriptor, mode, length, stream.tell()))
                stub = SimpleNamespace(LK_NBLCK=17, locking=Mock(side_effect=locking))
                with patch.object(R, 'fcntl', None), patch.dict(sys.modules, {'msvcrt': stub}):
                    R.acquire_lock(stream)
                    self.assertEqual(path.read_bytes(), b'\0')
                    stream.seek(0, os.SEEK_END)
                    R.acquire_lock(stream)
                    self.assertEqual(path.read_bytes(), b'\0')
                    self.assertEqual(positions, [(stream.fileno(), 17, 1, 0)]*2)
                    stub.locking.side_effect = OSError('Artificial lock owned by another process')
                    with self.assertRaisesRegex(BlockingIOError, 'Another relay owns'):
                        R.acquire_lock(stream)

    def test_json_atomic_windows_branch_flushes_file_and_skips_directory_open(self):
        # Patching only this module's os binding avoids changing pathlib's
        # platform choice on the Linux machine running these artificial checks.
        fake_os = SimpleNamespace(name='nt', fsync=Mock(wraps=os.fsync), O_RDONLY=os.O_RDONLY,
            open=Mock(side_effect=AssertionError('Windows directory open must be skipped')),
            close=Mock(side_effect=AssertionError('No Windows directory fd should exist')))
        with tempfile.TemporaryDirectory(prefix='relay-json-windows-stub-') as temporary:
            path = Path(temporary)/'STATUS.json'
            with patch.object(Q, 'os', fake_os):
                Q.atomic_json(path, dict(state='STARTING', host='人工ホスト'))
                self.assertEqual(json.loads(path.read_text()), dict(state='STARTING', host='人工ホスト'))
                Q.atomic_json(path, dict(state='STOPPED'))
            self.assertEqual(json.loads(path.read_text()), dict(state='STOPPED'))
            self.assertFalse(path.with_name(path.name+'.tmp').exists())
            self.assertEqual(fake_os.fsync.call_count, 2)
            fake_os.open.assert_not_called()
            fake_os.close.assert_not_called()



if __name__=='__main__':
    unittest.main()
