"""Simulated SSH relay with real controller RPC/mailbox/artificial adapters.

The fake qg1 SSH command executes locally. external_tunnel=True substitutes an
already-reachable loopback endpoint; these tests establish no actual host link.
"""
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks import rpe_optimizer_concurrent as C
from trottertracks import rpe_q3_worker as W

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


if __name__=='__main__':
    unittest.main()
