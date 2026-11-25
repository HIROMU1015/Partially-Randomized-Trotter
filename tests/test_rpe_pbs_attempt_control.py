"""Attempt stop with real local processes and an artificial PBS reservation."""
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from unittest.mock import patch

import pytest

from test_rpe_q3_worker import fixture, ROOT

sys.path.insert(0, str(ROOT/'scripts'))
import rpe_pbs_gateway as gateway
from trottertracks import rpe_q3_worker as worker
from trottertracks import rpe_pbs_attempt_control as control
from trottertracks.rpe_q3_store import JobStore


PBS_JOB = '123.prism'


def until(predicate, tick=lambda: None, timeout=10):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        tick()
        if predicate():
            return
        time.sleep(.02)
    raise AssertionError('Artificial PBS attempt did not reach the expected state')


def test_stop_one_attempt_and_adopt_another_on_the_same_reservation(tmp_path, monkeypatch):
    profile, first = fixture(tmp_path)
    root = tmp_path/'pbs-run'
    directory = root/'allocations'/'session1'
    directory.mkdir(parents=True)
    profile['work_dir'] = str(root/'workers'/profile['host_id'])
    profile['route_profiles'] = [dict(route_kind='artificial', profile=dict(profile))]
    profile['pbs'] = dict(allocation_dir=str(directory))
    gateway.atomic(directory/'PROFILE.json', profile)
    gateway.atomic(directory/'ALLOCATION.json', dict(job_id=PBS_JOB))
    gateway.atomic(directory/'STATE.json', dict(worker.stamp(), state='READY'))
    monkeypatch.setenv('PBS_JOBID', PBS_JOB)
    first['request']['route']['artificial']['sleep_seconds'] = 45
    first['request_sha256'] = worker.digest(first['request'])
    second = deepcopy(first)
    second['job_id'] = second['request']['request_id'] = 'independent'
    second['request']['route']['artificial']['sleep_seconds'] = .02
    second['request_sha256'] = worker.digest(second['request'])
    assignments = {item['job_id']: item for item in (first, second)}
    store = JobStore(tmp_path/'queue.sqlite3')
    assert store.reserve(first['method'], first['job_id'], first['request'], 4, True)

    def rpc(endpoint, value):
        if endpoint == '/v1/poll':
            rows = store.claim(profile['host_id'], value['slots'])
            return dict(stop=False, assignments=[dict(assignments[row['job_id']], attempt_id=row['attempt_id'])
                                                  for row in rows])
        if endpoint == '/v1/state':
            store.update(profile['host_id'], value['job_id'], value['attempt_id'], value['state'], value['detail'])
        elif endpoint == '/v1/complete':
            store.complete(profile['host_id'], value['job_id'], value['attempt_id'], value['request_sha256'], value['files'])
            store.adopt(value['job_id'])
        return dict(ok=True)

    instance = worker.Worker(profile, 'http://127.0.0.1:1', transport=rpc)
    unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
    try:
        instance.tick()
        row = store.row(first['job_id'])
        attempt = instance.work/first['job_id']/row['attempt_id']
        until(lambda: (attempt/'WORKER_LAUNCH.json').is_file(), lambda: instance.tick(acquire=False))
        (attempt/'partial.txt').write_text('artificial partial evidence')
        request = dict(action='attempt_stop', token=directory.name, pbs_job_id=PBS_JOB,
                       **{k: row[k] for k in control.IDENTITY})
        with patch.object(gateway, 'qstat', return_value={PBS_JOB:dict(job_state='R')}):
            with pytest.raises(ValueError, match='selected request'):
                gateway.handle(root, dict(request, request_sha256='wrong'))
            assert gateway.handle(root, request)['stop_requested']
            assert gateway.handle(root, request)['stop_requested']
        assert store.reserve(second['method'], second['job_id'], second['request'], 4, True)
        until(lambda: store.row(second['job_id'])['adopted_epoch'] is not None, instance.tick)
        stopped = gateway.handle(root, dict(request, action='attempt_status'))
        assert stopped['worker_state']['detail']['pbs_attempt_exit']['operator_stop'] is True
        assert worker.read(attempt/'PROCESS_RETURN.json')['signal'] == signal.SIGTERM
        assert (attempt/'partial.txt').read_text() == 'artificial partial evidence'
        assert store.row(first['job_id'])['state'] == 'UNKNOWN'
        assert store.row(first['job_id'])['attempt_id'] == row['attempt_id']
        assert store.row(first['job_id'])['adopted_epoch'] is None
        assert store.row(first['job_id'])['detail']['resource_released'] is True
        assert [e['event'] for e in store.events()].count('ASSIGN') == 2
        assert [e['event'] for e in store.events()].count('ADOPT') == 1
        assert worker.read(directory/'ALLOCATION.json')['job_id'] == PBS_JOB
        assert unrelated.poll() is None and not (directory/'CONTROL.json').exists()
    finally:
        for process in instance.processes.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        instance.close()
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_missing_receipt_releases_only_after_verified_session_exit(tmp_path, monkeypatch):
    attempt = tmp_path/'attempt'
    attempt.mkdir()
    profile = dict(host_id='worker-test', work_dir=str(tmp_path),
                   pbs=dict(allocation_dir=str(tmp_path/'allocation')))
    assignment = dict(job_id='j', attempt_id='a', request_sha256='fixed')
    monkeypatch.setenv('PBS_JOBID', PBS_JOB)
    process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)'], start_new_session=True)
    instance = object.__new__(worker.Worker)
    instance.profile, instance.processes = profile, {attempt: process}
    owner = dict(assignment, token='allocation', allocation_dir=profile['pbs']['allocation_dir'], pbs_job_id=PBS_JOB,
                 supervisor_pid=process.pid, clock_id=worker.stamp()['clock_id'])
    detached = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)'], start_new_session=True,
                                env=control.adapter_environment(owner))
    try:
        worker.atomic(attempt/control.OWNER, owner)
        for name in ('SUPERVISOR_STARTED.json', 'SUPERVISOR_LAUNCH.json'):
            worker.atomic(attempt/name, dict(pid=process.pid, session_id=process.pid))
        (attempt/'TRANSPORT_TIMING.jsonl').write_text(json.dumps(worker.stamp())+'\n')
        (attempt/'WORKER_PROCESS.lock').touch()
        worker.save_state(attempt, 'RUNNING')
        assert instance._reconcile(attempt, assignment)['state'] == 'RUNNING'
        process.terminate()
        process.wait(timeout=5)
        state = instance._reconcile(attempt, assignment)
        assert state['state'] == 'UNKNOWN' and not state['detail'].get('resource_released')
    finally:
        for child in (process, detached):
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=5)
    until(lambda: instance._reconcile(attempt, assignment)['detail'].get('resource_released'))
    state = instance._reconcile(attempt, assignment)
    assert state['state'] == 'UNKNOWN' and state['detail']['resource_released'] is True
    assert state['detail']['pbs_attempt_exit']['operator_stop'] is False
    assert not (attempt/'PROCESS_RETURN.json').exists()
