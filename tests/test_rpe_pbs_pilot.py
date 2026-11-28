import pytest
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import rpe_pbs_pilot as pilot
import run_rpe_pbs_storage_worker as collector

G = pilot.gateway


def test_pilot_control_pause_resume_does_not_erase_permanent_stop(tmp_path):
    (tmp_path/'SESSION_START.json').write_text('{}')
    pilot.update_session_control(tmp_path,dict(stop=False,pause=True,release_when_idle=False))
    assert pilot.read(tmp_path/'CONTROL.json')['pause'] is True
    pilot.update_session_control(tmp_path,dict(stop=False,pause=False,release_when_idle=False))
    assert pilot.read(tmp_path/'CONTROL.json')['pause'] is False
    pilot.update_session_control(tmp_path,dict(stop=True,release_when_idle=True))
    pilot.update_session_control(tmp_path,dict(stop=False,pause=False,release_when_idle=False))
    assert pilot.read(tmp_path/'CONTROL.json')['stop'] is True


class PilotTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name)
        self.pool = self.base/pilot.POOL_NAME
        self.a, self.b = (self.base/(c*20) for c in 'ab')
        self.jobs = {}
        self.submissions = []
        run = subprocess.run

        def scheduler(command, *args, **kwargs):
            if command[0] == '/opt/pbs/bin/qsub':
                token = Path(command[1]).parent.name
                job = str(len(self.submissions)+1)+'.prism'
                self.submissions.append(token)
                resources = G.read(Path(command[1]).parent/'SPEC.json')['resources']
                self.jobs[job] = dict(Job_Name='RPE'+token[:12], job_state='Q',
                                      Resource_List=dict(resources))
                return subprocess.CompletedProcess(command, 0, job, '')
            if command[0] == '/opt/pbs/bin/qdel':
                self.jobs[command[1]]['job_state'] = 'F'
                return subprocess.CompletedProcess(command, 0, '', '')
            return run(command, *args, **kwargs)

        for ctx in [patch.object(pilot, 'owned_live_jobs', return_value=self.jobs),
                    patch.object(G, 'qstat', side_effect=lambda *ids: {
                        k:v for k,v in self.jobs.items() if not ids or k in ids}),
                    patch.object(subprocess, 'run', side_effect=scheduler)]:
            ctx.start()
            self.addCleanup(ctx.stop)

    def attach(self, root, token, reason=None):
        spec = dict(token=token, resources=pilot.RESOURCES, run_id=root.name, reason=reason or {},
                    profile=dict(host_id='pbs-0', work_dir=str(root/'workers'/'pbs-0')),
                    python=sys.executable, transport_root=str(self.base/'code'))
        pilot.handle(root, dict(action='ensure', spec=spec), {})
        return root/'allocations'/token

    def stop(self, root, token):
        pilot.handle(root, dict(action='exchange', controls={token:dict(stop=True)}), {})
        pilot.handle(root, dict(action='delete', token=token), {})

    def test_retained_pilot_attaches_without_reserving_or_releasing(self):
        current = pilot.admin(self.pool, 'reserve')
        self.jobs[current['job_id']]['job_state'] = 'R'
        reason = dict(retained_pilot=dict(token=current['token'], job_id=current['job_id']))
        # An explicitly retained live reservation is valid even when a different
        # generation is selected for future ordinary allocations.
        with patch.object(pilot, 'reserve_locked', side_effect=AssertionError('unexpected reserve')):
            directory = self.attach(self.b, 'retained', reason)
        self.assertEqual(G.read(directory/'PILOT.json')['token'], current['token'])
        self.assertEqual(len(self.submissions), 1)
        self.assertFalse((self.pool/'allocations'/current['token']/'RELEASE.json').exists())

    def test_attempt_replay_binds_only_to_the_original_live_pilot(self):
        current = pilot.admin(self.pool, 'reserve')
        job_id = current['job_id']
        self.jobs[job_id]['job_state'] = 'R'
        replay = dict(replay_pilot=dict(token=current['token'], job_id=job_id))
        directory = self.attach(self.a, 'replay', replay)
        self.assertEqual(G.read(directory/'PILOT.json')['token'], current['token'])
        self.attach(self.a, 'replay', replay)
        self.jobs[job_id]['job_state'] = 'F'
        with self.assertRaisesRegex(ValueError, 'Original PBS reservation is unavailable'):
            self.attach(self.a, 'obsoleteReplay', replay)
        self.assertEqual(len(self.submissions), 1)

    def test_prequeue_stop_switch_lost_reply_and_explicit_release(self):
        with ThreadPoolExecutor(2) as executor:
            rows = list(executor.map(lambda _: pilot.admin(self.pool,'reserve'), range(2)))
        self.assertEqual(rows[0]['job_id'], rows[1]['job_id'])
        generation = self.pool/'allocations'/rows[0]['token']
        (generation/'QSUB_RETURN.json').unlink()
        (generation/'JOB_ID.json').unlink()
        self.attach(self.a, 'tokenA')
        self.stop(self.a, 'tokenA')
        b = self.attach(self.b, 'tokenB')
        self.attach(self.b, 'tokenB')
        self.assertEqual(pilot.claim(self.pool,generation,'1.prism'), b)
        self.assertEqual(pilot.session_snapshot(self.a)[0]['status'], 'F')
        self.assertEqual(pilot.session_snapshot(self.b)[0]['job_id'], '1.prism')
        self.assertEqual(self.jobs['1.prism']['job_state'], 'Q')
        self.assertEqual(len(self.submissions), 1)
        pilot.admin(self.pool, 'release')
        self.assertEqual(self.jobs['1.prism']['job_state'], 'F')
        self.assertEqual(pilot.admin(self.pool, 'reserve')['job_id'], '2.prism')

    def test_compute_handoff_and_saved_results_stay_with_original_run(self):
        script = self.base/'code/scripts/run_rpe_pbs_worker.py'
        script.parent.mkdir(parents=True)
        script.write_text("""import json, pathlib, sys, time
p=pathlib.Path(sys.argv[-1])
profile=json.loads(p.read_text())
d=p.parent
work=pathlib.Path(profile['work_dir'])/'artificial'/'original-attempt'
work.mkdir(parents=True)
for name,data in [('ASSIGNMENT',{'job_id':'artificial','attempt_id':'original-attempt','request':{}}),
                  ('WORKER_STATE',{'state':'COMPLETED_PENDING_TRANSFER'}),
                  ('OUTCOME',{'owner':str(d),'artifacts':{}}),('PROCESS_RETURN',{'returncode':0})]:
    (work/(name+'.json')).write_text(json.dumps(data))
while not (d/'CONTROL.json').is_file():
    time.sleep(.02)
""")
        row = pilot.admin(self.pool, 'reserve')
        generation = self.pool/'allocations'/row['token']
        a = self.attach(self.a, 'a1')
        self.jobs['1.prism']['job_state'] = 'R'
        self.jobs['1.prism']['resources_used'] = dict(walltime='01:00:00')

        def wait(path):
            until = time.monotonic()+10
            while not path.exists() and time.monotonic() < until:
                time.sleep(.025)
            self.assertTrue(path.exists(), str(path))

        with patch.dict(os.environ, PBS_JOBID='1.prism'), ThreadPoolExecutor(1) as executor:
            server = executor.submit(pilot.serve, generation)
            try:
                wait(self.a/'workers/pbs-0/artificial/original-attempt/PROCESS_RETURN.json')
                b = self.attach(self.b, 'b1')
                self.assertEqual(pilot.session_snapshot(self.b)[0]['status'], 'Q')
                self.assertFalse((b/'SESSION_START.json').exists())
                self.stop(self.a, 'a1')
                wait(self.b/'workers/pbs-0/artificial/original-attempt/PROCESS_RETURN.json')
                self.assertTrue((a/'SESSION_END.json').exists())
                finished = pilot.session_snapshot(self.a)[0]
                self.assertEqual(finished['session_status'], 'F')
                self.assertEqual(finished['reservation_status'], 'R')
                import base64
                for root, token, directory in [(self.a,'a1',a),(self.b,'b1',b)]:
                    data = pilot.handle(root, dict(action='harvest', token=token), {})
                    outcome = json.loads(base64.b64decode(data['attempts'][0]['files']['OUTCOME.json']))
                    self.assertEqual(outcome['owner'], str(directory))
                    self.assertEqual(G.read(directory/'ALLOCATION.json')['job_id'], '1.prism')
                    self.assertLessEqual(G.read(directory/'ALLOCATION.json')['remaining_seconds'], 23*3600)
                self.assertEqual(len(self.submissions), 1)
            finally:
                pilot.admin(self.pool, 'release')
            server.result(timeout=10)

    def test_unrelated_live_allocation_and_ambiguous_history_do_not_resubmit(self):
        self.jobs['other'] = dict(job_state='R', Job_Name='RPEunrelated')
        with self.assertRaises(ValueError):
            pilot.admin(self.pool, 'reserve')
        self.jobs.clear()
        row = pilot.admin(self.pool, 'reserve')
        self.jobs.clear()
        self.assertEqual(pilot.admin(self.pool, 'reserve')['status'], 'HISTORY_MISSING_RECONCILE_SPOOL')
        self.assertEqual(self.submissions, [row['token']])

    def test_reacquire_inherits_ended_resources_alongside_identified_search_only(self):
        token = pilot.prepare_generation(self.pool, pilot.SUCCESSOR_RESOURCES)
        G.atomic(self.pool/'CURRENT.json', dict(token=token))
        G.submit_once(self.pool/'allocations'/token)
        self.jobs['1.prism']['job_state'] = 'F'
        self.jobs['8.prism'] = dict(job_state='R', Job_Name='RPEsearch-0ee3f')
        self.jobs['9.prism'] = dict(job_state='Q', Job_Name='RPEabcdef012345')

        with self.assertRaises(ValueError):
            pilot.admin(self.pool, 'reacquire', replace_job_id='1.prism', alongside_search_job_id='8.prism')
        del self.jobs['9.prism']
        with self.assertRaises(ValueError):
            pilot.admin(self.pool, 'reacquire', replace_job_id='2.prism', alongside_search_job_id='8.prism')
        self.assertEqual(len(self.submissions), 1)
        new = pilot.admin(self.pool, 'reacquire', replace_job_id='1.prism', alongside_search_job_id='8.prism')
        self.assertEqual((new['job_id'], new['resources']), ('2.prism', pilot.SUCCESSOR_RESOURCES))
        self.assertEqual(pilot.admin(self.pool, 'reserve')['job_id'], '2.prism')
        self.assertEqual(self.jobs['8.prism']['job_state'], 'R')
        self.assertEqual(len(self.submissions), 2)

    def test_failed_worker_exit_keeps_node_for_next_session(self):
        script = self.base/'code/scripts/run_rpe_pbs_worker.py'
        script.parent.mkdir(parents=True)
        script.write_text("import pathlib, sys\nraise SystemExit(int(pathlib.Path(sys.argv[-1]).parent.name == 'a1'))\n")
        a = self.attach(self.a, 'a1')
        b = self.attach(self.b, 'b1')
        generation = self.pool/'allocations'/self.submissions[0]
        self.jobs['1.prism'].update(job_state='R', resources_used=dict(walltime='01:00:00'))
        with patch.dict(os.environ, PBS_JOBID='1.prism'), ThreadPoolExecutor(1) as executor:
            server = executor.submit(pilot.serve, generation)
            try:
                end = b/'SESSION_END.json'
                until = time.monotonic()+5
                while not end.is_file() and time.monotonic() < until:
                    time.sleep(.025)
                self.assertEqual(G.read(a/'SESSION_END.json')['reason'], 'WORKER_ERROR')
                self.assertEqual(G.read(end)['returncode'], 0)
                self.assertEqual(G.read(a/'ALLOCATION.json')['job_id'], G.read(b/'ALLOCATION.json')['job_id'])
                self.assertEqual(len(self.submissions), 1)
            finally:
                pilot.admin(self.pool, 'release')
            server.result(timeout=5)

    def test_staged_72h_handoff_waits_for_allocation_and_old_exit(self):
        first = pilot.admin(self.pool, 'reserve')
        self.jobs['1.prism']['job_state'] = 'R'
        old = self.pool/'allocations'/first['token']
        session = self.attach(self.a, 'oldSession')
        self.assertEqual(pilot.claim(self.pool, old, '1.prism'), session)
        with ThreadPoolExecutor(2) as executor:
            results = list(executor.map(lambda _: pilot.stage_successor(self.pool, '1.prism'), range(2)))
        self.assertEqual([r['job_id'] for r in results], ['2.prism', '2.prism'])
        self.assertEqual(len(self.submissions), 2)
        self.assertEqual(pilot.admin(self.pool, 'status')['job_id'], '1.prism')
        self.assertFalse((old/'RELEASE.json').exists())
        self.assertFalse((session/'CONTROL.json').exists())
        new = self.pool/'allocations'/self.submissions[-1]
        self.assertIn('#PBS -l walltime=72:00:00', (new/'worker.pbs').read_text())
        self.jobs['2.prism']['job_state'] = 'R'
        with patch.dict(os.environ, PBS_JOBID='2.prism'), ThreadPoolExecutor(1) as executor:
            server = executor.submit(pilot.serve, new)
            try:
                until = time.monotonic() + 5
                while not (new/'HANDOVER.json').exists() and time.monotonic() < until:
                    if server.done():
                        server.result()
                    time.sleep(.02)
                self.assertEqual(G.read(new/'HANDOVER.json')['state'], 'DRAINING_PREDECESSOR')
                self.assertTrue(G.read(session/'CONTROL.json')['stop'])
                self.assertIsNone(pilot.claim(self.pool, old, '1.prism'))
                self.assertEqual(pilot.stage_successor(self.pool, '1.prism')['job_id'], '2.prism')
                self.assertFalse((session/'SESSION_END.json').exists())
                another = self.attach(self.b, 'nextSession')
                self.assertEqual(G.read(another/'PILOT.json')['token'], new.name)
                self.assertEqual(pilot.session_snapshot(self.b)[0]['status'], 'Q')
                self.assertFalse((another/'SESSION_START.json').exists())
                self.stop(self.b, 'nextSession')
                pilot.end(session, 'WORKER_EXIT', returncode=0)
                self.jobs['1.prism']['job_state'] = 'F'
                until = time.monotonic() + 8
                while G.read(new/'HANDOVER.json')['state'] != 'ACTIVE' and time.monotonic() < until:
                    time.sleep(.02)
                self.assertEqual(G.read(new/'HANDOVER.json')['old_scheduler_state'], 'F')
                self.assertEqual(pilot.admin(self.pool, 'status')['resources']['walltime'], '72:00:00')
            finally:
                pilot.admin(self.pool, 'release')
            server.result(timeout=7)

    def test_after_exit_reservation_leaves_old_session_and_other_pool_alone(self):
        initial = pilot.admin(self.pool, 'reserve')
        old = self.pool/'allocations'/initial['token']
        spec = G.read(old/'SPEC.json')
        G.atomic(old/'SPEC.json', dict(spec, resources=pilot.SUCCESSOR_RESOURCES))
        self.jobs['1.prism'].update(job_state='R', Resource_List=dict(pilot.SUCCESSOR_RESOURCES))
        self.jobs['9.prism'] = dict(job_state='R', Job_Name='RPEIndependent')
        session = self.attach(self.a, 'oldSession')
        self.assertEqual(pilot.claim(self.pool, old, '1.prism'), session)
        control = dict(stop=True, release_when_idle=True)
        G.atomic(session/'CONTROL.json', control)

        staged = pilot.stage_successor(self.pool, '1.prism', after_exit=True)
        self.assertEqual(staged['job_id'], '2.prism')
        self.assertEqual(pilot.stage_successor(self.pool, '1.prism', after_exit=True)['job_id'], '2.prism')
        new = self.pool/'allocations'/staged['token']
        self.assertIn('#PBS -W depend=afterany:1.prism\n', (new/'worker.pbs').read_text())
        self.assertEqual(pilot.admin(self.pool, 'status')['job_id'], '1.prism')
        self.assertEqual(len(self.submissions), 2)

        self.jobs['2.prism']['job_state'] = 'R'
        with patch.dict(os.environ, PBS_JOBID='2.prism'), ThreadPoolExecutor(1) as executor:
            server = executor.submit(pilot.serve, new)
            try:
                until = time.monotonic()+3
                while not (new/'NODE.lock').exists() and time.monotonic() < until:
                    time.sleep(.02)
                self.assertFalse((new/'HANDOVER.json').exists())
                self.assertEqual(G.read(self.pool/'CURRENT.json')['token'], old.name)
                self.jobs['1.prism']['job_state'] = 'F'
                until = time.monotonic()+8
                while not (new/'HANDOVER.json').exists() and time.monotonic() < until:
                    time.sleep(.02)
                self.assertEqual(G.read(new/'HANDOVER.json')['old_scheduler_state'], 'F')
                self.assertEqual(G.read(self.pool/'CURRENT.json')['token'], new.name)
                self.assertEqual(G.read(session/'CONTROL.json'), control)
                self.assertFalse((old/'RELEASE.json').exists())
                self.assertEqual(self.jobs['9.prism']['job_state'], 'R')
            finally:
                G.atomic(new/'RELEASE.json', {})
            server.result(timeout=7)

    def test_available_successor_replaces_only_held_reservation_and_keeps_old_running(self):
        first = pilot.admin(self.pool, 'reserve')
        old = self.pool/'allocations'/first['token']
        self.jobs['1.prism']['job_state'] = 'R'
        session = self.attach(self.a, 'oldSession')
        self.assertEqual(pilot.claim(self.pool, old, '1.prism'), session)
        pilot.stage_successor(self.pool, '1.prism', after_exit=True)
        self.jobs['2.prism'].update(job_state='H', depend='afterany:1.prism@prism')
        replacement = pilot.stage_successor(self.pool, '1.prism', when_available=True)
        self.assertEqual(replacement['job_id'], '3.prism')
        self.assertEqual(pilot.stage_successor(self.pool, '1.prism', when_available=True)['job_id'], '3.prism')
        self.assertEqual(self.jobs['2.prism']['job_state'], 'F')
        new = self.pool/'allocations'/replacement['token']
        self.assertNotIn('depend=', (new/'worker.pbs').read_text())
        self.assertEqual(G.read(self.pool/'CURRENT.json')['token'], old.name)

        self.jobs['3.prism']['job_state'] = 'R'
        with patch.dict(os.environ, PBS_JOBID='3.prism'), ThreadPoolExecutor(1) as executor:
            server = executor.submit(pilot.serve, new)
            try:
                until = time.monotonic()+3
                while not (new/'HANDOVER.json').exists() and time.monotonic() < until:
                    time.sleep(.02)
                self.assertEqual(G.read(new/'HANDOVER.json')['retained_predecessor'], old.name)
                self.assertEqual(G.read(self.pool/'CURRENT.json')['token'], new.name)
                self.assertEqual(self.jobs['1.prism']['job_state'], 'R')
                self.assertFalse((old/'RELEASE.json').exists())
                self.assertFalse((session/'CONTROL.json').exists())
                G.atomic(session/'CONTROL.json', dict(stop=True))
                old_row, = pilot.session_snapshot(self.a)
                self.assertEqual(old_row['retained_replacement'], new.name)
                reason = dict(retained_replacement=dict(pilot_token=new.name, predecessor=old.name))
                with patch.object(pilot, 'claim', return_value=None):
                    attached = self.attach(self.a, 'newSession', reason)
                    self.assertEqual(G.read(attached/'PROFILE.json')['work_dir'],
                                     str(pilot.successor_work_dir(self.a, new)))
                self.assertEqual(G.read(session/'PROFILE.json')['work_dir'], str(self.a/'workers'/'pbs-0'))
            finally:
                G.atomic(new/'RELEASE.json', {})
            server.result(timeout=3)

    def test_wrong_successor_walltime_preserves_old_and_purged_history_needs_unlock(self):
        first = pilot.admin(self.pool, 'reserve')
        self.jobs['1.prism']['job_state'] = 'R'
        with self.assertRaises(ValueError):
            pilot.stage_successor(self.pool, 'wrong.prism')
        staged = pilot.stage_successor(self.pool, '1.prism')
        old = self.pool/'allocations'/first['token']
        new = self.pool/'allocations'/staged['token']
        self.jobs['2.prism']['Resource_List']['walltime'] = '24:00:00'
        with patch.dict(os.environ, PBS_JOBID='2.prism'), self.assertRaises(ValueError):
            pilot.serve(new)
        self.assertFalse((old/'RELEASE.json').exists())
        self.assertEqual(G.read(self.pool/'CURRENT.json')['token'], old.name)
        with patch.dict(os.environ, PBS_JOBID='1.prism'):
            G.record_allocation(old)
        del self.jobs['1.prism']
        predecessor = G.read(new/'SPEC.json')['predecessor']
        with (old/'NODE.lock').open('a') as lock:
            pilot.fcntl.flock(lock, pilot.fcntl.LOCK_EX)
            self.assertIsNone(pilot.predecessor_finished(self.pool, predecessor))
        self.assertEqual(pilot.predecessor_finished(self.pool, predecessor),
                         'ABSENT_FROM_LIVE_QUEUE_AND_NODE_LOCK_FREE')

    def test_offline_drain_requires_both_files_and_leaves_ack_to_controller(self):
        from trottertracks import rpe_pbs_worker as module
        with patch.object(module, 'AllocationWorker', module.AllocationWorker):
            collector.configure_allocation_worker({'pbs':{}}, Path('worker.py'))
            worker = object.__new__(module.AllocationWorker)
            worker.directory = self.base
            worker.draining = True
            attempt = self.base/'attempt'
            worker.records = {attempt:{}}
            G.atomic(self.base/'PILOT.json', {})
            G.atomic(self.base/'CONTROL.json', dict(stop=True))
            G.atomic(attempt/'WORKER_STATE.json', dict(state='COMPLETED_PENDING_TRANSFER'))
            G.atomic(attempt/'OUTCOME.json', {})
            self.assertTrue(worker.pending())
            G.atomic(attempt/'PROCESS_RETURN.json', {})
            self.assertFalse(worker.pending())
            self.assertEqual(G.read(attempt/'WORKER_STATE.json')['state'], 'COMPLETED_PENDING_TRANSFER')


def test_managed_rejection_requires_return_receipt_and_checked_history(tmp_path):
    pool = tmp_path/'pool'
    generation = pool/'allocations'/'test-token'
    generation.mkdir(parents=True)
    G.atomic(generation/'SPEC.json', {'run_managed':True})
    receipt = {'returncode':1, 'epoch':0, 'stdout':'', 'stderr':'artificial rejection'}
    row = {'status':'SUBMIT_REJECTED', 'receipt':receipt, 'job_id':None}
    with patch.object(G, 'snapshot', return_value=[row.copy()]), \
         patch.object(pilot, 'owned_jobs_by_name', return_value={}):
        assert pilot.physical(pool, generation.name)['status'] == 'SUBMIT_LOST'
    with patch.object(G, 'snapshot', return_value=[row.copy()]), \
         patch.object(pilot, 'owned_jobs_by_name', side_effect=RuntimeError('scheduler unavailable')):
        with pytest.raises(RuntimeError):
            pilot.physical(pool, generation.name)
    with patch.object(G, 'snapshot', return_value=[{'status':'SUBMIT_AMBIGUOUS_HOLD','receipt':None}]), \
         patch.object(pilot, 'owned_jobs_by_name', side_effect=AssertionError('no absence proof')):
        assert pilot.physical(pool, generation.name)['status'] == 'SUBMIT_AMBIGUOUS_HOLD'


def test_managed_rejected_generation_allows_new_session_token(tmp_path, monkeypatch):
    root = tmp_path/'run'
    submits = []
    def submit(directory):
        if (directory/'QSUB_STARTED.json').exists():
            return
        submits.append(directory.name)
        G.atomic(directory/'QSUB_STARTED.json', {'epoch':0})
        G.atomic(directory/'QSUB_RETURN.json', dict(returncode=1, stdout='', stderr='fixture', epoch=0))
    def spec(token):
        return dict(token=token, resources=pilot.RESOURCES, profile=dict(host_id='pbs-0',
            work_dir=str(root/'workers'/'pbs-0')), python=sys.executable, transport_root=str(tmp_path))
    monkeypatch.setattr(G, 'submit_async', submit)
    monkeypatch.setattr(pilot, 'owned_jobs_by_name', lambda token:{})
    first = pilot.attach_managed(root, spec('first'))
    rows = pilot.session_snapshot(root)
    assert rows[0]['status'] == 'F' and rows[0]['reservation_status'] == 'SUBMIT_LOST'
    assert rows[0]['session_end']['reason'] == 'SUBMIT_REJECTED_BEFORE_SESSION_START'
    generation = pilot.managed_pool(root, 'pbs-0')/'allocations'/first['pilot']
    assert (generation/'RELEASE.json').exists() and (generation/'SUBMIT_LOST.json').exists()
    second = pilot.attach_managed(root, spec('second'))
    assert second['pilot'] != first['pilot']
    assert len(submits) == 2
    assert pilot.attach_managed(root, spec('second'))['pilot'] == second['pilot']
    assert len(submits) == 2


def test_fence_unstarted_requires_reaped_session_and_no_launch_marker(tmp_path):
    root = tmp_path/'run'; directory = root/'allocations'/'session'
    generation = pilot.managed_pool(root,'pbs-0')/'allocations'/'generation'
    work = root/'workers'/'pbs-0'; attempt = work/'job'/'attempt'
    attempt.mkdir(parents=True)
    (work/'WORKER.lock').touch()
    G.atomic(directory/'PILOT.json', {'managed_host':'pbs-0','token':'generation'})
    G.atomic(generation/'sessions'/'session.json', {'directory':str(directory)})
    G.atomic(directory/'PROFILE.json', {'host_id':'pbs-0','work_dir':str(work)})
    identity = dict(job_id='job', attempt_id='attempt',request_sha256='fixed')
    G.atomic(attempt/'ASSIGNMENT.json',dict(identity,request={'fixture':True}))
    G.atomic(attempt/'ORIGINAL_REQUEST.json',{'fixture':True})
    G.atomic(attempt/'WORKER_PROFILE.json',{'pbs':{'allocation_dir':str(directory)}})
    G.atomic(attempt/'WORKER_STATE.json',{'state':'ASSIGNED','detail':{}})
    G.atomic(directory/'SESSION_START.json',{'job_id':'1.fixture','epoch':1})
    G.atomic(directory/'ALLOCATION.json',{'job_id':'1.fixture'})
    value = dict(identity,token='session')
    assert pilot.fence_unstarted(root,value) is None
    G.atomic(directory/'SESSION_END.json',{'reason':'WORKER_ERROR','returncode':1,'epoch':2})
    (attempt/'SUPERVISOR_LAUNCH.json').touch()
    assert pilot.fence_unstarted(root,value) is None
    (attempt/'SUPERVISOR_LAUNCH.json').unlink()
    successor = root/'allocations'/'successor'
    G.atomic(generation/'sessions'/'successor.json',{'directory':str(successor)})
    G.atomic(successor/'SESSION_START.json',{'job_id':'1.fixture','epoch':3})
    assert pilot.fence_unstarted(root,value) is None
    G.atomic(successor/'SESSION_END.json',{'reason':'WORKER_EXIT','returncode':0,'epoch':4})
    with (work/'WORKER.lock').open('rb') as lock:
        pilot.fcntl.flock(lock,pilot.fcntl.LOCK_EX | pilot.fcntl.LOCK_NB)
        assert pilot.fence_unstarted(root,value) is None
    fenced = pilot.fence_unstarted(root,value)
    assert fenced['state'] == 'UNKNOWN'
    assert fenced['detail']['pbs_admission_exit']['kind']=='KEEPER_SESSION_ENDED_BEFORE_LAUNCH'
    assert pilot.fence_unstarted(root,value) == fenced
    assert pilot.fence_unstarted(root,dict(value,request_sha256='different')) is None


def test_positive_terminal_scheduler_receipt_is_exact_and_never_a_retry(tmp_path):
    directory=tmp_path/'session'; attempt=tmp_path/'attempt'; attempt.mkdir()
    G.atomic(directory/'ALLOCATION.json',{'job_id':'1.fixture'})
    identity=dict(job_id='j',attempt_id='a',request_sha256='fixed')
    G.atomic(attempt/'PBS_ATTEMPT_OWNER.json',dict(identity,allocation_dir=str(directory),token='session',pbs_job_id='1.fixture',supervisor_pid=42))
    G.atomic(attempt/'SUPERVISOR_STARTED.json',{'pid':42})
    assert G.record_scheduler_end(directory,'1.fixture',{'job_state':'R','Job_Name':'RPEgeneration'},physical_token='generation') is None
    assert G.scheduler_attempt_exit(directory,attempt,identity) is None
    assert G.record_scheduler_end(directory,'1.fixture',{'job_state':'F','Job_Name':'RPEwrong'},physical_token='generation') is None
    receipt=G.record_scheduler_end(directory,'1.fixture',{'job_state':'F','Job_Name':'RPEgeneration'},physical_token='generation')
    proof=G.scheduler_attempt_exit(directory,attempt,identity)
    assert proof['scheduler_allocation_end']==receipt and proof['session_ended']
    assert G.scheduler_attempt_exit(directory,attempt,dict(identity,attempt_id='different')) is None


def test_managed_submit_returns_before_qsub_and_named_recovery(tmp_path,monkeypatch):
    directory=tmp_path/'allocations'/'token'; directory.mkdir(parents=True)
    G.atomic(directory/'SPEC.json',{'token':'token','run_managed':True,'profile':{'host_id':'pbs-0'}})
    with patch.object(G.subprocess,'Popen') as child:
        G.submit_async(directory)
        assert child.call_args.args[0][-2:]==['--submit',str(directory)]
        assert child.call_args.kwargs['start_new_session'] is True
        G.atomic(directory/'QSUB_STARTED.json',{'epoch':1})
        G.submit_async(directory)
        assert child.call_count==1
    with patch.object(G,'qstat',side_effect=AssertionError('global qstat prohibited')), \
         patch.object(pilot,'owned_jobs_by_name',return_value={}) as named:
        assert G.snapshot(tmp_path)[0]['status']=='SUBMIT_AMBIGUOUS_HOLD'
        named.assert_called_once_with('token')
    matching={'1.fixture':{'Job_Name':'RPEtoken','job_state':'Q'}}
    with patch.object(pilot,'owned_jobs_by_name',return_value=matching):
        assert G.snapshot(tmp_path)[0]['job_id']=='1.fixture'
    assert G.read(directory/'JOB_ID.json')['recovered'] is True


def test_direct_unstarted_fence_requires_positive_terminal_identity(tmp_path):
    root=tmp_path/'run';directory=root/'allocations'/'token';work=root/'workers'/'pbs';attempt=work/'job'/'attempt'
    attempt.mkdir(parents=True);(work/'WORKER.lock').touch()
    G.atomic(directory/'SPEC.json',{'token':'token','profile':{'host_id':'pbs'}})
    G.atomic(directory/'PROFILE.json',{'host_id':'pbs','work_dir':str(work)})
    G.atomic(directory/'ALLOCATION.json',{'job_id':'1.fixture'});G.atomic(directory/'JOB_ID.json',{'job_id':'1.fixture'})
    identity=dict(job_id='job',attempt_id='attempt',request_sha256='fixture')
    G.atomic(attempt/'ASSIGNMENT.json',dict(identity,request={'fixture':True}))
    G.atomic(attempt/'ORIGINAL_REQUEST.json',{'fixture':True})
    G.atomic(attempt/'WORKER_PROFILE.json',{'pbs':{'allocation_dir':str(directory)}})
    G.atomic(attempt/'WORKER_STATE.json',{'state':'ASSIGNED','detail':{}})
    value=dict(identity,token='token')
    assert pilot.fence_legacy_unstarted(root,value) is None
    G.record_scheduler_end(directory,'1.fixture',{'job_state':'F','Job_Name':'RPEtoken'})
    (attempt/'SUPERVISOR_LAUNCH.json').touch()
    assert pilot.fence_legacy_unstarted(root,value) is None
    (attempt/'SUPERVISOR_LAUNCH.json').unlink()
    got=pilot.fence_legacy_unstarted(root,value)
    assert got['detail']['pbs_admission_exit']['kind']=='DIRECT_ALLOCATION_ENDED_BEFORE_LAUNCH'
    assert pilot.fence_legacy_unstarted(root,value)==got


def test_managed_afterany_successor_is_same_resource_and_idempotent(tmp_path, monkeypatch):
    root = tmp_path/'run'
    submits = []
    monkeypatch.setattr(G, 'submit_async', lambda directory: submits.append(directory.name))
    spec = dict(token='session', resources=dict(pilot.RESOURCES, walltime='11:00:00'),
                profile=dict(host_id='pbs-0'), python=sys.executable, transport_root=str(tmp_path))
    attached = pilot.attach_managed(root, spec)
    pool = pilot.managed_pool(root, 'pbs-0')
    old = pool/'allocations'/attached['pilot']
    G.atomic(old/'ALLOCATION.json', {'job_id':'1.fixture'})
    monkeypatch.setattr(pilot, 'physical', lambda pool, token, **kw:
        dict(job_id='1.fixture',status='R') if token==old.name else dict(job_id=None,status='SUBMIT_PENDING'))
    request = dict(token='session', job_id='1.fixture')
    first = pilot.stage_managed_successor(root, request)
    second = pilot.stage_managed_successor(root, request)
    assert first == second
    successor = pool/'allocations'/first['token']
    assert G.read(successor/'SPEC.json')['resources'] == spec['resources']
    assert '#PBS -W depend=afterany:1.fixture' in (successor/'worker.pbs').read_text()
    assert not list(pilot.sessions(successor))
    assert G.read(pool/'CURRENT.json')['token'] == old.name
    G.atomic(root/'allocations'/'session'/'CONTROL.json', {'stop':True})
    with pytest.raises(ValueError, match='not accepting'):
        pilot.stage_managed_successor(root, request)


def test_managed_successor_explicit_release_fences_unsubmitted_target(tmp_path, monkeypatch):
    root = tmp_path/'run'
    monkeypatch.setattr(G,'submit_async',lambda directory:None)
    spec = dict(token='session', resources=pilot.RESOURCES, profile=dict(host_id='pbs-0'),
                python=sys.executable, transport_root=str(tmp_path))
    attached = pilot.attach_managed(root,spec)
    pool = pilot.managed_pool(root,'pbs-0')
    old = pool/'allocations'/attached['pilot']
    G.atomic(old/'ALLOCATION.json',{'job_id':'1.fixture'})
    monkeypatch.setattr(pilot,'physical',lambda pool,token,**kw:
        dict(job_id='1.fixture',status='R') if token==old.name else dict(job_id=None,status='SUBMIT_PENDING'))
    result = pilot.stage_managed_successor(root,dict(token='session',job_id='1.fixture'))
    successor = pool/'allocations'/result['token']
    G.atomic(root/'allocations'/'session'/'SESSION_END.json',{'returncode':0})
    pilot.release_managed_node(root,'pbs-0')
    assert (old/'RELEASE.json').exists() and (successor/'RELEASE.json').exists()
    with patch.object(G.subprocess,'run',side_effect=AssertionError('no scheduler')):
        G.submit_once(successor)
    assert not (successor/'QSUB_STARTED.json').exists()
