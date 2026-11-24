import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from recover_stopped_rpe_run import (PBS_PRELAUNCH_PROBE, accept, check_assignment, owner,
                                    pbs_results, probe_local, publish_received, read_rows, repair_dependencies,
                                    recover_live_attempts)
from run_rpe_central_maintenance import archive_candidates, publish_due, run_pass, wait_summary
from trottertracks.rpe_completion_acceptance import validate_preparation_result
from trottertracks.rpe_local0_search import supervise_run
from trottertracks.rpe_preparation_artifacts import ArtifactCache
from trottertracks.rpe_q3_store import JobStore


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def encoded(value):
    return base64.b64encode(json.dumps(value).encode()).decode()


class CentralMaintenanceTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.run = Path(temp.name)/'search-artificial'
        self.store = JobStore(self.run/'distributed/QUEUE.sqlite3')
        self.cache = ArtifactCache(self.run/'distributed/artifacts')
        self.source = {'commit':'a'*40,'files':{}}

    def reserve(self, job, stage, **settings):
        request = dict(route={'kind':'preparation'}, preparation=dict(stage=stage,source=self.source,artificial=True,**settings))
        self.assertTrue(self.store.reserve('PREPARATION:'+stage,job,request,2,True))
        write(self.run/'jobs'/('PREPARATION:'+stage)/job/'REQUEST.json',request)
        return request

    def test_live_recovery_delivers_to_existing_owner(self):
        launch = dict(pid=123)
        write(self.run/'LAUNCH.json', launch)
        write(self.run/'distributed/ENDPOINT.json', {'url': 'http://127.0.0.1:8763'})
        row = dict(state='UNKNOWN', job_id='q3', attempt_id='existing', host_id='local0-wsl',
                   request_sha256='request')
        files = {'OUTCOME.json': encoded({'status': 'fixture'})}
        observed = [(row, {'state': 'COMPLETED_PENDING_TRANSFER'}, files)]
        handshake = unittest.mock.mock_open(read_data='{"ok": true}')
        with patch('recover_stopped_rpe_run.worker_results', return_value=iter(observed)), \
             patch('recover_stopped_rpe_run.urlopen', handshake) as send:
            result = recover_live_attempts(self.run, launch, [row], self.cache)
        request = send.call_args.args[0]
        self.assertTrue(request.full_url.endswith('/v1/complete'))
        self.assertEqual(json.loads(request.data), dict(owner(row), files=files))
        self.assertEqual(result[0]['recovery'], 'LIVE_CONTROLLER_RECONCILED')
        self.assertIsNone(self.store.row('q3'))

    def test_receive_existing_attempt_without_new_execution_or_adoption(self):
        self.reserve('compile','compile')
        row, = self.store.claim('local',1)
        check_assignment(row,dict(owner(row),request=row['request']))
        with self.assertRaisesRegex(ValueError,'different|differs'):
            check_assignment(row, dict(owner(row), attempt_id='other'))
        payload = b'previously compiled artificial circuit'
        sha = hashlib.sha256(payload).hexdigest()
        self.cache.put({sha:base64.b64encode(payload).decode()})
        outcome = dict(request_id='compile',request_sha256=row['request_sha256'],stage='compile',
                       source_manifest=self.source,status='PREPARATION_COMPLETE',artifacts={'circuit':sha})
        clean = dict(returncode=0,signal=None)
        files = {'OUTCOME.json':encoded(outcome),'PROCESS_RETURN.json':encoded(clean),
                 'PREPARATION_RECEIPT.json':encoded(clean),'ARTIFACT_MANIFEST.json':encoded({'circuit':sha})}
        args = (self.store,self.cache,validate_preparation_result,{'code_root':str(ROOT)})
        incomplete = dict(files, **{'PROCESS_RETURN.json':encoded({'returncode':None,'signal':9})})
        self.assertEqual(accept(self.run,row,incomplete,*args), 'HELD_NO_CLEAN_RECEIPT')
        self.assertEqual(self.store.row('compile')['state'], 'ASSIGNED')
        self.assertEqual(accept(self.run,row,files,*args), 'RECEIVED')
        self.assertEqual(accept(self.run,row,files,*args), 'ALREADY_RECEIVED')
        accepted = self.store.row('compile')
        self.assertIsNone(accepted['adopted_epoch'])
        self.assertEqual(accepted['attempt_id'], row['attempt_id'])
        self.assertEqual([event['event'] for event in self.store.events()].count('ASSIGN'),1)

        result = self.run/'jobs/PREPARATION:compile/compile/OUTCOME.json'
        result.unlink()  # Interrupt only the test between ACCEPT and publish.
        self.assertTrue(publish_received(self.run,accepted))
        self.assertEqual(json.loads(result.read_bytes()),outcome)
        restored = result.parent/'artifacts/circuit'
        restored.parent.mkdir()
        restored.write_bytes(payload)
        self.cache.path(sha).unlink()
        self.reserve('audit','audit',dependency_inputs={'compile':{'circuit':'circuit'}})
        repaired = repair_dependencies(self.run,read_rows(self.run),self.cache)
        self.assertEqual(repaired,[dict(job_id='audit',predecessor='compile',sha256=sha)])
        self.assertEqual(self.cache.path(sha).read_bytes(),payload)
        self.assertEqual(self.store.row('audit')['state'],'QUEUED')

    def test_mailbox_observation_does_not_start_an_assigned_attempt(self):
        self.reserve('compile','compile')
        row, = self.store.claim('local',1)
        profile = self.run/'profile.json'
        spool = self.run/'spool'
        write(profile,dict(host_id='local',work_dir=str(spool)))
        attempt = spool/'compile'/row['attempt_id']
        write(attempt/'ASSIGNMENT.json',dict(owner(row),request=row['request']))
        write(attempt/'WORKER_STATE.json',{'state':'ASSIGNED'})
        before = sorted(p.name for p in attempt.iterdir())
        self.assertEqual(probe_local(profile,row)['state'],'ASSIGNED')
        self.assertEqual(sorted(p.name for p in attempt.iterdir()),before)
        self.assertEqual(self.store.row('compile')['state'],'ASSIGNED')

    def test_pbs_collection_uses_only_snapshot_and_harvest(self):
        self.reserve('compile','compile')
        row, = self.store.claim('pbs-0',1)
        root = self.run/'distributed/pbs'
        write(root/'SETTINGS.json',{'gateway_command':['artificial']})
        write(root/'STATE.json',dict(allocations={'allocation':{'spec':{'profile':{'host_id':'pbs-0'}}}}))
        assignment = dict(owner(row),request=row['request'])
        files = {'OUTCOME.json':encoded({'artifacts':{}})}
        with patch('trottertracks.rpe_pbs.Gateway') as gateway:
            connection = gateway.return_value
            connection.call.side_effect = [
                {'allocations':[{'token':'allocation','status':'F'}]},
                {'host_id':'pbs-0','attempts':[{'assignment':assignment,'files':files}]}]
            recovered, = pbs_results(self.run,[row],self.cache)
            self.assertEqual(recovered[2],files)
            self.assertEqual([call.args[0]['action'] for call in connection.call.call_args_list],['snapshot','harvest'])
            self.assertEqual(connection.call.call_args.args[0]['owners'],
                             [dict(job_id='compile', attempt_id=row['attempt_id'])])
        self.assertEqual(self.store.row('compile')['state'],'ASSIGNED')

    def test_wait_summary_and_archiving_reject_incomplete_invocation(self):
        self.reserve('signal','signal')
        write(self.run/'LAUNCH.json',{'invocation_id':'new'})
        write(self.run/'EXIT.json',{'invocation_id':'old','returncode':0})
        (self.run/'PROGRESS.jsonl').write_text(json.dumps(dict(event='QUEUE_WAIT_STATE',
              reasons={'signal':'PREPARATION_BUDGET_HOLD'},hosts={'pbs-0':{'idle_reason':'PREPARATION_BUDGET_HOLD'}}))+'\n')
        self.assertEqual(wait_summary(self.run,{})['queue_wait_reasons'],{'PREPARATION_BUDGET_HOLD':1})
        self.assertEqual(archive_candidates([self.run],None,{self.run:read_rows(self.run)})['count'],0)

    def test_legacy_run_publishes_with_the_current_resume_publisher(self):
        scripts = self.run.parent/'deployment/scripts'
        scripts.mkdir(parents=True)
        names = ('publish_romutail_search.py', 'publish_h4_search_run.py')
        hashes = {}
        for name in names:
            (scripts/name).write_text(name)
            hashes[name] = hashlib.sha256(name.encode()).hexdigest()
        config = dict(code_source='fixed', python=sys.executable,
                      publication=dict(repository='/fixture/results', branch='results', push=True))
        write(self.run/'CONTROLLER_CONFIG.json', config)
        write(self.run/'ROUND_RESULT.json', {})
        selection = dict(run_id=self.run.name, root=str(scripts.parent), base_source='fixed',
                         source='operations', publisher_sha256=hashes)
        command = [sys.executable, '-B', str(scripts/'run_rpe_local0_search.py'),
                   '--execute-run', str(self.run), '--invocation', 'fixture']
        marker = self.run/'LAUNCH.json'
        write(marker, dict(command=command, invocation_id='fixture'))
        with patch('run_rpe_central_maintenance.subprocess.run') as run:
            self.assertEqual(publish_due(self.run, active=True, previous={}), {})
            run.assert_not_called()
            write(marker, dict(command=command+['--schedule-resume', json.dumps(selection)], invocation_id='fixture'))
            run.return_value.stdout = '{"status":"PUBLISHED","head":"fixture-result"}'
            status = publish_due(self.run, active=True, previous={})
            self.assertEqual(status['status'], 'PUBLISHED')
            self.assertEqual(run.call_args.args[0][2], str(scripts/names[0]))
            self.assertEqual(run.call_args.args[0][-3:], ['--operations-source', 'operations', '--push'])
            self.assertEqual(publish_due(self.run, active=True, previous=status), status)
            run.assert_called_once()
            (scripts/names[0]).write_text('different')
            self.assertEqual(publish_due(self.run, active=True, previous={})['status'], 'PUBLICATION_PENDING')
            run.assert_called_once()

    def test_publication_only_keeps_queue_and_archive_maintenance_out_of_the_pass(self):
        state = self.run.parent
        write(state/'RUNS.json', {self.run.name: {'run_directory': str(self.run)}})
        write(state/'CURRENT_TARGET.json', {'run_id': self.run.name})
        report = state/'maintenance'/(self.run.name+'.json')
        write(report, {'status': 'EXISTING_RECOVERY_RECORD'})
        with patch('run_rpe_central_maintenance.publish_due',
                   return_value={'status': 'PUBLISHED'}) as publish, \
             patch('run_rpe_central_maintenance.subprocess.run') as process:
            result = run_pass(state, publication_only=True)
        process.assert_not_called()
        publish.assert_called_once_with(self.run, active=False, previous={})
        self.assertEqual(result['publication_checks'][self.run.name]['status'], 'PUBLISHED')
        self.assertEqual(result['new_scientific_calls'], 0)
        self.assertEqual(json.loads(report.read_bytes())['status'], 'EXISTING_RECOVERY_RECORD')

        other = state/'search-other'
        other.mkdir()
        write(state/'RUNS.json', {self.run.name: {'run_directory': str(self.run)},
                                  other.name: {'run_directory': str(other)}})
        write(state/'CURRENT_TARGET.json', {'run_id': other.name})
        with patch('run_rpe_central_maintenance.live_driver',
                   side_effect=lambda run: run == self.run), \
             patch('run_rpe_central_maintenance.publish_due', return_value={}) as publish:
            result = run_pass(state, publication_only=True)
        self.assertEqual(set(result['publication_checks']), {self.run.name, other.name})
        publish.assert_any_call(self.run, active=True, previous={'status': 'PUBLISHED'})

    def test_legacy_pbs_timeout_does_not_hide_another_allocation_result(self):
        from queue import Empty

        self.reserve('compile', 'compile')
        first, = self.store.claim('pbs-0', 1)
        self.reserve('audit', 'audit')
        second, = self.store.claim('pbs-1', 1)
        root = self.run/'distributed/pbs'
        write(root/'SETTINGS.json', {'gateway_command':['artificial']})
        write(root/'STATE.json', dict(allocations={
            'first': {'spec':{'profile':{'host_id':'pbs-0'}}},
            'second': {'spec':{'profile':{'host_id':'pbs-1'}}},
        }))
        files = {'OUTCOME.json':encoded({'artifacts':{}})}
        assignment = dict(owner(second), request=second['request'])
        with patch('trottertracks.rpe_pbs.Gateway') as gateway:
            connection = gateway.return_value
            connection.call.side_effect = [
                {'allocations':[{'token':'first','status':'F'}, {'token':'second','status':'F'}]},
                Empty(),
                {'host_id':'pbs-1','attempts':[{'assignment':assignment,'files':files}]},
            ]
            results = {row['job_id']: (status, result)
                       for row, status, result in pbs_results(self.run, [first,second], self.cache)}
            self.assertEqual(results['compile'][0]['state'], 'PBS_RECOVERY_PENDING')
            self.assertIsNone(results['compile'][1])
            self.assertEqual(results['audit'][1], files)
            self.assertEqual(connection.close.call_count, 2)

    def test_old_pbs_archive_observes_original_prelaunch_receipt_without_launch(self):
        self.reserve('audit','audit')
        row, = self.store.claim('pbs-0', 1)
        root = self.run/'distributed/pbs'
        spool = self.run/'pbs-spool'
        write(root/'SETTINGS.json',dict(gateway_command=['artificial'], ssh_alias='artificial',
                                        gateway_python=sys.executable))
        write(root/'STATE.json',dict(allocations={'allocation': {'status':'F',
             'spec': {'profile': {'host_id':'pbs-0', 'work_dir':str(spool)}}}}))
        detail = dict(automatic_retry=False, resource_released=True,
                      reason='ValueError: Existing adapter dry-check failed')
        attempt = spool/'audit'/row['attempt_id']
        write(attempt/'ASSIGNMENT.json', dict(owner(row), request=row['request']))
        write(attempt/'WORKER_STATE.json',dict(state='UNKNOWN',detail=detail))
        write(attempt/'WORKER_DRY-CHECK_RETURN.json',dict(returncode=2,signal=None))
        for name in ('SUPERVISOR_STARTED.json','SUPERVISOR_LAUNCH.json'):
            write(attempt/name,dict(pid=314))
        (attempt/'WORKER_PROCESS.lock').touch()
        self.store.update('pbs-0', 'audit', row['attempt_id'], 'UNKNOWN', detail)
        row = self.store.row('audit')
        run_python = subprocess.check_output
        with (patch('trottertracks.rpe_pbs.Gateway') as gateway,
              patch('recover_stopped_rpe_run.subprocess.check_output', side_effect=lambda _, **kw:
                    run_python([sys.executable,'-B','-c',PBS_PRELAUNCH_PROBE], **kw))):
            gateway.return_value.call.side_effect = [
                {'allocations':[{'token':'allocation','status':'F'}]},
                {'host_id':'pbs-0','attempts':[],'incomplete':[]}]
            (_, observed, files), = pbs_results(self.run, [row], self.cache)
            self.assertIsNone(files)
            self.assertEqual(observed['prelaunch_detail']['prelaunch_rejection']['attempt_id'], row['attempt_id'])
        self.assertEqual(self.store.row('audit'), row)

    def test_supervisor_uses_the_actual_child_wait_status(self):
        state = self.run.parent/'state'
        invocation = '6'*32
        history = self.run/'invocations'/invocation
        write(self.run/'CONTROLLER_CONFIG.json', dict(state_directory=str(state)))
        write(self.run/'LAUNCH.json', dict(invocation_id=invocation))
        command = [sys.executable, '-B', '-c', 'import os, signal; os.kill(os.getpid(), signal.SIGKILL)']
        self.assertEqual(supervise_run(self.run, invocation, command), -9)
        ended = json.loads((history/'EXIT.json').read_text())
        self.assertEqual(ended['returncode'], -9)
        self.assertEqual(ended['command'], command)
        self.assertEqual(json.loads((self.run/'EXIT.json').read_text()), ended)


if __name__ == '__main__':
    unittest.main()
