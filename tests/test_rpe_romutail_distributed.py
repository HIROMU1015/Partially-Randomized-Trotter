"""Metadata-only preparation of the actual-host artificial acceptance harness."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout
import io

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src'),str(ROOT/'tests')]
import check_rpe_romutail_distributed as H


class RealHostFixtureTests(unittest.TestCase):
    def test_prepare_is_artificial_only_and_uses_remote_mailboxes_without_server1(self):
        source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        deployment=dict(source_commit=source,hosts={host:dict(root=str(ROOT)if host=='local'else'/SYNTHETIC/'+host+'/source',
            python=sys.executable,directory='/SYNTHETIC/'+host+'/dedicated-state',ssh_alias='qg2'if host=='server2'else'Romulaptop',
            **(dict(transport='ssh-wsl',distribution='Ubuntu',participation_wsl_file='/root/.romutail-controller-acceptance/settings.json')if host=='local0-wsl'else{}))
            for host in H.HOSTS})
        with tempfile.TemporaryDirectory()as temporary:
            base=Path(temporary)/'fixture'
            with patch.object(H.F,'free_port',return_value=19876):
                result=H.prepare(base,deployment)
            self.assertEqual(result['status'],'PREPARED')
            self.assertFalse((base/'run').exists())
            round=json.loads((base/'ROUND.json').read_text())
            self.assertTrue(round['artificial_fixture']['only'])
            self.assertEqual(round['artificial_preparation']['compile_host'],'server2')
            self.assertEqual(round['artificial_preparation']['audit_host'],'local0-wsl')
            dist=round['executor']['distributed']
            self.assertNotIn('max_concurrent_jobs',round['executor'])
            self.assertEqual(dist['auto_allocation']['discovery_grace_seconds'],0)
            self.assertEqual(dist['affinity_wait_seconds'],0)
            self.assertNotIn('server1',dist.get('required_hosts',[]))
            relay=json.loads((base/'RELAY.json').read_text())
            self.assertTrue(relay['external_tunnel'])
            workers={row['host_id']:row for row in relay['workers']}
            self.assertFalse(workers['server1']['enabled'])
            self.assertEqual(workers['local0-wsl']['transport'],'ssh-wsl')
            self.assertFalse(json.loads((base/'HOST_ADMISSION.json').read_text())['hosts']['local0-wsl']['accepting'])

    def test_participation_writer_rejects_the_real_worker_settings(self):
        worker=dict(participation_wsl_file='/root/projects/existing/.local0-wsl/SETTINGS.json')
        with self.assertRaisesRegex(ValueError,'dedicated synthetic'):
            H.synthetic_participation(worker,True)

    def test_synthetic_participation_code_writes_valid_json_and_only_owned_path(self):
        with tempfile.TemporaryDirectory()as temporary:
            path=Path(temporary)/'.romutail-controller-acceptance/settings.json'
            path.parent.mkdir()
            worker=dict(python=sys.executable,participation_wsl_file=str(path),ssh_alias='synthetic')
            seen={}
            def remote(worker,command,action):
                seen['command']=command;seen['action']=action
                return 'synthetic-command'
            original=subprocess.run
            def run(*args,**kwargs):
                return original(seen['command'],input=kwargs.get('input'),text=True,capture_output=True,check=True)
            with patch.object(H.R,'ssh_wsl_command',side_effect=remote),patch.object(H.subprocess,'run',side_effect=run):
                result=H.synthetic_participation(worker,True)
            self.assertTrue(result['accepting_new_jobs'])
            self.assertEqual(json.loads(path.read_text()),result)

    def test_gateway_planonly_uses_normal_schedule_without_worker_or_queue(self):
        source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        deployment=dict(source_commit=source,hosts={host:dict(root=str(ROOT),python=sys.executable,
            directory='/SYNTHETIC/'+host+'/dedicated-state',ssh_alias='unused')for host in H.HOSTS})
        with tempfile.TemporaryDirectory()as temporary:
            base=Path(temporary)/'fixture'
            with patch.object(H.F,'free_port',return_value=19876):H.prepare(base,deployment)
            control=Path(temporary)/'gateway-run';control.mkdir()
            (control/'PROBLEM.json').write_bytes((base/'PROBLEM.json').read_bytes())
            (control/'ROUND.requested.json').write_bytes((base/'ROUND.json').read_bytes())
            with patch.object(H.R.Relay,'remote',side_effect=AssertionError('PlanOnly contacted remote worker')),redirect_stdout(io.StringIO()):
                H.controller(base,plan_only=True,control_run=control,max_cycles=1)
            self.assertTrue((control/'plan-only/KAPPA_STATE.json').exists())
            self.assertFalse((base/'run').exists())
            self.assertFalse((control/'plan-only/distributed/QUEUE.sqlite3').exists())

    def test_gateway_resume_clears_only_transient_mirrored_artificial_user_stop(self):
        with tempfile.TemporaryDirectory()as temporary:
            base=Path(temporary);control=base/'gateway';run=base/'run';control.mkdir();run.mkdir()
            (base/'ROUND.json').write_text(json.dumps(dict(artificial_fixture=dict(only=True))))
            marker=run/'STOP_REQUESTED.json'
            marker.write_text(json.dumps(dict(reason='NUMERICAL_UNKNOWN_MANDATORY_STOP')))
            H.clear_mirrored_stop(base,control,run,explicit_control=True)
            self.assertTrue(marker.exists())
            marker.write_text(json.dumps(dict(reason='USER_REQUESTED_STOP')))
            H.clear_mirrored_stop(base,control,run,explicit_control=False)
            self.assertTrue(marker.exists())
            H.clear_mirrored_stop(base,control,run,explicit_control=True)
            self.assertFalse(marker.exists())
            self.assertEqual(json.loads((run/'STOP_REQUESTED_BEFORE_GATEWAY_RESUME.json').read_text())['reason'],'USER_REQUESTED_STOP')

    def test_admission_targets_the_gateway_owned_copy_without_touching_real_policy(self):
        with tempfile.TemporaryDirectory()as temporary:
            base=Path(temporary);gateway=base/'gateway/state/runs/search-test';gateway.mkdir(parents=True)
            policy=gateway/'HOST_ADMISSION.json';policy.write_text('{}')
            (gateway/'ROUND.json').write_text(json.dumps(dict(executor=dict(distributed=dict(host_admission=str(policy))))))
            (base/'CONTROL_INPUTS.json').write_text(json.dumps(dict(directory=str(gateway),round=str(gateway/'ROUND.json'))))
            self.assertFalse((base/'run/ROUND.json').exists())
            H.synthetic_admission(base,True)
            self.assertTrue(json.loads(policy.read_text())['hosts']['local0-wsl']['accepting'])
            outside=base.parent/'REAL_WORKER.json'
            (gateway/'ROUND.json').write_text(json.dumps(dict(executor=dict(distributed=dict(host_admission=str(outside))))))
            with self.assertRaisesRegex(ValueError,'owned fixture'):H.synthetic_admission(base,False)


if __name__=='__main__':unittest.main(verbosity=2)
