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

    def test_gateway_planonly_uses_normal_schedule_without_worker_or_queue(self):
        source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        deployment=dict(source_commit=source,hosts={host:dict(root=str(ROOT),python=sys.executable,
            directory='/SYNTHETIC/'+host+'/dedicated-state',ssh_alias='unused')for host in H.HOSTS})
        with tempfile.TemporaryDirectory()as temporary:
            base=Path(temporary)/'fixture'
            with patch.object(H.F,'free_port',return_value=19876):H.prepare(base,deployment)
            control=Path(temporary)/'gateway-run';control.mkdir()
            for name in ('PROBLEM.json','ROUND.json'):(control/name).write_bytes((base/name).read_bytes())
            with patch.object(H.R.Relay,'remote',side_effect=AssertionError('PlanOnly contacted remote worker')),redirect_stdout(io.StringIO()):
                H.controller(base,plan_only=True,control_run=control,max_cycles=1)
            self.assertTrue((control/'plan-only/KAPPA_STATE.json').exists())
            self.assertFalse((base/'run').exists())
            self.assertFalse((control/'plan-only/distributed/QUEUE.sqlite3').exists())


if __name__=='__main__':unittest.main(verbosity=2)
