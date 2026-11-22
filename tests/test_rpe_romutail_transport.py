"""Direct RomuTail relay configuration; synthetic transports only."""
import base64
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('_romutail_relay', ROOT/'scripts/run_rpe_q3_relay.py')
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


class DirectRelayTests(unittest.TestCase):
    def test_normal_executor_owns_relay_endpoint_and_preserves_received_jobs_on_stop(self):
        from trottertracks.rpe_optimizer_concurrent import DistributedExecutor
        with tempfile.TemporaryDirectory()as temporary:
            executor=DistributedExecutor.__new__(DistributedExecutor)
            executor.run_dir=Path(temporary);executor.root=ROOT;executor.state=SimpleNamespace()
            executor.distributed={'relay':{'workers':[{'host_id':'server2','ssh_alias':'qg2'}]}}
            executor.log=lambda **event:None
            child=SimpleNamespace(pid=999999,wait=lambda **kwargs:0)
            with patch('trottertracks.rpe_optimizer_concurrent.subprocess.Popen',return_value=child)as launch:
                executor.start_controller_relay(12345)
            path=executor.run_dir/'distributed/relay/CONFIG.json'
            config=json.loads(path.read_text())
            self.assertEqual(config['controller'],{'local_port':12345})
            self.assertTrue(config['external_tunnel']);self.assertEqual(launch.call_count,1)
            self.assertNotIn('--start',launch.call_args.args[0])
            # The actual relay's lock stands in for a survivor after controller loss.
            import fcntl
            with (path.parent/'RELAY.lock').open('a')as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with patch('trottertracks.rpe_optimizer_concurrent.subprocess.Popen',side_effect=AssertionError('duplicate relay')):
                    executor.start_controller_relay(12345)
                    with self.assertRaisesRegex(ValueError,'configuration differs'):
                        executor.start_controller_relay(54321)
            executor.stop_controller_relay()
            self.assertTrue((path.parent/'STOP').exists())

    def relay(self, directory, workers=()):
        path = Path(directory)/'relay.json'
        path.write_text(json.dumps(dict(state_directory=str(Path(directory)/'state'),
            external_tunnel=True,controller=dict(local_port=19876),workers=list(workers),poll_seconds=.001)))
        return R.Relay(path)

    def test_per_worker_ssh_command_keeps_linux_mailbox_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            relay = self.relay(directory)
            worker = dict(host_id='server2',ssh_alias='qg2',ssh_command=['ssh','-i','owned key'],
                root='/owned/source checkout',profile='/owned/profile path.json',python='/owned/python')
            with patch.object(R.subprocess,'run',return_value=SimpleNamespace(
                    returncode=0,stdout='{"host_id":"server2"}',stderr='')) as call:
                observed,_ = relay.remote(worker,'status')
            self.assertEqual(observed['host_id'],'server2')
            argv = call.call_args.args[0]
            self.assertEqual(argv[:3],['ssh','-i','owned key'])
            self.assertEqual(argv[-2],'qg2')
            self.assertIn("'/owned/source checkout/scripts/run_rpe_q3_worker.py'",argv[-1])
            self.assertEqual(json.loads(call.call_args.kwargs['input']),{})

    def test_remote_windows_wsl_reads_remote_pause_without_local_file_access(self):
        with tempfile.TemporaryDirectory() as directory:
            relay = self.relay(directory)
            worker = dict(host_id='local0-wsl',ssh_alias='Romulaptop',transport='ssh-wsl',
                distribution='Ubuntu',root='/owned/WSL source',profile='/owned/WSL profile.json',
                python='/usr/bin/python3',participation_file=r'C:\Users\romuk\研究\participation.json',
                ssh_command=['ssh','-i','owned windows key','-l','romuk'])
            response=dict(host_id='local0-wsl',slots=0,accepting_new_jobs=False)
            with patch.object(R.subprocess,'run',return_value=SimpleNamespace(
                    returncode=0,stdout=json.dumps(response),stderr='')) as call:
                observed,_ = relay.remote(worker,'status')
            self.assertEqual(observed,response)
            argv = call.call_args.args[0]
            self.assertEqual(argv[:5],['ssh','-i','owned windows key','-l','romuk'])
            self.assertEqual(argv[-2],'Romulaptop')
            encoded = argv[-1].split()[-1]
            script = base64.b64decode(encoded).decode('utf-16-le')
            self.assertIn("$si.FileName='wsl.exe'",script)
            self.assertIn('/owned/WSL source/scripts/run_rpe_q3_worker.py',script)
            self.assertIn(worker['participation_file'],script)
            self.assertIn('if(-not $enabled){$value.slots=0}',script)
            self.assertNotIn('[Console]::In.ReadToEnd()',script)
            self.assertIn('wslpath -u',script)
            self.assertNotIn('RedirectStandardInput',script)
            self.assertIn('finally{Remove-Item',script)
            self.assertIn('WaitForExit((LeftMs))',script)
            self.assertIn('$proc.Kill()',script)
            call.assert_called_once()  # the small status payload is sent inline
            self.assertEqual(call.call_args.kwargs['stdin'],subprocess.DEVNULL)

    def test_large_wsl_request_is_transferred_intact_without_console_stdin(self):
        with tempfile.TemporaryDirectory()as directory:
            relay=self.relay(directory)
            worker=dict(host_id='local0-wsl',ssh_alias='owned-windows',transport='ssh-wsl',
                distribution='Ubuntu',root='/owned/source',profile='/owned/profile',python='python3',
                ssh_command=['ssh','-p','2222','-l','owned-user'])
            payload=dict(blobs={'artificial':'a'*180224});saved={}
            def run(command,**kwargs):
                self.assertEqual(kwargs['stdin'],subprocess.DEVNULL)
                if command[0]=='scp':
                    self.assertEqual(command[1:3],['-P','2222'])
                    saved['path']=Path(command[-2]);saved['payload']=json.loads(saved['path'].read_text())
                return SimpleNamespace(returncode=0,stdout='{"ok":true}',stderr='')
            with patch.object(R.subprocess,'run',side_effect=run):
                result,_=relay.remote(worker,'artifacts-put',payload)
            self.assertTrue(result['ok'])
            self.assertEqual(saved['payload'],payload)
            self.assertFalse(saved['path'].exists())

    def test_disabled_server_is_never_contacted(self):
        with tempfile.TemporaryDirectory() as directory:
            relay = self.relay(directory,[dict(host_id='server1',enabled=False,
                disabled_reason='Server 1 is stopped; inspect after return')])
            with patch.object(relay,'remote',side_effect=AssertionError('Offline host contacted')):
                relay.cycle()
            saved=json.loads((relay.directory/'STATUS.json').read_text())
            self.assertEqual(saved['workers']['server1']['phase'],'DISABLED')
            self.assertFalse(saved['workers']['server1']['accepting_new_jobs'])
            self.assertFalse(relay.pending_discovery)
            if relay.discovery_pool is not None:
                relay.discovery_pool.shutdown(wait=True)

    def test_remote_wsl_participation_file_is_read_in_its_own_distribution(self):
        worker=dict(distribution='Ubuntu',participation_wsl_file='/root/owned/settings.json')
        command=R.ssh_wsl_command(worker,['python3','-B','/owned/worker.py'],'status')
        script=base64.b64decode(command.split()[-1]).decode('utf-16-le')
        self.assertIn("'-d','Ubuntu','--exec','cat','/root/owned/settings.json'",script)
        self.assertNotIn('Test-Path',script)
        self.assertIn('if(-not $enabled){$value.slots=0}',script)


if __name__ == '__main__':
    unittest.main(verbosity=2)
