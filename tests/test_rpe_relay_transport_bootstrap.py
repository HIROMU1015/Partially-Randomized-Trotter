"""Normal operation transport hook dispatch with stub launches; no science."""
import base64
import contextlib
import io
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from trottertracks.rpe_local0_search import Gateway


class RelayTransportBootstrap(unittest.TestCase):
    def invoke(self,action,result,run,*,sync=False):
        with tempfile.TemporaryDirectory(prefix='rpe-bootstrap-transport-')as temporary:
            service=Path(temporary)/'SERVICE.json'
            service.write_text(json.dumps(dict(code_root=str(ROOT),state_directory=str(Path(temporary)/'state'),
                relay_transport_command=['ARTIFICIAL_RELAY_HOOK'],relay_transport_timeout_seconds=45,
                confirmed_run_hook=['ARTIFICIAL_SYNC_HOOK'])))
            envelope=dict(server_config=str(service),action=action,invoke_result_sync=sync,
                request_base64=base64.b64encode(b'{}').decode())
            output=io.StringIO()
            with patch.object(Gateway,'handle',return_value=dict(result)), \
                    patch('trottertracks.rpe_release_registry.select',side_effect=lambda config,request:config), \
                    patch('subprocess.run',side_effect=run),contextlib.redirect_stdout(output):
                runpy.run_path(str(ROOT/'tools/local0-search/remote.py'),init_globals={'CONFIG':envelope})
            response=json.loads(output.getvalue())
            self.assertTrue(response['ok'])
            return response['result']

    def test_resume_transport_is_applied_before_existing_result_sync(self):
        calls=[]
        original=dict(status='STARTED',started=True,run_id='ARTIFICIAL_RUN',
            run_directory='/ARTIFICIAL/run',runtime={'invocation_id':'SAME_FIXED_INVOCATION'})
        def run(command,**options):
            calls.append((command,options))
            if command==['ARTIFICIAL_RELAY_HOOK']:
                self.assertEqual(json.loads(options['input']),original)
                self.assertEqual(options['timeout'],45)
                return SimpleNamespace(returncode=0,stdout='{"status":"RELAY_TRANSPORT_READY"}')
            self.assertEqual(command,['ARTIFICIAL_SYNC_HOOK'])
            self.assertEqual(json.loads(options['input'])['relay_transport']['status'],'RELAY_TRANSPORT_READY')
            return SimpleNamespace(returncode=0,stdout='{"status":"ON_CONFIRMED_RUN"}')
        result=self.invoke('resume',original,run,sync=True)
        self.assertEqual([call[0]for call in calls],[['ARTIFICIAL_RELAY_HOOK'],['ARTIFICIAL_SYNC_HOOK']])
        self.assertEqual(result['runtime'],original['runtime'])
        self.assertEqual(result['result_sync']['status'],'ON_CONFIRMED_RUN')

    def test_transport_does_not_depend_on_result_sync_flag(self):
        calls=[]
        def run(command,**options):
            calls.append(command)
            return SimpleNamespace(returncode=0,stdout='{"status":"RELAY_TRANSPORT_READY"}')
        result=self.invoke('start',dict(status='ALREADY_RUNNING',started=True),run)
        self.assertEqual(calls,[['ARTIFICIAL_RELAY_HOOK']])
        self.assertEqual(result['relay_transport']['status'],'RELAY_TRANSPORT_READY')
        self.assertNotIn('result_sync',result)

    def test_status_stop_rejected_and_completed_operations_do_not_start_transport(self):
        for action,status in [('status','STARTED'),('stop','ALREADY_RUNNING'),
                              ('start','NOT_STARTED'),('resume','REUSED_COMPLETED')]:
            with self.subTest(action=action,status=status):
                def no_launch(*args,**kwargs):raise AssertionError('No operational hook may run')
                result=self.invoke(action,dict(status=status,started=False),no_launch)
                self.assertNotIn('relay_transport',result)

    def test_hook_failure_preserves_run_and_reports_transport_pending(self):
        failures=[SimpleNamespace(returncode=2,stdout=json.dumps({'error':'x'*2000})),
                  SimpleNamespace(returncode=0,stdout='[]'),
                  subprocess.TimeoutExpired('ARTIFICIAL_RELAY_HOOK',45)]
        for failure in failures:
            with self.subTest(failure=type(failure).__name__):
                def run(*args,**kwargs):
                    if isinstance(failure,Exception):raise failure
                    return failure
                result=self.invoke('resume',dict(status='ALREADY_RUNNING',started=True,
                    run_id='ARTIFICIAL_RUN'),run)
                self.assertEqual(result['status'],'ALREADY_RUNNING')
                self.assertTrue(result['started'])
                self.assertEqual(result['relay_transport']['status'],'RELAY_TRANSPORT_PENDING')
                self.assertLessEqual(len(result['relay_transport']['error']),1500)


if __name__=='__main__':unittest.main()
