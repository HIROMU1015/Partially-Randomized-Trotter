"""Central common entry reuses the existing confirmed-run monitor protocol."""
from pathlib import Path
import base64
import json
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import run_rpe_windows_sync_hook as H


class ConfirmedRunHook(unittest.TestCase):
    def test_only_confirmed_real_runs_reach_the_existing_hook(self):
        with patch.object(H.subprocess,'run',side_effect=AssertionError('unexpected SSH')):
            self.assertEqual(H.confirmed_run({},dict(status='NOT_STARTED'))['status'],'NOT_A_CONFIRMED_RUN')
            self.assertEqual(H.confirmed_run({},dict(status='STARTED',artificial_only=True))['status'],'ARTIFICIAL_SKIPPED')

    def test_preserves_result_and_reuses_existing_sync_command_and_identity(self):
        config=dict(ssh_alias='existing-Windows',ssh_command=['ssh','-l','existing-user'],
            result_sync_script='C:/owned/tools/local0-result-sync/Invoke-H4ResultSync.ps1',
            result_sync_config='C:/owned/tools/local0-result-sync/CONFIG.json')
        result=dict(status='STARTED',run_id='same-run',run_directory='/central/state/same-run')
        commands=[]
        def run(command,**kwargs):
            commands.append(command)
            if command[0]=='scp':
                self.assertEqual(json.loads(Path(command[-2]).read_text()),result)
                self.assertTrue(command[-1].startswith('existing-user@existing-Windows:.rpe-confirmed-run-'))
                return subprocess.CompletedProcess(command,0,b'',b'')
            source=base64.b64decode(command[-1].split()[-1]).decode('utf-16-le')
            self.assertIn('-Action OnConfirmedRun',source)
            self.assertIn(config['result_sync_script'],source)
            self.assertIn('finally{Remove-Item -LiteralPath $p',source)
            return subprocess.CompletedProcess(command,0,b'{"state":"RUNNING","task_name":"one-existing-identity"}',b'')
        with patch.object(H.subprocess,'run',side_effect=run):reply=H.confirmed_run(config,result)
        self.assertEqual(reply['result']['task_name'],'one-existing-identity')
        self.assertEqual(len(commands),2)


if __name__=='__main__':unittest.main()
