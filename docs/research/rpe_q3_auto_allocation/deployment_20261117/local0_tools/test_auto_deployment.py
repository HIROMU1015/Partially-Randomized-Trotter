"""Synthetic deployment transport/bundle tests; no SSH, WSL or numerical work."""
import base64
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_auto_deploy',HERE/'run_auto_deployment.py')
R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)


class DeploymentTests(unittest.TestCase):
    def test_bundle_exact_source_real_requests_and_three_host_profiles(self):
        manifest,files=R.unpack((HERE/'AUTO_DEPLOY_349558B74.zip').read_bytes())
        self.assertEqual(manifest['source'],R.SOURCE)
        self.assertEqual({h['host_id']for h in manifest['hosts']},{'server1','local','local0-wsl'})
        self.assertEqual(sum(len(h['assignments'])for h in manifest['hosts']),12)
        for host in manifest['hosts']:
            profile=json.loads(files[host['profile']])
            self.assertNotEqual(profile['root'],host['transport_root'])
            self.assertEqual(profile['slot_policy'],dict(mode='auto',initial_slots=1))
            for item in [profile]+[r['profile']for r in profile.get('route_profiles',[])]:
                self.assertEqual(item['adapter_root'],host['transport_root'])
                self.assertEqual(item['adapter_manifest']['commit'],R.SOURCE)
                self.assertIn('src/trottertracks/rpe_q3_timing.py',item['adapter_manifest']['files'])
            for name in host['assignments']:
                assignment=json.loads(files[name])
                self.assertEqual(assignment['source_manifest']['commit'],'28d976b99cdf8329adcc1b79327e329e947ef774')
                self.assertNotEqual(assignment['request']['route']['kind'],'artificial')
                if assignment['method']=='PR_PAULI':self.assertEqual(assignment['request']['route']['view_commit'],'33e324b43690caeaa353ecdb9e3d42f3139198b6')

    def test_changed_payload_rejected_before_host_contact(self):
        with zipfile.ZipFile(HERE/'AUTO_DEPLOY_349558B74.zip')as original:
            files={name:original.read(name)for name in original.namelist()}
        files['profiles/server1.json']+=b' '
        buffer=io.BytesIO()
        with zipfile.ZipFile(buffer,'w')as changed:
            for name,data in files.items():changed.writestr(name,data)
        with self.assertRaisesRegex(ValueError,'Payload hash differs'):
            R.unpack(buffer.getvalue())

    def test_ssh_and_wsl_use_argv_stdin_without_shell_or_daemon(self):
        calls=[]
        def run(command,**kwargs):
            calls.append((command,kwargs));return subprocess.CompletedProcess(command,0,b'{"ok":true}',b'')
        with patch.object(R,'run',side_effect=run):
            self.assertTrue(R.ssh('RomuTail','/path with spaces/python3',dict(action='host'))['ok'])
            self.assertTrue(R.wsl(dict(distribution='Ubuntu',transport_python='/existing/python3'),dict(action='host'))['ok'])
        self.assertEqual(calls[0][0][:2],['ssh','-T'])
        self.assertIn('RomuTail',calls[0][0])
        self.assertIn("'/path with spaces/python3'",calls[0][0][-1])
        self.assertEqual(calls[1][0][:6],['wsl.exe','-d','Ubuntu','--exec','/existing/python3','-c'])
        for command,kwargs in calls:
            self.assertNotIn('shell',kwargs)
            code,payload=kwargs['input'].split(b'\n',1)
            self.assertEqual(base64.b64decode(code),R.Path(R.__file__).read_bytes())
            self.assertEqual(json.loads(payload)['action'],'host')
            self.assertNotIn('--start',command)

    def test_new_worktree_setup_uses_git_only_and_preserves_old_scientific_root(self):
        with tempfile.TemporaryDirectory()as temporary:
            repo=Path(temporary);root=repo/'owned'/'fixed';calls=[]
            def git(where,*args):
                calls.append((str(where),args))
                if args[:2]==('worktree','add'):
                    root.mkdir(parents=True);(root/'.git').write_text('gitdir: owned')
                if args==('rev-parse','HEAD'):return R.SOURCE
                return ''
            with patch.object(R,'git',side_effect=git):
                self.assertEqual(R.ensure_transport(dict(repository=str(repo),transport_root=str(root))),root)
            self.assertTrue(all(args[0]in('cat-file','worktree','sparse-checkout','checkout','rev-parse','diff')for _,args in calls))
            self.assertFalse(list(repo.rglob('spool')))

    def test_return_manifest_is_idempotent_and_changed_bytes_are_preserved(self):
        with tempfile.TemporaryDirectory()as temporary:
            raw=b'{"status":"REAL_HOST_DRY_CHECK_PASS"}\n'
            payload=dict(directory=str(Path(temporary)/'return'),manifest={'server1/dry/READINESS.json':R.sha(raw)},
                         files={'server1/dry/READINESS.json':base64.b64encode(raw).decode()})
            first=R.receive_action(payload);second=R.receive_action(payload)
            self.assertEqual(first,second)
            changed=dict(payload,files={'server1/dry/READINESS.json':base64.b64encode(b'changed').decode()})
            with self.assertRaisesRegex(ValueError,'Transfer bytes differ'):R.receive_action(changed)
            self.assertEqual((Path(payload['directory'])/'server1/dry/READINESS.json').read_bytes(),raw)


if __name__=='__main__':unittest.main()
