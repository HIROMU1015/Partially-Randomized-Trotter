"""A failed SSH transport must not lose other hosts' saved inspections."""
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('preparation_deployment',ROOT/'scripts/run_rpe_preparation_deployment.py')
D=importlib.util.module_from_spec(spec);spec.loader.exec_module(D)


class DeploymentRecovery(unittest.TestCase):
    def fixture(self,root):
        flag=root/'PARTICIPATION.json';flag.write_text('{"accepting_new_jobs":false}\n')
        hosts=[dict(host_id=h,profile=h+'.profile.json')for h in ('server1','local','local0-wsl')]
        files={'legacy_auto_deployment.py':(ROOT/'docs/research/rpe_distributed_preparation/deployment_f956f5197/legacy_auto_deployment.py').read_bytes(),'TASKS.json':b'[]'}
        files.update({h['profile']:b'{}'for h in hosts})
        manifest=dict(source='FIXED_ARTIFICIAL',hosts=hosts,participation_file=str(flag),
            return_directory=str(root/'returned'),return_transport={'ssh_alias':'qg2'},files={n:hashlib.sha256(v).hexdigest()for n,v in files.items()})
        files['MANIFEST.json']=json.dumps(manifest).encode();bundle=root/'package.zip'
        with zipfile.ZipFile(bundle,'w')as z:
            for n,v in files.items():z.writestr(n,v)
        return bundle,hashlib.sha256(bundle.read_bytes()).hexdigest(),manifest

    def saved_result(self,host,source='FIXED_ARTIFICIAL'):
        row=dict(host_id=host,stage='signal',status='SAVED_INPUT_RUNTIME_INSPECTION_PASS')
        return dict(ok=True,result=dict(host_id=host,source=source,rows=[row]),
            files={host+'/READINESS.json':base64.b64encode(json.dumps(row).encode()).decode()})

    def invoke(self,args):
        with patch.object(D.sys,'argv',['run_rpe_preparation_deployment.py',*map(str,args)]):D.main()

    def test_transport_exception_returns_successful_hosts_and_saves_failure(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);bundle,sha,manifest=self.fixture(root);calls=[]
            def remote(host,payload,U,utility):
                calls.append((host.get('host_id'),payload['action']))
                if payload['action']=='receive':return U.receive_action(payload)
                if host['host_id']=='server1':raise RuntimeError('No route to host')
                return self.saved_result(host['host_id'])
            with patch.object(D,'remote',side_effect=remote):
                self.invoke(['--bundle',bundle,'--bundle-sha256',sha,'--output',root/'results'])
            saved=json.loads((root/'results/server1.json').read_text())
            self.assertFalse(saved['ok']);self.assertFalse(saved['automatic_retry'])
            summary=json.loads((root/'returned/SUMMARY.json').read_text())
            self.assertEqual(sum(r['status']=='SAVED_INPUT_RUNTIME_INSPECTION_PASS'for r in summary['rows']),2)
            self.assertEqual(calls.count(('server1','host')),1);self.assertIn((None,'receive'),calls)
            self.assertEqual(summary['wsl_accepting_new_jobs'],False)

    def test_collect_saved_only_contacts_qg2_and_preserves_original_results(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);bundle,sha,manifest=self.fixture(root);output=root/'results';output.mkdir()
            for host in ('local','local0-wsl'):(output/(host+'.json')).write_text(json.dumps(self.saved_result(host)))
            original={p.name:p.read_bytes()for p in output.iterdir()};calls=[]
            def remote(host,payload,U,utility):
                calls.append(host['ssh_alias']);self.assertEqual(payload['action'],'receive');return U.receive_action(payload)
            with patch.object(D,'remote',side_effect=remote):
                self.invoke(['--bundle',bundle,'--bundle-sha256',sha,'--collect-saved',output])
            self.assertEqual(calls,['qg2'])
            for name,raw in original.items():self.assertEqual((output/name).read_bytes(),raw)
            self.assertFalse((output/'server1.json').exists())
            summary=json.loads((root/'returned/SUMMARY.json').read_text())
            self.assertEqual(summary['collection_scope'],'SAVED_LOCAL_RESULTS_NO_HOST_RECONNECT')
            self.assertIsNone(summary['wsl_paused_unchanged'])
            self.assertEqual(summary['wsl_accepting_new_jobs'],False)

    def test_collect_saved_rejects_different_source_before_return(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);bundle,sha,manifest=self.fixture(root);output=root/'results';output.mkdir()
            (output/'local.json').write_text(json.dumps(self.saved_result('local','OTHER_SOURCE')))
            with patch.object(D,'remote')as remote:
                with self.assertRaisesRegex(ValueError,'SOURCE differs'):
                    self.invoke(['--bundle',bundle,'--bundle-sha256',sha,'--collect-saved',output])
            remote.assert_not_called()


if __name__=='__main__':unittest.main()
