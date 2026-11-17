"""Concrete cross-host preparation transport checks; no molecular computation."""
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import rpe_preparation_worker as P
from trottertracks.rpe_preparation_artifacts import ArtifactCache
from trottertracks import rpe_q3_worker as W


def repo(root):
    root.mkdir()
    subprocess.run(['git','init','-q',str(root)],check=True)
    (root/'leaf.py').write_text('print("synthetic fixed source")\n')
    subprocess.run(['git','add','leaf.py'],cwd=root,check=True)
    subprocess.run(['git','-c','user.name=Test','-c','user.email=test@example.invalid','commit','-qm','fixed'],cwd=root,check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    return {'commit':commit,'files':{'leaf.py':hashlib.sha256((root/'leaf.py').read_bytes()).hexdigest()}}


class PreparationTransportTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.source=repo(self.root/'source')

    def tearDown(self):self.tmp.cleanup()

    def profile(self,host):
        return dict(host_id=host,root=str(self.root/'source'),interpreter=sys.executable,
                    work_dir=str(self.root/host/'spool'),slots=1,dp_workers=1)

    def task(self,stage,code,inputs=None,outputs=None,**fields):
        request=dict(request_id='prep-'+stage,method='FULL_PF',route={'kind':'preparation'},
            preparation=dict(stage=stage,provider='synthetic',artificial=True,source=self.source,
                command=['@PYTHON@','-c',code],outputs=outputs or ['result.json'],inputs=inputs or {},**fields))
        return dict(job_id=request['request_id'],method=request['method'],attempt_id='attempt',
                    source_manifest=self.source,request=request,request_sha256=P.digest(request))

    def test_compile_artifacts_move_to_another_host_once(self):
        compile=self.task('compile',"from pathlib import Path;Path(r'@WORK@/result.json').write_text('{\"compiled\":7}')")
        compile_dir=self.root/'compiler-attempt'
        result=P.execute(compile,self.profile('local'),compile_dir)
        source_cache=ArtifactCache(self.root/'local/spool/artifacts')
        central=ArtifactCache(self.root/'central/artifacts');audit_cache=ArtifactCache(self.root/'server1/spool/artifacts')
        manifest=result['artifacts'];hashes=list(manifest.values())
        first=central.put(source_cache.get(central.missing(hashes)))
        self.assertEqual(first['files'],1)
        self.assertEqual(central.put(source_cache.get(central.missing(hashes)))['bytes'],0)
        audit_cache.put(central.get(audit_cache.missing(hashes)))
        audit=self.task('audit',"from pathlib import Path;import json;x=json.loads(Path(r'@INPUTS@/result.json').read_text());Path(r'@WORK@/result.json').write_text(str(x['compiled']*2))",inputs=manifest)
        audit_dir=self.root/'auditor-attempt'
        audited=P.execute(audit,self.profile('server1'),audit_dir)
        self.assertEqual((audit_dir/'result.json').read_text(),'14')
        self.assertEqual(audited['status'],'PREPARATION_COMPLETE')
        self.assertEqual(P.collect(audit,self.profile('server1'),audit_dir),audited)
        with self.assertRaises(FileExistsError):P.execute(audit,self.profile('server1'),audit_dir)
        self.assertEqual((audit_dir/'result.json').read_text(),'14')

    def test_git_workspace_exports_new_objects_without_changing_source(self):
        code="""from pathlib import Path
import subprocess
Path('proof.json').write_text('{"receipt":"synthetic"}')
subprocess.run(['git','add','proof.json'],check=True)
subprocess.run(['git','commit','-qm','one synthetic receipt'],check=True)
"""
        task=self.task('compile',code,source_workspace=True,outputs=['workspace/proof.json'])
        outcome=P.execute(task,self.profile('local'),self.root/'git-attempt')
        self.assertEqual(len(outcome['git_bundles']),1)
        self.assertFalse((self.root/'source/proof.json').exists())
        self.assertEqual(subprocess.check_output(['git','rev-parse','HEAD'],cwd=self.root/'source',text=True).strip(),self.source['commit'])
        bundle=self.root/'git-attempt'/outcome['git_bundles'][0]['path']
        consumer=self.root/'consumer'
        subprocess.run(['git','clone','-q',str(self.root/'source'),str(consumer)],check=True)
        subprocess.run(['git','fetch','-q',str(bundle),'HEAD'],cwd=consumer,check=True)
        returned=json.loads(subprocess.check_output(['git','show',outcome['git_bundles'][0]['head']+':proof.json'],cwd=consumer))
        self.assertEqual(returned,{'receipt':'synthetic'})

    def test_dry_check_binds_stage_source_runtime_and_leaves_work_absent(self):
        profile=self.profile('server2');task=self.task('signal',"raise RuntimeError('must not execute')")
        first=P.current_preparation_fingerprint(task,profile)
        self.assertEqual(first['stage'],'signal')
        self.assertFalse(Path(profile['work_dir']).exists())
        changed=json.loads(json.dumps(task));changed['request']['preparation']['source']['files']['leaf.py']='0'*64
        changed['request_sha256']=P.digest(changed['request']);changed['source_manifest']=changed['request']['preparation']['source']
        with self.assertRaisesRegex(ValueError,'SOURCE hash'):P.current_preparation_fingerprint(changed,profile)
        task['request']['preparation']['runtime_requirements']={'version_info_prefix':[99]}
        task['request_sha256']=P.digest(task['request'])
        with self.assertRaisesRegex(ValueError,'Python version'):P.current_preparation_fingerprint(task,profile)

    def test_missing_stage_profile_does_not_reuse_q3_pass(self):
        task=self.task('signal','pass');task['request']['preparation']['artificial']=False
        with self.assertRaisesRegex(ValueError,'No verified preparation profile'):
            P.select_preparation_profile(self.profile('server1'),task)

    def test_artifact_hash_corruption_and_path_escape_rejected(self):
        cache=ArtifactCache(self.root/'cache');data=b'circuit';sha=hashlib.sha256(data).hexdigest()
        with self.assertRaisesRegex(ValueError,'hash differs'):cache.put({sha:base64.b64encode(b'other').decode()})
        cache.put({sha:base64.b64encode(data).decode()})
        with self.assertRaises(ValueError):cache.materialize({'../outside':sha},self.root/'inputs')
        cache.path(sha).write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError,'Corrupt'):cache.get([sha])

    def test_preparation_result_bundle_does_not_repeat_large_output(self):
        attempt=self.root/'attempt';attempt.mkdir()
        task=self.task('compile','pass');(attempt/'ASSIGNMENT.json').write_text(json.dumps(task))
        (attempt/'OUTCOME.json').write_text('{}');(attempt/'PROCESS_RETURN.json').write_text('{}')
        (attempt/'large-circuit').write_bytes(b'x'*1000000)
        bundle=W.result_bundle(attempt)
        self.assertEqual(set(bundle),{'OUTCOME.json','PROCESS_RETURN.json'})
        self.assertLess(sum(len(x)for x in bundle.values()),100)


if __name__=='__main__':unittest.main()
