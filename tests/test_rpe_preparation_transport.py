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

    def test_concurrent_adoption_materializes_shared_large_artifact_once(self):
        from concurrent.futures import ThreadPoolExecutor
        import threading
        cache=ArtifactCache(self.root/'cache');raw=b'circuit-receipt-'*500000
        sha=hashlib.sha256(raw).hexdigest();cache.put({sha:base64.b64encode(raw).decode()})
        start=threading.Barrier(10)
        def adopt(_):
            start.wait();cache.materialize({'shared/raw/circuit.json':sha},self.root/'adoption')
        with ThreadPoolExecutor(max_workers=10)as pool:list(pool.map(adopt,range(10)))
        self.assertEqual((self.root/'adoption/shared/raw/circuit.json').read_bytes(),raw)
        self.assertEqual(list((self.root/'adoption').rglob('*.materialize-*')),[])

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

    def test_input_git_bundle_cannot_replace_fixed_scientific_code(self):
        malicious=self.root/'changed-code'
        subprocess.run(['git','clone','-q',str(self.root/'source'),str(malicious)],check=True)
        (malicious/'leaf.py').write_text('raise RuntimeError("different scientific source")\n')
        subprocess.run(['git','add','leaf.py'],cwd=malicious,check=True)
        subprocess.run(['git','-c','user.name=Test','-c','user.email=test@example.invalid','commit','-qm','changed-code'],cwd=malicious,check=True)
        bundle=self.root/'changed.bundle'
        subprocess.run(['git','bundle','create',str(bundle),'HEAD','^'+self.source['commit']],cwd=malicious,check=True,capture_output=True)
        raw=bundle.read_bytes();sha=hashlib.sha256(raw).hexdigest()
        ArtifactCache(self.root/'server1/spool/artifacts').put({sha:base64.b64encode(raw).decode()})
        task=self.task('audit',"from pathlib import Path;Path(r'@WORK@/result.json').write_text('must not run')",
                       source_workspace=True,input_git_bundles=[{'sha256':sha,'ref':'HEAD'}])
        with self.assertRaisesRegex(ValueError,'changed fixed scientific SOURCE'):
            P.execute(task,self.profile('server1'),self.root/'blocked-attempt')
        self.assertFalse((self.root/'blocked-attempt/result.json').exists())
        self.assertEqual((self.root/'source/leaf.py').read_text(),'print("synthetic fixed source")\n')

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

    def test_inspection_examples_cannot_be_mistaken_for_executable_tasks(self):
        spec=importlib.util.spec_from_file_location('prep_checker',ROOT/'scripts/check_rpe_preparation_host_readiness.py')
        checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
        task=self.task('compile',"raise RuntimeError('never execute')",dry_check_only=True)
        task['request'].pop('request_id')
        row=checker.check(task['request'],self.profile('server2'))
        self.assertEqual(row['status'],'SAVED_INPUT_RUNTIME_INSPECTION_PASS')
        self.assertFalse(row['execution_request_validated'])
        self.assertEqual(row['execution_readiness'],'UNCONFIRMED')
        from trottertracks.rpe_preparation_queue import normalize_preparation_request
        request=normalize_preparation_request(task['request'])
        assignment=dict(request=request,request_sha256=P.digest(request),job_id=request['request_id'],source_manifest=self.source)
        with self.assertRaisesRegex(ValueError,'Inspection-only'):
            P.execute(assignment,self.profile('server2'),self.root/'never-run')
        self.assertFalse((self.root/'never-run/PREPARATION_STARTED.json').exists())

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

    def test_relay_cross_host_artifact_protocol_only_transfers_missing_bytes(self):
        spec=importlib.util.spec_from_file_location('prep_relay',ROOT/'scripts/run_rpe_q3_relay.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        relay=object.__new__(module.Relay)
        central=ArtifactCache(self.root/'central-cas');remote=ArtifactCache(self.root/'worker-cas')
        raw=b'compiled-circuit'*100000;sha=hashlib.sha256(raw).hexdigest()
        central.put({sha:base64.b64encode(raw).decode()});transfers=[]
        def post(path,value):
            if path.endswith('/missing'):return {'missing':central.missing(value['hashes'])}
            if path.endswith('/get'):
                transfers.append(('central-download',value['hashes']));return {'blobs':central.get(value['hashes'])}
            if path.endswith('/put'):
                transfers.append(('central-upload',list(value['blobs'])));return {'ok':True,**central.put(value['blobs'])}
            raise AssertionError(path)
        def mailbox(worker,action,value):
            if action=='artifacts-missing':result={'missing':remote.missing(value['hashes'])}
            elif action=='artifacts-get':
                transfers.append(('worker-download',value['hashes']));result={'blobs':remote.get(value['hashes'])}
            elif action=='artifacts-put':
                transfers.append(('worker-upload',list(value['blobs'])));result={'ok':True,**remote.put(value['blobs'])}
            else:raise AssertionError(action)
            return result,0.0
        relay.post=post;relay.remote=mailbox;relay.log=lambda *args,**kwargs:None
        task=self.task('audit','pass',inputs={'circuit.json':sha});worker={'host_id':'server1'}
        relay._deliver_artifacts(worker,task);relay._deliver_artifacts(worker,task)
        self.assertEqual(len(transfers),2)
        extra=b'new-audit-receipt';other=hashlib.sha256(extra).hexdigest()
        remote.put({other:base64.b64encode(extra).decode()})
        relay._recover_artifacts(worker,{'artifact_manifest':{'audit.json':other}})
        relay._recover_artifacts(worker,{'artifact_manifest':{'audit.json':other}})
        self.assertEqual(len(transfers),4)
        self.assertEqual(base64.b64decode(central.get([other])[other]),extra)

    def test_preparation_result_bundle_does_not_repeat_large_output(self):
        attempt=self.root/'attempt';attempt.mkdir()
        task=self.task('compile','pass');(attempt/'ASSIGNMENT.json').write_text(json.dumps(task))
        (attempt/'OUTCOME.json').write_text('{}');(attempt/'PROCESS_RETURN.json').write_text('{}')
        (attempt/'large-circuit').write_bytes(b'x'*1000000)
        bundle=W.result_bundle(attempt)
        self.assertEqual(set(bundle),{'OUTCOME.json','PROCESS_RETURN.json'})
        self.assertLess(sum(len(x)for x in bundle.values()),100)


if __name__=='__main__':unittest.main()
