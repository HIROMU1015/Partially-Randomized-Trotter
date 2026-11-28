"""Concrete cross-host preparation transport checks; no molecular computation."""
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import rpe_preparation_worker as P
from trottertracks.rpe_preparation_bridge import DistributedPreparationBridge, fixed_input_hashes, pauli_audit_raw_inputs
from trottertracks.rpe_preparation_artifacts import ArtifactCache
from trottertracks import rpe_q3_worker as W
from trottertracks import rpe_worker_storage as Storage


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

    def test_fixed_git_inputs_keep_saved_tree_listing_hash(self):
        root = self.root/'source'
        (root/'nested').mkdir()
        (root/'nested/leaf').write_bytes(b'contents\x00\n')
        subprocess.run(['git', 'add', 'nested'], cwd=root, check=True)
        subprocess.run(['git', '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                        'commit', '-qm', 'tree'], cwd=root, check=True)
        locators = []
        for path in ('nested', 'nested/leaf'):
            raw = subprocess.check_output(['git', 'show', 'HEAD:'+path], cwd=root)
            locators.append(dict(commit='HEAD', path=path, sha256=hashlib.sha256(raw).hexdigest()))
        P._verify_fixed_git_inputs(root, locators)
        locators[1]['sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'nested/leaf'):
            P._verify_fixed_git_inputs(root, locators)

    def test_compaction_preserves_git_objects_and_reclaims_exact_duplicate_pack(self):
        root = (self.root/'source').resolve()
        subprocess.run(['git', 'repack', '-ad'], cwd=root, check=True)
        packdir = root/'.git/objects/pack'
        for path in list(packdir.iterdir()):
            shutil.copy2(path, path.with_name('pack-'+'b'*40+path.suffix))
        before = Storage._packed_objects(packdir)
        self.assertGreater(Storage._compact_pack_directory(packdir, self.root.resolve()), 0)
        self.assertEqual(Storage._packed_objects(packdir), before)
        self.assertEqual(subprocess.check_output(['git', 'show', 'HEAD:leaf.py'], cwd=root),
                         (root/'leaf.py').read_bytes())

    def test_quota_reclaims_abandoned_git_temporaries_even_with_only_one_indexed_pack(self):
        root = self.root.resolve()
        spool = root/'spool'
        spool.mkdir()
        (spool/'WORKER.lock').touch()
        cache = root/'preparation-metadata-cache'
        owner = cache/'private'
        packs = owner/'objects.git/objects/pack'
        packs.mkdir(parents=True)
        (owner/'IMPORT.lock').touch()
        old = packs/'tmp_pack_abc'
        in_use = packs/'tmp_pack_busy'
        recent = packs/'tmp_pack_new'
        indexed = packs/'pack-example.pack'
        for path in (old, in_use, recent, indexed):
            path.write_bytes(b'pack contents')
        for path in (old, in_use):
            os.utime(path, (1, 1))
        policy = dict(roots=[str(cache)], sample_path=str(root/'USAGE.json'), limit_bytes=10_000)
        # This fixture owns one artificial spool. Other tasks on the shared
        # host may have an unreadable Git process; production intentionally
        # refuses cleanup in that case. Scope this unit's process inventory to
        # the test process, which actually holds the in_use file descriptor.
        original_iterdir = Path.iterdir
        def fixture_iterdir(path):
            if path == Path('/proc'):
                return iter([Path('/proc')/str(os.getpid())])
            return original_iterdir(path)
        with patch.object(Path, 'iterdir', fixture_iterdir), in_use.open('rb'):
            result = Storage.report(policy, spools=(str(spool),), compact_metadata=True, proactive=True)
        self.assertFalse(old.exists())
        self.assertTrue(in_use.exists() and recent.exists() and indexed.exists())
        self.assertGreater(result['reclaimed_metadata_temporary_bytes'], 0)

    def test_only_matching_packs_in_transferred_attempts_share_storage(self):
        spool = self.root/'spool'
        packname = 'pack-'+'a'*40+'.pack'
        pairs = {}
        for job, state, matching in (('complete', 'TRANSFERRED', True),
                                     ('different', 'TRANSFERRED', False),
                                     ('pending', 'UNKNOWN', True)):
            attempt = spool/job/'attempt'
            paths = [attempt/p/'.git/objects/pack'/packname
                     for p in ('workspace', 'workspaces/provider')]
            for index, path in enumerate(paths):
                path.parent.mkdir(parents=True)
                path.write_bytes(b'git pack' if index == 0 or matching else b'other...')
                path.chmod(0o444)
            (attempt/'WORKER_STATE.json').write_text(json.dumps(dict(state=state)))
            (attempt/'WORKER_PROCESS.lock').touch()
            pairs[job] = paths

        Storage._share_transferred_git_packs([str(spool)], [str(spool)], 2**20)
        for job, (original, copy) in pairs.items():
            self.assertEqual(os.path.samefile(original, copy), job == 'complete')
        self.assertEqual(pairs['different'][1].read_bytes(), b'other...')

    def test_pauli_binding_publishes_only_owned_metadata_history(self):
        spec=importlib.util.spec_from_file_location('pauli_candidate_prices',ROOT/'scripts/run_pr_pauli_candidate_prices.py')
        prices=importlib.util.module_from_spec(spec);spec.loader.exec_module(prices)
        workspace=self.root/'source'
        git=lambda *args:subprocess.check_output(['git',*args],cwd=workspace,text=True).strip()
        bare=self.root/'publication.git'
        git('clone','--bare',str(workspace),str(bare))
        base=self.source['commit']; reference='refs/heads/codex/fixed-provider'
        subprocess.run(['git','-C',str(bare),'update-ref',reference,base],check=True)
        env=P.prepare_pauli_binding_publication(dict(pauli_binding_publication=dict(origin=str(bare),base_ref=reference)),
                                                dict(job_id='prep-test',attempt_id='a1'),workspace,base)
        candidate=workspace/'docs/research/d2_automatic_price_20261116/test-candidate'
        candidate.mkdir(parents=True)
        binding=candidate/'BINDING.json';binding.write_text('{}\n')
        prices.OPS.commit(workspace,[binding],'Freeze synthetic binding')
        with patch.dict(os.environ,env):
            prices._publish_compile_binding(workspace,candidate)
            published=git('ls-remote','--heads','origin','codex/rpe-pauli-binding-prep-test-a1').split()[0]
            self.assertEqual(published,git('rev-parse','HEAD'))
            private=candidate/'PRIVATE.json';private.write_text('private\n')
            prices.OPS.commit(workspace,[private],'Synthetic out-of-scope commit')
            git('rm',str(private));git('-c','user.name=Test','-c','user.email=test@example.invalid','commit','-qm','remove')
            with self.assertRaisesRegex(ValueError,'outside the compile binding'):
                prices._publish_compile_binding(workspace,candidate)
            self.assertEqual(git('ls-remote','--heads','origin','codex/rpe-pauli-binding-prep-test-a1').split()[0],published)

    def test_pauli_group_publication_excludes_each_groups_raw_result(self):
        spec=importlib.util.spec_from_file_location('pauli_candidate_prices',ROOT/'scripts/run_pr_pauli_candidate_prices.py')
        prices=importlib.util.module_from_spec(spec);spec.loader.exec_module(prices)
        workspace=self.root/'source'
        git=lambda *args:subprocess.check_output(['git',*args],cwd=workspace,text=True).strip()
        bare=self.root/'publication.git'
        git('clone','--bare',str(workspace),str(bare))
        git('config','user.name','Test');git('config','user.email','test@example.invalid')
        base=self.source['commit']; reference='refs/heads/codex/fixed-provider'
        subprocess.run(['git','-C',str(bare),'update-ref',reference,base],check=True)
        env=P.prepare_pauli_binding_publication(dict(pauli_binding_publication=dict(origin=str(bare),base_ref=reference)),
                                                dict(job_id='prep-test',attempt_id='a2'),workspace,base)
        request=dict(workspace=str(workspace),directory='docs/research/d2_automatic_price_20261116/test',
                     private_root=str(self.root/'private'),scoped_registration=True,configuration={'id':'artificial'})
        raw_commits=[]

        def synthetic_group(child, *, execute, push, **options):
            if not execute:
                return {'calibration_requests':[{'request_id':'test:'+x}for x in ('first','second')]}
            self.assertEqual(git('rev-parse','HEAD'),base)
            group=workspace/child['directory'];group.mkdir(parents=True)
            binding=group/'BINDING.json';binding.write_text('{}\n')
            prices.OPS.commit(workspace,[binding],'Freeze artificial binding')
            prices._publish_compile_binding(workspace,group)
            result=group/'RAW.json';result.write_text('{}\n')
            raw_commits.append(prices.OPS.commit(workspace,[result],'Save artificial raw result'))
            return {'status':'COMPILE_READY_FOR_AUDIT'}

        with patch.dict(os.environ,env),patch.object(prices,'_run_one',side_effect=synthetic_group):
            result=prices.run(request,execute=True,push=False,phase='compile',publish_binding=True)
        self.assertEqual(result['status'],'STEP_COMPILES_READY_FOR_AUDIT')
        self.assertEqual(git('branch','--show-current'),'codex/rpe-pauli-binding-prep-test-a2')
        published=git('ls-remote','--heads','origin','codex/rpe-pauli-binding-prep-test-a2*').splitlines()
        self.assertEqual(len(published),2)
        for line in published:
            files=git('diff','--name-only',base,line.split()[0]).splitlines()
            self.assertEqual([Path(path).name for path in files],['BINDING.json'])
        for commit in raw_commits:
            git('merge-base','--is-ancestor',commit,'HEAD')
        self.assertEqual(len(list((workspace/request['directory']).rglob('RAW.json'))),2)

    @unittest.skipUnless(shutil.which('bwrap'), 'bubblewrap is unavailable')
    def test_fixed_pauli_runtime_refuses_changed_input(self):
        path = self.root/'retained.json'
        path.write_text('{}\n')
        original = '/home/AbeHiromu/projects/artificial/retained.json'
        mount = dict(controller_path=original, path=str(path), kind='file',
                     sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        profile = dict(pauli_binding_publication=dict(legacy_runtime={'saved_config': mount}))
        self.assertEqual(P.pauli_calibration_runtime(profile)[original]['path'],str(path))
        path.write_text('{"changed":true}\n')
        with self.assertRaisesRegex(ValueError,'runtime file hash differs'):
            P.pauli_calibration_runtime(profile)

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
            start.wait();cache.materialize({'shared/raw/circuit.json':sha},self.root/'adoption',immutable=True)
        with ThreadPoolExecutor(max_workers=10)as pool:list(pool.map(adopt,range(10)))
        self.assertEqual((self.root/'adoption/shared/raw/circuit.json').read_bytes(),raw)
        self.assertTrue((self.root/'adoption/shared/raw/circuit.json').samefile(cache.path(sha)))
        cache.materialize({'shared/raw/circuit.json':sha}, self.root/'workspace')
        private = self.root/'workspace/shared/raw/circuit.json'
        self.assertFalse(private.samefile(cache.path(sha)))
        private.write_bytes(b'workspace change')
        self.assertEqual(cache.path(sha).read_bytes(), raw)
        self.assertEqual(list((self.root/'adoption').rglob('*.materialize-*')),[])

    def test_fixed_blas_profile_reaches_the_actual_synthetic_child(self):
        code="""import json,os
from pathlib import Path
keys=['OPENBLAS_CORETYPE','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']
Path(r'@WORK@/result.json').write_text(json.dumps({k:os.environ.get(k) for k in keys}))
"""
        task=self.task('signal',code)
        profile=dict(self.profile('local'),openblas_coretype='HASWELL')
        directory=self.root/'fixed-blas-attempt'
        outcome=P.execute(task,profile,directory)
        self.assertEqual(outcome['status'],'PREPARATION_COMPLETE')
        self.assertEqual(json.loads((directory/'result.json').read_text()),dict(
            OPENBLAS_CORETYPE='HASWELL',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'))

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

    def test_imported_bundle_is_reused_without_original_bytes_on_all_worker_profiles(self):
        from trottertracks import _pipeline_pbs_q3_worker as pipeline
        producer = self.root/'producer'
        subprocess.run(['git','clone','-q',str(self.root/'source'),str(producer)],check=True)
        (producer/'docs').mkdir()
        (producer/'docs'/'partial.json').write_text('fixed synthetic input')
        subprocess.run(['git','add','docs'],cwd=producer,check=True)
        subprocess.run(['git','-c','user.name=Test','-c','user.email=test@example.invalid',
                        'commit','-qm','partial'],cwd=producer,check=True)
        head = P._head(producer)
        bundle = self.root/'input.bundle'
        subprocess.run(['git','bundle','create',str(bundle),'HEAD','^'+self.source['commit']],
                       cwd=producer,check=True)
        sha = hashlib.sha256(bundle.read_bytes()).hexdigest()
        task = self.task('audit', "from pathlib import Path;Path('result.json').write_text(Path('docs/partial.json').read_text())",
                         source_workspace=True, return_git_delta=True, outputs=['workspace/result.json','workspace/docs/partial.json'],
                         input_git_bundles=[dict(sha256=sha,ref='HEAD')],
                         input_hashes=[dict(commit=head,path='docs/partial.json',
                                            sha256=hashlib.sha256(b'fixed synthetic input').hexdigest())])
        for host, worker in [('local',W),('local0-wsl',W),('server1',W),('server2',W),('pbs',pipeline)]:
            with self.subTest(host=host):
                profile = self.profile(host)
                cache = P.worker_cache(profile)
                self.assertEqual(P.missing_metadata_bundle_hashes(task,profile),[sha])
                cache.put_file(bundle,sha)
                self.assertEqual(P.dry_check(task['request'],profile)['scientific_calls'],0)
                metadata=P._metadata_repository_path(profile,self.root/'source')
                # Simulate metadata imported by a previous release that only
                # wrote IMPORTED.json. Preserve ref selectors on cleanup.
                (metadata.parent/'BUNDLE_HEADS.json').unlink()
                self.assertTrue(P.remember_imported_bundle(metadata,cache.path(sha),sha))
                cache.path(sha).unlink()
                self.assertEqual(P.missing_metadata_bundle_hashes(task,profile),[])
                payload = dict(hashes=[sha],direct_hashes=[],preparation_challenges=[task])
                self.assertEqual(worker.mailbox(profile,'artifacts-missing',payload),dict(missing=[]))
                # A regular input with the same bytes still needs the raw file.
                self.assertEqual(worker.mailbox(profile,'artifacts-missing',
                    dict(payload,direct_hashes=[sha])),dict(missing=[sha]))
                self.assertEqual(P.dry_check(task['request'],profile)['scientific_calls'],0)
                attempt=self.root/host/'spool'/task['job_id']/'attempt'
                result=P.execute(task,profile,attempt)
                self.assertEqual(result['status'],'PREPARATION_COMPLETE')
                self.assertEqual((attempt/'workspace/result.json').read_text(),'fixed synthetic input')
                metadata=P._metadata_repository_path(profile,self.root/'source')
                subprocess.run(['git','update-ref','-d','refs/rpe-metadata/'+sha+'/0'],cwd=metadata,check=True)
                self.assertEqual(P.missing_metadata_bundle_hashes(task,profile),[sha])
                self.assertEqual(worker.mailbox(profile,'artifacts-missing',payload),dict(missing=[sha]))
                cache.put_file(bundle,sha)
                self.assertEqual(P.dry_check(task['request'],profile)['scientific_calls'],0)
                self.assertEqual(P.missing_metadata_bundle_hashes(task,profile),[])
                if host == 'local':
                    # Keep the imported commit but remove one required blob
                    # from the derived store. Tip checks alone cannot prove it.
                    object_dir=metadata/'objects'
                    for pack in (object_dir/'pack').glob('*.pack'):
                        saved=self.root/pack.name
                        pack.rename(saved)
                        pack.with_suffix('.idx').unlink(missing_ok=True)
                        with saved.open('rb') as stream:
                            subprocess.run(['git','unpack-objects'],cwd=metadata,stdin=stream,check=True)
                    oid=subprocess.check_output(['git','rev-parse',head+':docs/partial.json'],
                                                cwd=metadata,text=True).strip()
                    (object_dir/oid[:2]/oid[2:]).unlink()
                    cache.path(sha).unlink()
                    with self.assertRaisesRegex(ValueError,'fixed Git input is unavailable'):
                        P.dry_check(task['request'],profile)
                    self.assertEqual(worker.mailbox(profile,'artifacts-missing',payload),dict(missing=[sha]))
                    cache.put_file(bundle,sha)
                    self.assertEqual(P.dry_check(task['request'],profile)['scientific_calls'],0)

    def test_failed_attempt_compacts_only_unchanged_source_and_restores_for_inspection(self):
        source=self.root/'source'
        for name in ('src/large.txt','scripts/keep.py','configs/keep.json','docs/results/seed.json'):
            path=source/name
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes((name.encode()+b'\n')*65536)
        (source/'.gitignore').write_text('*.tmp\n')
        subprocess.run(['git','add','.'],cwd=source,check=True)
        subprocess.run(['git','-c','user.name=Test','-c','user.email=test@example.invalid',
                        'commit','-qm','static tree'],cwd=source,check=True)
        self.source['commit']=P._head(source)
        profile=self.profile('local')
        task=self.task('compile', """from pathlib import Path
Path('scripts/partial.tmp').write_text('ignored partial')
Path('configs/keep.json').write_text('modified input')
Path('docs/results/result.json').write_text('partial evidence')
raise SystemExit(2)
""",source_workspace=True,outputs=['workspace/docs/results'])
        attempt=Path(profile['work_dir'])/task['job_id']/task['attempt_id']
        outcome=P.execute(task,profile,attempt)
        self.assertEqual(outcome['status'],'PREPARATION_FAILED')
        W.atomic(attempt/'ASSIGNMENT.json',task)
        W.atomic(attempt/'PROCESS_RETURN.json',dict(returncode=2))
        (attempt/'WORKER_PROCESS.lock').touch()
        work=attempt/'workspace'
        before=(work/'src/large.txt').read_bytes()
        W.save_state(attempt,'UNKNOWN')
        self.assertEqual(Storage.compact_failed_preparation_workspaces(attempt),0)
        self.assertTrue((work/'src/large.txt').exists())
        W.save_state(attempt,'TRANSFERRED')
        Storage.reclaim_transferred_attempt(dict(roots=[profile['work_dir']]),attempt)
        self.assertTrue((work/'src/large.txt').exists())  # ACK remains inexpensive
        used = Storage._used_bytes([profile['work_dir']])
        after, removed = Storage._reclaim([profile['work_dir']], [profile['work_dir']], used, 0)
        self.assertGreater(removed, 0)
        self.assertLess(after, used)
        self.assertFalse((work/'src/large.txt').exists())
        self.assertEqual((work/'scripts/partial.tmp').read_text(),'ignored partial')
        self.assertEqual((work/'configs/keep.json').read_text(),'modified input')
        self.assertEqual((work/'docs/results/result.json').read_text(),'partial evidence')
        self.assertEqual(P.read(attempt/'OUTCOME.json'),outcome)
        self.assertEqual(Storage.compact_failed_preparation_workspaces(attempt),0)
        Storage.restore_failed_workspace(attempt)
        self.assertEqual((work/'src/large.txt').read_bytes(),before)

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

    def test_failed_output_checkout_is_collectable_without_starting_provider(self):
        task=self.task('signal',"raise AssertionError('provider must not run')",source_workspace=True)
        attempt=self.root/'missing-output-workspace'
        failure=subprocess.CalledProcessError(254,['git','sparse-checkout','add'],stderr='missing object')
        with patch.object(P,'_include_declared_outputs',side_effect=failure):
            outcome=P.execute(task,self.profile('server1'),attempt)
        self.assertEqual(outcome['status'],'PREPARATION_FAILED')
        self.assertEqual(outcome['workspace_error'],dict(step='git sparse-checkout add',
            returncode=254,scientific_body_started=False))
        self.assertFalse((attempt/'JOB_STAGES.jsonl').exists())
        self.assertEqual(P.collect(task,self.profile('server1'),attempt),outcome)

    def test_failed_calibration_returns_only_partial_receipts(self):
        code = """from pathlib import Path
base = Path(r'@WORK@/workspace/case/df_prices/calibration/acquisition')
point = base/'m0/p0/axes_X'
point.mkdir(parents=True)
for name in ('PATH_RECEIPT.json', 'cosine.json', 'events_PRIVATE.json'):
    (point/name).write_text('{}')
(base/'STARTED_m0_p0_axesX.json').write_text('{}')
(base/'RESULT.json').write_text('{}')
raise SystemExit(2)
"""
        task = self.task('compile', code, outputs=['workspace/case'])
        task['request']['preparation']['provider'] = 'DF_CALIBRATION'
        task['request_sha256'] = P.digest(task['request'])
        profile = self.profile('local')
        attempt = self.root/'interrupted-calibration'
        outcome = P.execute(task, profile, attempt)
        self.assertEqual(outcome['status'], 'PREPARATION_FAILED')
        base = 'workspace/case/df_prices/calibration/acquisition/'
        self.assertEqual(set(outcome['artifacts']),
                         {base+name for name in ('m0/p0/axes_X/PATH_RECEIPT.json',
                                                 'm0/p0/axes_X/cosine.json', 'RESULT.json')})
        self.assertEqual(P.collect(task, profile, attempt), outcome)

    def test_worker_readiness_holds_missing_sparse_output_objects(self):
        source = self.root/'source'
        sibling = source/'docs/research/sibling.md'
        sibling.parent.mkdir(parents=True)
        sibling.write_text('tracked sibling of the audit output')
        subprocess.run(['git','add','docs'], cwd=source, check=True)
        subprocess.run(['git','-c','user.name=Test','-c','user.email=test@example.invalid',
                        'commit','-qm','fixed output ancestors'], cwd=source, check=True)
        self.source['commit'] = P._head(source)
        oid = subprocess.check_output(['git','rev-parse','HEAD:docs/research/sibling.md'],
                                      cwd=source, text=True).strip()
        (source/'.git/objects'/oid[:2]/oid[2:]).unlink()
        profile = self.profile('wsl')
        task = self.task('audit', "raise AssertionError('preflight must not run the provider')",
                         source_workspace=True, outputs=['workspace/docs/research/new-audit'])

        held = W.host_fingerprints(profile, [task])
        self.assertEqual(held['host_fingerprints'], {})
        self.assertIn('Git workspace preflight failed at git sparse-checkout add',
                      held['readiness_errors'][task['job_id']])
        with self.assertRaisesRegex(ValueError, 'Git workspace preflight failed'):
            P.dry_check(task['request'], profile)

        restored = subprocess.check_output(['git','hash-object','-w','docs/research/sibling.md'],
                                           cwd=source, text=True).strip()
        self.assertEqual(restored, oid)
        checked = W.host_fingerprints(profile, [task])
        self.assertEqual(checked['readiness_errors'], {})
        self.assertIn(task['job_id'], checked['host_fingerprints'])
        self.assertEqual(P.dry_check(task['request'], profile)['scientific_calls'], 0)

    def test_sparse_clone_keeps_only_declared_output_cones_for_plain_git_add(self):
        source=self.root/'source';archive=source/'docs/archive/unneeded.json';archive.parent.mkdir(parents=True);archive.write_text('{"historical":true}')
        subprocess.run(['git','add','docs'],cwd=source,check=True)
        subprocess.run(['git','-c','user.name=Test','-c','user.email=test@example.invalid','commit','-qm','historical-data'],cwd=source,check=True)
        self.source['commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
        blob=subprocess.check_output(['git','rev-parse','HEAD:docs/archive/unneeded.json'],cwd=source,text=True).strip()
        # Model a metadata-only worker seed with no historical artifact blob.
        (source/'.git/objects'/blob[:2]/blob[2:]).unlink()
        code="""from pathlib import Path
import subprocess
assert not Path('docs/archive/unneeded.json').exists()
p=Path('docs/provider-output/result.json');p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{"done":true}')
subprocess.run(['git','add',str(p)],check=True)
subprocess.run(['git','commit','--only','-qm','new result','--',str(p)],check=True)
"""
        task=self.task('audit',code,source_workspace=True,outputs=['workspace/docs/provider-output'])
        result=P.execute(task,self.profile('server1'),self.root/'sparse-attempt')
        self.assertEqual(result['status'],'PREPARATION_COMPLETE')
        self.assertFalse((self.root/'sparse-attempt/workspace/docs/archive/unneeded.json').exists())
        self.assertTrue((self.root/'sparse-attempt/workspace/docs/provider-output/result.json').is_file())
        self.assertEqual(archive.read_text(),'{"historical":true}')

    def test_signal_pack_restores_sparse_checkout_and_output_ancestors(self):
        source = self.root/'source'
        wanted = ['README.md', 'docs/research/sibling.md', 'LICENSES/license.txt']
        unrelated = 'docs/archive/large.json'
        for name in [*wanted, unrelated]:
            path = source/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name)
        subprocess.run(['git','add','.'], cwd=source, check=True)
        subprocess.run(['git','-c','user.name=Test','-c','user.email=test@example.invalid',
                        'commit','-qm','fixed tree'], cwd=source, check=True)
        self.source['commit'] = P._head(source)
        seed = self.root/'worker-source'
        subprocess.run(['git','clone','-q','--no-checkout',str(source),str(seed)], check=True)
        (seed/'leaf.py').write_bytes((source/'leaf.py').read_bytes())
        for name in [*wanted, unrelated]:
            oid = subprocess.check_output(['git','rev-parse','HEAD:'+name],cwd=source,text=True).strip()
            (seed/'.git/objects'/oid[:2]/oid[2:]).unlink()
        profile = dict(self.profile('wsl'), preparation_sources={self.source['commit']: str(seed)})
        cache = P.worker_cache(profile)
        bridge = DistributedPreparationBridge(SimpleNamespace(run_dir=self.root/'central',artifact_cache=cache), source)
        outputs = ['workspace/docs/research/new-result']
        view = dict(source=self.source, input_git_bundles=[])
        bundle = bridge._workspace_bundle(source, view, Path('workspace'), outputs)
        self.assertEqual(bundle, bridge._workspace_bundle(source, view, Path('workspace'), outputs))
        code = """from pathlib import Path
import subprocess
p=Path('docs/research/new-result/output.json')
p.parent.mkdir(parents=True)
p.write_text('{}')
subprocess.run(['git','add',str(p)],check=True)
subprocess.run(['git','commit','--only','-qm','signal','--',str(p)],check=True)
"""
        task = self.task('signal',code,outputs=outputs,source_workspace=True,input_git_bundles=[bundle])
        attempt = self.root/'wsl-attempt'
        outcome = P.execute(task,profile,attempt)
        self.assertEqual(outcome['status'], 'PREPARATION_COMPLETE')
        checkout = attempt/'workspace'
        self.assertEqual((checkout/'docs/research/sibling.md').read_text(), wanted[1])
        self.assertFalse((checkout/unrelated).exists())

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

    def test_required_readonly_source_view_is_resolved_before_body(self):
        inner=self.root/'inner';manifest=repo(inner)
        (inner/'leaf.py').write_text('print("inner fixed source")\n')
        subprocess.run(['git','add','leaf.py'],cwd=inner,check=True)
        subprocess.run(['git','-c','user.name=Test','-c','user.email=test@example.invalid','commit','-qm','inner'],cwd=inner,check=True)
        manifest={'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=inner,text=True).strip(),
                  'files':{'leaf.py':hashlib.sha256((inner/'leaf.py').read_bytes()).hexdigest()}}
        task=self.task('signal',"from pathlib import Path;import os,json;v=json.loads(os.environ['RPE_PREPARATION_PATH_MAP']);Path(r'@WORK@/result.json').write_text(next(iter(v.values())))",
            required_views={'inner':{'source':manifest,'controller_path':'/old/server2/inner'}})
        profile=self.profile('server1')
        with self.assertRaisesRegex(ValueError,'SOURCE view missing'):P.current_preparation_fingerprint(task,profile)
        profile['preparation_sources']={self.source['commit']:str(self.root/'source'),manifest['commit']:str(inner)}
        fp=P.current_preparation_fingerprint(task,profile)
        self.assertEqual(fp['views']['view:inner'],str(inner))
        P.execute(task,profile,self.root/'view-attempt')
        self.assertEqual((self.root/'view-attempt/result.json').read_text(),str(inner))
        self.assertEqual((inner/'leaf.py').read_text(),'print("inner fixed source")\n')

    def test_required_private_provider_config_needs_mapping_and_matching_bytes(self):
        config=self.root/'private-config.json';config.write_text('{"fixed":"original-data"}')
        provider=self.root/'private-provider';provider.mkdir();(provider/'provider.py').write_text('FIXED=1\n')
        required={'config':{'controller_path':'/server2/config.json','kind':'file','sha256':hashlib.sha256(config.read_bytes()).hexdigest()},
                  'provider':{'controller_path':'/server2/provider','kind':'directory','files':{'provider.py':hashlib.sha256((provider/'provider.py').read_bytes()).hexdigest()}}}
        task=self.task('audit','pass',required_paths=required);profile=self.profile('local')
        with self.assertRaisesRegex(ValueError,'not mapped'):P.current_preparation_fingerprint(task,profile)
        profile['preparation_paths']={'config':str(config),'provider':str(provider)}
        fp=P.current_preparation_fingerprint(task,profile)
        self.assertEqual(fp['runtime_paths']['provider']['path'],str(provider))
        config.write_text('{"fixed":"different-data"}')
        with self.assertRaisesRegex(ValueError,'file hash differs'):P.current_preparation_fingerprint(task,profile)
        self.assertFalse(Path(profile['work_dir']).exists())

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

    def test_audit_bundle_carries_raw_commit_from_another_branch(self):
        source = self.root/'source'
        def git(*args, cwd=source):
            return subprocess.check_output(['git', *args], cwd=cwd, text=True).strip()
        group = Path('docs/research/pauli/group')
        raw = group/'calibration/run01'
        (source/raw).mkdir(parents=True)
        for name in (group/'CONFIGURATION.json', group/'calibration/BINDING.json',
                     raw/'RESULT.json', raw/'COMPILE_RECEIPTS.json'):
            (source/name).write_text('{}\n')
        git('add', str(group))
        git('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'saved compile')
        raw_commit = git('rev-parse', 'HEAD')
        git('checkout', '--detach', self.source['commit'])
        state = dict(status='COMPILE_READY_FOR_AUDIT', driver_spec={'base':str(group/'calibration')},
                     raw_commit=raw_commit, inputs_commit=raw_commit)
        declarations = pauli_audit_raw_inputs(str(Path('workspaces/provider')/group/'PRICE_STATE.json'), state)
        locators, _ = fixed_input_hashes(source, declarations, {})
        self.assertEqual(len(locators), 4)
        cache = ArtifactCache(self.root/'cache')
        bridge = DistributedPreparationBridge(SimpleNamespace(artifact_cache=cache, run_dir=self.root/'run'), source)
        bundles = bridge._bundle(source, self.source['commit'], locators)
        worker = self.root/'worker'
        git('init', str(worker))
        git('fetch', '--depth=1', str(source), self.source['commit'], cwd=worker)
        for bundle in bundles:
            git('bundle', 'unbundle', str(cache.path(bundle['sha256'])), cwd=worker)
        for locator in locators:
            self.assertEqual(git('show', locator['commit']+':'+locator['path'], cwd=worker), '{}')

        # A new worker starts with the fixed SOURCE only, while the central
        # checkout has advanced to the producer commit.
        git('checkout', '--detach', raw_commit)
        bridge.settings['bundle_fixed_source_only'] = True
        bundles = bridge._bundle(source, self.source['commit'], locators)
        self.assertEqual(len(bundles), 1)
        clean = self.root/'fixed-worker'
        git('init', str(clean))
        git('fetch', '--depth=1', str(source), self.source['commit'], cwd=clean)
        bundle = str(cache.path(bundles[0]['sha256']))
        git('bundle', 'verify', bundle, cwd=clean)
        git('bundle', 'unbundle', bundle, cwd=clean)
        for locator in locators:
            self.assertEqual(git('show', locator['commit']+':'+locator['path'], cwd=clean), '{}')


if __name__=='__main__':unittest.main()


def test_nested_worker_path_is_not_rewritten_twice():
    from trottertracks.rpe_preparation_worker import rewrite_bound_argument
    root='/controller'
    worker=root+'/parts/one/worker/attempt'
    script=root+'/scripts/run_signal.py'
    bindings={root:worker+'/workspace',script:worker+'/private/run_signal.py'}
    assert rewrite_bound_argument(script,bindings)==worker+'/private/run_signal.py'
    assert rewrite_bound_argument(root+'/data/INPUT.json',bindings)==worker+'/workspace/data/INPUT.json'
    assert rewrite_bound_argument('--input='+script,bindings)=='--input='+worker+'/private/run_signal.py'


def test_in_progress_stage_hold_revisits_only_its_received_owner(tmp_path):
    from unittest.mock import Mock
    bridge=DistributedPreparationBridge.__new__(DistributedPreparationBridge)
    bridge.executor=SimpleNamespace(_user_stop_requested=lambda:False)
    bridge.recovery_evidence=Mock(return_value={'job_id':'prep-artificial','state':'RUNNING'})
    reason='DISTRIBUTED_PREPARATION_IN_PROGRESS:DF_SIGNALS:prep-artificial'
    assert bridge.hold_evidence(tmp_path,reason)['state']=='RUNNING'
    assert not bridge.recoverable_transport(tmp_path,reason)
    bridge.recovery_evidence.return_value={'job_id':'prep-artificial','state':'ACCEPTED'}
    assert bridge.recoverable_transport(tmp_path,reason)


def test_fixed_runtime_hash_cache_rechecks_changed_file(tmp_path,monkeypatch):
    from trottertracks import rpe_preparation_worker as worker
    path=tmp_path/'fixture';path.write_bytes(b'first')
    expected=hashlib.sha256(b'first').hexdigest()
    with monkeypatch.context() as patcher:
        patcher.setattr(worker.time,'time_ns',lambda:path.stat().st_ctime_ns+2_000_000_000)
        assert worker.fixed_file_sha(path)==expected
        patcher.setattr(worker,'file_digest',lambda *_:pytest.fail('unchanged file rehashed'))
        assert worker.fixed_file_sha(path)==expected
    path.write_bytes(b'other')
    assert worker.fixed_file_sha(path)==hashlib.sha256(b'other').hexdigest()


def test_fixed_openblas_runtime_requires_profile_and_actual_dispatch(monkeypatch):
    import sys,json
    from types import SimpleNamespace
    profile={'interpreter':sys.executable,'openblas_coretype':'HASWELL'}
    need={'openblas_coretype':'HASWELL'}
    probe={'executable':sys.executable,'version_info':[3,11],'packages':{}}
    observed=[]
    def run(*args,**kwargs):
        observed.append(kwargs)
        return SimpleNamespace(returncode=0,stdout=json.dumps(probe),stderr='Core: Haswell\n')
    monkeypatch.setattr(P.subprocess,'run',run)
    runtime,_=P._runtime(profile,need)
    assert runtime['openblas_coretype']=='HASWELL'
    assert observed[0]['env']['OPENBLAS_CORETYPE']=='HASWELL'
    assert observed[0]['env']['OPENBLAS_VERBOSE']=='2'
    assert 'numpy' in json.loads(observed[0]['input'])['imports']
    import pytest
    with pytest.raises(ValueError,match='profile differs'):
        P._runtime(dict(profile,openblas_coretype='NEHALEM'),need)
    monkeypatch.setattr(P.subprocess,'run',lambda *_a,**_k:
        SimpleNamespace(returncode=0,stdout=json.dumps(probe),stderr='Core: Nehalem\n'))
    with pytest.raises(ValueError,match='actual OpenBLAS'):
        P._runtime(profile,need)


def test_verified_own_hardlink_reuses_stable_identity_but_changed_bytes_rehash(tmp_path,monkeypatch):
    from trottertracks import rpe_preparation_artifacts as A
    import time,hashlib,base64,pytest
    cache=A.ArtifactCache(tmp_path/'cache')
    raw=b'fixture-immutable'
    sha=hashlib.sha256(raw).hexdigest()
    cache.put({sha:base64.b64encode(raw).decode()})
    real=time.time_ns
    monkeypatch.setattr(A.time,'time_ns',lambda:real()+2_000_000_000)
    calls=[]
    digest=A.file_digest
    monkeypatch.setattr(A,'file_digest',lambda p:(calls.append(p),digest(p))[1])
    assert cache.missing([sha])==[]
    cache.materialize({'input.json':sha},tmp_path/'final',immutable=True)
    assert cache.missing([sha])==[]
    assert len(calls)==1
    cache.path(sha).write_bytes(b'changed-content')
    with pytest.raises(ValueError,match='Corrupt cached artifact'):
        cache.missing([sha])
