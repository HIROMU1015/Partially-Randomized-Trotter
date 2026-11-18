"""Preparation concurrency and exact reuse with artificial provider leaves."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from contextlib import nullcontext
import importlib.util
import json
import hashlib
import shutil
import subprocess
from pathlib import Path
from random import Random
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import pr_candidate_completion as C
from trottertracks import rpe_preparation_bridge as B
from trottertracks.rpe_preparation_artifacts import ArtifactCache
from trottertracks import rpe_preparation_worker as W
from trottertracks.rpe_q3_store import JobStore


class CompletionBridgeTests(unittest.TestCase):
    def test_native_runtime_paths_reuse_original_seals_and_ignore_optional_cache(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);provider=root/'provider';provider.mkdir();code=provider/'worker.py';code.write_text('# fixed native fixture\n')
            code_sha=hashlib.sha256(code.read_bytes()).hexdigest();config=root/'INPUT_CONFIG.json'
            C.put(config,{'sealed_files':[{'role':'provider/runtime','path':str(code),'sha256':code_sha},
                {'role':'unrelated-history','path':str(root/'absent'),'sha256':'0'*64}]})
            document={'runtime':{'provider_root':str(provider),'input_config':str(config),'old_cache':str(root/'optional-missing-cache')}}
            paths=B.collect_required_paths([document]);self.assertEqual(len(paths),2)
            directory=next(row for row in paths.values()if row['kind']=='directory')
            self.assertEqual(directory['files'],{'worker.py':code_sha})
            file=next(row for row in paths.values()if row['kind']=='file')
            self.assertEqual(file['sha256'],hashlib.sha256(config.read_bytes()).hexdigest())
            self.assertFalse(any('old_cache'in row['controller_path']for row in paths.values()))
            code.write_text('# changed source\n')
            with self.assertRaisesRegex(ValueError,'sealed provider/runtime file missing or changed'):
                B.collect_required_paths([document])
            config.unlink()
            with self.assertRaisesRegex(ValueError,'input_config missing'):
                B.collect_required_paths([document])

    def test_nested_git_locators_and_runtime_views_are_bound_without_import(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp)
            def git(*args):return subprocess.check_output(['git',*args],cwd=root,text=True,stderr=subprocess.DEVNULL).strip()
            git('init');git('config','user.name','Fixture');git('config','user.email','fixture@example.invalid')
            payload=root/'CONFIGURATION.json';payload.write_text('{"selector": {"a": 1}, "preserve": true}\n')
            view_file=root/'src/trottertracks/pr_price_rules.py';view_file.parent.mkdir(parents=True);view_file.write_text('# metadata only, never imported\n')
            git('add','.');git('commit','-qm','fixed nested fixture');commit=git('rev-parse','HEAD')
            original={'nested':[{'input':{'commit':commit,'path':'CONFIGURATION.json','selector':['selector']}}],
                      'd2_view':str(root)}
            before=deepcopy(original)
            inputs,working=B.collect_fixed_inputs(root,[original])
            self.assertEqual(inputs,[{'commit':commit,'path':'CONFIGURATION.json','sha256':hashlib.sha256(payload.read_bytes()).hexdigest()}])
            self.assertEqual(working,[]);self.assertEqual(original,before)
            views=B.collect_required_views([original]);row=next(iter(views.values()))
            self.assertEqual(row['source']['commit'],commit);self.assertEqual(row['controller_path'],str(root))
            self.assertEqual(row['source']['files']['src/trottertracks/pr_price_rules.py'],hashlib.sha256(view_file.read_bytes()).hexdigest())
            view_file.write_text('# changed metadata SOURCE\n')
            with self.assertRaisesRegex(ValueError,'operational SOURCE view changed'):
                B.collect_required_views([original])

    def test_missing_fixed_input_metadata_holds_before_submission(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);engine=type('Engine',(),{'run_dir':root/'run'})()
            bridge=B.DistributedPreparationBridge(engine,root)
            completion=type('Completion',(),{'source':'fixed'})()
            with patch.object(bridge,'build_request',side_effect=ValueError('missing fixed input object')):
                with self.assertRaisesRegex(C.Held,'OPERATIONAL_HOLD.*INPUT_OR_SOURCE_METADATA_UNAVAILABLE'):
                    bridge.call(completion,['fixed'],root,'DF_SIGNALS')
            self.assertFalse((root/'DF_SIGNALS.DISTRIBUTED.json').exists())

    def test_prior_owner_markers_are_checked_at_original_path_before_submission(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);owner=root/'original-owner';output=owner/'docs/selected';output.mkdir(parents=True)
            directory=root/'new-controller';directory.mkdir()
            input_file=directory/'REQUEST.json';C.put(input_file,{'workspace':str(owner),'directory':'docs/selected'})
            marker=output/'SIGNALS_STARTED.json';C.put(marker,{'pid':123,'automatic_retry':False})
            engine=type('Engine',(),{'run_dir':root/'run'})()
            bridge=B.DistributedPreparationBridge(engine,root)
            completion=type('Completion',(),{'source':'fixed'})()
            with patch.object(bridge,'build_request',side_effect=AssertionError('must not clone/submit')):
                with self.assertRaisesRegex(C.Held,'ORIGINAL_OWNER_RUNNING_OR_UNKNOWN'):
                    bridge.call(completion,['fixed','--request',str(input_file)],directory,'PAULI_SIGNALS')
            self.assertFalse((directory/'PAULI_SIGNALS.DISTRIBUTED.json').exists())
            terminal=output/'run01/result.json';C.put(terminal,{'status':'ENTRY_CARRIERS_COMPLETE_NOT_RPE_QUALIFIED'})
            observation=bridge.inspect_prior_owner(['fixed',str(input_file)],directory,'PAULI_SIGNALS')
            self.assertFalse(observation['pending']);self.assertFalse(observation['failed'])
            C.put(output/'STOP.json',{'status':'PRESERVED_STOP'})
            observation=bridge.inspect_prior_owner(['fixed',str(input_file)],directory,'PAULI_SIGNALS')
            self.assertTrue(observation['failed'])
            self.assertTrue(marker.is_file());self.assertEqual(json.loads(input_file.read_text())['workspace'],str(owner))

    def test_operational_hold_reconciles_same_attempt_but_failed_science_does_not_retry(self):
        for first_status in ('HOST_ELIGIBILITY_HOLD','PREPARATION_FAILED','UNKNOWN'):
            with self.subTest(first_status=first_status),tempfile.TemporaryDirectory()as tmp:
                root=Path(tmp)
                class Engine:
                    run_dir=root/'run'
                    def __init__(self):self.store=JobStore(root/'store.sqlite3');self.requests=[];self.result_calls=0;self.starts=0
                    def submit_preparation(self,request,**kwargs):
                        self.requests.append(deepcopy(request))
                        self.store.reserve('PREPARATION:signal','job',request,1,True)
                        if self.store.row('job')['state']=='QUEUED':
                            row=self.store.claim('original-host',1)[0];self.attempt=row['attempt_id'];self.starts+=1
                            self.store.update('original-host','job',self.attempt,'RUNNING')
                        return 'job'
                    def preparation_result(self,*args,**kwargs):
                        self.result_calls+=1
                        if self.result_calls==1:
                            if first_status=='UNKNOWN':self.store.update('original-host','job',self.attempt,'UNKNOWN')
                            return {'status':first_status}
                        self.store.complete('original-host','job',self.attempt,W.digest(self.requests[-1]),{})
                        self.store.adopt('job')
                        return {'status':'PREPARATION_COMPLETE','artifact_directory':str(root),'artifacts':{}}
                class Fake(C.Completion):
                    def pauli_configuration(self,request,d,allow_science=True):return request['configuration'],{}
                    def initial_quote(self,*args):return {'complete':True,'rows':[{'unit_price_RZ':1}]}
                    def pauli_signals(self,config,prepared,d):
                        self.call(['fixed'],d,'PAULI_SIGNALS',science=True)
                        return {'packet_source':{'commit':'saved','path':'PACKET'}}
                    def price(self,config,cref,prepared,r,d,initial_quote=None):
                        q=self.initial_quote();C.put(d/'FINAL_QUOTE.json',q);return q
                    def service(self,mode,*args,**kwargs):return {'binding_id':'binding','total_configuration_id':'total'}
                c=Fake.__new__(Fake);c.directory=root;c.settings={};c.source='fixed';c.lock=threading.RLock()
                c.shared_preparation_locks={};c.state={'records':{}};c.state_path=root/'STATE.json';c.registry=root/'REGISTRY.json'
                C.put(c.registry,{'entries':[]})
                ex=Engine();bridge=B.DistributedPreparationBridge(ex,root,{'budgets':{'signal':1}},approved=True);c.bind_stage_executor(bridge)
                frozen={'route':{'kind':'preparation'},'preparation':{'artificial':True},'original_contract':{'source':'fixed'}}
                request={'method':'PR_PAULI','configuration':{'id':'a','spec':{'L_D':4,'formula':'2nd','kappa':{'value':'2'},'delta_pf':'1'},'rounds':[]}}
                with patch.object(C,'ROOT',root),patch.object(C,'publish',return_value='saved'),\
                     patch.object(C,'value',side_effect=lambda loc:C.read(root/loc['path'])),\
                     patch.object(bridge,'build_request',return_value=frozen)as builder,\
                     patch.object(bridge,'_import_result'):
                    first=c.complete(request)
                    attempt=ex.store.row('job')['attempt_id']
                    second=c.complete(request)
                self.assertEqual(ex.starts,1);self.assertEqual(ex.store.row('job')['attempt_id'],attempt)
                self.assertEqual(builder.call_count,1)
                if first_status=='HOST_ELIGIBILITY_HOLD':
                    self.assertEqual(first['status'],'WAITING_DISTRIBUTED_PREPARATION')
                    self.assertEqual(second['status'],'READY_FOR_SEARCH')
                    self.assertEqual(len(ex.requests),2);self.assertEqual(ex.requests[0],ex.requests[1])
                    self.assertEqual(len([e for e in ex.store.events()if e['event']=='ADOPT']),1)
                else:
                    self.assertTrue(first['scientific_failure']);self.assertTrue(second['resumed'])
                    self.assertEqual(len(ex.requests),1)

    def test_runtime_requirements_follow_original_input_and_fixed_worker_guard(self):
        with tempfile.TemporaryDirectory()as tmp:
            script=Path(tmp)/'fixed.py';script.write_text("if (numpy.__version__, scipy.__version__) != ('1.26.4','1.14.1'): raise ValueError('runtime')\n")
            inputs=[{'configuration':{'profile':{'qiskit_version':'1.3.0','compiler':'a'*64}}}]
            requirements,bindings=B.contract_runtime(inputs,script,{})
            self.assertEqual(requirements['packages'],{'qiskit':'1.3.0','numpy':'1.26.4','scipy':'1.14.1'})
            self.assertEqual(bindings,['a'*64])
            with self.assertRaisesRegex(ValueError,'conflicts with original input'):
                B.contract_runtime(inputs,script,{'packages':{'qiskit':'2.0.0'}})

    def test_pf_wrapper_carries_distinct_provider_manifest_and_rejects_change(self):
        with tempfile.TemporaryDirectory()as tmp:
            top=Path(tmp);transport=top/'transport';transport.mkdir();provider=top/'provider';provider.mkdir()
            def git(*args):
                return subprocess.check_output(['git',*args],cwd=provider,text=True,stderr=subprocess.DEVNULL).strip()
            git('init');git('config','user.name','Fixture');git('config','user.email','fixture@example.invalid')
            files={}
            for index in range(5):
                name='scripts/fixed_provider_'+str(index)+'.py';path=provider/name;path.parent.mkdir(exist_ok=True)
                path.write_text('# fixed provider fixture '+str(index)+'\n');files[name]=hashlib.sha256(path.read_bytes()).hexdigest()
            git('add','.');git('commit','-qm','fixed PF fixture');source=git('rev-parse','HEAD')
            wrapper=transport/'wrapper.py';wrapper.write_text('# transport adapter fixture\n')
            d=provider/'docs/owned';d.mkdir(parents=True);C.put(d/'REQUEST.json',{'scope':'fixture'})
            executor=type('Executor',(),{'run_dir':top/'run','artifact_cache':ArtifactCache(top/'cas')})()
            bridge=B.DistributedPreparationBridge(executor,transport)
            completion=type('Completion',(),{'source':source,'preparation_root':provider,
                'preparation_source_manifest':{'commit':source,'files':files}})()
            task=bridge.build_request(completion,[sys.executable,'-B',str(wrapper),'--input',str(d/'REQUEST.json')],d,'PF_COMPILE')
            self.assertEqual(task['preparation']['source'],{'commit':source,'files':files})
            self.assertEqual(task['preparation']['path_bindings'][str(transport)],'@TRANSPORT@')
            task['preparation']['artificial']=True;task['request_id']='pf-fixture'
            assignment={'job_id':'pf-fixture','request':task,'request_sha256':W.digest(task),'source_manifest':task['preparation']['source']}
            profile={'host_id':'fixture','root':str(provider),'interpreter':sys.executable,'work_dir':str(top/'spool')}
            W.current_preparation_fingerprint(assignment,profile)
            (provider/next(iter(files))).write_text('# changed source\n')
            with self.assertRaisesRegex(ValueError,'SOURCE hash differs'):
                W.current_preparation_fingerprint(assignment,profile)

    def test_new_locator_bundle_reaches_distinct_owned_provider_workspace(self):
        with tempfile.TemporaryDirectory()as tmp:
            top=Path(tmp);root=top/'controller';root.mkdir()
            def git(*args,cwd=root):
                return subprocess.check_output(['git',*args],cwd=cwd,text=True,stderr=subprocess.DEVNULL).strip()
            git('init');git('config','user.name','Fixture');git('config','user.email','fixture@example.invalid')
            helper=root/'src/trottertracks/rpe_preparation_paths.py';helper.parent.mkdir(parents=True)
            shutil.copy2(ROOT/'src/trottertracks/rpe_preparation_paths.py',helper)
            fixed_provider=root/'scripts/run_pauli_split_cal.py';fixed_provider.parent.mkdir()
            fixed_provider.write_text('# artificial source guard only\n')
            script=root/'scripts/artificial_stage.py'
            script.write_text('''import hashlib,json,os,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from trottertracks.rpe_preparation_paths import resolve
raw=Path(sys.argv[1]).read_bytes();request=json.loads(raw)
workspace=Path(resolve(request['workspace']))
config=json.loads(subprocess.check_output(['git','show',request['configuration']['commit']+':'+request['configuration']['path']],cwd=workspace))
out=workspace/request['directory'];out.mkdir(parents=True,exist_ok=True)
(out/'TOUCHED.json').write_text(json.dumps({'fixture':True}))
subprocess.run(['git','add',str(out/'TOUCHED.json')],cwd=workspace,check=True)
subprocess.run(['git','commit','-qm','artificial provider receipt'],cwd=workspace,check=True)
publication=subprocess.check_output(['git','rev-parse','HEAD'],cwd=workspace,text=True).strip()
Path(sys.argv[2]).write_text(json.dumps({'configuration':config,'input_sha256':hashlib.sha256(raw).hexdigest(),'workspace':str(workspace),'publication':publication}))
''')
            historic=root/'old'/'CONFIGURATION.json';historic.parent.mkdir()
            historic.write_text('{"previous_proof": 17}\n')
            git('add','.');git('commit','-qm','historical fixed input');historic_commit=git('rev-parse','HEAD')
            historic_blob=git('rev-parse',historic_commit+':old/CONFIGURATION.json')
            historic.unlink()
            git('add','.');git('commit','-qm','fixed artificial provider');source=git('rev-parse','HEAD')
            provider=top/'original-provider';git('clone','-q',str(root),str(provider))
            # A real remote host has a separate Git object database. Clone
            # before the controller creates its unpublished input commits.
            view=top/'fixed-view';git('clone','-q',str(root),str(view))
            # A source checkout can resolve the historical commit and lack the
            # particular historical object. The new bundle must supply it.
            (view/'.git'/'objects'/historic_blob[:2]/historic_blob[2:]).unlink()
            config=root/'docs/CONFIGURATION.json';config.parent.mkdir();config.write_text('{"scientific_fixture": "immutable"}\n')
            git('add','.');git('commit','-qm','new controller input');config_commit=git('rev-parse','HEAD')
            d=root/'docs/candidate';d.mkdir()
            request=d/'REQUEST.json';request.write_text(json.dumps({'workspace':str(provider),'directory':'docs/provider-output',
                'configuration':{'commit':config_commit,'path':'docs/CONFIGURATION.json'},
                'historical':{'commit':historic_commit,'path':'old/CONFIGURATION.json'}})+'\n')
            pinned_helper=hashlib.sha256(helper.read_bytes()).hexdigest()
            helper.write_bytes(helper.read_bytes()+b'\n# New controller transport declaration.\n')
            (root/'shared-history.bin').write_bytes(Random(20261118).randbytes(1024 * 1024))
            git('add','.');git('commit','-qm','frozen task input')
            executor=type('Executor',(),{'run_dir':top/'run','artifact_cache':ArtifactCache(top/'controller-cas')})()
            git('checkout', '-b', 'earlier-result', source)
            (root/'unrelated.json').write_text('{"another_candidate": true}\n')
            git('add', 'unrelated.json');git('commit', '-qm', 'unrelated producer result')
            unrelated = git('rev-parse', 'HEAD')
            earlier = 'refs/rpe-preparation/'+B.digest(str(executor.run_dir))[:16]+'/earlier-result'
            git('update-ref', earlier, unrelated)
            git('checkout', '-')
            bridge=B.DistributedPreparationBridge(executor,root,{'bundle_validation_sources':{source:str(view)}})
            completion=type('Completion',(),{'source':source})()
            task=bridge.build_request(completion,[sys.executable,'-B',str(script),str(request),str(d/'RESULT.json')],d,'PAULI_SIGNALS')
            bundles = task['preparation']['input_git_bundles']
            base, inputs = bundles
            self.assertGreater(executor.artifact_cache.path(base['sha256']).stat().st_size, 1024 * 1024)
            self.assertLess(executor.artifact_cache.path(inputs['sha256']).stat().st_size, 64 * 1024)
            other_run = type('Executor', (), {'run_dir': top/'other-run', 'artifact_cache': executor.artifact_cache})()
            other_bridge = B.DistributedPreparationBridge(other_run, root)
            locators = task['preparation']['input_hashes']
            self.assertEqual(other_bridge._bundle(root, source, locators), bundles)
            tracked_base, tracked = bridge._bundle(root, source, locators[1:], tracked_workspace=True)
            self.assertEqual(tracked_base, base)
            self.assertNotEqual(tracked['sha256'], inputs['sha256'])
            self.assertLess(executor.artifact_cache.path(tracked['sha256']).stat().st_size, 64 * 1024)
            self.assertTrue(tracked['checkout_input_head'])
            tracked_view = W._clone(view, top/'tracked-view', [tracked_base, tracked], executor.artifact_cache)
            self.assertEqual(git('rev-parse', 'HEAD^{tree}', cwd=tracked_view), git('rev-parse', 'HEAD^{tree}'))
            self.assertEqual(task['preparation']['workspaces']['provider']['source']['files'][
                'src/trottertracks/rpe_preparation_paths.py'],pinned_helper)
            fixed_provider.write_text('# changed provider body\n')
            with self.assertRaisesRegex(ValueError,'Provider workspace SOURCE differs'):
                bridge.build_request(completion,[sys.executable,'-B',str(script),str(request),str(d/'RESULT.json')],d,'PAULI_SIGNALS')
            fixed_provider.write_text('# artificial source guard only\n')
            root_hashes={v['sha256']for v in task['preparation']['input_git_bundles']}
            provider_hashes={v['sha256']for v in task['preparation']['workspaces']['provider']['input_git_bundles']}
            self.assertTrue(root_hashes<=provider_hashes)
            task['preparation']['artificial']=True;task['request_id']='fixture'
            assignment={'job_id':'fixture','request':task,'request_sha256':W.digest(task),'source_manifest':task['preparation']['source']}
            profile={'host_id':'simulated-other-host','root':str(view),'interpreter':sys.executable,
                     'work_dir':str(top/'worker-spool'),'preparation_sources':{source:str(view)},
                     'artifact_cache_directory':str(top/'worker-cas')}
            hashes=set(task['preparation']['inputs'].values())|root_hashes|provider_hashes
            ArtifactCache(top/'worker-cas').put(executor.artifact_cache.get(hashes))
            self.assertNotEqual(subprocess.run(['git','cat-file','-e',config_commit],cwd=view,
                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode,0)
            attempt=top/'worker-attempt';outcome=W.execute(assignment,profile,attempt)
            self.assertEqual(outcome['status'],'PREPARATION_COMPLETE')
            result=json.loads((attempt/'workspace/docs/candidate/RESULT.json').read_text())
            self.assertEqual(result['configuration'],{'scientific_fixture':'immutable'})
            self.assertEqual(result['input_sha256'],hashlib.sha256(request.read_bytes()).hexdigest())
            self.assertEqual(git('show',historic_commit+':old/CONFIGURATION.json',cwd=attempt/'workspace'),'{"previous_proof": 17}')
            unexpected = subprocess.run(['git', 'cat-file', '-e', unrelated], cwd=attempt/'workspace', capture_output=True)
            self.assertNotEqual(unexpected.returncode, 0)
            self.assertNotEqual(result['workspace'],str(provider))
            self.assertEqual(git('rev-parse','HEAD',cwd=view),source)
            self.assertFalse((provider/'docs/provider-output').exists())
            original_head=git('rev-parse','HEAD')
            bridge._import_result(dict(outcome,artifact_directory=str(attempt)),task)
            published=json.loads(git('show',result['publication']+':docs/provider-output/TOUCHED.json'))
            self.assertEqual(published,{'fixture':True})
            self.assertEqual(git('rev-parse','HEAD'),original_head)
            self.assertEqual(request.read_bytes(),(attempt/'workspace/docs/candidate/REQUEST.json').read_bytes())
            # A returned --output file describes the previous execution host.
            # It is dependency evidence, not the next task's runtime request.
            returned=d/'RESULT.json';returned.write_text(json.dumps(dict(result,workspace='/not-on-controller/old-attempt')))
            next_task=bridge.build_request(completion,[sys.executable,'-B',str(script),str(request),'--output',str(returned)],d,'PAULI_AUDIT')
            self.assertNotIn('/not-on-controller/old-attempt',next_task['preparation']['path_bindings'])
            self.assertIn(str(provider),next_task['preparation']['path_bindings'])

    def test_incomplete_descriptor_locator_still_holds_before_signal(self):
        c=C.Completion.__new__(C.Completion);c.settings={};c.state={'records':{}}
        config={'id':'fixture','spec':{'L_D':4,'binding':{'H':'fixed'}}}
        prepared={'descriptor':{'source':{'commit':'fixed'},'descriptor':{'tail':'own'}}}
        with patch.object(c,'call',side_effect=AssertionError('must not start signal')):
            with self.assertRaisesRegex(C.Held,'PAULI_DESCRIPTOR_SOURCE_UNRESOLVED'):
                c.pauli_signals(config,prepared,Path('/unused'))

    def test_two_declared_candidates_share_inflight_physical_work(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp)
            class Fake(C.Completion):
                def pauli_configuration(self,request,d,allow_science=True):return request['configuration'],{}
                def initial_quote(self,*args):return {'complete':True,'rows':[{'unit_price_RZ':1}]}
                def pauli_signals(self,config,prepared,d):
                    with self.lock:self.signal_count+=1
                    self.branch_barrier.wait(timeout=2)
                    return {'packet_source':{'commit':'saved','path':'PACKET'}}
                def price(self,config,cref,prepared,r,d,initial_quote=None):
                    with self.lock:self.compile_count+=1
                    self.branch_barrier.wait(timeout=2)
                    q=self.initial_quote();C.put(d/'FINAL_QUOTE.json',q);return q
                def service(self,mode,*args,**kwargs):
                    if mode=='intake':return {'binding_id':'binding','total_configuration_id':'total'}
                    raise AssertionError(mode)
            c=Fake.__new__(Fake);c.directory=root;c.settings={};c.source='fixed'
            c.lock=threading.RLock();c.shared_preparation_locks={};c.stage_executor=object()
            c.state={'records':{}};c.state_path=root/'STATE.json';c.registry=root/'REGISTRY.json'
            C.put(c.registry,{'entries':[]});c.branch_barrier=threading.Barrier(2)
            c.signal_count=c.compile_count=0
            cfg={'id':'a','spec':{'L_D':4,'formula':'2nd','kappa':{'value':'2'},'delta_pf':'1',
                'binding':{'H':'fixed'},'split':{'dictionary':'fixed'}},'rounds':[]}
            other=deepcopy(cfg);other['id']='b';other['spec']['kappa']['value']='3'
            with patch.object(C,'ROOT',root),patch.object(C,'publish',return_value='saved'),\
                 patch.object(C,'value',side_effect=lambda loc:C.read(root/loc['path'])):
                with ThreadPoolExecutor(max_workers=2)as pool:
                    results=list(pool.map(c.complete,[{'method':'PR_PAULI','configuration':cfg},
                                                      {'method':'PR_PAULI','configuration':other}]))
            self.assertEqual([r['status']for r in results],['READY_FOR_SEARCH']*2)
            self.assertEqual((c.signal_count,c.compile_count),(1,1))
            self.assertEqual(sum('reused_physical_configuration'in r for r in results),1)

    def test_heavy_call_uses_queue_recovery_before_started_marker(self):
        with tempfile.TemporaryDirectory()as tmp:
            c=C.Completion.__new__(C.Completion)
            runtime=type('Runtime',(),{'handles':staticmethod(lambda stage:True),
                'call':staticmethod(lambda *args,**kwargs:{'status':'RECONCILED'})})()
            c.stage_executor=runtime
            C.put(Path(tmp)/'DF_SIGNALS.STARTED.json',{'pid':123})
            self.assertEqual(c.call(['fixed'],Path(tmp),'DF_SIGNALS',science=True),{'status':'RECONCILED'})

    def test_budget_default_is_hold_and_no_result_import(self):
        with tempfile.TemporaryDirectory()as tmp:
            class Executor:
                run_dir=tmp
                def submit_preparation(self,request,**kwargs):self.args=kwargs;return 'task'
                def preparation_result(self,*args,**kwargs):return {'status':'PREPARATION_BUDGET_OR_PERMISSION_HOLD'}
            ex=Executor();bridge=B.DistributedPreparationBridge(ex,tmp,approved=True)
            completion=type('C',(),{'source':'fixed'})()
            with patch.object(bridge,'build_request',return_value={'original_contract':{}}):
                with self.assertRaisesRegex(C.Held,'BUDGET_OR_PERMISSION_HOLD'):
                    bridge.call(completion,[],tmp,'DF_SIGNALS')
            self.assertEqual(ex.args['budget'],0)

    def test_pauli_audit_never_acquires_missing_compile(self):
        path=ROOT/'scripts/run_pr_pauli_candidate_prices.py'
        spec=importlib.util.spec_from_file_location('_prep_pauli_phase',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp)
            for name in module.A_FILES+module.AUDIT_FILES:
                target=root/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b'')
            cfg={'id':'fixture','spec':{'L_D':4}}
            request={'workspace':tmp,'directory':'owned','configuration':cfg,
                     'price_quote':{'configuration_id':'fixture','policy':'MODEL','complete':False},
                     'prepared_item':{'configuration':cfg}}
            with patch.object(module.OPS,'owned_checkout_lock',return_value=nullcontext()),\
                 patch.object(module.subprocess,'check_output',return_value=b''),\
                 patch.object(module,'command',side_effect=AssertionError('audit must not acquire')):
                result=module._run_one(request,execute=True,push=False,phase='audit')
            self.assertEqual(result['status'],'HELD_AUDIT_COMPILE_DEPENDENCY_MISSING')
            self.assertEqual(result['new_compiles_this_invocation'],0)
        self.assertIn('PF_HOLDOUT_COMPILE',B.STAGES)
        self.assertIn('PF_HOLDOUT_AUDIT',B.STAGES)


if __name__=='__main__':unittest.main()
