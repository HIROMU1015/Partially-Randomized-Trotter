"""Artificial metadata and fake compilation only; never load real H2 arrays/results."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h2_p_contract as c
import h2_p_science as science
import h2_panels_science as inherited
import prepare_h2_p_seal as seal
import run_h2_p as runner
from prepare_h2_panels import row
import test_h2_panels as regression


def summaries():
    a=dict(binding=deepcopy(c.base.BINDING),accounting=dict(winners={}),signals=[],preparation_identities={})
    d=dict(binding=deepcopy(c.base.BINDING),accounting=dict(winners={}))
    s=dict(plan=dict(binding=deepcopy(c.base.BINDING)),recommended=dict(n_fixed=dict(c.N_FIXED),
        proposed_P_calls=dict(c.CALLS,seed_slots=140,control_probe_maximum=8,independent_event_oracle_maximum=2)),
        reference_directions=[],P_executed=False)
    for (epsilon,cid,pf,n),cell in zip(c.TARGETS,c.cells()):
        shots=dict(cosine=3,sine=5)
        pr=dict(cell_id=cid,epsilon=epsilon,eligible=True,cohort='EXPLORATION_N8',n=8,
            G_RZ_exact='20',G_RZ_SE=2.,shots=shots)
        control=dict(cell_id=pf,epsilon=epsilon,eligible=True,cohort='DETERMINISTIC',n=1,G_RZ_exact='100')
        a['accounting']['winners'][str(epsilon)]=pr;d['accounting']['winners'][str(epsilon)]=control
        a['signals'].append(dict(cell=cell,precision_rows={str(epsilon):{'1.0':dict(cost_acquisition_eligible=True,
            axes=dict(real=dict(shots=3),imag=dict(shots=5)))}},log_B=.5,allowance=dict(log_B_margin=1e-10)))
        a['preparation_identities'][cid]='toy-preparation'
        s['reference_directions'].append(dict(epsilon=epsilon,cell_id=cid,PR_mean_exact='20',PF_cell_id=pf,
            PF_G_exact='100',expected_direction=1))
    return a,d,s


def pairs(registry):
    out=[]
    for t in registry['tasks']:
        axes={axis:dict(RZ=2+t['replica']%2,compiler_hash=c.base.BINDING['compiler'],
            fingerprint='a'*64,seed=t['seed'],event_digest='toy-event') for axis in ('cosine','sine')}
        out.append(dict(cell_id=t['cell']['id'],replica=t['replica'],seed=t['seed'],axes=axes,
            event_digest='toy-event',cohort=c.COHORT,preparation_hash='toy-preparation'))
    return out


class SeedAndAccounting(unittest.TestCase):
    def setUp(self):
        self.tasks=c.tasks('a'*40,'b'*64)
        self.registry=dict(SOURCE='a'*40,seed_plan_digest='b'*64,tasks=self.tasks)
        self.endpoints=c.selected_endpoints(*summaries())

    def test_h2_native_fixed_budget(self):
        self.assertEqual((len(self.tasks),len(c.base.seed_values(self.tasks))),(92,140))
        self.assertEqual(sum(t['cell']['R'] for t in self.tasks),256)
        self.assertEqual(sum(t['cell']['q'] for t in self.tasks),128)
        self.assertEqual(c.CALLS['wrapper_compile'],184)

    def test_namespace_source_plan_and_replica_change_seed(self):
        self.assertNotEqual(self.tasks,c.tasks('c'*40,'b'*64))
        self.assertNotEqual(self.tasks,c.tasks('a'*40,'d'*64))
        self.assertNotEqual(self.tasks[0]['seed'],self.tasks[1]['seed'])

    def test_step_hierarchy_q1_and_q4(self):
        self.assertEqual(self.tasks[0]['step_seeds'],[self.tasks[0]['seed']])
        self.assertEqual(len(self.tasks[-1]['step_seeds']),4)
        c.check_tasks(self.registry,'a'*40,'b'*64,[])

    def test_old_collision_refused_no_redraw(self):
        before=deepcopy(self.tasks)
        with self.assertRaisesRegex(ValueError,'P_SEED_COLLISION'):
            c.check_tasks(self.registry,'a'*40,'b'*64,[self.tasks[0]['seed']])
        self.assertEqual(before,self.tasks)

    def test_changed_step_seed_refused(self):
        self.tasks[-1]['step_seeds'][0]+=1
        with self.assertRaisesRegex(ValueError,'SEALED_P_TASKS'):c.check_tasks(self.registry,'a'*40,'b'*64,[])

    def test_missing_or_extra_seed_task_refused(self):
        for tasks in (self.tasks[:-1],self.tasks+[self.tasks[0]]):
            with self.assertRaises(ValueError):c.check_tasks(dict(self.registry,tasks=tasks),'a'*40,'b'*64,[])

    def test_saved_rows_copy_and_unknown_tail(self):
        plan=c.plan();self.assertIsNone(plan['unseen_event_cost']);self.assertIsNone(plan['unseen_event_variance'])
        self.assertEqual(self.endpoints[0]['saved_shots'],dict(cosine=3,sine=5))
        self.assertFalse(plan['population_mean_certified'])

    def test_wrong_sector_refused(self):
        a,d,s=summaries();d['binding']['sector_dimension']=6
        with self.assertRaisesRegex(ValueError,'SECTOR4'):c.selected_endpoints(a,d,s)

    def test_n_or_PF_or_shots_substitution_refused(self):
        for kind in ('n','PF','shots'):
            a,d,s=summaries()
            if kind=='n':s['recommended']['n_fixed'][c.TARGETS[0][1]]=81
            if kind=='PF':d['accounting']['winners']['0.05']['G_RZ_exact']='101'
            if kind=='shots':a['accounting']['winners']['0.05']['shots']['cosine']=4
            with self.assertRaises(ValueError):c.selected_endpoints(a,d,s)

    def test_artificial_complete_fixed_n_accounting(self):
        result=c.account(pairs(self.registry),self.registry,self.endpoints)
        self.assertEqual(len(result['comparisons']),2)
        self.assertEqual([r['n'] for r in result['comparisons']],[80,12])
        self.assertTrue(all(r['status']=='REPRODUCED' for r in result['comparisons']))
        self.assertIsNone(result['unseen_event_cost']);self.assertFalse(result['population_mean_certified'])

    def test_incomplete_duplicate_or_A_measurements_refused(self):
        full=pairs(self.registry)
        for kind in ('missing','duplicate','pooled','seed','unpaired','preparation'):
            values=deepcopy(full)
            if kind=='missing':values.pop()
            if kind=='duplicate':values[-1]=values[0]
            if kind=='pooled':values[0]['cohort']='EXPLORATION_N8'
            if kind=='seed':values[0]['seed']+=1
            if kind=='unpaired':values[0]['axes']['sine']['event_digest']='different'
            if kind=='preparation':values[0]['preparation_hash']='wrong'
            with self.assertRaises(ValueError):c.account(values,self.registry,self.endpoints)

    def test_conservative_integer_inventory_includes_derived(self):
        value=dict(tasks=[dict(seed=123,cell=dict(q=4))],values=[1,2],extra=dict(marker=99))
        found=seal.seed_metadata_values(value)
        self.assertTrue(set(c.base.stats.seed_hierarchy(123,4))<=found)
        self.assertTrue({1,2,99}<=found)

    def test_F_FP_and_current_public_manifest_formats(self):
        expected=[dict(path='toy.json',sha256='a'*64)]
        for files in ({'toy.json':'a'*64},[dict(file='toy.json',sha256='a'*64)],expected):
            self.assertEqual(c.public_manifest_rows(dict(files=files)),expected)

    def test_duplicate_or_escaping_public_manifest_rejected(self):
        for files in ([dict(path='../toy',sha256='a'*64)], [dict(path='toy',sha256='a'*64)]*2):
            with self.assertRaises(ValueError):c.public_manifest_rows(dict(files=files))


class FullMetadataGate(unittest.TestCase):
    def setUp(self):
        parent=c.PRIVATE/'artificial_p_seal';parent.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=parent);self.root=Path(self.temp.name);self.patchers=[]
        store=self.root/'base';kernel=store/'A_kernel/scripts';kernel.mkdir(parents=True)
        for obj,name,value in [(c,'ROOT',self.root),(c,'PRIVATE',self.root),(c.base,'STORE',store)]:
            p=patch.object(obj,name,value);p.start();self.patchers.append(p)
        source=self.root/'scripts/toy.py';source.parent.mkdir();source.write_text('# ARTIFICIAL SOURCE\n')
        p=patch.object(c,'source_closure',return_value={'scripts/toy.py':c.sha(source)});p.start();self.patchers.append(p)
        p=patch.object(c.subprocess,'check_output',return_value=source.read_bytes());p.start();self.patchers.append(p)
        provider=self.root/'provider';provider.mkdir();(provider/'toy.py').write_text('# ARTIFICIAL PROVIDER\n')
        (kernel/'toy.py').write_text('# ARTIFICIAL KERNEL\n')
        self.snapshot=self.root/'toy.npz';self.snapshot.write_bytes(b'NOT A NUMERICAL ARRAY')
        p=patch.dict(c.base.BINDING,dict(snapshot=c.sha(self.snapshot)));p.start();self.patchers.append(p)
        self.inputs=[]
        def fixed(name,value):
            path=self.root/(name+'.json');c.save(path,value);self.inputs.append(row(path,name));return str(path)
        a,d,s=summaries();deps={}
        for stage,(commit,folder,scientific_source) in c.DEPENDENCIES.items():
            result={'A':a,'D':d,'S_STATS':s}.get(stage,dict(binding=deepcopy(c.base.BINDING)))
            result.update(SOURCE=scientific_source,status='COMPLETE_MANDATORY_STOP')
            if stage=='F':result.update(rank=3,sector=dict(dimension=4),hamiltonian_hash=c.base.BINDING['hamiltonian'],state_hash=c.base.BINDING['state'])
            result_path=fixed(stage+'_result',result)
            audit=dict(SOURCE=scientific_source,status='COMPLETE_MANDATORY_STOP',result_sha256=c.sha(result_path),
                native_exit_code=0,same_parent_reaped=True,wait4_complete=True)
            deps[stage]=dict(commit=commit,folder=folder,SOURCE=scientific_source,result=result_path,
                audit=fixed(stage+'_audit',audit),native=fixed(stage+'_native',dict(exit_code=0,same_parent_reaped=True,wait4_complete=True)),
                binding=fixed(stage+'_binding',dict(SOURCE=scientific_source,source_hashes={})))
        endpoints=c.selected_endpoints(a,d,s)
        exclusions=fixed('exclusions',dict(values=[1,2]))
        sp=c.seed_plan('a'*40,endpoints,deps,c.sha(exclusions));pd=c.digest(sp)
        registry=dict(SOURCE='a'*40,seed_plan_digest=pd,tasks=c.tasks('a'*40,pd))
        registry_path=fixed('registry',registry)
        self.receipt=dict(status='COLLISION_ZERO',new_values_digest=c.digest(sorted(c.base.seed_values(registry['tasks']))),
            seed_registry_sha256=c.sha(registry_path),seed_redrawn=False,new_RTE_sampling=0,
            geometry_checks=[dict(old_seed_slots_checked=7737216,collisions=[])])
        check=fixed('seed_check',self.receipt)
        origins=fixed('origins',[]);prior=fixed('prior',[])
        adoption=fixed('adoption',dict(role='user',accepted_S_STATS_commit=c.STATS_COMMIT,adopted_n=c.N_FIXED,P_scientific_execution_authorized=False))
        tol_receipt=fixed('tol_receipt',dict(artificial=True));self.inputs.append(row(self.snapshot,'snapshot'))
        self.runtime_file=self.root/'toy-runtime';self.runtime_file.write_text('ARTIFICIAL BYTES')
        runtime=self.root/'runtime.json';c.save(runtime,{str(self.runtime_file):c.sha(self.runtime_file)})
        self.spec=dict(scope=c.SCOPE,stage='P',status='SEALED_TECHNICAL_PENDING_USER_APPROVAL',pending=list(c.PENDING),
            SOURCE='a'*40,plan=c.plan(),calls=dict(c.CALLS),source_hashes=c.source_closure(),inputs=self.inputs,
            dependencies=deps,endpoints=endpoints,seed_plan=sp,seed_plan_digest=pd,seed_registry=registry_path,
            seed_check=check,seed_exclusions=exclusions,seed_origins=origins,prior_manifest=prior,adoption_receipt=adoption,
            native_config=dict(binding=deepcopy(c.base.BINDING),compiler_options=c.base.COMPILER,provider_root=str(provider),
                mapping=dict(tol_snapshot=str(self.snapshot),tol_receipt=tol_receipt)),
            provider_roots=[str(provider)],provider_files=[row(provider/'toy.py','provider')],
            kernel_files=[row(kernel/'toy.py','kernel')],A_kernel_SOURCE=c.base.A_SOURCE,A_kernel_scripts=str(kernel),
            runtime_manifest=str(runtime),runtime_manifest_sha256=c.sha(runtime),python=sys.executable,
            python_sha256=c.sha(sys.executable),attempt=str(self.root/'attempts/P_run01'),worker_cpus=[],resource_receipt=None)
        self.pending=self.root/'pending.json';c.save(self.pending,self.spec)
        self.spec.update(status='SEALED_FOR_EXECUTION',pending=[],pending_spec=str(self.pending),pending_spec_sha256=c.sha(self.pending))
        resource=self.root/'resource.json';c.save(resource,dict(admitted=True))
        refresh=self.root/'refresh.json';c.save(refresh,dict(self.receipt,checked_at_UTC='ARTIFICIAL',known_campaigns_checked=['toy'],exclusion_values_checked=2,
            exclusions_path=exclusions,exclusions_sha256=c.sha(exclusions),origins_path=origins,origins_sha256=c.sha(origins)))
        self.spec.update(worker_cpus=[min(os.sched_getaffinity(0))],resource_receipt=str(resource),
            resource_receipt_sha256=c.sha(resource),seed_refresh=str(refresh),seed_refresh_sha256=c.sha(refresh))
        approval=self.root/'approval.json';c.save(approval,dict(role='user',stage='P',SOURCE='a'*40,
            plan_digest=c.digest(c.plan()),P_scientific_execution_authorized=True,verbatim_user_message='ARTIFICIAL FIXTURE ONLY'))
        self.grant=dict(scope=c.SCOPE,stage='P',approved_by_user=True,runtime_authorization=True,one_shot=True,
            automatic_retry=False,next_stage_authorized=False,SOURCE='a'*40,spec_digest=c.digest(self.spec),
            approval_receipt=str(approval),approval_receipt_sha256=c.sha(approval))

    def tearDown(self):
        for p in reversed(self.patchers):p.stop()
        self.temp.cleanup()

    def test_artificial_valid_gate(self):c.authorize(self.spec,self.grant)

    def test_no_grant_before_all_reads(self):
        with patch.object(c,'verify',side_effect=AssertionError('TOO_EARLY')):
            with self.assertRaisesRegex(ValueError,'SEPARATE_EXPLICIT_P'):c.authorize({},c.invalid_grant())

    def test_foreign_retry_and_next_grants_rejected(self):
        for change in (dict(scope='H4'),dict(stage='A'),dict(automatic_retry=True),dict(next_stage_authorized=True),dict(one_shot=False)):
            with self.assertRaises(ValueError):c.authorize(self.spec,dict(self.grant,**change))

    def test_pending_seal_cannot_execute(self):
        with self.assertRaisesRegex(ValueError,'P_STILL_PENDING'):c.authorize(c.read(self.pending),self.grant)

    def test_original_seal_scope_budget_or_n_cannot_change(self):
        for change in ('budget','n','endpoint','seed_plan'):
            spec=deepcopy(self.spec)
            if change=='budget':spec['calls']['trajectory']=93
            if change=='n':spec['plan']['n_fixed'][c.TARGETS[0][1]]=81
            if change=='endpoint':spec['endpoints'][0]['saved_shots']['cosine']=4
            if change=='seed_plan':spec['seed_plan_digest']='c'*64
            with self.assertRaisesRegex(ValueError,'ACTIVATION_MAY_NOT_CHANGE'):c.authorize(spec,dict(self.grant,spec_digest=c.digest(spec)))

    def test_input_bytes_change_rejected(self):
        self.snapshot.write_bytes(b'CHANGED')
        with self.assertRaisesRegex(ValueError,'P_BYTES_CHANGED'):c.verify(self.spec)

    def test_runtime_change_rejected(self):
        self.runtime_file.write_text('CHANGED')
        with self.assertRaisesRegex(ValueError,'P_RUNTIME_CHANGED'):c.verify(self.spec)

    def test_extra_provider_file_rejected(self):
        (Path(self.spec['provider_roots'][0])/'extra.py').write_text('# extra')
        with self.assertRaisesRegex(ValueError,'PROVIDER_FILE_SET'):c.verify(self.spec)

    def test_source_bytes_change_rejected(self):
        (self.root/'scripts/toy.py').write_text('# changed')
        with self.assertRaisesRegex(ValueError,'SOURCE_BYTES'):c.verify(self.spec)

    def test_missing_native_reap_rejected(self):
        path=Path(self.spec['dependencies']['A']['native']);v=c.read(path);v['same_parent_reaped']=False;path.write_text(json.dumps(v))
        next(r for r in self.inputs if r['path']==str(path))['sha256']=c.sha(path)
        with self.assertRaisesRegex(ValueError,'NATIVE_DEPENDENCY'):c.verify(self.spec)

    def test_unrelated_HEAD_and_status_are_not_gates(self):
        (self.root/'unrelated_HEAD').write_text('new unrelated revision')
        c.verify(self.spec)

    def test_approval_seal_only_receipt_not_science(self):
        path=Path(self.grant['approval_receipt']);v=c.read(path);v['P_scientific_execution_authorized']=False;path.write_text(json.dumps(v))
        self.grant['approval_receipt_sha256']=c.sha(path)
        with self.assertRaisesRegex(ValueError,'SCIENCE_RECEIPT'):c.authorize(self.spec,self.grant)

    def test_fresh_seed_or_resource_receipt_required(self):
        for key in ('seed_refresh_sha256','resource_receipt_sha256'):
            spec=dict(self.spec);spec[key]='0'*64
            with self.assertRaises(ValueError):c.authorize(spec,dict(self.grant,spec_digest=c.digest(spec)))

    def test_fresh_seed_actual_collision_not_just_status_is_checked(self):
        refresh=Path(self.spec['seed_refresh']);r=c.read(refresh)
        extra=self.root/'fresh_exclusions.json';values=[1,2,c.read(self.spec['seed_registry'])['tasks'][0]['seed']]
        c.save(extra,dict(values=values));r.update(exclusions_path=str(extra),exclusions_sha256=c.sha(extra),exclusion_values_checked=3)
        refresh.write_text(json.dumps(r));self.spec['seed_refresh_sha256']=c.sha(refresh);self.grant['spec_digest']=c.digest(self.spec)
        with self.assertRaisesRegex(ValueError,'COLLISION_OR_LOST'):c.authorize(self.spec,self.grant)

    def test_fresh_seed_exclusion_cannot_be_dropped(self):
        refresh=Path(self.spec['seed_refresh']);r=c.read(refresh);extra=self.root/'fresh_exclusions.json'
        c.save(extra,dict(values=[1,3]));r.update(exclusions_path=str(extra),exclusions_sha256=c.sha(extra),exclusion_values_checked=2)
        refresh.write_text(json.dumps(r));self.spec['seed_refresh_sha256']=c.sha(refresh);self.grant['spec_digest']=c.digest(self.spec)
        with self.assertRaisesRegex(ValueError,'COLLISION_OR_LOST'):c.authorize(self.spec,self.grant)

    def test_one_shot_claim_and_stop_preserved(self):
        attempt=runner.claim_attempt(self.spec,self.grant);marker=c.sha(attempt/'ONE_SHOT.json')
        c.save(attempt/'driver_STOP.json',dict(reason='ARTIFICIAL FAILURE',automatic_retry=False))
        with self.assertRaisesRegex(ValueError,'CONSUMED_NO_RETRY'):runner.claim_attempt(self.spec,self.grant)
        self.assertEqual(marker,c.sha(attempt/'ONE_SHOT.json'));self.assertTrue((attempt/'driver_STOP.json').exists())


class FakeKernelAndEntry(unittest.TestCase):
    def test_kernel_clone_preserves_original_globals_and_P_cohort(self):
        before=dict(inherited.__dict__)
        def fake(port,m,settings,task,directory,stage):
            return dict(seed=task['seed'],preparation_hash=task['preparation_hash'],cohort=c.plan(stage)['cohort'])
        with tempfile.TemporaryDirectory() as td,patch.object(inherited,'compile_one',fake):
            t=dict(seed=123,preparation_hash='toy',step_seeds=[123],cell=c.cells()[0],replica=0,seed_plan_digest='toy')
            result=science.compile_one(None,None,None,t,Path(td))
            self.assertEqual(result['cohort'],c.COHORT)
            self.assertEqual(c.read(Path(td)/'confirmation_seed_identity.json')['master_seed'],123)
        self.assertEqual(inherited.__dict__,before)

    def test_execute_requires_grant_before_context(self):
        with patch.object(science,'context',side_effect=AssertionError('SCIENCE')):
            with self.assertRaisesRegex(ValueError,'SEPARATE_EXPLICIT_P'):science.execute({},c.invalid_grant(),None)

    def test_fake_P_pipeline_controls_first_and_exact_budget(self):
        registry=dict(tasks=c.tasks('a'*40,'b'*64));endpoints=c.selected_endpoints(*summaries());waves=[]
        fakepairs={(r['cell_id'],r['replica']):r for r in pairs(registry)}
        def context(spec,output,calls):
            calls.take('saved_input_load');calls.take('preparation');return None,None,None
        def batch(spec,tasks,callback,directory):
            waves.append([t['replica'] for t in tasks])
            return [fakepairs[t['cell']['id'],t['replica']] for t in tasks]
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'registry.json';c.save(path,registry);output=Path(td)/'output';output.mkdir()
            spec=dict(SOURCE='a'*40,calls=c.CALLS,seed_registry=str(path),endpoints=endpoints,seed_plan_digest='toy')
            with patch.object(c,'authorize'),patch.object(science,'context',side_effect=context),patch.object(inherited,'batch',side_effect=batch):
                result=science.execute(spec,{},output)
            self.assertEqual(waves[0],[0,0]);self.assertEqual(len(waves[1]),90)
            self.assertTrue(all(rep>0 for rep in waves[1]));self.assertEqual(result['calls_reserved'],c.CALLS)
            self.assertEqual(len(result['measurements']),92)

    def test_first_control_failure_prevents_remaining_wave(self):
        registry=dict(tasks=c.tasks('a'*40,'b'*64));endpoints=c.selected_endpoints(*summaries())
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'registry.json';c.save(path,registry)
            spec=dict(calls=c.CALLS,seed_registry=str(path),endpoints=endpoints,seed_plan_digest='toy')
            with patch.object(c,'authorize'),patch.object(science,'context',return_value=(None,None,None)),patch.object(inherited,'batch',side_effect=ValueError('ARTIFICIAL CONTROL FAILURE')) as batch:
                with self.assertRaisesRegex(ValueError,'CONTROL FAILURE'):science.execute(spec,{},Path(td))
            self.assertEqual(batch.call_count,1)

    def test_invalid_driver_worker_cli_stdlib_only(self):
        entry=Path(__file__).resolve().parents[1]/'scripts/run_h2_p.py'
        with tempfile.TemporaryDirectory() as td:
            spec=Path(td)/'spec.json';grant=Path(td)/'grant.json'
            spec.write_text('{}');grant.write_text(json.dumps(c.invalid_grant()))
            for flags in ([],['--worker']):
                p=subprocess.run([sys.executable,'-B','-S',str(entry),'--spec',str(spec),'--grant',str(grant),*flags],capture_output=True,text=True)
                self.assertEqual(p.returncode,1);self.assertIn('SEPARATE_EXPLICIT_P_SCIENCE_APPROVAL_REQUIRED',p.stderr)
                self.assertNotIn('ModuleNotFoundError',p.stderr)

    def test_seal_module_does_not_import_scientific_adapter(self):
        import ast
        tree=ast.parse(Path(seal.__file__).read_text())
        imported=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        imported += [a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
        self.assertNotIn('h2_p_science',imported);self.assertNotIn('numpy',imported)


class NativeArtificialPAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):regression.NativeArtificialTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):regression.NativeArtificialTests.tearDownClass.__func__(cls)

    def test_H2_P_context_loads_once_and_reuses_one_prefix(self):
        output=self.root/'P_context';output.mkdir()
        prep=self.m._prepare(self.ham,1)
        with patch.dict(c.base.BINDING,dict(hamiltonian=prep.hamiltonian_hash)):
            config=dict(provider_root=str(self.provider),binding=deepcopy(c.base.BINDING),compiler_options=c.base.COMPILER)
            spec=dict(native_config=config,A_kernel_scripts=str(self.kernel/'scripts'),
                endpoints=[dict(preparation_hash=prep.preparation_hash)]*2)
            calls=c.base.Calls(output,c.CALLS)
            loader=lambda config:(self.ham,self.sector,self.full,self.state,
                dict(hamiltonian_hash=prep.hamiltonian_hash,state_hash=c.base.BINDING['state']))
            port,m,compiler=science.context(spec,output,calls,loader=loader)
            self.assertEqual(calls.used['saved_input_load'],1);self.assertEqual(calls.used['preparation'],1)
            self.assertIs(port.preps[c.cells()[0]['id']],port.preps[c.cells()[1]['id']])
            self.assertEqual(sum(calls.used.values()),2)

    def test_H2_P_q1_q4_full_wrappers_paired_with_original_kernel(self):
        import math
        import pr_cost_native as kernel
        from h4_dp_science import compiler
        prep=self.m._prepare(self.ham,1);port=SimpleNamespace(ham=self.ham,
            preps={x['id']:prep for x in c.cells()},qstate=self.port.qstate)
        settings=compiler(self.m,self.spec['native_config'])
        for index,cell in enumerate(c.cells()):
            master=12345+index;requests,distribution=kernel.steps(prep,cell,master)
            task=dict(cell=cell,seed=master,replica=0,preparation_hash=prep.preparation_hash,
                step_seeds=c.base.stats.seed_hierarchy(master,cell['q'])[(0 if cell['q']==1 else 1):],
                seed_plan_digest='ARTIFICIAL',expected=dict(log_B=cell['R']*math.log(distribution.exact_finite_distribution),
                    allowance=dict(log_B_margin=1e-10)))
            directory=self.root/('P_compile_'+str(index));directory.mkdir()
            result=science.compile_one(port,self.m,settings,task,directory)
            self.assertEqual(result['cohort'],c.COHORT);c.base.check_pair(result)
            self.assertTrue(c.read(directory/'control.json')['max_error']<1e-9)
            self.assertEqual(c.read(directory/'trajectory.json')['cohort'],c.COHORT)


if __name__=='__main__':unittest.main()
