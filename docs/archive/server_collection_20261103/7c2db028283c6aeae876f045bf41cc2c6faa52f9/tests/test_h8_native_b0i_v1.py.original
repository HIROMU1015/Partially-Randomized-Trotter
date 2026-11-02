"""Bounded artificial H8 scope, scalar, four-worker waves and resource tests."""
from copy import deepcopy
import errno
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
import unittest
import weakref
from unittest.mock import Mock,patch

spec=importlib.util.spec_from_file_location('h8_tests_fixture',Path(__file__).with_name('h8_b0i_fixtures_v1.py'))
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
r,g,b,p=f.r,f.g,f.b,f.parallel
observer=f.load('h8_tests_observer',f.ROOT/'scripts/h8_b0i_family_external_monitor_v1.py')


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=os.environ['TMPDIR'],prefix='h8-b0i-artificial-')
        self.base=Path(self.tmp.name);self.root=self.base/'source';self.out=self.base/'output'
        self.c,self.a,self.mapping=f.setup(self.root,self.out)
        self.backend=type('Backend',(f.old_parallel.FakeBackend,),dict(instances=[],failure=None,cleanup_error=False))
        f.FakeAdapter.instances.clear()
    def tearDown(self):self.tmp.cleanup()
    def refresh(self):self.a=f.authorization(self.c,self.out)
    def batch(self,*args):return p.compile_batch(*args,g=g,metrics_check=r.adapter_module.metrics_checked,backend_factory=self.backend)
    def execute(self,factory=f.FakeAdapter):
        return r.run(self.c,self.a,contract_sha=g.d0.digest(self.c),authorization_sha=g.d0.digest(self.a),
            output=self.out,root=self.root,mapping_loader=lambda:self.mapping,synthetic=True,
            monitor_factory=f.FakeMonitor,factory=factory,compile_batch=self.batch)
    def read(self,name):return json.loads((self.out/name).read_bytes())
    def test_fixed_twelve_cases_twenty_four_new_axes_and_no_reuse(self):
        self.assertEqual(self.execute(),0);result=self.read('full_family_results_v1.json')
        self.assertEqual(result['calls'],r.LIMITS);self.assertEqual(len(result['accounting']),120)
        self.assertEqual(len(self.backend.instances),6)
        self.assertTrue(all(x.poll_widths==[4] for x in self.backend.instances))
        self.assertTrue(all(len({h['task']['cpu'] for h in x.handles})==4 for x in self.backend.instances))
        self.assertTrue(all(c['acquisition']=='NEW_MEASURED' for c in result['cases']))
        self.assertEqual(len(result['comparison']['comparisons']),60)
    def test_all_screen_complete_before_first_native_case(self):
        execute=self.backend.start
        def start(backend,task):
            self.assertEqual(len(list(self.out.glob('case_*_screen.json'))),12)
            self.assertEqual(len(list(self.out.glob('case_*_correctness.json'))),2*len(self.backend.instances))
            return execute(backend,task)
        with patch.object(self.backend,'start',start):self.assertEqual(self.execute(),0)
    def test_all_ineligible_skip_build_and_compile_not_zero_measured_cost(self):
        self.c['fixture_skip']=[list(x) for x in r.family.REGISTRY];self.refresh()
        self.assertEqual(self.execute(),0);v=self.read('full_family_results_v1.json')
        self.assertEqual(v['calls']['compile'],0);self.assertEqual(self.backend.instances,[])
        self.assertTrue(all(x['cost_status']=='UNMEASURED' and x['metrics']=={} for x in v['accounting']))
    def test_selected_only_calls_and_first_correctness_rule(self):
        self.c['fixture_skip']=[list(x) for x in r.family.REGISTRY[:5]];self.refresh()
        self.assertEqual(self.execute(),0);v=self.read('full_family_results_v1.json')
        self.assertEqual(v['calls']['compile'],14);self.assertEqual(v['calls']['control_probe'],28)
        self.assertEqual([x.poll_widths for x in self.backend.instances],[[4],[4],[4],[2]])
        self.assertEqual(self.read('compile_plan_after_screen_v1.json')['first_native_correctness_case'],list(r.family.REGISTRY[5]))
    def test_last_screen_failure_starts_no_native_or_compile(self):
        class Fail(f.FakeAdapter):
            def prepare(obj,prefix,q):
                if (prefix,q)==r.family.REGISTRY[-1]:raise g.Stop('LAST_SCREEN')
                return super().prepare(prefix,q)
        self.assertEqual(self.execute(Fail),2);self.assertEqual(f.FakeAdapter.instances[-1].builds,[])
    def test_first_native_failure_compile_zero(self):
        class Fail(f.FakeAdapter):
            def validate(obj,*args):raise g.Stop('FIRST_NATIVE_FAILURE')
        self.assertEqual(self.execute(Fail),2)
        self.assertEqual(self.read('terminal_v1.json')['calls_reserved_before_operations']['compile'],0)
    def test_second_case_failure_prevents_all_four_wave_compiles(self):
        class Fail(f.FakeAdapter):
            def validate(obj,case,built):
                if (case['prefix'],case['q'])==r.family.REGISTRY[1]:raise g.Stop('SECOND_CASE_FAILURE')
                return super().validate(case,built)
        self.assertEqual(self.execute(Fail),2)
        self.assertEqual(self.backend.instances,[])
        self.assertEqual(self.read('terminal_v1.json')['calls_reserved_before_operations']['compile'],0)
        self.assertTrue(f.FakeAdapter.instances[-1].closed)
    def test_only_one_selected_case_launches_two_workers(self):
        self.c['fixture_skip']=[list(x) for x in r.family.REGISTRY[1:]];self.refresh()
        self.assertEqual(self.execute(),0)
        self.assertEqual([x.poll_widths for x in self.backend.instances],[[2]])
        self.assertEqual(self.read('full_family_results_v1.json')['calls']['compile'],2)
    def test_at_most_two_native_cases_held_and_released_between_waves(self):
        refs=[];held=[]
        class Built(dict):pass
        class Track(f.FakeAdapter):
            def build(obj,case):
                held.append(sum(ref() is not None for ref in refs))
                value=Built(super().build(case));refs.append(weakref.ref(value));return value
        self.assertEqual(self.execute(Track),0)
        self.assertEqual(held,[0,1]*6)
        self.assertTrue(all(ref() is None for ref in refs))
    def test_four_worker_count_and_memory_batch_cannot_be_silently_changed(self):
        for count in (2,6):
            c=deepcopy(self.c);c['workers']=count
            grant=SimpleNamespace(contract=c,synthetic=True,require=Mock())
            with self.assertRaisesRegex(g.Stop,'FIXED_SCIENTIFIC'):r.validate_contract(grant)
        c=deepcopy(self.c);c['parallel_compile']['max_cases_in_memory']=3
        with self.assertRaisesRegex(g.Stop,'FOUR_WORKER_TWO_CASE'):
            r.validate_contract(SimpleNamespace(contract=c,synthetic=True,require=Mock()))
    def test_helper_rejects_three_cases_before_any_worker_starts(self):
        grant=SimpleNamespace(contract=self.c,synthetic=True,require=Mock())
        cases=[dict(prefix=prefix,q=q,validation=dict(status='PASS',fingerprints=dict(cosine='a',sine='b')),
            built=dict(wrappers=dict(cosine='ARTIFICIAL',sine='ARTIFICIAL'))) for prefix,q in r.family.REGISTRY[:3]]
        with self.assertRaisesRegex(g.Stop,'AT_MOST_TWO_CASES'):
            self.batch(grant,cases,Mock(),f.FakeMonitor(grant),Mock(),Mock())
        self.assertEqual(self.backend.instances,[])
    def test_late_native_failure_keeps_completed_costs_but_no_complete_result(self):
        class Fail(f.FakeAdapter):
            def validate(obj,case,built):
                if (case['prefix'],case['q'])==r.family.REGISTRY[2]:raise g.Stop('LATE_NATIVE_FAILURE')
                return super().validate(case,built)
        self.assertEqual(self.execute(Fail),2)
        self.assertEqual(len(list(self.out.glob('case_*_cost.json'))),4)
        self.assertFalse((self.out/'full_family_results_v1.json').exists())
    def test_missing_saved_reference_prevents_first_build(self):
        class Missing(f.FakeAdapter):
            def prepare(obj,*args):
                v=super().prepare(*args);v['independent']=None;return v
        self.assertEqual(self.execute(Missing),2);self.assertEqual(f.FakeAdapter.instances[-1].builds,[])
    def test_previous_H6_grant_is_not_H8_authorization(self):
        self.a['scope']='H6_NATIVE_B0I_FULL_FAMILY12_V1';factory=Mock()
        with self.assertRaises(g.Stop):self.execute(factory)
        factory.assert_not_called();self.assertFalse(self.out.exists())
    def test_draft_rejected_before_input_mapping_or_science(self):
        self.c['approved']=False;self.refresh();factory=Mock()
        with self.assertRaises(g.Stop):self.execute(factory)
        factory.assert_not_called()
    def test_missing_grant_no_source_or_input_read(self):
        self.a={};factory=Mock()
        with self.assertRaises(g.Stop):self.execute(factory)
        factory.assert_not_called()
    def test_contract_content_change_is_bound(self):
        self.c['execution_order']='SILENT_NEW_ORDER'
        self.a['contract_sha256']=g.d0.digest(self.c)
        with self.assertRaisesRegex(g.Stop,'CONTENT_BINDING'):self.execute()
    def test_wrong_prefix_registry_rejected(self):
        self.c['candidates'][0]['prefix']=5;self.refresh()
        with self.assertRaisesRegex(g.Stop,'FIXED_SCIENTIFIC'):self.execute()
    def test_wrong_q_domain_rejected(self):
        self.c['new_candidates'][0]['q']=16;self.refresh()
        with self.assertRaisesRegex(g.Stop,'FIXED_SCIENTIFIC'):self.execute()
    def test_real_provider_change_rejected_before_CPU_and_arrays(self):
        grant=SimpleNamespace(contract=self.c,synthetic=False,require=Mock())
        with self.assertRaisesRegex(g.Stop,'PINNED_SCIENTIFIC_PROVIDER'):r.validate_contract(grant)
    def test_compiler_change_rejected(self):
        self.c['compiler_options']['optimization_level']=2;self.refresh()
        with self.assertRaisesRegex(g.Stop,'FIXED_COMPILER'):self.execute()
    def test_matched_B0_not_implicitly_authorized(self):
        self.c['matched_B0']['approved']=True;self.refresh()
        with self.assertRaises(g.Stop):self.execute()
    def test_input_tamper_no_scientific_factory(self):
        Path(self.mapping['signal_summary.json']).write_text('{}');factory=Mock()
        self.assertEqual(self.execute(factory),2);factory.assert_not_called()
    def test_source_tamper_no_scientific_factory(self):
        (self.root/'synthetic_source.txt').write_text('CHANGED');factory=Mock()
        self.assertEqual(self.execute(factory),2);factory.assert_not_called()
    def test_repeated_output_never_retries(self):
        self.assertEqual(self.execute(),0)
        with self.assertRaisesRegex(g.Stop,'ONE_SHOT'):self.execute()
    def test_twenty_fifth_compile_forbidden_and_no_refund(self):
        journal=[];calls=r.Calls(r.LIMITS,journal.append);calls.take('compile',24)
        with self.assertRaises(g.Stop):calls.take('compile')
        self.assertEqual(calls.used['compile'],24);self.assertEqual(journal[-1]['refund'],0)
    def test_wave_cleanup_files_unique_and_total_CPU_not_reset(self):
        self.assertEqual(self.execute(),0)
        self.assertEqual(len(list(self.out.glob('wave_*_parallel_cleanup_v1.json'))),6)
        self.assertAlmostEqual(self.read('terminal_v1.json')['child_CPU_seconds'],.24)
    def test_terminal_binds_final_result_and_completed_cost_bytes(self):
        self.assertEqual(self.execute(),0);receipts=self.read('terminal_v1.json')['output_file_receipts']
        self.assertIn('full_family_results_v1.json',{r['name'] for r in receipts})
        self.assertEqual(sum(r['name'].endswith('_cost.json') for r in receipts),24)
        for row in receipts:
            raw=(self.out/row['name']).read_bytes()
            self.assertEqual((len(raw),g.sha(raw)),(row['bytes'],row['sha256']))
    def test_worker_failure_and_cleanup_error_preserve_first_STOP(self):
        self.backend.failure='native';self.backend.cleanup_error=True
        self.assertEqual(self.execute(),2);self.assertEqual(len(self.backend.instances),1)
        self.assertEqual(len(self.backend.instances[0].handles),4)
        self.assertTrue(self.backend.instances[0].cleaned)
        self.assertTrue(all(h['reaped'] for h in self.backend.instances[0].handles))
        self.assertIn('ORIGINAL_COMPILE_FAILURE',self.read('terminal_v1.json')['first_stop']['reason'])
    def test_fourth_worker_start_failure_cleans_all_preceding_workers(self):
        start=self.backend.start;cleanup=self.backend.cleanup
        def partial(backend,task):
            if len(backend.handles)==3:raise g.Stop('FOURTH_START_FAILED')
            return start(backend,task)
        def reap(backend):
            for h in backend.handles:h.update(reaped=True,exit_code=2)
            return cleanup(backend)
        with patch.object(self.backend,'start',partial),patch.object(self.backend,'cleanup',reap):
            self.assertEqual(self.execute(),2)
        backend=self.backend.instances[0]
        self.assertTrue(backend.cleaned);self.assertEqual(len(backend.handles),3)
        self.assertTrue(all(h['reaped'] for h in backend.handles))
        terminal=self.read('terminal_v1.json')
        self.assertIn('FOURTH_START_FAILED',terminal['first_stop']['reason'])
        self.assertEqual(terminal['calls_reserved_before_operations']['compile'],4)
        self.assertFalse((self.out/'full_family_results_v1.json').exists())
    def test_observer_records_all_four_worker_identities(self):
        c=self.c;c['external_monitor_memory_baseline']={};c['log_cap_bytes']=16*2**20
        control=Path(c['control_directory']);(control/'launch_stdout.log').write_bytes(b'')
        driver=dict(pid=1,uid=os.getuid(),parent_pid=99,start_ticks=1,boot_id='ARTIFICIAL')
        identities=[]
        for i,(prefix,q,axis) in enumerate((p,q,axis) for p,q in r.family.REGISTRY[:2] for axis in ('cosine','sine')):
            identity=dict(driver,pid=i+2,parent_pid=1,start_ticks=i+2);identities.append(identity)
            record=dict(identity=identity,cpu=c['parallel_compile']['worker_cpus'][i],name=f'{prefix}_{q}_{axis}')
            (control/f'observed_compile_started_{prefix}_{q}_{axis}.json').write_text(json.dumps(record))
        memory=dict(effective_available_bytes=10**12)
        fakeg=SimpleNamespace(Stop=g.Stop,d0=SimpleNamespace(memory_snapshot=lambda:memory,check_memory_snapshot=Mock()),
            validate_memory_observation=Mock(),filesystem_snapshot=lambda _:dict(available_bytes=10**12),check_filesystem=Mock())
        grant=SimpleNamespace(contract=c,contract_sha='CONTRACT',authorization_sha='AUTH',output=self.out)
        obs=observer.Observer(grant,driver,dict(pid=99),fakeg,clock=lambda:1.)
        def process(identity,allowed,**kwargs):return dict(identity=identity,CPU_seconds=.01,peak_RSS_bytes=1024,AS_bytes=2048,affinity=allowed)
        with patch.object(obs,'process',side_effect=process):self.assertTrue(obs.step())
        record=json.loads((control/observer.HEARTBEAT).read_bytes())
        self.assertEqual([row['identity'] for row in record['workers']],identities)
        ticket=dict(parent_identity=dict(pid=99),identity=driver);c['SOURCE']=record['SOURCE']
        for identity in identities:self.assertEqual(observer.verify_record(record,grant,ticket,identity,1.),record)
    def test_proof_tamper_blocks_final_comparison(self):
        poll=self.backend.poll
        def bad(backend,active):
            handles=poll(backend,active)
            for h in handles:
                row=json.loads(h['payload']);row['row']['input_fingerprint_check']['after']='0'*64
                h['payload']=bytearray(json.dumps(row).encode())
            return handles
        with patch.object(self.backend,'poll',bad):self.assertEqual(self.execute(),2)
        self.assertIn('COMPILED_IDENTITY',self.read('terminal_v1.json')['first_stop']['reason'])
    def test_exploration_n8_cannot_enter_PR_confirmation(self):
        path=Path(self.mapping['d0_candidate_accounting.json']);v=json.loads(path.read_bytes())
        for row in v['H8_CONFIRMATION_UNION6']:
            if row['method']=='B2':row['sample_phase']='exploration';row['sample_count']=8;break
        raw=json.dumps(v).encode();path.write_bytes(raw)
        receipt=next(x for x in self.c['input_receipts'] if x['logical_name']=='d0_candidate_accounting.json')
        receipt.update(bytes=len(raw),sha256=g.sha(raw));self.refresh()
        self.assertEqual(self.execute(),2)
        self.assertIn('CONFIRMATION_NOT_N32',self.read('terminal_v1.json')['first_stop']['reason'])
    def test_recipe_mutation_rejected_before_science(self):
        self.c['binding']['b0_fingerprints']['7/1']='0'*64;self.refresh()
        with self.assertRaisesRegex(g.Stop,'RECIPE'):self.execute()
    def test_wrong_retained_block_order_fails(self):
        bad=f.prepared(7);bad.deterministic_block_order=tuple(reversed(bad.deterministic_block_order))
        grant=g.authorization_gate(self.c,self.a,contract_sha=g.d0.digest(self.c),authorization_sha=g.d0.digest(self.a),output=self.out,scope=r.SCOPE,synthetic=True)
        with self.assertRaisesRegex(g.Stop,'BLOCK_ORDER'):b.validate_recipe(grant,7,1,bad,b.binding_for(grant,7,1))


class PureBridgeAndCPU(unittest.TestCase):
    def test_signed_full_Fock_identity_not_fixed_sector_trace(self):
        self.assertEqual(b.exact_identity(((1,1j),(-1j,2)),-2.),-8)
    def test_nonHermitian_parameter_STOP(self):
        with self.assertRaises(ValueError):b.exact_identity(((1,1),(0,2)),1.)
    def test_nonfinite_parameter_STOP(self):
        with self.assertRaises(ValueError):b.exact_identity(((math.inf,0),(0,2)),1.)
    def test_rotation_sign_and_unrotated_full_H_reference(self):
        record=dict(signals=dict(corrected=dict(real=1.,imag=0.),exact_truncated=dict(real=.5,imag=0.)),reference=dict(real=.2,imag=.7),
            allowance=dict(log_B_margin=0.),error_components=dict(kind='EMPIRICAL',certified=False,
            B0_signal=dict(real=0.,imag=0.),full_H_reference=dict(real=0.,imag=0.),signal_abs_upper=1.,evidence={}))
        out=b.saved_signal_overlay(record,C=math.pi/(2*.8),C_error=0.,rotation_roundoff=0.,phase_roundoff=0.)
        self.assertAlmostEqual(out['signals']['corrected']['imag'],-1.)
        self.assertEqual(out['reference'],record['reference']);self.assertFalse(out['allowance']['certified'])
    def test_negative_error_bound_STOP(self):
        with self.assertRaises(ValueError):b.upper_float(-1)
    def test_H6_dimensions_cannot_pass_H8_gate(self):
        for n,rank,dim in [(12,28,4900),(16,19,4900),(16,28,400)]:
            with self.assertRaisesRegex(g.Stop,'H8_DF28'):r.adapter_module.validate_h8_dimensions(SimpleNamespace(n_qubits=n,lambdas=[0]*rank),SimpleNamespace(dimension=dim))
        r.adapter_module.validate_h8_dimensions(SimpleNamespace(n_qubits=16,lambdas=[0]*28),SimpleNamespace(dimension=4900))
    def test_completed_CPU_plus_live_next_wave_no_double_count(self):
        carry=dict(contract_sha256='BOUND',driver_identity={'pid':1},all_reaped=True,
            completed_names=['7_1_cosine','7_1_sine'],child_cpu_seconds=7.)
        seen={'7_1_cosine':2.,'7_1_sine':3.,'7_2_cosine':5.,'7_2_sine':6.}
        self.assertEqual(observer.child_cpu_total(seen,carry,'BOUND',{'pid':1},set(seen)),18.)
    def test_missing_or_changed_CPU_binding_is_STOP(self):
        carry=dict(contract_sha256='BOUND',driver_identity={'pid':1},all_reaped=True,completed_names=['7_1_cosine'],child_cpu_seconds=2.)
        for field,value in [('contract_sha256','OTHER'),('driver_identity',{'pid':2}),('all_reaped',False),('completed_names',['ILLEGAL']),('child_cpu_seconds',-2.),('child_cpu_seconds',math.nan)]:
            changed=dict(carry);changed[field]=value
            with self.assertRaises(ValueError):observer.child_cpu_total({'7_1_cosine':20},changed,'BOUND',{'pid':1},{'7_1_cosine'})
    def test_pidfd_only_ESRCH_is_already_exited_and_FD_closed(self):
        identity=dict(pid=123456,uid=os.getuid(),parent_pid=123,start_ticks=4,boot_id='ARTIFICIAL')
        for number in (errno.ESRCH,errno.EPERM):
            with patch.object(g,'identity',return_value=identity),patch.object(p.os,'pidfd_open',return_value=876),\
                    patch.object(p.signal,'pidfd_send_signal',side_effect=OSError(number,'FIXTURE')),patch.object(p.os,'close') as close:
                if number==errno.ESRCH:p.terminate_owned(identity['pid'],identity,signal.SIGTERM,g)
                else:
                    with self.assertRaises(OSError):p.terminate_owned(identity['pid'],identity,signal.SIGTERM,g)
                close.assert_called_once_with(876)
    def test_cleanup_failure_keeps_STOP_signals_deferred(self):
        with patch.object(p.signal,'pthread_sigmask',return_value=set()) as mask,patch.object(p.signal,'setitimer',return_value=(.25,0.)) as timer:
            state=p.CleanupSignals(True);state.defer();state.finish(resume=False)
        mask.assert_called_once_with(signal.SIG_BLOCK,p.CLEANUP_SIGNALS)
        timer.assert_called_once_with(signal.ITIMER_REAL,0)
