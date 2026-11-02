"""Limited fabricated full-family integration and independent small arithmetic."""
from copy import deepcopy
import errno
import importlib.util
import json
import os
from pathlib import Path
import signal
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch

spec=importlib.util.spec_from_file_location('family_fixture',Path(__file__).with_name('h6_b0i_family_fixtures_v1.py'))
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
r,g,p=f.r,f.g,f.parallel


class FamilyTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=os.environ['TMPDIR'],prefix='full-family-mock-')
        self.base=Path(self.temp.name);self.root=self.base/'source';self.output=self.base/'out'
        self.c,self.a,self.mapping,self.payloads=f.setup(self.root,self.output)
        self.backend=type('Backend',(f.old_parallel.FakeBackend,),dict(instances=[],failure=None,cleanup_error=False))
        f.FakeAdapter.instances.clear()
    def tearDown(self):self.temp.cleanup()
    def refresh(self):self.a=f.authorization(self.c,self.output)
    def batch(self,*args):
        return p.compile_batch(*args,g=g,metrics_check=r.adapter_module.metrics_checked,backend_factory=self.backend)
    def execute(self,factory=f.FakeAdapter):
        return r.run(self.c,self.a,contract_sha=g.d0.digest(self.c),authorization_sha=g.d0.digest(self.a),
            output=self.output,root=self.root,mapping_loader=lambda:self.mapping,synthetic=True,
            monitor_factory=f.old.FakeMonitor,factory=factory,compile_batch=self.batch)
    def read(self,name):return json.loads((self.output/name).read_text())
    def change_input(self,name,change):
        obj=json.loads(Path(self.mapping[name]).read_text());change(obj)
        raw=json.dumps(obj).encode();Path(self.mapping[name]).write_bytes(raw)
        row=next(x for x in self.c['input_receipts'] if x['logical_name']==name)
        row.update(bytes=len(raw),sha256=g.sha(raw));self.refresh()
    def test_twelve_registry_reuse_six_and_max_eighteen_new_axes(self):
        self.assertEqual(self.execute(),0);res=self.read('full_family_results_v1.json')
        self.assertEqual(len(res['cases']),12);self.assertEqual(res['calls'],r.LIMITS)
        self.assertEqual(self.backend.instances[-1].poll_widths,[6,6,6])
        self.assertEqual(len(f.FakeAdapter.instances[-1].compiles),18)
        self.assertFalse(set(f.FakeAdapter.instances[-1].builds)&set(r.family.REUSED))
        self.assertEqual(len(res['comparison']['comparisons']),60)
    def test_all_new_ineligible_skip_native_and_compile_retain_old_axes(self):
        self.c['fixture_skip']=[list(x) for x in r.family.NEW];self.refresh()
        self.assertEqual(self.execute(),0);res=self.read('full_family_results_v1.json')
        self.assertEqual(res['calls']['compile'],0);self.assertEqual(res['calls']['native_evolution_build'],0)
        self.assertEqual(self.backend.instances,[])
        self.assertEqual(sum(c['axes'] is not None for c in res['cases']),3)
        rows=[x for x in res['accounting'] if x['acquisition']=='SKIPPED_INELIGIBLE_UNMEASURED']
        self.assertTrue(all(x['status']=='INELIGIBLE' and x['cost_status']=='UNMEASURED' and x['metrics']=={} for x in rows))
    def test_partial_selection_counts_exact_adaptive_calls(self):
        self.c['fixture_skip']=[list(x) for x in r.family.NEW[:4]];self.refresh()
        self.assertEqual(self.execute(),0);calls=self.read('full_family_results_v1.json')['calls']
        self.assertEqual(calls['tail_scalar'],9);self.assertEqual(calls['compile'],10)
        self.assertEqual(calls['control_probe'],20)
    def test_missing_authorization_no_loader_or_factory(self):
        self.a={};factory=Mock()
        with self.assertRaises(g.Stop):self.execute(factory)
        factory.assert_not_called();self.assertFalse(self.output.exists())
    def test_draft_no_scientific_factory(self):
        self.c['approved']=False;self.refresh();factory=Mock()
        with self.assertRaises(g.Stop):self.execute(factory)
        factory.assert_not_called()
    def test_legacy_scope_grant_rejected(self):
        self.a['scope']='H6_NATIVE_B0I_PILOT6_V1'
        with self.assertRaises(g.Stop):self.execute()
    def test_canonical_contract_tamper_prevents_otherwise_unbound_field(self):
        self.c['fixture_skip']=[[5,2]]
        self.a['contract_sha256']=g.d0.digest(self.c)
        with self.assertRaisesRegex(g.Stop,'CONTENT_BINDING'):self.execute()
    def test_registry_mutation_before_provider(self):
        self.c['candidates'].append(dict(prefix=5,q=16,formula='2nd',T=.8));self.refresh()
        with self.assertRaisesRegex(g.Stop,'REGISTRY'):self.execute()
    def test_old_case_cannot_be_added_to_new_acquisition(self):
        self.c['new_candidates'][0]=dict(prefix=5,q=1,formula='2nd',T=.8);self.refresh()
        with self.assertRaises(g.Stop):self.execute()
    def test_compiler_options_mutation_before_provider(self):
        self.c['compiler_options']['optimization_level']=2;self.refresh()
        with self.assertRaisesRegex(g.Stop,'COMPILER'):self.execute()
    def test_old_wall_four_hours_not_implicitly_inherited(self):
        self.c['budget']['wall_seconds']=14400;self.refresh()
        with self.assertRaisesRegex(g.Stop,'WALL_POLICY'):self.execute()
    def test_more_workers_than_six_rejected(self):
        self.c['workers']=7;self.c['parallel_compile']['max_workers']=7;self.refresh()
        with self.assertRaisesRegex(g.Stop,'PARALLEL'):self.execute()
    def test_input_byte_tamper_no_factory(self):
        Path(self.mapping['pilot_results.json']).write_text('{}');factory=Mock()
        self.assertEqual(self.execute(factory),2);factory.assert_not_called()
    def test_source_byte_tamper_no_factory(self):
        (self.root/'synthetic_source.txt').write_text('ALTERED');factory=Mock()
        self.assertEqual(self.execute(factory),2);factory.assert_not_called()
    def test_failed_prior_run_not_reusable_before_science_loader(self):
        self.change_input('pilot_native_exit.json',lambda x:x.update(shell_wait_status=2));factory=Mock()
        self.assertEqual(self.execute(factory),2);factory.assert_not_called()
    def test_reuse_profile_mismatch_no_factory(self):
        self.change_input('pilot_authorization.json',lambda x:x.update(profile_digest='0'*64));factory=Mock()
        self.assertEqual(self.execute(factory),2);factory.assert_not_called()
    def test_reuse_hamiltonian_mismatch_no_factory(self):
        self.change_input('pilot_results.json',lambda x:x['old_science_binding'].update(hamiltonian='0'*64));factory=Mock()
        self.assertEqual(self.execute(factory),2);factory.assert_not_called()
    def test_reuse_axis_fingerprint_mismatch_no_factory(self):
        self.change_input('pilot_results.json',lambda x:x['cases'][0]['axes']['cosine'].update(fingerprint='0'*64));factory=Mock()
        self.assertEqual(self.execute(factory),2);factory.assert_not_called()
    def test_reuse_T_mismatch_before_loader(self):
        self.change_input('pilot_scalar_5_1.json',lambda x:x['signal']['cell'].update(T=1.))
        factory=Mock();self.assertEqual(self.execute(factory),2);factory.assert_not_called()
    def test_reuse_full_Fock_convention_required(self):
        self.change_input('pilot_scalar_5_1.json',lambda x:x['scalar'].update(full_Fock_identity=False))
        factory=Mock();self.assertEqual(self.execute(factory),2);factory.assert_not_called()
    def test_reuse_preparation_blueprint_must_match_fixed_expected(self):
        self.change_input('pilot_scalar_5_1.json',lambda x:x['blueprint'].update(q=16))
        factory=Mock();self.assertEqual(self.execute(factory),2);factory.assert_not_called()
    def test_reuse_provenance_keeps_original_source_not_new_measurement(self):
        self.assertEqual(self.execute(),0)
        for c in self.read('full_family_results_v1.json')['cases'][:3]:
            self.assertEqual(c['origin_SOURCE'],r.family.OLD_SOURCE)
            self.assertEqual(c['acquisition'],'REUSE_WITH_DECLARED_LIMITS')
            self.assertFalse(c['new_SOURCE_compiled_output_bit_identity_verified'])
    def test_last_screen_failure_blocks_all_builds_and_compiles(self):
        class Fail(f.FakeAdapter):
            def prepare(obj,prefix,q):
                if (prefix,q)==r.family.NEW[-1]:raise g.Stop('LAST_SCREEN_FAIL')
                return super().prepare(prefix,q)
        self.assertEqual(self.execute(Fail),2);self.assertEqual(f.FakeAdapter.instances[-1].builds,[])
        self.assertEqual(self.read('terminal_v1.json')['calls_reserved_before_operations']['compile'],0)
    def test_last_correctness_failure_blocks_all_compile(self):
        class Fail(f.FakeAdapter):
            def validate(obj,case,built):
                if (case['prefix'],case['q'])==r.family.NEW[-1]:raise g.Stop('LAST_CORRECTNESS_FAIL')
                return super().validate(case,built)
        self.assertEqual(self.execute(Fail),2);self.assertEqual(self.backend.instances,[])
    def test_native_failure_stops_queue_and_preserves_first_reason(self):
        self.c['workers']=2;self.c['parallel_compile'].update(max_workers=2,worker_cpus=[2,4]);self.refresh()
        self.backend.failure='native';self.backend.cleanup_error=True
        self.assertEqual(self.execute(),2);self.assertEqual(len(self.backend.instances[-1].handles),2)
        self.assertIn('ORIGINAL_COMPILE_FAILURE',self.read('terminal_v1.json')['first_stop']['reason'])
    def test_cleanup_failure_never_complete(self):
        self.backend.cleanup_error=True;self.assertEqual(self.execute(),2)
        self.assertIn('CLEANUP_UNCONFIRMED',self.read('terminal_v1.json')['first_stop']['reason'])
    def test_consumed_output_cannot_retry(self):
        self.assertEqual(self.execute(),0)
        with self.assertRaisesRegex(g.Stop,'ONE_SHOT'):self.execute()
    def test_missing_eligible_compiled_axis_not_complete(self):
        original=p.compile_batch
        def broken(*args):
            rows=original(*args,g=g,metrics_check=r.adapter_module.metrics_checked,backend_factory=self.backend)
            rows.pop(next(iter(rows)));return rows
        self.batch=broken;self.assertEqual(self.execute(),2)
    def test_PR_exploration_cannot_enter_confirmation_comparison(self):
        def mutate(x):
            next(v for v in x['H6_CONFIRMATION_UNION8'] if v['method']=='B2')['sample_phase']='exploration'
        self.change_input('d0_candidate_accounting.json',mutate)
        self.assertEqual(self.execute(),2)
        self.assertIn('CONFIRMATION_NOT_N32',self.read('terminal_v1.json')['first_stop']['reason'])
    def test_eligible_baseline_missing_cost_never_complete(self):
        self.change_input('d0_candidate_accounting.json',lambda x:x['H6_CONFIRMATION_UNION8'][0].update(status='MISSING'))
        self.assertEqual(self.execute(),2)
    def test_matched_B0_deltas_remain_null(self):
        self.assertEqual(self.execute(),0)
        self.assertTrue(all(x['matched_B0_delta'] is None for x in self.read('full_family_results_v1.json')['accounting']))
    def test_deterministic_denominator_includes_old_baselines(self):
        self.assertEqual(self.execute(),0)
        first=self.read('full_family_results_v1.json')['comparison']['comparisons'][0]
        self.assertEqual(first['best_registered_deterministic']['G_point'],100)
        self.assertEqual(first['confirmed_PR_union8']['G_point'],80)
        self.assertAlmostEqual(first['PR_over_deterministic'],.8)
    def test_new_operation_eighteenth_axis_is_hard_maximum(self):
        journal=[];calls=r.Calls(r.LIMITS,journal.append);calls.take('compile',18)
        with self.assertRaises(g.Stop):calls.take('compile')
        self.assertEqual(calls.used['compile'],18);self.assertEqual(journal[0]['refund'],0)
    def test_child_CPU_persists_after_successful_cleanup(self):
        self.assertEqual(self.execute(),0)
        self.assertAlmostEqual(self.read('parallel_cleanup_v1.json')['child_cpu_seconds'],.18)
    def test_only_native_validation_and_worker_compile_traverse_inputs(self):
        class WorkerChecks(f.FakeAdapter):
            checking=False
            def fingerprint(obj,w):
                if not obj.checking:raise AssertionError('SERIAL_PARENT_FINGERPRINT')
                return super().fingerprint(w)
            def validate(obj,*args):
                obj.checking=True
                try:return super().validate(*args)
                finally:obj.checking=False
            def compile(obj,*args):
                obj.checking=True
                try:return super().compile(*args)
                finally:obj.checking=False
        self.assertEqual(self.execute(WorkerChecks),0)
        for c in self.read('full_family_results_v1.json')['cases'][3:]:
            for row in c['axes'].values():
                self.assertEqual(row['input_fingerprint_check'],dict(before=row['fingerprint'],after=row['fingerprint']))
    def test_worker_detects_input_mutation_after_all_native_validation(self):
        start=self.backend.start
        def changed(backend,task):
            task['wrapper']+='MUTATION_AFTER_NATIVE_VALIDATION'
            return start(backend,task)
        with patch.object(self.backend,'start',changed):self.assertEqual(self.execute(),2)
        self.assertIn('PRECOMPILE_FINGERPRINT_CHANGED',self.read('terminal_v1.json')['first_stop']['reason'])
        self.assertFalse((self.output/'full_family_results_v1.json').exists())
    def test_worker_detects_input_mutation_during_compile(self):
        class Changed(f.FakeAdapter):
            changed=False
            def fingerprint(obj,w):return super().fingerprint(w+('MUTATED' if obj.changed else ''))
            def fabricated_cost(obj,*args,**kwargs):
                cost=super().fabricated_cost(*args,**kwargs);obj.changed=True;return cost
        self.assertEqual(self.execute(Changed),2)
        self.assertIn('COMPILED_INPUT_OR_COMPILER_BINDING',self.read('terminal_v1.json')['first_stop']['reason'])
    def test_worker_result_requires_both_before_and_after_proofs(self):
        poll=self.backend.poll
        def missing(backend,active):
            handles=poll(backend,active)
            for h in handles:
                obj=json.loads(h['payload']);obj['row'].pop('input_fingerprint_check')
                h['payload']=bytearray(json.dumps(obj).encode())
            return handles
        with patch.object(self.backend,'poll',missing):self.assertEqual(self.execute(),2)
        self.assertIn('PARALLEL_COMPILED_IDENTITY',self.read('terminal_v1.json')['first_stop']['reason'])
    def test_tampered_worker_after_proof_blocks_completion(self):
        poll=self.backend.poll
        def tampered(backend,active):
            handles=poll(backend,active)
            for h in handles:
                obj=json.loads(h['payload']);obj['row']['input_fingerprint_check']['after']='0'*64
                h['payload']=bytearray(json.dumps(obj).encode())
            return handles
        with patch.object(self.backend,'poll',tampered):self.assertEqual(self.execute(),2)
        self.assertIn('PARALLEL_COMPILED_IDENTITY',self.read('terminal_v1.json')['first_stop']['reason'])
    def test_final_accounting_also_requires_input_proofs(self):
        original=self.batch
        def missing(*args):
            rows=original(*args);next(iter(rows.values())).pop('input_fingerprint_check');return rows
        self.batch=missing;self.assertEqual(self.execute(),2)
        self.assertIn('COMPILED_IDENTITY_MISMATCH',self.read('terminal_v1.json')['first_stop']['reason'])
    def test_two_worker_retry_configuration_retains_all_eighteen_axes(self):
        self.c['workers']=2;self.c['parallel_compile'].update(max_workers=2,worker_cpus=[2,4]);self.refresh()
        self.assertEqual(self.execute(),0)
        self.assertEqual(self.backend.instances[-1].poll_widths,[2]*9)
        self.assertEqual(self.read('full_family_results_v1.json')['calls']['compile'],18)
    def test_repeated_STOP_signals_deferred_through_failed_cleanup(self):
        self.backend.failure='native';original=self.backend.cleanup
        real=p.CleanupSignals;blocked=set()
        def mask(how,values):
            previous=set(blocked)
            if how==signal.SIG_BLOCK:blocked.update(values)
            else:blocked.clear();blocked.update(values)
            return previous
        def cleanup(backend):
            self.assertTrue(p.CLEANUP_SIGNALS.issubset(blocked))
            return original(backend)
        with patch.object(p,'CleanupSignals',lambda enabled:real(True)),\
                patch.object(p.signal,'pthread_sigmask',side_effect=mask),\
                patch.object(p.signal,'setitimer',return_value=(.5,0.)),\
                patch.object(self.backend,'cleanup',cleanup):
            self.assertEqual(self.execute(),2)
        self.assertIn('ORIGINAL_COMPILE_FAILURE',self.read('terminal_v1.json')['first_stop']['reason'])
        self.assertTrue(self.read('parallel_cleanup_v1.json')['all_reaped'])
        self.assertTrue(self.read('terminal_v1.json')['cleanup']['signal_deferral']['retained_until_process_exit'])
    def test_numba_cache_is_confined_to_exclusive_run_temp(self):
        grant=g.authorization_gate(self.c,self.a,contract_sha=g.d0.digest(self.c),
            authorization_sha=g.d0.digest(self.a),output=self.output,scope=r.SCOPE,synthetic=True)
        self.output.mkdir()
        previous=tempfile.tempdir
        try:
            with patch.dict(os.environ):
                r.prepare_private_temp(grant)
                self.assertEqual(Path(os.environ['NUMBA_CACHE_DIR']),self.output/'temp/numba')
                self.assertTrue(Path(os.environ['NUMBA_CACHE_DIR']).is_dir())
        finally:tempfile.tempdir=previous
    def font_seed(self):
        raw=json.dumps(dict(_version='3.11.0',__class__='FontManager',afmlist=[],ttflist=[])).encode()
        path=self.base/'display-font-cache.json';path.write_bytes(raw)
        self.c['startup_font_cache']=dict(path=str(path),filename='fontlist-v3.11.0.json',
            format_version='3.11.0',scope='DISPLAY_FONT_METADATA_ONLY',bytes=len(raw),sha256=g.sha(raw))
        self.refresh();return path,raw
    def test_byte_bound_font_metadata_seed_without_science_import(self):
        path,raw=self.font_seed()
        grant=g.authorization_gate(self.c,self.a,contract_sha=g.d0.digest(self.c),
            authorization_sha=g.d0.digest(self.a),output=self.output,scope=r.SCOPE,synthetic=True)
        directory=self.base/'font-destination';directory.mkdir()
        g.seed_font_cache(grant,directory)
        self.assertEqual((directory/'fontlist-v3.11.0.json').read_bytes(),raw)
    def test_changed_font_metadata_rejected_before_copy(self):
        path,raw=self.font_seed();path.write_bytes(raw+b' ')
        grant=g.authorization_gate(self.c,self.a,contract_sha=g.d0.digest(self.c),
            authorization_sha=g.d0.digest(self.a),output=self.output,scope=r.SCOPE,synthetic=True)
        directory=self.base/'font-destination';directory.mkdir()
        with self.assertRaisesRegex(g.Stop,'BYTES_SHA'):g.seed_font_cache(grant,directory)
        self.assertEqual(list(directory.iterdir()),[])
    def test_unresolved_screen_is_not_ineligible_skip(self):
        rows=[dict(epsilon=e,factor=f,accounting=dict(cost_acquisition_eligible=False,
            eligibility_under_declared_allowance='NUMERICALLY_UNRESOLVED')) for e in r.family.EPS for f in r.family.FACTORS]
        with self.assertRaisesRegex(g.Stop,'UNRESOLVED'):r.family.screen(rows,g)
    def test_direct_mock_cannot_fall_through_to_real_fork(self):
        grant=g.authorization_gate(self.c,self.a,contract_sha=g.d0.digest(self.c),
            authorization_sha=g.d0.digest(self.a),output=self.output,scope=r.SCOPE,synthetic=True)
        with self.assertRaisesRegex(g.Stop,'MOCK_BACKEND'):p.ForkBackend(grant,None,None,g)


class SmallIndependentTests(unittest.TestCase):
    def test_cleanup_success_restores_timer_and_original_signal_mask(self):
        calls=[];original_mask={signal.SIGINT}
        def mask(how,values):calls.append(('mask',how,set(values)));return original_mask
        def timer(*args):calls.append(('timer',args));return (.25,0.)
        with patch.object(p.signal,'pthread_sigmask',side_effect=mask),patch.object(p.signal,'setitimer',side_effect=timer):
            state=p.CleanupSignals(True);state.defer();state.finish(resume=True)
        self.assertEqual(calls,[('mask',signal.SIG_BLOCK,p.CLEANUP_SIGNALS),
            ('timer',(signal.ITIMER_REAL,0)),('timer',(signal.ITIMER_REAL,.25,0.)),
            ('mask',signal.SIG_SETMASK,original_mask)])
        self.assertFalse(state.receipt()['retained_until_process_exit'])
    def test_cleanup_failure_does_not_restore_pending_signals_or_timer(self):
        with patch.object(p.signal,'pthread_sigmask',return_value=set()) as mask,\
                patch.object(p.signal,'setitimer',return_value=(.25,0.)) as timer:
            state=p.CleanupSignals(True);state.defer();state.finish(resume=False)
        mask.assert_called_once_with(signal.SIG_BLOCK,p.CLEANUP_SIGNALS)
        timer.assert_called_once_with(signal.ITIMER_REAL,0)
        self.assertTrue(state.receipt()['retained_until_process_exit'])
    def test_cleanup_reap_all_after_TERM_and_KILL_and_close_pipes(self):
        backend=object.__new__(p.ForkBackend);backend.parent=123;backend.g=g
        backend.handles=[dict(pid=900000+i,identity=dict(pid=900000+i),fd=100+i,
            task=dict(name=str(i)),reaped=False,exit_code=None,cpu_seconds=0.,peak_rss_bytes=0,payload=bytearray()) for i in range(6)]
        sent={};clock=[0.];closed=[]
        def terminate(pid,identity,sig,guard):sent[pid]=sig
        def reap(h):
            if sent.get(h['pid'])==(signal.SIGTERM if h['pid']%2 else signal.SIGKILL):
                h.update(reaped=True,exit_code=-int(sent[h['pid']]),cpu_seconds=.25)
        def drain(h):closed.append(h['fd']);h['fd']=None
        backend.reap=reap;backend.drain=drain
        with patch.object(p,'terminate_owned',side_effect=terminate),\
                patch.object(p.time,'monotonic',side_effect=lambda:clock[0]),\
                patch.object(p.time,'sleep',side_effect=lambda seconds:clock.__setitem__(0,clock[0]+seconds)):
            receipt=backend.cleanup()
        self.assertTrue(receipt['all_reaped']);self.assertEqual(receipt['errors'],[])
        self.assertEqual(set(closed),set(range(100,106)));self.assertLess(clock[0],3.1)
        self.assertEqual(sum(x['cpu_seconds'] for x in receipt['workers']),1.5)
    def test_unreaped_cleanup_never_reports_success_after_bounded_waits(self):
        backend=object.__new__(p.ForkBackend);backend.parent=123;backend.g=g
        h=dict(pid=900000,identity=dict(pid=900000),fd=100,task=dict(name='fixture'),
            reaped=False,exit_code=None,cpu_seconds=0.,peak_rss_bytes=0,payload=bytearray())
        backend.handles=[h];backend.reap=lambda h:None;backend.drain=lambda h:None;clock=[0.]
        with patch.object(p,'terminate_owned'),patch.object(p.os,'close') as close,\
                patch.object(p.time,'monotonic',side_effect=lambda:clock[0]),\
                patch.object(p.time,'sleep',side_effect=lambda seconds:clock.__setitem__(0,clock[0]+seconds)):
            receipt=backend.cleanup()
        self.assertFalse(receipt['all_reaped']);self.assertIsNone(receipt['workers'][0]['native_exit_code'])
        close.assert_called_once_with(100);self.assertLess(clock[0],3.1)
    def test_production_cannot_omit_bound_font_seed(self):
        grant=SimpleNamespace(require=lambda scope:None,synthetic=False,contract={})
        with self.assertRaisesRegex(g.Stop,'BOUND_STARTUP_FONT_CACHE_REQUIRED'):
            g.seed_font_cache(grant,Path('/home/AbeHiromu/unused'))
    def test_thread_and_affinity_failures_have_distinct_diagnostics(self):
        e=r.load('family_observer_error_test','h6_b0i_family_external_monitor_v1.py')
        observer=object.__new__(e.Observer);observer.g=g;observer.clock=lambda:100.
        observer.resource_last_seen={};observer.worker_first_seen={};observer.worker_affinity_confirmed=set()
        observer.grant=SimpleNamespace(contract=dict(budget=dict(rss_bytes=2**30,address_space_bytes=2**30,cpu_seconds=100)))
        identity=dict(pid=123456,uid=os.getuid(),start_ticks=1,parent_pid=123,boot_id='FIXTURE')
        for threads,affinity,reason in ((2,{63},'EXTERNAL_THREAD_COUNT_CHANGED:threads=2'),
                (1,{62},'EXTERNAL_AFFINITY_CHANGED')):
            status=f'State:\tR\nVmHWM:\t16 kB\nVmSize:\t32 kB\nThreads:\t{threads}\n'
            def read(path,*args,**kwargs):
                return status if path.name=='status' else '123456 (fixture) '+' '.join(['R']+['0']*30)
            with patch.object(g,'identity',return_value=identity),patch.object(e.Path,'read_text',read),\
                    patch.object(e.os,'sched_getaffinity',return_value=affinity):
                with self.assertRaisesRegex(g.Stop,reason):observer.process(identity,[63])
    def test_Phase_D_still_charges_completed_children(self):
        monitor=object.__new__(g.Monitor)
        monitor.clock=lambda:100.;monitor.last=100.;monitor.started=0.;monitor.max_interval=0.
        monitor.budget=dict(monitor_max_seconds=5.,wall_seconds=None,cpu_seconds=10)
        monitor.child_cpu_seconds=9.;monitor.first_stop=None;monitor.phase='D_COMPARISON'
        with patch.object(g.resource,'getrusage',return_value=SimpleNamespace(ru_utime=2.,ru_stime=0.)):
            with self.assertRaisesRegex(g.Stop,'AGGREGATE_STAGE_CPU_CAP'):monitor.tick(force=True)
        self.assertEqual(monitor.first_stop['phase'],'D_COMPARISON')
    def test_CPU_stop_does_not_replace_first_stop(self):
        monitor=object.__new__(g.Monitor)
        monitor.clock=lambda:100.;monitor.first_stop=dict(reason='ORIGINAL_STOP',phase='B',monotonic=1.)
        with self.assertRaisesRegex(g.Stop,'ORIGINAL_STOP'):monitor.fail('AGGREGATE_STAGE_CPU_CAP')
    def test_signed_trace_identity_two_mode_hand_calculation(self):
        # trG=3, trG^2=7 -> signed lambda*(-) 16/4 = -8.
        self.assertEqual(r.adapter_module.b.exact_identity(((1,1j),(-1j,2)),-2.),-8)
    def test_phase_rotation_preserves_full_H_reference(self):
        record=dict(signals=dict(corrected=dict(real=1.,imag=0.),exact_truncated=dict(real=.5,imag=0.)),
            reference=dict(real=.2,imag=.7),allowance=dict(log_B_margin=0.),
            error_components=dict(kind='EMPIRICAL',certified=False,B0_signal=dict(real=0.,imag=0.),
                full_H_reference=dict(real=0.,imag=0.),signal_abs_upper=1.,evidence={}))
        import math
        out=r.adapter_module.b.saved_signal_overlay(record,C=math.pi/(2*.8),C_error=0.,rotation_roundoff=0.,phase_roundoff=0.)
        self.assertAlmostEqual(out['signals']['corrected']['imag'],-1.)
        self.assertEqual(out['reference'],record['reference']);self.assertFalse(out['allowance']['certified'])
    def test_allowance_never_becomes_certificate(self):
        record=dict(allowance=dict(kind='EMPIRICAL',certified=False,real=1e-6,imag=2e-6),
            signals=dict(corrected=dict(real=.5,imag=.2)))
        value=r.adapter_module.b.conservative_saved_error_components(record)
        self.assertEqual(value['B0_signal'],value['full_H_reference'])
        self.assertFalse(value['certified'])
    def test_scope_api_forbids_old_pilot_reacquisition(self):
        self.assertEqual(set(r.adapter_module.b.PILOT),set(r.family.NEW))
        self.assertTrue(set(r.adapter_module.b.PILOT).isdisjoint(r.family.REUSED))
    def test_pidfd_only_ESRCH_is_exit_and_FD_closed(self):
        identity=dict(pid=123456,uid=os.getuid(),parent_pid=123,start_ticks=4,boot_id='ARTIFICIAL')
        for number in (errno.ESRCH,errno.EPERM):
            with patch.object(g,'identity',return_value=identity),patch.object(p.os,'pidfd_open',return_value=876),\
                    patch.object(p.signal,'pidfd_send_signal',side_effect=OSError(number,'ARTIFICIAL')),\
                    patch.object(p.os,'close') as close:
                if number==errno.ESRCH:p.terminate_owned(identity['pid'],identity,signal.SIGTERM,g)
                else:
                    with self.assertRaises(OSError):p.terminate_owned(identity['pid'],identity,signal.SIGTERM,g)
                close.assert_called_once_with(876)
    def test_pid_owner_change_forbids_signal(self):
        with patch.object(g,'identity',return_value={'pid':1}),patch.object(p.os,'pidfd_open') as opened:
            with self.assertRaises(g.Stop):p.terminate_owned(1,{'pid':2},signal.SIGTERM,g)
            opened.assert_not_called()


if __name__=='__main__':unittest.main()
