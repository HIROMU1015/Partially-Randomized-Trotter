"""Bounded artificial integration tests; never loads H6 or imports Qiskit."""
from copy import deepcopy
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch

spec=importlib.util.spec_from_file_location('h6_new_fixture',Path(__file__).with_name('h6_pilot_fixtures_v1.py'))
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
r=f.r


class PilotRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=os.environ['TMPDIR'],prefix='h6-pilot-artificial-')
        self.base=Path(self.tmp.name);self.root=self.base/'source';self.out=self.base/'out'
        self.c,self.a=f.setup(self.root,self.out);f.FakeAdapter.instances.clear()
    def tearDown(self):self.tmp.cleanup()
    def refresh(self):self.a=f.f.authorization(self.c,self.out)
    def execute(self,factory=f.FakeAdapter,loader=lambda:{}):
        return r.run(self.c,self.a,contract_sha=r.g.d0.digest(self.c),authorization_sha=r.g.d0.digest(self.a),
            output=self.out,root=self.root,mapping_loader=loader,synthetic=True,
            monitor_factory=f.f.FakeMonitor,factory=factory)
    def read(self,name):return json.loads((self.out/name).read_text())

    def test_complete_three_cases_six_axes_and_no_matched_build(self):
        self.assertEqual(self.execute(),0)
        result=self.read('pilot_results_v1.json');self.assertEqual(result['calls'],r.LIMITS)
        self.assertEqual(len(result['cases']),3);api=f.FakeAdapter.instances[-1]
        self.assertEqual(len(api.compiles),6);self.assertEqual(len(api.native_calls),3);self.assertTrue(api.closed)
        for case in result['cases']:
            self.assertIsNone(case['matched_B0']['deltas']);self.assertFalse(case['matched_B0']['automatic_build'])
            self.assertEqual(len(case['accounting']),10)
        self.assertEqual(self.read('terminal_v1.json')['status'],'COMPLETE_MANDATORY_STOP')

    def test_missing_authorization_never_reaches_mapping_or_provider(self):
        self.a={};loader=Mock();factory=Mock()
        with self.assertRaises(r.g.Stop):self.execute(factory,loader)
        loader.assert_not_called();factory.assert_not_called();self.assertFalse(self.out.exists())

    def test_draft_never_reaches_real_loader(self):
        self.c['approved']=False;self.c['runtime_authorization']=False;self.refresh();loader=Mock()
        with self.assertRaises(r.g.Stop):self.execute(loader=loader)
        loader.assert_not_called()

    def test_budget_profile_carry_input_binding_mutants_rejected(self):
        for field in ('budget','profile','carry','input_receipts','source_hashes','allowed_cpus'):
            auth=deepcopy(self.a);auth[field+'_digest']='0'*64
            with patch.object(self,'a',auth),self.subTest(field=field),self.assertRaises(r.g.Stop):self.execute()

    def test_registry_extension_rejected_before_loader(self):
        self.c['candidates'].append(dict(prefix=15,q=4,formula='2nd',T=.8));self.refresh();loader=Mock()
        with self.assertRaisesRegex(r.g.Stop,'REGISTRY'):self.execute(loader=loader)
        loader.assert_not_called()

    def test_compiler_options_changed_rejected_before_loader(self):
        self.c['compiler_options']['seed_transpiler']=19;self.refresh();loader=Mock()
        with self.assertRaisesRegex(r.g.Stop,'COMPILER_OPTIONS'):self.execute(loader=loader)
        loader.assert_not_called()

    def test_source_tamper_stops_before_loader(self):
        (self.root/'synthetic_source.txt').write_text('TAMPER');loader=Mock()
        self.assertEqual(self.execute(loader=loader),2);loader.assert_not_called()
        self.assertIn('SOURCE_BYTES_SHA',self.read('terminal_v1.json')['first_stop']['reason'])

    def test_last_phase_a_failure_keeps_compile_zero(self):
        class Fail(f.FakeAdapter):
            def prepare(obj,p,q):
                if p==15:raise r.g.Stop('LAST_SCALAR_STOP')
                return super(Fail,obj).prepare(p,q)
        self.assertEqual(self.execute(Fail),2);api=f.FakeAdapter.instances[-1]
        self.assertEqual(api.builds,[]);self.assertEqual(api.compiles,[]);self.assertTrue(api.closed)

    def test_last_phase_b_failure_keeps_compile_zero(self):
        class Fail(f.FakeAdapter):
            def validate(obj,case,built):
                if case['prefix']==15:raise r.g.Stop('LAST_NATIVE_STOP')
                return super(Fail,obj).validate(case,built)
        self.assertEqual(self.execute(Fail),2);api=f.FakeAdapter.instances[-1]
        self.assertEqual(api.compiles,[]);self.assertTrue(api.closed)
        self.assertEqual(self.read('terminal_v1.json')['calls_reserved_before_operations']['compile'],0)

    def test_input_tamper_never_loads_adapter(self):
        p=self.base/'input.json';p.write_bytes(b'{}');self.c['input_receipts']=[dict(logical_name='input',bytes=2,sha256='0'*64)];self.refresh();factory=Mock()
        self.assertEqual(self.execute(factory,lambda:{'input':str(p)}),2);factory.assert_not_called()

    def test_numeric_fingerprint_mutation_stops_before_compile(self):
        class Fail(f.FakeAdapter):
            def validate(obj,case,built):
                v=super(Fail,obj).validate(case,built);v['fingerprints']['cosine']='0'*64;return v
        self.assertEqual(self.execute(Fail),2);self.assertEqual(f.FakeAdapter.instances[-1].compiles,[])

    def test_compiled_hash_mismatch_stops_and_keeps_partial(self):
        class Fail(f.FakeAdapter):
            def compile(obj,w,fp):
                v=super(Fail,obj).compile(w,fp);v['compiler_hash']='0'*64;return v
        self.assertEqual(self.execute(Fail),2)
        terminal=self.read('terminal_v1.json');self.assertEqual(terminal['calls_reserved_before_operations']['compile'],1)
        self.assertIn('COMPILED_IDENTITY',terminal['first_stop']['reason'])
        self.assertFalse((self.out/'pilot_results_v1.json').exists())

    def test_invalid_compiled_metric_is_not_missing_zero(self):
        class Fail(f.FakeAdapter):
            def compile(obj,w,fp):
                v=super(Fail,obj).compile(w,fp);v['metrics']['RZ']=None;return v
        self.assertEqual(self.execute(Fail),2);self.assertFalse((self.out/'pilot_results_v1.json').exists())

    def test_one_shot_cannot_retry_after_provider_failure(self):
        class Fail(f.FakeAdapter):
            def build(obj,case):raise r.g.Stop('PROVIDER_FAILURE')
        self.assertEqual(self.execute(Fail),2)
        with self.assertRaisesRegex(r.g.Stop,'ONE_SHOT_ALREADY_USED'):self.execute()

    def test_output_collision_preserves_old_file(self):
        self.out.mkdir();(self.out/'KEEP').write_text('OLD')
        with self.assertRaisesRegex(r.g.Stop,'ONE_SHOT_ALREADY_USED'):self.execute()
        self.assertEqual((self.out/'KEEP').read_text(),'OLD')

    def test_call_budget_reservation_persists_on_failed_operation(self):
        journal=[];calls=r.Calls(dict(compile=1),journal.append);calls.take('compile')
        with self.assertRaises(r.g.Stop):calls.take('compile')
        self.assertEqual(calls.used['compile'],1);self.assertEqual(journal[0]['refund'],0)

    def test_stored_independent_reference_required_and_finite(self):
        for record in ({},{'independent_forward_signals':{'corrected':{'real':math.nan,'imag':0.}}}):
            with self.assertRaises(ValueError):r.adapter_module.stored_independent(record)
        self.assertEqual(r.adapter_module.stored_independent({'independent_forward_signals':{'corrected':{'real':1.,'imag':.5}}}),1+.5j)

    def test_absent_native_wait_does_not_become_success(self):
        self.assertEqual(self.execute(),0)
        terminal=self.read('terminal_v1.json');self.assertIsNone(terminal['native_exit_code'])
        self.assertFalse((self.out/'native_exit_v1.json').exists())

    def test_signal_identity_is_separate_from_b0_request(self):
        self.assertEqual(self.execute(),0)
        scalar=self.read('case_5_1_scalar_signal.json')
        self.assertEqual(scalar['signal']['cell']['id'],'B0I_p5_q1_2nd')
        self.assertFalse(scalar['signal']['allowance']['certified'])
        self.assertEqual(scalar['signal']['reference'],self.c['fixture_record']['reference'])

    def test_eligible_cost_uses_both_axis_shots_without_zero_fallback(self):
        for fragment in self.c['fixture_fragments']:fragment['lambda']=0.
        self.refresh();self.assertEqual(self.execute(),0)
        row=self.read('case_5_1_scalar_signal.json')['precision'][0]['accounting']
        value=self.read('pilot_results_v1.json')['cases'][0]['accounting'][0]
        self.assertEqual(value['status'],'DERIVED_SAVED_SCALAR')
        self.assertEqual(value['metrics']['RZ'],10*(row['axes']['real']['shots']+row['axes']['imag']['shots']))

    def test_final_provider_tamper_is_not_complete(self):
        class Fail(f.FakeAdapter):
            def verify_bindings(obj):raise r.g.Stop('FINAL_PROVIDER_TAMPER')
        self.assertEqual(self.execute(Fail),2)
        self.assertEqual(self.read('terminal_v1.json')['status'],'FAIL_CLOSED_STOP')

    def test_journal_cap_stops_before_first_scientific_load(self):
        self.c['journal_cap_bytes']=1;self.refresh()
        self.assertEqual(self.execute(),2)
        terminal=self.read('terminal_v1.json')
        self.assertIn('JOURNAL_CAP',terminal['first_stop']['reason'])
        self.assertEqual(terminal['calls_reserved_before_operations']['compile'],0)

    def test_private_temp_is_created_before_tempfile_can_fallback(self):
        grant=r.g.authorization_gate(self.c,self.a,contract_sha=r.g.d0.digest(self.c),
            authorization_sha=r.g.d0.digest(self.a),output=self.out,scope=r.SCOPE,synthetic=True)
        self.out.mkdir()
        with patch.dict(os.environ),patch.object(r.tempfile,'tempdir',None):
            r.prepare_private_temp(grant)
            self.assertEqual(r.tempfile.gettempdir(),str(self.out/'temp'))
            for key,name in (('TMPDIR','temp'),('MPLCONFIGDIR','temp/mpl'),('XDG_CACHE_HOME','temp/cache')):
                self.assertEqual(os.environ[key],str(self.out/name));self.assertTrue((self.out/name).is_dir())

if __name__=='__main__':unittest.main()
