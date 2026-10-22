"""Artificial records and operation logs only; no saved scientific evaluation."""
import builtins
import copy
from dataclasses import asdict, replace
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
real_import = builtins.__import__
forbidden_import_attempts = []


def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'numpy', 'scipy', 'qiskit', 'trotterlib', 'trottertracks'} or name.startswith('h46_'):
        forbidden_import_attempts.append(name)
        raise AssertionError('FORBIDDEN_SCIENTIFIC_IMPORT:' + name)
    return real_import(name, *args, **kwargs)


builtins.__import__ = guarded_import
spec = importlib.util.spec_from_file_location('h2_common_under_test', ROOT / 'src/trotterlib/h2_common_native_policy.py')
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


def draft():
    return dict(schema='h2_common_preflight_v1', status='DRAFT_NOT_EXECUTABLE',
        science_authorized=False, native_adopted=False, allowance_adopted=False,
        physical=copy.deepcopy(m.PHYSICAL), compiler=copy.deepcopy(m.COMPILER),
        candidates=[asdict(c) for c in m.CANDIDATES], native_policy=m.NATIVE_POLICY,
        allowance_policy=m.ALLOWANCE_POLICY, production_dispatch_bound=False,
        grant=None, ledger=None, marker=None, retry=False, mandatory_STOP=True,
        new_call_counts=None, missing_inputs=['synthetic_missing_reference'])


def prep(candidate):
    return dict(native_policy=m.NATIVE_POLICY, prefix=candidate.prefix,
        fragment_order=list(range(candidate.prefix)), n_qubits=4, ancilla=4,
        preparation_sha256='a'*64, basis_sha256='b'*64, operations_sha256='c'*64,
        basis_order='SAVED_JW_PROVIDER_ORDER', scalar_policy='ORIGINAL_PLUS_PR_TAIL_ONCE')


def pieces(candidate):
    # Invented times, deliberately not a molecular/Yoshida numerical fixture.
    signs = (0.125,) if candidate.formula == '2nd' else (0.125, -0.0625, 0.125)
    counts = (32,) if len(signs) == 1 else (10, 12, 10)
    return [dict(outer=o, piece=i, signed_time=t,
                 rte_steps=counts[i] if candidate.method == 'PR' else 0)
            for o in range(candidate.q) for i, t in enumerate(signs)]


def requests(candidate, schedule, preparation):
    return [] if candidate.method == 'FULL' else [dict(piece=s, K=4, controlled=True,
        ancilla=4, preparation_sha256=preparation['preparation_sha256'],
        conditioning='UNCONDITIONAL_SAVED_LAW') for s in schedule]


def diagnostics():
    def x(v, unit, scope):return m.Observation(v, unit, scope, 'synthetic:test')
    return m.AdaptiveInputs(m.CANDIDATES[2].id, 'SYNTHETIC_ONLY', 'synthetic:reference',
        x(1e-6, 'CORRECTED_SIGNAL', 'H2_REFERENCE_COMBINED_DIAGNOSTIC'),
        (x(2e-6, 'CORRECTED_SIGNAL', 'PRIMARY_VS_INDEPENDENT_SIGNAL'),),
        (x(3e-6, 'CORRECTED_STATE_VECTOR', 'PRIMARY_VS_INDEPENDENT_VECTOR'),),
        x(4e-6, 'CORRECTED_SIGNAL', 'B_RAW_MINUS_CORRECTED_SIGNAL'),
        x(5e-6, 'LOG_NORMALIZATION', 'TWO_PATH_LOG_B_DIFFERENCE'),
        dict(deterministic=2, tail=3), dict(deterministic=4, tail=5),
        x(1.25, 'CORRECTED_STATE_VECTOR', 'BOTH_PATHS_ALL_REQUIRED_STAGES'),
        'BOTH_PATHS_ALL_REQUIRED_STAGES', 'synthetic:both-path-traces',
        m.NATIVE_POLICY, 'e'*64)


class DraftTests(unittest.TestCase):
    def test_valid_is_not_ready(self):
        r=m.validate_draft(draft());self.assertFalse(r['science_GO']);self.assertFalse(r['execution_ready'])

    def test_science_or_adoption_flags_cannot_authorize(self):
        for key in ['science_authorized','native_adopted','allowance_adopted','production_dispatch_bound','retry']:
            with self.subTest(key=key):
                d=draft();d[key]=True
                with self.assertRaises(m.Stop):m.validate_draft(d)

    def test_live_or_guessed_reservations_rejected(self):
        for key,value in [('grant',{}),('ledger','x'),('marker','x'),('new_call_counts',{'compile':4})]:
            with self.subTest(key=key):
                d=draft();d[key]=value
                with self.assertRaises(m.Stop):m.validate_draft(d)

    def test_fixed_conditions_and_candidates(self):
        for modify in [lambda d:d['physical'].update(epsilon=.001),
                       lambda d:d['compiler'].update(optimization_level=2),
                       lambda d:d['candidates'].pop(),
                       lambda d:d['candidates'][2].update(q=8),
                       lambda d:d.update(missing_inputs=[])]:
            d=draft();modify(d)
            with self.assertRaises(m.Stop):m.validate_draft(d)

    def test_unknown_fields_and_numeric_type_substitution(self):
        for modify in [lambda d:d.update(science_command='run'),
                       lambda d:d['physical'].update(P=False),
                       lambda d:d['candidates'][0].update(q=4.0)]:
            d=draft();modify(d)
            with self.assertRaises(m.Stop):m.validate_draft(d)

    def test_missing_inputs_must_be_unique_explicit_names(self):
        for value in ['missing', [None], ['x', 'x']]:
            d=draft();d['missing_inputs']=value
            with self.assertRaises(m.Stop):m.validate_draft(d)

    def test_cli_denies_execute_before_read(self):
        r=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/prepare_h2_common_native.py'),
                          '--plan','/does/not/exist','--execute'],capture_output=True,text=True)
        self.assertEqual(r.returncode,2);self.assertIn('SCIENCE_DISPATCH_UNBOUND',r.stderr)
        self.assertNotIn('FileNotFound',r.stderr)

    def test_cli_draft(self):
        with tempfile.TemporaryDirectory(prefix='h2-common-synthetic-') as d:
            p=Path(d)/'synthetic.json';p.write_text(json.dumps(draft()))
            r=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/prepare_h2_common_native.py'),
                              '--plan',str(p)],capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr);self.assertFalse(json.loads(r.stdout)['science_GO'])


class NativeTests(unittest.TestCase):
    def build(self,c,**changes):
        p=prep(c);s=pieces(c);data=dict(domain='SYNTHETIC_ONLY',scalar=.375)
        r=requests(c,s,p);args=dict(candidate_record=asdict(c),evidence=data,
                                 preparation=p,pieces=s,requests=r);args.update(changes)
        port=m.MockNativePort();a=m.CommonNativeAdapter(port,m.SyntheticAuthority())
        return port,a,args

    def test_four_methods_share_policy_scalar_and_wrapper(self):
        for c in m.CANDIDATES:
            with self.subTest(candidate=c.id):
                port,a,args=self.build(c);out=a.build(**args)
                events=port.events
                self.assertEqual(events[0]['binding'],m.BINDINGS['prepare'])
                scalar=[e for e in events if e['operation']=='scalar_once']
                self.assertEqual(len(scalar),1);self.assertEqual(scalar[0]['scalar'],.375)
                fragments=[e for e in events if e['operation']=='signed_s2_piece']
                self.assertTrue(all(e['scalar']==0.0 for e in fragments))
                self.assertEqual([e['piece'] for e in fragments],args['pieces'])
                self.assertEqual([e['axis'] for e in events if e['operation']=='guarded_compile'],['cosine','sine'])
                self.assertFalse(out['compiled_identity_attested'])

    def test_no_authority_no_partial_output(self):
        port,a,args=self.build(m.CANDIDATES[0]);a.authority=None
        with self.assertRaises(m.Stop):a.build(**args)
        self.assertEqual(port.events,[])

    def test_real_domain_no_partial_output(self):
        port,a,args=self.build(m.CANDIDATES[0]);args['evidence']['domain']='SAVED_SCIENCE'
        with self.assertRaises(m.Stop):a.build(**args)
        self.assertEqual(port.events,[])

    def test_unordered_or_sign_changed_schedule(self):
        for mutation in [lambda s:s.reverse(),lambda s:s[1].update(signed_time=.0625),
                         lambda s:s[0].update(signed_time=math.nan),lambda s:s.pop()]:
            port,a,args=self.build(m.CANDIDATES[1]);mutation(args['pieces'])
            with self.assertRaises(m.Stop):a.build(**args)
            self.assertEqual(port.events,[])

    def test_outer32_and_no_tail_for_full(self):
        for c in [m.CANDIDATES[0],m.CANDIDATES[3]]:
            port,a,args=self.build(c);args['pieces'][0]['rte_steps']+=1
            with self.assertRaises(m.Stop):a.build(**args)

    def test_request_identity_and_unconditional_law(self):
        for key,value in [('ancilla',3),('controlled',False),('K',2),('conditioning','S0_ONLY'),('preparation_sha256','d'*64)]:
            port,a,args=self.build(m.CANDIDATES[2]);args['requests'][0][key]=value
            with self.assertRaises(m.Stop):a.build(**args)
            self.assertEqual(port.events,[])

    def test_basis_wire_and_fragment_order(self):
        for key,value in [('ancilla',0),('basis_order','REVERSED'),('fragment_order',[2,1,0]),
                          ('operations_sha256',None),('scalar_policy','PER_PIECE')]:
            port,a,args=self.build(m.CANDIDATES[0]);args['preparation'][key]=value
            with self.assertRaises(m.Stop):a.build(**args)

    def test_prefixes_need_not_have_equal_preparation_hash(self):
        for i,c in enumerate(m.CANDIDATES):
            port,a,args=self.build(c);args['preparation']['preparation_sha256']=str(i+1)*64
            args['requests']=requests(c,args['pieces'],args['preparation'])
            self.assertTrue(a.build(**args)['synthetic_only'])

    def test_full_random_requests_rejected(self):
        port,a,args=self.build(m.CANDIDATES[0]);args['requests']=[{}]
        with self.assertRaises(m.Stop):a.build(**args)

    def test_deferred_recipe_preserves_method_specific_bindings(self):
        for c in m.CANDIDATES:
            r=m.deferred_kernel_recipe(asdict(c))
            self.assertFalse(r['executable'])
            self.assertEqual(r['prepare']['prefix'],c.prefix)
            self.assertEqual(r['compile']['guard'],m.BINDINGS['compiler_guard'])
            self.assertEqual(r['compile']['apply_guard_to_method'],c.method)
            self.assertEqual(r['compose']['scalar_per_piece'],0.0)
            self.assertEqual(r['schedule']['r_value'],32 if c.method=='PR' else 1)
            if c.method=='FULL':
                self.assertEqual(r['requests']['value'],[])
                self.assertIsNone(r['requests']['binding'])
                self.assertEqual(r['compose']['scalar'],'$ham.constant')
            else:
                self.assertEqual(r['requests']['binding'],m.BINDINGS['requests'])
                self.assertFalse(r['requests']['sampling_authorized'])
                self.assertEqual(r['compose']['scalar'],'$ham.constant + tail_identity')

    def test_preparation_types_do_not_alias_true_and_one(self):
        port,a,args=self.build(m.CANDIDATES[2]);args['preparation']['prefix']=True
        with self.assertRaises(m.Stop):a.build(**args)


class AllowanceTests(unittest.TestCase):
    def evaluate(self,d):return m.evaluate_synthetic_allowance(d,m.SyntheticAuthority(),common_reference_id='synthetic:reference')

    def test_artificial_formula_and_no_certificate(self):
        r=self.evaluate(diagnostics())
        self.assertAlmostEqual(r['u_real'],8e-5)
        self.assertAlmostEqual(r['log_B_margin'],5e-5)
        self.assertFalse(r['certified']);self.assertFalse(r['formally_adopted'])

    def test_floor_is_times_safety(self):
        d=diagnostics();zero=lambda x:replace(x,value=0.0)
        d=replace(d,reference_error=zero(d.reference_error),
          path_signal_errors=tuple(zero(x) for x in d.path_signal_errors),
          path_vector_errors=tuple(zero(x) for x in d.path_vector_errors),
          corrected_signal_closure=zero(d.corrected_signal_closure),log_B_discrepancy=zero(d.log_B_discrepancy),
          work_primary=dict(deterministic=0,tail=0),work_independent=dict(deterministic=0,tail=0))
        r=self.evaluate(d);self.assertEqual(r['u_real'],1e-11)

    def test_saved_science_blocked_before_evaluation(self):
        d=replace(diagnostics(),domain='SAVED_SCIENCE',work_primary=None)
        with self.assertRaisesRegex(m.Stop,'SAVED_SCIENCE_VALUES'):self.evaluate(d)

    def test_missing_values_do_not_become_zero(self):
        d=diagnostics()
        for kw in [dict(reference_error=replace(d.reference_error,value=None)),
                   dict(path_vector_errors=()),dict(work_independent=None),
                   dict(maximum_norm=replace(d.maximum_norm,value=None)),dict(trace_evidence=None)]:
            with self.subTest(field=next(iter(kw))):
                with self.assertRaises(m.Stop):self.evaluate(replace(d,**kw))

    def test_nonfinite_negative_boolean(self):
        d=diagnostics()
        for value in [math.nan,math.inf,-1.,True]:
            with self.subTest(value=value):
                with self.assertRaises(m.Stop):self.evaluate(replace(d,reference_error=replace(d.reference_error,value=value)))

    def test_unit_and_coverage_mismatch(self):
        d=diagnostics()
        for kw in [dict(corrected_signal_closure=replace(d.corrected_signal_closure,unit='RAW_VECTOR')),
                   dict(norm_coverage='PIECE_ENDPOINTS_ONLY'),dict(work_primary={'stage_proxy':7}),
                   dict(reference_id='DIFFERENT_REFERENCE')]:
            with self.assertRaises(m.Stop):self.evaluate(replace(d,**kw))

    def test_mapping_pr_does_not_use_saved_u_or_raw_closure(self):
        projection={'saved':{'candidate':asdict(m.CANDIDATES[2]),
          'empirical_eligibility':{'u':1e-9},'normalization_error':0.0,
          'independent_forward_abs_errors':[1e-12],
          'signal':{'allowance_evidence':{'raw_corrected_vector_closure':1e-12,'maximum_intermediate_norm':1.0}}}}
        mapped=m.map_pr_diagnostics(projection,'synthetic:projection')
        self.assertIsNone(mapped.corrected_signal_closure.value)
        self.assertIsNone(mapped.maximum_norm.value)
        self.assertIsNone(mapped.work_primary)
        self.assertIn('path_vector_errors',mapped.missing())

    def test_overflow_stops(self):
        d=diagnostics();d=replace(d,maximum_norm=replace(d.maximum_norm,value=1e308),
                              work_primary=dict(deterministic=10**100,tail=0))
        with self.assertRaises(m.Stop):self.evaluate(d)

    def test_saved_native_diagnostics_need_new_series_binding(self):
        d=replace(diagnostics(),native_policy='LEGACY_NATIVE')
        with self.assertRaises(m.Stop):self.evaluate(d)


class ReuseTests(unittest.TestCase):
    def receipt(self):
        keys=('candidate_id','hamiltonian_sha256','state_sha256','native_policy',
          'preparation_sha256','schedule_sha256','wrapper_sha256','compiler_source_sha256',
          'compiler_settings_sha256','runtime_closure_sha256','event_law_sha256','cohort_id')
        return dict({k:'a'*64 if k.endswith('_sha256') else 'synthetic:'+k for k in keys},compiled_correctness_accepted=True)

    def test_identical_accepted_is_reusable(self):
        r=self.receipt();self.assertTrue(m.check_cost_reuse(r,r)['reusable'])

    def test_every_identity_difference_blocks_reuse_without_recompile(self):
        r=self.receipt()
        for k in r:
            x=dict(r);x[k]=None
            result=m.check_cost_reuse(x,r)
            self.assertFalse(result['reusable']);self.assertFalse(result['automatic_recompile'])

    def test_no_actual_imports(self):
        self.assertEqual(forbidden_import_attempts,[])


class ResultSchemaTests(unittest.TestCase):
    def record(self):
        return dict(schema='h2_common_result_proposal_v1',domain='SYNTHETIC_ONLY',
            candidate=asdict(m.CANDIDATES[0]),native_policy=m.NATIVE_POLICY,
            identity={key:'a'*64 for key in m.IDENTITY_FIELDS},
            checks={key:dict(error=0.0,evidence_sha256='b'*64) for key in m.GATE_LIMITS},
            phase_alignment=False,state_fit=False,retry=False,mandatory_STOP=True)

    def check(self,r):return m.validate_synthetic_result(r,m.SyntheticAuthority())

    def test_complete_synthetic_receipt_cannot_become_science_pass(self):
        r=self.check(self.record())
        self.assertFalse(r['science_GO']);self.assertFalse(r['real_correctness_accepted'])

    def test_each_gate_is_required_and_strict(self):
        for key,limit in m.GATE_LIMITS.items():
            r=self.record();r['checks'].pop(key)
            with self.assertRaises(m.Stop):self.check(r)
            r=self.record();r['checks'][key]['error']=limit
            with self.assertRaises(m.Stop):self.check(r)

    def test_identity_and_phase_flags_fail_closed(self):
        for key in m.IDENTITY_FIELDS:
            r=self.record();r['identity'][key]=None
            with self.assertRaises(m.Stop):self.check(r)
        for key in ['phase_alignment','state_fit','retry']:
            r=self.record();r[key]=True
            with self.assertRaises(m.Stop):self.check(r)

    def test_real_result_blocked_before_gate_inspection(self):
        r=self.record();r['domain']='SAVED_SCIENCE';r['checks']=None
        with self.assertRaisesRegex(m.Stop,'SAVED_SCIENCE_VALUES'):self.check(r)

    def test_nonfinite_and_unrepresentable_gate_values_stop(self):
        for value in [math.nan, math.inf, True, 10**1000]:
            r=self.record();r['checks']['native_low']['error']=value
            with self.assertRaises(m.Stop):self.check(r)


if __name__ == '__main__':
    unittest.main(verbosity=2)
