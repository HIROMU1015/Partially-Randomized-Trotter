"""Synthetic contracts and fake provider objects only; no molecular/Qiskit calls."""
import csv
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import pf_direct_compile_contract as c
import pf_direct_compile_adapter as a


def proposals(system='H6_DF19', candidate='B0_p11_q4_2nd', prefix='11', q='4', PF='2nd'):
    return [dict(task_id=f'{system}:{candidate}:{axis}', system=system, candidate_id=candidate,
                 axis=axis, prefix=prefix, q=q, PF=PF, execution_authorized='False') for axis in c.AXES]


def raw_csv(rows):
    handle=io.StringIO(); writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return handle.getvalue().encode()


class FakeBackend:
    def __init__(self, calls, *, control_pass=True, bad_metric=False, fail_axis=None):
        self.calls=calls;self.control_pass=control_pass;self.bad_metric=bad_metric;self.fail_axis=fail_axis
    def restore(self, task):self.calls.append('restore')
    def build(self, task):self.calls.append('build');return object()
    def control(self, evolution):
        self.calls.append('control');return dict(status='PASS' if self.control_pass else 'FAIL', probes=4)
    def compile_axis(self, evolution, axis, journal):
        journal.take('wrapper_build');journal.take('wrapper_compile');self.calls.append(axis)
        if self.fail_axis==axis:raise RuntimeError('synthetic compiler failure')
        return dict(axis=axis,RZ=True if self.bad_metric else 123,CX=45,RZ_depth=80,CX_depth=30,
                    depth=150,size=300,posttranspile_gate_counts={'rz':123,'cx':45,'sx':132},
                    compiler_hash=c.COMPILER,fingerprint_before='fixture_fp_'+axis,
                    fingerprint_after='fixture_fp_'+axis,compiled_input_fingerprint='fixture_fp_'+axis,
                    compile_seconds=0.0,native_exit_code=0,reaped=True,
                    saved_model_prediction='arbitrarily inaccurate fixture; MUST NOT evaluate')


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.rows=proposals()
        self.raw=raw_csv(self.rows);self.sha=hashlib.sha256(self.raw).hexdigest()
        self.draft=c.draft(self.rows,self.sha,'f'*64)
    def tearDown(self):self.tmp.cleanup()
    def sealed_fixture(self):
        # This is fabricated TEST metadata, never a real scientific authorization.
        contract=deepcopy(self.draft)
        f=self.root/'bound-fixture.txt';f.write_text('synthetic fixture')
        item=dict(path=str(f),sha256=c.sha(f))
        contract.update(acquisition_scope_approved=True,execution_authorized=True,grant_issued=True,
                        SOURCE='a'*40, selected_task_ids=[r['task_id'] for r in self.rows],
                        call_caps=c.call_caps(self.rows),loader_binding={'sealed':True,'kind':'FAKE_TEST_ONLY'},
                        output_root=str(self.root),run_id='synthetic-only')
        for key in ('source_closure','input_files','provider_files','runtime_files','profile_files'):
            contract[key]=[dict(item)]
        approval=dict(action='PF_DIRECT_COMPILE_SCIENTIFIC_EXECUTION',approved_by_user=True,
                      contract_digest=c.digest(contract),fixture_only=True)
        return contract,approval
    def live(self,contract):
        return dict(status='PASS',contract_digest=c.digest(contract),SOURCE=contract['SOURCE'],run_id=contract['run_id'])
    def run_fake(self,contract,approval,backend=None,live=None):
        calls=[]
        def factory(*args):calls.append('factory');return backend or FakeBackend(calls)
        result=a.acquire_pair(contract,approval,self.rows,('H6_DF19','B0_p11_q4_2nd'),
                              live_gate=live or self.live,backend_factory=factory)
        return result,calls
    def test_proposal_hash_required(self):
        with self.assertRaisesRegex(ValueError,'PROPOSAL_BYTES'):c.parse_proposals(self.raw,'0'*64)
    def test_proposal_parameter_identity(self):
        rows=deepcopy(self.rows);rows[0]['q']='16';raw=raw_csv(rows)
        with self.assertRaisesRegex(ValueError,'PF_PARAMETERS'):c.parse_proposals(raw,hashlib.sha256(raw).hexdigest())
    def test_duplicate_proposal_rejected(self):
        rows=self.rows+[self.rows[0]];raw=raw_csv(rows)
        with self.assertRaisesRegex(ValueError,'DUPLICATION'):c.parse_proposals(raw,hashlib.sha256(raw).hexdigest())
    def test_no_q_cutoff(self):
        rows=proposals('H4_DF10','B0_p2_q16_4th','2','16','4th');raw=raw_csv(rows)
        self.assertEqual(len(c.parse_proposals(raw,hashlib.sha256(raw).hexdigest())),2)
    def test_df_series_key_separation(self):
        other=proposals('H4_LEGACY_DF12')
        self.assertEqual(len(c.select_pairs(self.rows+other,[r['task_id'] for r in self.rows+other])),4)
    def test_half_pair_rejected(self):
        with self.assertRaisesRegex(ValueError,'PAIRED'):c.select_pairs(self.rows,[self.rows[0]['task_id']])
    def test_conditional_reacquisition_not_auto_added(self):
        with self.assertRaisesRegex(ValueError,'NOT_IN_PROPOSAL'):c.select_pairs(self.rows,['H6_DF19:B0I_p7_q1_2nd:cosine'])
    def test_suitability_is_not_execution_approval(self):
        self.assertTrue(self.draft['plan_suitable_for_research'])
        with self.assertRaisesRegex(ValueError,'NOT_AUTHORIZED'):c.authorize_metadata(self.draft,{},self.rows)
    def test_rejection_precedes_factory_output_and_input_read(self):
        def forbidden(*args):raise AssertionError('must not call')
        contract=deepcopy(self.draft);contract['input_files']=[{'path':'/does/not/exist'}]
        with self.assertRaisesRegex(ValueError,'NOT_AUTHORIZED'):
            a.acquire_pair(contract,{},self.rows,('H6_DF19','B0_p11_q4_2nd'),live_gate=forbidden,backend_factory=forbidden)
        self.assertEqual(list(self.root.iterdir()),[])
    def test_publication_approval_not_scientific(self):
        contract,approval=self.sealed_fixture();approval['action']='PUBLISH_FILES'
        with self.assertRaisesRegex(ValueError,'EXPLICIT_SCIENTIFIC'):self.run_fake(contract,approval)
    def test_approval_scope_digest(self):
        contract,approval=self.sealed_fixture();contract['run_id']='altered'
        with self.assertRaisesRegex(ValueError,'CONTRACT_MISMATCH'):self.run_fake(contract,approval)
    def test_unsealed_loader_rejected(self):
        contract,approval=self.sealed_fixture();contract['loader_binding']=None;approval['contract_digest']=c.digest(contract)
        with self.assertRaisesRegex(ValueError,'LOADER_NOT_BOUND'):self.run_fake(contract,approval)
    def test_missing_profile_rejected(self):
        contract,approval=self.sealed_fixture();contract['profile_files']=[];approval['contract_digest']=c.digest(contract)
        with self.assertRaisesRegex(ValueError,'UNSEALED_profile'):self.run_fake(contract,approval)
    def test_approved_scope_cannot_change_candidate_parameters(self):
        contract,approval=self.sealed_fixture();rows=deepcopy(self.rows);rows[0]['q']='16'
        with self.assertRaisesRegex(ValueError,'PROPOSAL_RECORDS_CHANGED'):
            a.acquire_pair(contract,approval,rows,('H6_DF19','B0_p11_q4_2nd'),
                live_gate=lambda value:self.fail('gate must not run'),backend_factory=lambda *args:self.fail('factory must not run'))
    def test_duplicate_json_key_rejected(self):
        p=self.root/'invalid.json';p.write_text('{"approved":false,"approved":true}')
        with self.assertRaisesRegex(ValueError,'DUPLICATE_JSON_KEY'):c.read_json(p)
    def test_stdlib_only_import_path(self):
        code="""import builtins,sys
sys.path.insert(0,sys.argv[1])
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in {'qiskit','numpy','scipy','pyscf','trotterlib','trottertracks'}:
        raise AssertionError('scientific import: '+name)
    return original(name,*args,**kwargs)
builtins.__import__=guarded
import pf_direct_compile_contract,pf_direct_compile_adapter,check_pf_direct_compile_preparation
"""
        proc=subprocess.run([sys.executable,'-c',code,str(ROOT/'scripts')],capture_output=True,text=True)
        self.assertEqual(proc.returncode,0,proc.stderr)
    def test_call_scope_cannot_include_sampling(self):
        contract,approval=self.sealed_fixture();contract['call_caps']['sampling']=1;approval['contract_digest']=c.digest(contract)
        with self.assertRaisesRegex(ValueError,'EXACT_CALL_SCOPE'):self.run_fake(contract,approval)
    def test_changed_bound_file_rejected(self):
        contract,approval=self.sealed_fixture();(self.root/'bound-fixture.txt').write_text('changed')
        with self.assertRaisesRegex(ValueError,'BOUND_FILE_CHANGED'):self.run_fake(contract,approval)
        self.assertFalse((self.root/'H6_DF19__B0_p11_q4_2nd').exists())
    def test_bad_live_gate_rejected_before_output(self):
        contract,approval=self.sealed_fixture()
        with self.assertRaisesRegex(ValueError,'FRESH_GATE'):self.run_fake(contract,approval,live=lambda x:{'status':'PASS'})
        self.assertFalse((self.root/'H6_DF19__B0_p11_q4_2nd').exists())
    def test_fake_pair_complete_without_model_evaluation(self):
        contract,approval=self.sealed_fixture();result,calls=self.run_fake(contract,approval)
        self.assertEqual(result['status'],'COMPLETE_MANDATORY_STOP')
        self.assertEqual(calls,['factory','restore','build','control','cosine','sine'])
        self.assertTrue(all(r['model_evaluated'] is False for r in result['rows']))
        self.assertEqual(result['calls']['wrapper_compile'],2)
    def test_one_shot_no_output_reuse(self):
        contract,approval=self.sealed_fixture();self.run_fake(contract,approval)
        with self.assertRaisesRegex(ValueError,'OUTPUT_ALREADY_USED'):self.run_fake(contract,approval)
    def test_failed_control_stops_before_compile(self):
        contract,approval=self.sealed_fixture();calls=[]
        with self.assertRaisesRegex(ValueError,'CONTROL_FAILED'):self.run_fake(contract,approval,FakeBackend(calls,control_pass=False))
        self.assertEqual(calls,['restore','build','control'])
    def test_failed_cosine_preserves_reservation_no_sine(self):
        contract,approval=self.sealed_fixture();calls=[]
        with self.assertRaisesRegex(RuntimeError,'synthetic'):self.run_fake(contract,approval,FakeBackend(calls,fail_axis='cosine'))
        self.assertNotIn('sine',calls)
        stop=json.loads((self.root/'H6_DF19__B0_p11_q4_2nd/STOP.json').read_text())
        self.assertEqual(stop['calls']['wrapper_compile'],1);self.assertFalse(stop['reservations_refunded'])
    def test_boolean_metric_not_integer_measurement(self):
        contract,approval=self.sealed_fixture()
        with self.assertRaisesRegex(ValueError,'NONINTEGER'):self.run_fake(contract,approval,FakeBackend([],bad_metric=True))
    def test_stale_fingerprint_rejected(self):
        row=FakeBackend([]).compile_axis(None,'cosine',SimpleNamespace(take=lambda *args:None));row['fingerprint_after']='wrong'
        with self.assertRaisesRegex(ValueError,'FINGERPRINT'):a.validate_measurement(row,c.COMPILER,'cosine')
    def test_owned_result_not_assumed(self):
        row=FakeBackend([]).compile_axis(None,'cosine',SimpleNamespace(take=lambda *args:None));row.update(native_exit_code=None,reaped=False)
        with self.assertRaisesRegex(ValueError,'EXIT_OR_REAP'):a.validate_measurement(row,c.COMPILER,'cosine')
        witness=dict(task_id='fixture',result_digest=c.digest(row),exit_code=0,reaped=True,cleanup_errors=[])
        accepted=a.accept_owned_measurement(row,witness,'fixture');a.validate_measurement(accepted,c.COMPILER,'cosine')
        witness['result_digest']='changed'
        with self.assertRaisesRegex(ValueError,'RESULT_IDENTITY'):a.accept_owned_measurement(row,witness,'fixture')
    def test_dynamic_admission_not_fixed_worker_limit(self):
        tasks=[dict(id=i,memory_admission_bytes=1,disk_admission_bytes=1) for i in range(12)]
        self.assertEqual(len(c.admissible_pairs(tasks,permitted_cpus=list(range(12)),available_memory_bytes=20,available_disk_bytes=20)),12)
        self.assertEqual(len(c.admissible_pairs(tasks,permitted_cpus=list(range(12)),available_memory_bytes=3,available_disk_bytes=20)),3)
    def test_unknown_memory_estimate_not_fabricated(self):
        self.assertEqual(c.admissible_pairs([{'q':16}],permitted_cpus=[0],available_memory_bytes=100,available_disk_bytes=100),[])
    def native_fixture(self):
        log=[]
        def build(blocks,**kwargs):log.append(('build',blocks,kwargs));return 'fake_evolution'
        def wrapper(evolution,**kwargs):log.append(('wrapper',kwargs));return dict(axis=kwargs['axis'])
        def fingerprint(value):return 'fixture_'+value['axis']
        def transpile(value,compiler,**kwargs):
            log.append(('transpile',value['axis'],kwargs))
            return SimpleNamespace(rz_count=9,cx_count=2,rz_depth=7,cx_depth=2,total_depth=15,circuit_size=22,
                pretranspile_gate_counts=(('fixture',1),),posttranspile_gate_counts=(('rz',9),('cx',2),('sx',11)),
                compiler_settings_hash=c.COMPILER,actual_circuit_fingerprint=kwargs['actual_circuit_fingerprint'])
        context=dict(system='H6_DF19',candidate_id='B0_p11_q4_2nd',q=4,PF='2nd',prefix=11,
            system_qubits=12,DF_rank=19,identity_gate_passed=True,
            scalar_policy='original_plus_own_tail_once',control_policy='symmetric_directional',
            independent_control_oracle_bound=True,native_instruction_limit=None,
            deterministic_blocks=('saved_fake_blocks',),saved_scalar='fixture_scalar',
            provider=SimpleNamespace(build_native_hadamard_wrapper=wrapper,
                canonical_qiskit_circuit_fingerprint=fingerprint,transpile_and_measure_cost=transpile),
            native=SimpleNamespace(build_deterministic_native=build),compiler=object(),
            control_oracle=lambda ev:dict(status='PASS',probes=4))
        return context,log
    def test_native_adapter_uses_only_build_wrapper_and_transpile(self):
        context,log=self.native_fixture();kernel=a.NativeKernel(context);kernel.restore(self.rows[0])
        evolution=kernel.build(self.rows[0]);kernel.control(evolution)
        row=kernel.compile_axis(evolution,'cosine',SimpleNamespace(take=lambda *args:None))
        self.assertEqual([r[0] for r in log],['build','wrapper','transpile'])
        self.assertEqual(log[0][2]['q'],4);self.assertEqual(log[0][2]['formula'],'2nd')
        self.assertEqual(log[0][2]['max_instructions'],float('inf'))
        self.assertEqual(row['posttranspile_gate_counts']['rz'],9)
        self.assertIsNone(row['native_exit_code']);self.assertFalse(row['reaped'])
    def test_native_adapter_rejects_df10_df12_transfer(self):
        context,_=self.native_fixture();context['DF_rank']=12
        with self.assertRaisesRegex(ValueError,'SERIES_MISMATCH'):a.NativeKernel(context).restore(self.rows[0])
    def test_native_adapter_requires_independent_oracle(self):
        context,_=self.native_fixture();context['independent_control_oracle_bound']=False
        with self.assertRaisesRegex(ValueError,'ORACLE_MISSING'):a.NativeKernel(context).restore(self.rows[0])
    def test_native_adapter_rejects_scalar_policy_change(self):
        context,_=self.native_fixture();context['scalar_policy']='tail_from_another_prefix'
        with self.assertRaisesRegex(ValueError,'SCALAR_NOT_BOUND'):a.NativeKernel(context).restore(self.rows[0])
    def test_native_adapter_does_not_inherit_historical_instruction_cap(self):
        context,_=self.native_fixture();context['native_instruction_limit']=1000000
        with self.assertRaisesRegex(ValueError,'INSTRUCTION_CEILING'):a.NativeKernel(context).restore(self.rows[0])
    def test_cli_rejects_execute(self):
        proc=subprocess.run([sys.executable,str(ROOT/'scripts/check_pf_direct_compile_preparation.py'),'--execute'],capture_output=True,text=True)
        self.assertEqual(proc.returncode,2);self.assertIn('unrecognized arguments',proc.stderr)


if __name__=='__main__':unittest.main()
