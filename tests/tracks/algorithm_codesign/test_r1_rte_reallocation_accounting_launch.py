"""Pure accounting and temporary launch fixtures, no R1 acquisition/synthesis."""
from fractions import Fraction as F
import hashlib,importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from trottertracks.algorithm_codesign.rte_reallocation.model import Event,events
from trottertracks.algorithm_codesign.rte_reallocation.native import Gate,Angle,planned_angles
from trottertracks.algorithm_codesign.rte_reallocation.numeric import configure,validate_saved,synthesis_key,enclosure
from trottertracks.algorithm_codesign.rte_reallocation.accounting import canonical_profile,confidence_budget,interval_ratio
from trottertracks.algorithm_codesign.rte_reallocation.launch import validate_binding,consume_marker,BudgetGuard

ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('r1_runner_under_test',ROOT/'scripts/tracks/algorithm_codesign/run_r1_rte_reallocation.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


class Accounting(unittest.TestCase):
    def setUp(self):configure(80)
    def test_canonical_probability_and_moment_are_charged_consistently(self):
        es=[Event('test','0',F(1),F(0),F(1)),Event('test','1',F(2),F(0),F(1))]
        p=canonical_profile(es,[[],[Gate('X',(0,))]],{},'1e-6',True)
        self.assertEqual(p['implemented_B'],'3');self.assertEqual(p['implemented_weight_second_moment'],'9')
        self.assertEqual([F(e['canonical_probability_exact']) for e in p['events']],[F(1,3),F(2,3)])
        self.assertEqual(F(p['E_native_cost']['1Q']),F(2,3));self.assertEqual(p['workspace_qubits_beyond_2_system'],1)
    def test_rounded_coefficients_still_factor_as_order_probability_times_IID_law(self):
        es=events('distinct_basis',F(1,3),1,'A')
        profile=canonical_profile(es,[[] for e in es],{},'1e-6',True)
        groups={}
        for row,e in zip(profile['events'],es):
            factor=F(row['canonical_probability_exact'])/e.label_probability
            previous=groups.setdefault((e.a,e.b),factor)
            self.assertEqual(factor,previous)
    def profile(self,bias='0'):
        return {'implemented_B':'2','coefficient_and_strict_synthesis_bias_upper':bias,'coefficient_L1_bias_upper':'0',
                'E_native_cost':{'T':'3','CX':'5','1Q':'7'}}
    def task(self):return {'epsilon_axis':'1/100','alpha_axis':'1/100','coefficient_bias_cap':'1e-25','shot_cap_per_axis':10**9}
    def test_common_confidence_budgets_include_both_axes_and_readout(self):
        r=confidence_budget(self.profile(),self.task());n=r['sufficient_shots_per_axis']
        self.assertEqual(F(r['G_T']),6*n);self.assertEqual(F(r['G_CX']),10*n)
        self.assertEqual(F(r['G_1Q_with_Hadamard_preparation_readout']),19*n);self.assertFalse(r['exact_signal_used'])
    def test_bias_consumes_accuracy_instead_of_reducing_shots(self):
        a=confidence_budget(self.profile(),self.task());b=confidence_budget(self.profile('1/200'),self.task())
        self.assertGreater(b['sufficient_shots_per_axis'],a['sufficient_shots_per_axis'])
    def test_bias_and_shot_caps_are_separate_infeasibility_records(self):
        self.assertEqual(confidence_budget(self.profile('1/100'),self.task())['status'],'INFEASIBLE_BIAS')
        task=dict(self.task(),shot_cap_per_axis=1)
        self.assertEqual(confidence_budget(self.profile(),task)['status'],'INFEASIBLE_SHOT_CAP')
    def test_zero_cost_baseline_and_interval_overlap_are_not_wins(self):
        self.assertIsNone(interval_ratio({'lo':'0','hi':'0'},{'lo':'0','hi':'0'})['ratio'])
        self.assertEqual(interval_ratio({'lo':'9/10','hi':'11/10'},{'lo':'1','hi':'1'})['status'],'OVERLAP_OR_EQUAL')
    def test_joint_synthesis_operator_error_is_doubled_for_coherent_measurement(self):
        cost={'T':0,'CX':0,'1Q':0,'strict_event_error_upper':F(1,100),'IR_sha256':'fixture','IR_gate_count':0}
        with patch('trottertracks.algorithm_codesign.rte_reallocation.accounting.native_cost',return_value=cost):
            p=canonical_profile([Event('test','I',F(1),F(0),F(1))],[[]],{},'1e-6',True)
        self.assertEqual(F(p['coefficient_and_joint_operator_error_upper']),F(1,100))
        self.assertEqual(F(p['coefficient_and_strict_synthesis_bias_upper']),F(1,50))
    def test_saved_sequence_count_hash_and_global_phase_metadata_are_verified(self):
        a=Angle('pi',0);row={'key':synthesis_key(a,'1e-6'),'angle_key':a.key,'epsilon':'1e-6',
          'sequence':'','sequence_sha256':hashlib.sha256(b'').hexdigest(),'T_count':0,'Tdagger_count':0,
          'one_qubit_count':0,'global_W_count':0,'strict_operator_error_upper':'0','error_pass':True}
        validate_saved(row,a,'1e-6')
        for key,value in [('T_count',1),('global_W_count',1),('sequence_sha256','bad'),('error_pass',False),('strict_operator_error_upper','1')]:
            with self.assertRaises(PermissionError):validate_saved(dict(row,**{key:value}),a,'1e-6')
    def test_frozen_registry_contains_no_targets_added_by_result(self):
        c=json.loads((ROOT/'artifacts/track_b_rte_reallocation_r1_source/2026-10-06/contract_v2.json').read_text())
        self.assertEqual(len(planned_angles(c['domain']['x'])),42);self.assertEqual(len(runner.requests(c)),126)
        runner.verify_plan(ROOT,c)
    def test_nonfinite_interval_cannot_be_serialized_as_zero_error(self):
        import mpmath as mp
        with self.assertRaises(ArithmeticError):enclosure(mp.iv.mpf([mp.inf,mp.inf]))


class Launch(unittest.TestCase):
    def setUp(self):
        self.s,self.a,self.hash='1'*40,'2'*40,'3'*64
        self.auth={'status':'APPROVED_FOR_ONE_R1_RUN','source_commit':self.s,'science_execution_authorized':True,
                   'runs':1,'retries':0,'mandatory_STOP':True,'contract_sha256':self.hash,
                   'explicit_execution_instruction':'固定R1 sourceで一回だけ実行し必ずSTOP'}
    def verify(self,**kw):
        params=dict(auth=self.auth,contract_hash=self.hash,head=self.a,parents=[self.s],
                    changed=['authorization.json'],dirty=False,allowed={'authorization.json','receipt.md'})
        params.update(kw);validate_binding(**params)
    def test_direct_authorization_only_child_is_required(self):
        self.verify()
        for kw in [dict(head=self.s),dict(parents=[self.a]),dict(parents=[self.s,self.a]),
                   dict(changed=['authorization.json','source.py']),dict(dirty=True)]:
            with self.assertRaises(PermissionError):self.verify(**kw)
    def test_pending_wrong_stage_retry_and_boolean_run_counts_are_refused(self):
        for field,value in [('source_commit',None),('status','APPROVED_FOR_ONE_SP1_RUN'),('retries',1),('runs',True),
                            ('mandatory_STOP',False),('science_execution_authorized',False),('explicit_execution_instruction',None)]:
            with self.assertRaises(PermissionError):self.verify(auth=dict(self.auth,**{field:value}))
    def test_contract_mismatch_is_refused(self):
        with self.assertRaises(PermissionError):self.verify(contract_hash='4'*64)
    def test_pending_runner_refuses_before_tool_marker_keys_circuits_or_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c=root/'contract.json';c.write_text(json.dumps({'authorization_path':'auth.json'}))
            (root/'auth.json').write_text(json.dumps({'source_commit':None,'science_execution_authorized':False}))
            with patch.object(runner,'ROOT',root),patch.object(runner,'CONTRACT',c), \
                 patch.object(runner,'verify_runtime') as tool,patch.object(runner,'verify_plan') as plan, \
                 patch.object(runner,'consume_marker') as marker,patch.object(runner,'synthesize') as synth, \
                 patch.object(runner,'build_rows') as rows:
                with self.assertRaises(PermissionError):runner.run()
                for mock in (tool,plan,marker,synth,rows):mock.assert_not_called()
    def test_one_shot_marker_cannot_be_replaced_or_used_to_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=consume_marker(Path(tmp)/'out',{'runs':1,'retries':0})
            with self.assertRaises(FileExistsError):consume_marker(p.parent,{'runs':1})
            self.assertEqual(json.loads(p.read_text())['retries'],0)
    def test_existing_evidence_without_marker_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);(d/'result.json').write_text('old evidence')
            with self.assertRaises(FileExistsError):consume_marker(d,{'runs':1})
            self.assertEqual((d/'result.json').read_text(),'old evidence')
    def test_per_key_timeout_has_no_retry_loop(self):
        guard=BudgetGuard({'wall_seconds':100,'cpu_seconds':100,'RSS_MiB':512,
                           'per_key_wall_seconds':1,'per_key_cpu_seconds':1})
        guard.begin_key()
        with patch('trottertracks.algorithm_codesign.rte_reallocation.launch.time.monotonic',return_value=guard.key_start+2):
            with self.assertRaises(TimeoutError):guard.check()


if __name__=='__main__':unittest.main()
