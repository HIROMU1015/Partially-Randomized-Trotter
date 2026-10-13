"""Artificial saved-cost fixtures only. No registered data or Hamiltonian imports."""
from copy import deepcopy
from pathlib import Path
import json,math,subprocess,sys,tempfile,unittest,os
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from trottertracks.representation_exploration.c1_saved_oracle_spec_v1 import Spec,SCORES,LABELS,FIXED,ApprovalError,InputError,OUTPUT,SCOPE,LIMITS,PREPARATION,BASE_COMMIT,SOURCE_FILES,OLD_SOURCE_LOCK
from trottertracks.representation_exploration.c1_saved_oracle_schema_v1 import validate
from trottertracks.representation_exploration.c1_saved_oracle_diagnostic_v1 import diagnose
from trottertracks.representation_exploration.c1_saved_oracle_checker_v1 import verify_result
from trottertracks.representation_exploration.c1_saved_oracle_execution_v1 import approval,authorize,safe_write,claim_audit

FIXTURE=Spec(('F_COST',),(3,),('00',),(1e-5,),2,('S0_m1',),('M01_1_4','M02_1_4'),('S4_0',),('C2_0',),True)

def fixture(costs=None,selected='S0_m1',eligible=True):
    costs=costs or {'S0_m1':100.,'M01_1_4':90.,'M02_1_4':120.,'S4_0':110.,'C2_0':115.}
    rows=[]
    for h in range(2):
        for method in FIXTURE.methods:
            rows.append({'method':method,'h_index':h,'eligible':eligible,'g_per_time':None,
                         'conditional_quantum_cost':costs[method], 'certified_PF_bound':None})
    front=lambda method:next((deepcopy(r) for r in rows if r['method']==method and r['eligible']),None)
    fr={}
    for score in SCORES:
        for label in LABELS:fr[score+':'+label]=front(selected if label in ('C1','strong_known_composition') else
                                                   'S0_m1' if label=='best_single' else 'M01_1_4')
    for label,method in [('all_single_subdivision','S0_m1'),('S4','S4_0'),('C2','C2_0')]:
        fr['oracle_diagnostic:'+label]=front(method)
    c={'input_id':'F_COST','price':3,'state':'00','epsilon':1e-5,'PF_budget':5e-6,'statistical_budget':5e-6,
       'rows':rows,'frontiers':fr,'coefficient_only_precision_choices':{l:{'method':selected,'h_index':None} for l in LABELS},
       'classifications':[],'full_ground_energy_cost':'UNKNOWN','novelty_adopted':False}
    seal={'F_COST':{'3':{'states':{'00':{'choices':{s:{l:selected if l in ('C1','strong_known_composition') else
                'S0_m1' if l=='best_single' else 'M01_1_4' for l in LABELS} for s in SCORES},
                'coefficient_only_precision_choices':{},'strong_known_composition_equals_C1_by_construction':True}}}}}
    return [c],seal

class Preparation(unittest.TestCase):
    def checked(self,*args,**kwargs):
        d,s=fixture(*args,**kwargs);r=diagnose(d,s,FIXTURE);verify_result(d,s,r,FIXTURE);return r['contexts'][0]
    def test_oracle_advantage_selector_miss_and_known_overlap(self):
        r=self.checked();self.assertEqual(r['C_M']['cost'],90.);self.assertEqual(r['C_K']['cost'],90.)
        self.assertIn('B_ORACLE_ADVANTAGE_SELECTOR_MISS',r['classifications'])
        self.assertIn('KNOWN_COMPOSITION_EQUIVALENT_ORACLE_ADVANTAGE',r['classifications'])
        self.assertEqual(r['selectors']['empirical']['method_regret'],'1/9')
    def test_no_oracle_advantage(self):
        r=self.checked({'S0_m1':100.,'M01_1_4':96.,'M02_1_4':120.,'S4_0':110.,'C2_0':115.})
        self.assertEqual(r['classifications'],['A_NO_ORACLE_5PCT_ADVANTAGE'])
    def test_selected_macro_success_has_no_miss(self):
        r=self.checked(selected='M01_1_4');self.assertEqual(r['selectors']['joint']['method_regret'],'0')
        self.assertNotIn('B_ORACLE_ADVANTAGE_SELECTOR_MISS',r['classifications'])
    def test_strong_S4_baseline_includes_overhead_already_saved(self):
        r=self.checked({'S0_m1':100.,'M01_1_4':90.,'M02_1_4':120.,'S4_0':80.,'C2_0':115.})
        self.assertEqual(r['C_B']['method'],'S4_0');self.assertEqual(r['macro_ratio']['gate'],'NO_ADVANTAGE')
    def test_strong_C2_baseline(self):
        r=self.checked({'S0_m1':100.,'M01_1_4':90.,'M02_1_4':120.,'S4_0':110.,'C2_0':70.})
        self.assertEqual(r['C_B']['method'],'C2_0')
    def test_same_catalog_denominator_is_not_strong_baseline(self):
        r=self.checked({'S0_m1':100.,'M01_1_4':90.,'M02_1_4':120.,'S4_0':80.,'C2_0':115.})
        self.assertEqual(r['selectors']['box']['method_regret'],'1/9')
    def test_stable_tie_preserves_all_equal_pointers(self):
        r=self.checked({'S0_m1':100.,'M01_1_4':90.,'M02_1_4':90.,'S4_0':110.,'C2_0':115.})
        self.assertEqual(r['C_M']['row_index'],1);self.assertEqual(r['C_M']['tied_row_indices'],[1,2,6,7])
    def test_gate_boundary_is_undetermined(self):
        r=self.checked({'S0_m1':100.,'M01_1_4':95.,'M02_1_4':120.,'S4_0':110.,'C2_0':115.})
        self.assertEqual(r['macro_ratio']['exact_ratio'],'19/20');self.assertEqual(r['classifications'],['UNDETERMINED'])
    def test_nearby_ulp_boundary_is_undetermined(self):
        r=self.checked({'S0_m1':100.,'M01_1_4':math.nextafter(95.,-math.inf),'M02_1_4':120.,'S4_0':110.,'C2_0':115.})
        self.assertEqual(r['macro_ratio']['reason'],'ONE_ULP_GATE_BOUNDARY')
    def test_no_eligible_candidates(self):
        r=self.checked(eligible=False);self.assertIsNone(r['C_M']);self.assertIsNone(r['selectors']['joint']['method_regret'])
    def test_ineligible_null_cost_is_retained(self):
        d,s=fixture(eligible=False)
        for row in d[0]['rows']:row['conditional_quantum_cost']=None
        r=diagnose(d,s,FIXTURE);verify_result(d,s,r,FIXTURE);self.assertIsNone(r['contexts'][0]['C_K'])
    def test_eligible_null_cost_rejected(self):
        d,s=fixture();d[0]['rows'][0]['conditional_quantum_cost']=None
        with self.assertRaises(InputError):validate(d,s,FIXTURE)
    def test_nonfinite_zero_negative_and_bool_cost_rejected(self):
        for x in (math.nan,math.inf,-math.inf,0.,-1.,True):
            d,s=fixture();d[0]['rows'][0]['conditional_quantum_cost']=x
            with self.subTest(x=x),self.assertRaises(InputError):validate(d,s,FIXTURE)
    def test_missing_row_field_rejected(self):
        d,s=fixture();del d[0]['rows'][0]['eligible']
        with self.assertRaises(InputError):validate(d,s,FIXTURE)
    def test_missing_row_or_context_rejected(self):
        d,s=fixture();d[0]['rows'].pop()
        with self.assertRaises(InputError):validate(d,s,FIXTURE)
        with self.assertRaises(InputError):validate([],s,FIXTURE)
    def test_duplicate_method_h_rejected(self):
        d,s=fixture();d[0]['rows'][1]=deepcopy(d[0]['rows'][0])
        with self.assertRaises(InputError):validate(d,s,FIXTURE)
    def test_unknown_method_or_h_rejected(self):
        for k,v in [('method','NEW_KERNEL'),('h_index',2),('h_index',True)]:
            d,s=fixture();d[0]['rows'][0][k]=v
            with self.subTest(k=k,v=v),self.assertRaises(InputError):validate(d,s,FIXTURE)
    def test_wrong_price_target_state_rejected(self):
        for k,v in [('price',6),('epsilon',1e-6),('state','11')]:
            d,s=fixture();d[0][k]=v
            with self.subTest(k=k),self.assertRaises(InputError):validate(d,s,FIXTURE)
    def test_known_control_mismatch_rejected(self):
        d,s=fixture();s['F_COST']['3']['states']['00']['choices']['empirical']['strong_known_composition']='M01_1_4'
        with self.assertRaises(InputError):validate(d,s,FIXTURE)
    def test_saved_selector_frontier_mismatch_rejected(self):
        d,s=fixture();d[0]['frontiers']['empirical:C1']=deepcopy(d[0]['rows'][1])
        with self.assertRaises(InputError):validate(d,s,FIXTURE)
    def test_independent_checker_rejects_forged_ratio_pointer_and_tags(self):
        for kind in ('ratio','pointer','tags'):
            d,s=fixture();r=diagnose(d,s,FIXTURE)
            if kind=='ratio':r['contexts'][0]['macro_ratio']['exact_ratio']='0'
            elif kind=='pointer':r['contexts'][0]['C_M']['row_index']=2
            else:r['contexts'][0]['classifications']=[]
            with self.subTest(kind=kind),self.assertRaises(InputError):verify_result(d,s,r,FIXTURE)
    def test_checker_rejects_nonminimal_saved_selected_h(self):
        d,s=fixture();d[0]['rows'][5]['conditional_quantum_cost']=80.
        r=diagnose(d,s,FIXTURE)
        with self.assertRaises(InputError):verify_result(d,s,r,FIXTURE)
    def test_input_readonly(self):
        d,s=fixture();before=deepcopy((d,s));r=diagnose(d,s,FIXTURE);verify_result(d,s,r,FIXTURE)
        self.assertEqual((d,s),before)
    def test_oracle_advantage_missing_selected_point_keeps_partial_undetermined(self):
        d,s=fixture()
        for row in d[0]['rows']:
            if row['method']=='S0_m1':row['eligible']=False
        for key,point in d[0]['frontiers'].items():
            if point is not None and point['method']=='S0_m1':d[0]['frontiers'][key]=None
        r=diagnose(d,s,FIXTURE);verify_result(d,s,r,FIXTURE);c=r['contexts'][0]
        self.assertIn('B_ORACLE_ADVANTAGE_SELECTOR_MISS',c['classifications'])
        self.assertIn('UNDETERMINED',c['classifications']);self.assertTrue(c['unresolved_parts'])
    def test_registered_evaluation_without_receipt_denied_before_schema(self):
        with self.assertRaises(ApprovalError):diagnose(None,None,FIXED)
        with self.assertRaises(ApprovalError):verify_result(None,None,None,FIXED)
    def test_forged_status_only_receipt_denied_before_data(self):
        receipt={'scope':SCOPE,'attempt_consumed':True,'authorization':{'status':'APPROVED'}}
        with self.assertRaises(ApprovalError):diagnose(None,None,FIXED,receipt)
        with self.assertRaises(ApprovalError):verify_result(None,None,None,FIXED,receipt)
    def test_missing_draft_and_old_pilot_authorization_rejected_before_inputs(self):
        with patch('trottertracks.representation_exploration.c1_saved_oracle_execution_v1.preflight',side_effect=AssertionError('INPUTS_MUST_NOT_BE_READ')):
            with self.assertRaises(ApprovalError):authorize(Path('/tmp'),None)
            for a in ({'status':'APPROVAL_TARGET_NOT_AUTHORIZATION'}, {'type':'C1_PILOT_EXECUTION_APPROVAL','status':'APPROVED'}):
                with tempfile.TemporaryDirectory() as t:
                    p=Path(t)/'approval.json';p.write_text(json.dumps(a))
                    with self.assertRaises(ApprovalError):authorize(Path(t),p)
                    self.assertFalse((Path(t)/OUTPUT).exists())
    def test_one_shot_receipt_and_second_attempt_rejected(self):
        a={'type':'C1_SAVED_ORACLE_POSTHOC_APPROVAL_V1','status':'APPROVED','scope':SCOPE,
           'limits':LIMITS,'explicit_user_approval':'SYNTHETIC_TEST_ONLY','identities':{}}
        with tempfile.TemporaryDirectory() as t,patch.dict(os.environ),patch(
            'trottertracks.representation_exploration.c1_saved_oracle_execution_v1.preflight',return_value={'identities':{}}),patch(
            'trottertracks.representation_exploration.c1_saved_oracle_execution_v1.resource.setrlimit'):
            p=Path(t)/'artificial_approval.json';p.write_text(json.dumps(a))
            out,r=authorize(Path(t),p);self.assertTrue(r['attempt_consumed'])
            self.assertTrue((out/'launch_receipt.json').exists())
            with self.assertRaises(FileExistsError):authorize(Path(t),p)
    def test_approval_identity_and_limits_mismatch(self):
        a={'type':'C1_SAVED_ORACLE_POSTHOC_APPROVAL_V1','status':'APPROVED','scope':SCOPE,
           'limits':dict(LIMITS),'explicit_user_approval':'SYNTHETIC_TEST_ONLY','identities':{}}
        with self.assertRaises(ApprovalError):approval(a,{'different':'identity'})
        a['limits']['retry']=1
        with self.assertRaises(ApprovalError):approval(a)
    def test_source_hash_failure_precedes_registered_schema(self):
        from trottertracks.representation_exploration.c1_saved_oracle_execution_v1 import preflight
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);folder=root/PREPARATION;folder.mkdir(parents=True)
            plan={'scope':SCOPE,'limits':LIMITS,'output':OUTPUT,'base_commit':BASE_COMMIT,'source_commit':'a'*40}
            (folder/'execution_plan.json').write_text(json.dumps(plan))
            (folder/'source_manifest.json').write_text(json.dumps({'files':{p:'0'*64 for p in SOURCE_FILES}}))
            legacy=root/OLD_SOURCE_LOCK;legacy.parent.mkdir(parents=True,exist_ok=True)
            legacy.write_text(json.dumps({'files':{f'legacy{i}':'0'*64 for i in range(20)}}))
            from trottertracks.representation_exploration.c1_saved_oracle_spec_v1 import COMPARISONS,SEAL,SUMMARY,PRIOR_STOP
            names={f'legacy{i}' for i in range(20)}|{COMPARISONS,SEAL,SUMMARY,PRIOR_STOP,OLD_SOURCE_LOCK}
            (folder/'input_manifest.json').write_text(json.dumps({'base_commit':BASE_COMMIT,'files':{p:{} for p in names}}))
            (folder/'runtime_lock.json').write_text('{}')
            first=root/SOURCE_FILES[0];first.parent.mkdir(parents=True,exist_ok=True);first.write_bytes(b'tampered')
            with patch('trottertracks.representation_exploration.c1_saved_oracle_execution_v1.subprocess.run'),patch(
                'trottertracks.representation_exploration.c1_saved_oracle_execution_v1.subprocess.check_output',return_value=
                    'git@github.com:HIROMU1015/Partially-Randomized-Trotter.git\n'),patch(
                'trottertracks.representation_exploration.c1_saved_oracle_execution_v1.validate',side_effect=AssertionError('NO_REGISTERED_SCHEMA')):
                with self.assertRaisesRegex(InputError,'SOURCE_HASH'):preflight(root)
    def test_independent_checker_does_not_import_evaluator(self):
        p=Path(__file__).resolve().parents[1]/'src/trottertracks/representation_exploration/c1_saved_oracle_checker_v1.py'
        self.assertNotIn('c1_saved_oracle_diagnostic_v1',p.read_text())
    def test_output_cap_keeps_failure_record_space(self):
        with tempfile.TemporaryDirectory() as t:
            out=Path(t)
            with self.assertRaises(InputError):safe_write(out,'diagnostic.json',{'large':'x'*(2*2**20)})
            safe_write(out,'failure.json',{'mandatory_stop':True})
            self.assertTrue((out/'failure.json').exists())
    def test_independent_audit_consumed_once(self):
        with tempfile.TemporaryDirectory() as t:
            out=Path(t);claim_audit(out)
            with self.assertRaises(FileExistsError):claim_audit(out)
            safe_write(out,'independent_audit_failure.json',{'mandatory_stop':True,'retry':0})
            self.assertTrue((out/'audit_attempt_consumed.json').exists())
    def test_fixed_scope_cardinality(self):
        self.assertEqual(len(FIXED.single),12);self.assertEqual(len(FIXED.macro),24)
        self.assertEqual(len(FIXED.methods)*6*4*2*2*2,8064)

if __name__=='__main__':unittest.main()
