"""All numerical fixtures in this module are invented; no saved science read."""
import copy
from fractions import Fraction as F
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/trotterlib'))
import mechanism_analysis as m
spec=importlib.util.spec_from_file_location('mechanism_launcher_test',ROOT/'scripts/run_mechanism_analysis.py')
cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)

def fixture(role='NEW_PR',system='H6'):
    cohort=('FIXED_ANCHOR_SELECTION_AND_ESTIMATION' if system=='H4' and role=='OLD_PR' else
            'CONFIRMATION_UNION_POST_SELECTION' if role=='OLD_PR' else
            'EXPLORATION_N8' if role=='NEW_PR' else 'INDEPENDENT_FIXED_CONFIRMATION_N48')
    n=32 if role=='OLD_PR' else 8 if role=='NEW_PR' else 48
    binding=dict(system=system,DF='ARTIFICIAL',compiler='ARTIFICIAL',hamiltonian='ARTIFICIAL',state='ARTIFICIAL')
    pairs=[]
    for i in range(n):
        pair=dict(replica=i,pair_id='artificial-'+str(i),candidate_id='PR',cohort=cohort,seed=i,event_digest='artificial-'+str(i))
        pair['axes']={a:dict(RZ=v,pair_id=pair['pair_id'],seed=i,event_digest=pair['event_digest']) for a,v in [('cosine',1),('sine',6)]};pairs.append(pair)
    return dict(artificial=True,contrast_id='ARTIFICIAL_1',system=system,H4_stage='H4_PRE_D_P_VALIDATION' if system=='H4' else 'NOT_APPLICABLE',
        binding=binding,PR_id='PR',PF_id='PF',PR_role=role,bias_mode='USUAL_SIGNED',factor='1',epsilon='0.05',cohorts_pooled=False,saved_SE='saved-text-not-recomputed',
        PR=dict(candidate_id='PR',binding=copy.deepcopy(binding),cohort=cohort,n=n,trajectories=pairs,shots=dict(cosine=2,sine=4),eligible=True,saved_G_exact='26',explanation=None),
        PF=dict(candidate_id='PF',binding=copy.deepcopy(binding),axis_costs=dict(cosine=2,sine=8),shots=dict(cosine=3,sine=1),eligible=True,saved_G_exact='14',cost_class='DIRECT_MEASURED',source_evidence='ARTIFICIAL',explanation=None))

def bundle(row):return dict(schema=m.SCHEMA,kind='ARTIFICIAL',target_ids=[row['contrast_id']],comparisons=[row])
def run(row):return m.analyze_contrast(row,m.artificial_permit())
def explain_fixture():
    row=fixture()
    for method in ['PR','PF']:
        side=row[method];h=float(row['epsilon'])/math.sqrt(2);B=1.0;alpha=.025
        bound=2*B**2/h**2*math.log(2/alpha);N=math.ceil(bound)
        side['shots']={a:N for a in m.AXES}
        side['explanation']=dict(B='1',log_B='0',signed_bias={a:'0' for a in m.AXES},allowance={a:'0' for a in m.AXES},headroom={a:repr(h) for a in m.AXES},alpha_axis=repr(alpha),log_shot_bound={a:repr(math.log(2)+2*math.log(B)-2*math.log(h)+math.log(math.log(2/alpha))) for a in m.AXES})
        side['saved_G_exact']=str(N*(7 if method=='PR' else 10))
    return row

class ExactCore(unittest.TestCase):
    def test_unequal_axes(self):self.assertEqual(run(fixture())['R_G'],'13/7')
    def test_effective_identity(self):
        r=run(fixture());self.assertEqual(F(r['R_G']),F(r['R_N'])*F(r['R_Ceff']))
    def test_pf_weight_identity(self):
        r=run(fixture());self.assertEqual(F(r['R_G']),sum(F(a['weighted_term']) for a in r['axes']))
    def test_weights_sum(self):self.assertEqual(sum(F(a['w_PF']) for a in run(fixture())['axes']),1)
    def test_not_simple_average(self):self.assertNotEqual(F(run(fixture())['R_G']),F(7,10)*F(6,4))
    def test_large_integer_precision(self):
        r=fixture();n=10**25+3;r['PR']['shots']={'cosine':n,'sine':n};r['PR']['saved_G_exact']=str(7*n)
        self.assertEqual(run(r)['G_PR'],str(7*n))
    def test_model_fraction(self):
        r=fixture();r['PF'].update(axis_costs={'cosine':'3/2','sine':8},saved_G_exact='25/2',cost_class='VALIDATED_MODEL_ESTIMATE')
        self.assertEqual(run(r)['R_G'],'52/25');self.assertEqual(run(r)['PF_cost_class'],'VALIDATED_MODEL_ESTIMATE')
    def test_pairing_mismatch(self):
        r=fixture();r['PR']['trajectories'][0]['axes']['sine']['pair_id']='different'
        with self.assertRaisesRegex(m.AnalysisError,'PAIRED'):run(r)
    def test_seed_mismatch(self):
        r=fixture();r['PR']['trajectories'][0]['axes']['sine']['seed']=999
        with self.assertRaisesRegex(m.AnalysisError,'SEED_EVENT'):run(r)
    def test_duplicate_replica(self):
        r=fixture();r['PR']['trajectories'][1]['replica']=0
        with self.assertRaisesRegex(m.AnalysisError,'DUPLICATE'):run(r)
    def test_n48_reused_replica_once(self):
        r=fixture('CONFIRMATION_N48','H8');r['PR']['trajectories'][0]['reused_complete_trajectory']=True
        self.assertEqual(run(r)['G_PR'],'26')
        r['PR']['trajectories'].append(copy.deepcopy(r['PR']['trajectories'][0]))
        with self.assertRaisesRegex(m.AnalysisError,'COHORT_COVERAGE'):run(r)
    def test_missing_axis(self):
        r=fixture();del r['PR']['trajectories'][0]['axes']['sine']
        with self.assertRaisesRegex(m.AnalysisError,'PAIRED_AXES'):run(r)
    def test_mixed_n8_n32(self):
        r=fixture();r['PR']['n']=32
        with self.assertRaisesRegex(m.AnalysisError,'COHORT_MIXING'):run(r)
    def test_mixed_n48_pair(self):
        r=fixture();r['PR']['trajectories'][0]['cohort']='INDEPENDENT_FIXED_CONFIRMATION_N48'
        with self.assertRaisesRegex(m.AnalysisError,'COHORT_MIXING'):run(r)
    def test_h4_anchor(self):self.assertEqual(run(fixture('OLD_PR','H4'))['core_status'],'CORE_COMPLETE')
    def test_h4_anchor_misclassified(self):
        r=fixture('OLD_PR','H4');r['PR']['cohort']='INDEPENDENT_FIXED_CONFIRMATION_N48'
        with self.assertRaisesRegex(m.AnalysisError,'CLASSIFICATION'):run(r)
    def test_h4_future_rejected(self):
        r=fixture(system='H4');r['H4_stage']='H4_POST_D_P'
        with self.assertRaisesRegex(m.AnalysisError,'FUTURE'):run(r)
    def test_binding_mismatch(self):
        r=fixture();r['PF']['binding']['DF']='DIFFERENT'
        with self.assertRaisesRegex(m.AnalysisError,'SOURCE_OR_BINDING'):run(r)
    def test_candidate_mismatch(self):
        r=fixture();r['PR_id']='NEW_WINNER'
        with self.assertRaisesRegex(m.AnalysisError,'CANDIDATE'):run(r)
    def test_source_evidence_required(self):
        r=fixture();r['PF']['source_evidence']=None
        with self.assertRaisesRegex(m.AnalysisError,'EVIDENCE'):run(r)
    def test_bad_cost_class(self):
        r=fixture();r['PF']['cost_class']='ALL_GRID_MEASURED'
        with self.assertRaisesRegex(m.AnalysisError,'PROVENANCE'):run(r)
    def test_saved_G_mismatch(self):
        r=fixture();r['PR']['saved_G_exact']='25'
        with self.assertRaisesRegex(m.AnalysisError,'EXACT_G'):run(r)
    def test_saved_reduction_mismatch(self):
        r=fixture();r['saved_reduction_percent_exact']='1'
        with self.assertRaisesRegex(m.AnalysisError,'REDUCTION'):run(r)
    def test_saved_N_mismatch(self):
        r=fixture();r['PR']['saved_N_total']=5
        with self.assertRaisesRegex(m.AnalysisError,'SHOT_TOTAL'):run(r)
    def test_no_new_statistics(self):
        r=run(fixture());self.assertEqual(r['saved_SE'],'saved-text-not-recomputed');self.assertFalse(r['new_statistics_computed'])
        self.assertNotIn('CI',r);self.assertNotIn('new_SE',r)

class MissingAndGuard(unittest.TestCase):
    def test_null_not_false(self):
        r=fixture();r['PR']['eligible']=None;self.assertEqual(run(r)['core_status'],'UNRESOLVED')
    def test_ineligible(self):
        r=fixture();r['PF']['eligible']=False;self.assertEqual(run(r)['core_status'],'INELIGIBLE')
    def test_integer_zero_is_not_false_flag(self):
        r=fixture();r['PF']['eligible']=0
        with self.assertRaisesRegex(m.AnalysisError,'ELIGIBILITY_TYPE'):run(r)
    def test_missing_shots(self):
        r=fixture();r['PR']['shots']['cosine']=None;self.assertEqual(run(r)['core_status'],'CORE_INPUT_MISSING')
    def test_missing_price(self):
        r=fixture();r['PF']['axis_costs']=None;self.assertEqual(run(r)['core_status'],'CORE_INPUT_MISSING')
    def test_display_only(self):
        r=fixture();r['PF']['saved_G_exact']=None;r['PF']['saved_G_display']='14.0';self.assertEqual(run(r)['core_status'],'DISPLAY_ONLY_SOURCE')
    def test_core_complete_without_explanation(self):
        r=run(fixture());self.assertEqual(r['core_status'],'CORE_COMPLETE');self.assertEqual(r['explanation_status'],'EXPLANATORY_INPUT_MISSING')
    def test_invalid_exact_inputs(self):
        for x in [True,1.0,'nan','1/0','0','-1']:
            with self.subTest(x=x),self.assertRaises(m.AnalysisError):m.exact(x)
    def test_noninteger_measured_RZ(self):
        r=fixture();r['PR']['trajectories'][0]['axes']['cosine']['RZ']='3/2'
        with self.assertRaisesRegex(m.AnalysisError,'INTEGER_MEASURED'):run(r)
    def test_noninteger_shots(self):
        for x in [True,2.0,'2',0,-1]:
            r=fixture();r['PR']['shots']['cosine']=x
            with self.subTest(x=x),self.assertRaises(m.AnalysisError):run(r)
    def test_secondary_accounting_rejected(self):
        r=fixture();r['bias_mode']='EMPIRICAL_NONCANCELLATION'
        with self.assertRaisesRegex(m.AnalysisError,'PRIMARY_32'):run(r)
    def test_production_permit_not_public(self):
        with self.assertRaisesRegex(m.AnalysisError,'AUTHORIZATION'):m.Permit('SAVED_REAL_VALUES')
    def test_synthetic_permit_rejects_real(self):
        r=fixture();r['artificial']=False
        with self.assertRaisesRegex(m.AnalysisError,'DATA_KIND'):run(r)
    def test_unapproved_gate_before_contract(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'grant.json';p.write_text(json.dumps({'approved':False}))
            with self.assertRaisesRegex(m.AnalysisError,'AUTHORIZATION'):cli.gate(p,Path(d)/'UNREADABLE_CONTRACT')
    def test_source_mismatch_gate(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'approval';p.write_text('ARTIFICIAL_USER_APPROVAL')
            c=Path(d)/'contract';c.write_text(json.dumps({'schema':m.SCHEMA,'scope':'TRACK_A_MECHANISM_PRIMARY32','source_commit':'x'*40,'input_binding_sha256':'x','target_ids':list(range(32)),'output_root':d+'/future'}))
            g=Path(d)/'grant';g.write_text(json.dumps(dict(approved=True,approved_by_user=True,runtime_authorization=True,one_shot=True,next_stage_authorized=False,retry=False,kind='SAVED_REAL_VALUES',scope='TRACK_A_MECHANISM_PRIMARY32',approval_receipt=str(p),approval_receipt_sha256=cli.sha(p),contract_sha256=cli.sha(c),source_commit='DIFFERENT',input_binding_sha256='x',target_ids=list(range(32)),output_root=d+'/future')))
            with self.assertRaisesRegex(m.AnalysisError,'SOURCE_OR_BINDING'):cli.gate(g,c)
    def test_protected_existing_files(self):
        with tempfile.TemporaryDirectory() as d:
            parent=Path(d);original=parent/'old_SOURCE';original.mkdir();(original/'STOP.json').write_text('UNCHANGED')
            for path in [original,original/'new',original/'STOP.json']:
                with self.assertRaises(m.AnalysisError):cli.prepare_output(path,path.parent,[str(original)])
            self.assertEqual((original/'STOP.json').read_text(),'UNCHANGED')
    def test_one_shot_reuse_refused(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d)/'run';cli.prepare_output(r,Path(d),[])
            with self.assertRaisesRegex(m.AnalysisError,'ALREADY_EXISTS'):cli.prepare_output(r,Path(d),[])
    def test_symlink_output_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'old').mkdir();(root/'link').symlink_to(root/'old',target_is_directory=True)
            with self.assertRaisesRegex(m.AnalysisError,'SYMLINK'):cli.prepare_output(root/'link'/'run',root/'link',[])

class Explanation(unittest.TestCase):
    def test_complete(self):self.assertEqual(run(explain_fixture())['explanation_status'],'EXPLANATION_COMPLETE')
    def test_integer_unchanged(self):
        r=explain_fixture();old=copy.deepcopy(r);run(r);self.assertEqual(r,old)
    def test_normalization_headroom_factors(self):
        out=run(explain_fixture())['explanation']['sides']['ratios']['cosine'];self.assertEqual(out['normalization_ratio_squared'],1);self.assertEqual(out['headroom_ratio_squared'],1)
    def test_missing_signed_bias_not_inverted(self):
        r=explain_fixture();r['PF']['explanation']['signed_bias']=None;self.assertEqual(run(r)['explanation_status'],'EXPLANATORY_INPUT_MISSING')
    def test_zero_headroom(self):
        r=explain_fixture();r['PF']['explanation']['headroom']['cosine']='0';self.assertEqual(run(r)['explanation_status'],'UNRESOLVED')
    def test_headroom_mismatch(self):
        r=explain_fixture();r['PF']['explanation']['headroom']['cosine']='0.01';self.assertEqual(run(r)['explanation_status'],'UNRESOLVED')
    def test_log_binding(self):
        r=explain_fixture();r['PF']['explanation']['log_B']='0.1';self.assertEqual(run(r)['explanation_status'],'UNRESOLVED')
    def test_rounding_boundary_wrong_saved_N(self):
        r=explain_fixture();r['PF']['shots']['cosine']+=1;r['PF']['saved_G_exact']=str(2*r['PF']['shots']['cosine']+8*r['PF']['shots']['sine']);self.assertEqual(run(r)['explanation_status'],'UNRESOLVED')
    def test_log_domain_guard(self):
        r=explain_fixture()['PR'];r['explanation'].update(B=None,log_B='1000',log_shot_bound=None,integer_status={a:'LOG_DOMAIN_ONLY' for a in m.AXES})
        e=m.explanation_side(r,.05);self.assertEqual(e['axes']['cosine']['integer_status'],'LOG_DOMAIN_ONLY');self.assertIsNone(e['axes']['cosine']['Nstar'])
    def test_nonfinite(self):
        r=explain_fixture();r['PF']['explanation']['headroom']['cosine']='NaN';self.assertEqual(run(r)['explanation_status'],'UNRESOLVED')
    def test_smallest_saved_integer(self):
        s=explain_fixture()['PR'];s['shots']={a:1 for a in m.AXES};s['explanation'].update(headroom={a:'10' for a in m.AXES},log_shot_bound=None)
        e=m.explanation_side(s,10*math.sqrt(2));self.assertEqual(e['axes']['cosine']['integer_status'],'SAVED_INTEGER_UNCHANGED')

class ExponentGuard(unittest.TestCase):
    def test_exp_overflow(self):self.assertIsNone(m.exp_safe(10000))
    def test_exp_underflow_not_missing_zero(self):self.assertIsNone(m.exp_safe(-10000))

class NativeArtificial(unittest.TestCase):
    def test_cli_native_exit_and_result_hash(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'fixture.json';p.write_text(json.dumps(bundle(fixture())))
            result=subprocess.run([sys.executable,'-I','-B',str(ROOT/'scripts/run_mechanism_analysis.py'),'--artificial-fixture',str(p),'--run-root',str(root/'run')],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr);t=cli.read(root/'run/terminal.json');self.assertEqual(t['status'],'ARTIFICIAL_ONLY')
            for name,digest in t['hashes'].items():self.assertEqual(cli.sha(root/'run'/name),digest)
    def test_cli_real_input_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);r=fixture();r['artificial']=False;b=bundle(r);b['kind']='SAVED_REAL_VALUES';p=root/'fixture.json';p.write_text(json.dumps(b))
            result=subprocess.run([sys.executable,'-I','-B',str(ROOT/'scripts/run_mechanism_analysis.py'),'--artificial-fixture',str(p),'--run-root',str(root/'run')],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0);self.assertFalse((root/'run').exists())
    def test_duplicate_targets(self):
        b=bundle(fixture());b['comparisons']*=2;b['target_ids']*=2
        with self.assertRaisesRegex(m.AnalysisError,'REGISTERED_TARGETS'):m.analyze_bundle(b,m.artificial_permit())

class SyntheticLauncherLifecycle(unittest.TestCase):
    def setup_case(self,root,exit_code=0,corrupt=False):
        grant=root/'grant_INVALID_ARTIFICIAL.txt';grant.write_text('ARTIFICIAL MOCK ONLY: NOT AN APPROVED GRANT')
        contract_path=root/'contract_ARTIFICIAL.txt';contract_path.write_text('ARTIFICIAL MOCK ONLY')
        input_file=root/'invented_input';input_file.write_text('ARTIFICIAL')
        target_ids=['ARTIFICIAL_'+str(i) for i in range(32)]
        contract=dict(output_root=str(root/'new_run'),allowed_run_parent=str(root),target_ids=target_ids,source_commit='ARTIFICIAL_MOCK',input_binding=str(input_file),runtime_profile=str(input_file))
        binding=dict(input_files=[dict(path=str(input_file),sha256=cli.sha(input_file))],protected_paths=[])
        class FakeOwnedChild:
            pid=123456789
            returncode=exit_code
            def __init__(self,*args,**kwargs):
                out=root/'new_run'
                if exit_code==0:
                    cli.write_new(out/'invented_result.json',dict(artificial=True))
                    cli.write_new(out/'worker_result.json',dict(target_ids=target_ids,hashes={'invented_result.json':cli.sha(out/'invented_result.json')}))
                if corrupt:input_file.write_text('CHANGED_ARTIFICIAL_INPUT')
            def communicate(self):return ('ARTIFICIAL_STDOUT','ARTIFICIAL_STDERR')
            def poll(self):return self.returncode
        return grant,contract_path,contract,binding,FakeOwnedChild
    def test_native_acceptance_lifecycle_mock(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);g,c,contract,binding,child=self.setup_case(root)
            with patch.object(cli,'gate',return_value=({},contract,binding)),patch.object(cli.subprocess,'Popen',child):cli.launch(g,c)
            terminal=cli.read(root/'new_run/terminal.json');self.assertEqual(terminal['native_exit_code'],0);self.assertTrue(terminal['reaped']);self.assertTrue((root/'new_run/ONE_SHOT.json').exists())
    def test_failure_retains_stop_and_reservation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);g,c,contract,binding,child=self.setup_case(root,exit_code=7)
            with patch.object(cli,'gate',return_value=({},contract,binding)),patch.object(cli.subprocess,'Popen',child),self.assertRaisesRegex(m.AnalysisError,'WORKER_FAILURE'):cli.launch(g,c)
            self.assertTrue((root/'new_run/STOP.json').exists());self.assertTrue((root/'new_run/ONE_SHOT.json').exists());self.assertFalse((root/'new_run/terminal.json').exists())
            self.assertEqual(cli.read(root/'new_run/native_wait.json')['native_exit_code'],7)
    def test_post_input_mismatch_cannot_pass(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);g,c,contract,binding,child=self.setup_case(root,corrupt=True)
            with patch.object(cli,'gate',return_value=({},contract,binding)),patch.object(cli.subprocess,'Popen',child),self.assertRaisesRegex(m.AnalysisError,'POST_INPUT'):cli.launch(g,c)
            self.assertTrue((root/'new_run/STOP.json').exists());self.assertFalse((root/'new_run/terminal.json').exists())
    def test_preparer_has_no_science_imports(self):
        import ast
        tree=ast.parse((ROOT/'scripts/prepare_mechanism_analysis.py').read_text())
        imports=[a.name for node in ast.walk(tree) if isinstance(node,ast.Import) for a in node.names]
        self.assertFalse({'mechanism_analysis','statistics','math','numpy','qiskit','fractions'} & set(imports))
    def test_input_not_mutated(self):
        row=fixture();before=copy.deepcopy(row);run(row);self.assertEqual(row,before)

if __name__=='__main__':unittest.main(verbosity=2)
