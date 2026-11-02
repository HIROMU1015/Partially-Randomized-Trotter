"""Invented fixtures only. No saved scientific input is loaded by these tests."""
import ast
import copy
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/trotterlib'))
import mechanism_explanation as m
import mechanism_analysis as old
spec=importlib.util.spec_from_file_location('explanation_cli_test',ROOT/'scripts/run_mechanism_explanation.py')
cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)


def invented():
    row=dict(artificial=True,contrast_id='INVENTED_1',system='H6',
        H4_stage='NOT_APPLICABLE',epsilon='0.05',factor='1',bias_mode='USUAL_SIGNED',
        PR_role='NEW_PR',PR_cohort='EXPLORATION_N8',PR_id='INVENTED_PR',PF_id='INVENTED_PF',
        binding=dict(system='H6',snapshot='INVENTED',compiler='INVENTED'),cohorts_pooled=False,sides={})
    for method in ('PR','PF'):
        step=1.125 if method=='PR' else 1.;steps=4 if method=='PR' else 0
        nom=steps*math.log(step);margin=.015;upper=nom+margin;alpha=.025
        h=.05/math.sqrt(2)-.001-.0003
        log_n=math.log(2)+2*upper-2*math.log(h)+math.log(math.log(2/alpha))
        n=max(1,math.ceil(2*math.exp(upper)**2/h**2*math.log(2/alpha)))
        row['sides'][method]=dict(identity=m.identity(row,method),eligible=True,
            normalization=dict(law='FINITE_RTE' if method=='PR' else 'DETERMINISTIC_PF',
                step_b=repr(step),step_count=steps,log_B_nominal=repr(nom),log_B_margin=repr(margin),
                factor='1',upper_route='EXP_OF_LOG_UPPER',normalization_upper_saved=None),
            signed_bias={a:'.001' for a in m.AXES},allowance={a:'.0003' for a in m.AXES},
            headroom={a:repr(h) for a in m.AXES},alpha_axis=repr(alpha),
            log_shot_bound={a:repr(log_n) for a in m.AXES},
            integer_status={a:'BINARY64_DIAGNOSTIC' for a in m.AXES},shots={a:n for a in m.AXES})
    saved=dict(contrast_id=row['contrast_id'],system=row['system'],H4_stage=row['H4_stage'],
        core_status='CORE_COMPLETE',saved_G_exact_match=True,exact_decompositions_match=True,
        G_PR='INVENTED_SAVED_TOKEN_NOT_RECOMPUTED',saved_SE='INVENTED_UNCHANGED_SE')
    return dict(schema=m.SCHEMA,kind='ARTIFICIAL',target_ids=[row['contrast_id']],comparisons=[row]),[saved]


def run(bundle,core):return m.analyze_bundle(bundle,core,m.artificial_permit())[0]


class DefinitionAndRegression(unittest.TestCase):
    def test_correct_upper_policy(self):
        b,c=invented();self.assertEqual(run(b,c)['explanation']['status'],'EXPLANATION_COMPLETE')
    def test_old_pr_guard_reproduced_on_invented_values(self):
        b,c=invented();side=b['comparisons'][0]['sides']['PR'];norm=side['normalization']
        legacy=dict(shots=side['shots'],explanation=dict(B=norm['step_b'],log_B=norm['log_B_nominal'],
            **{k:side[k] for k in ('signed_bias','allowance','headroom','alpha_axis','log_shot_bound','integer_status')}))
        with self.assertRaisesRegex(old.AnalysisError,'NORMALIZATION_LOG_BINDING'):old.explanation_side(legacy,.05)
        self.assertEqual(run(b,c)['explanation']['status'],'EXPLANATION_COMPLETE')
    def test_old_pf_margin_omission_reproduced(self):
        b,c=invented();side=b['comparisons'][0]['sides']['PF']
        legacy=dict(shots=side['shots'],explanation=dict(B='1',log_B='0',
            **{k:side[k] for k in ('signed_bias','allowance','headroom','alpha_axis','log_shot_bound','integer_status')}))
        with self.assertRaisesRegex(old.AnalysisError,'LOG_BOUND_MISMATCH'):old.explanation_side(legacy,.05)
    def test_nominal_and_upper_are_distinct(self):
        b,c=invented();n=run(b,c)['explanation']['sides']['PR']['normalization']
        self.assertGreater(n['log_B_upper'],n['log_B_nominal']);self.assertGreater(n['B_upper'],n['step_b'])
    def test_step_stage_mismatch_rejected(self):
        b,c=invented();b['comparisons'][0]['sides']['PR']['normalization']['step_count']=3
        self.assertEqual(run(b,c)['explanation']['sides']['PR']['reason'],'STEP_CUMULATIVE_LOG_MISMATCH')
    def test_deterministic_overlay_has_no_rte_fields(self):
        b,c=invented();b['comparisons'][0]['sides']['PF']['normalization'].update(step_b=None,step_count=None)
        n=run(b,c)['explanation']['sides']['PF']
        self.assertEqual(n['status'],'EXPLANATION_COMPLETE');self.assertIsNone(n['normalization']['step_b']);self.assertIsNone(n['normalization']['step_count'])
    def test_deterministic_nonzero_log_rejected(self):
        b,c=invented();b['comparisons'][0]['sides']['PF']['normalization']['log_B_nominal']='.01'
        self.assertEqual(run(b,c)['explanation']['sides']['PF']['reason'],'DETERMINISTIC_NORMALIZATION_SCHEMA')
    def test_deterministic_random_step_count_rejected(self):
        b,c=invented();b['comparisons'][0]['sides']['PF']['normalization']['step_count']=4
        self.assertEqual(run(b,c)['explanation']['status'],'UNRESOLVED')
    def test_margin_omission_rejected(self):
        b,c=invented();b['comparisons'][0]['sides']['PF']['normalization']['log_B_margin']='0'
        self.assertEqual(run(b,c)['explanation']['sides']['PF']['reason'],'LOG_BOUND_MISMATCH')
    def test_negative_margin_rejected(self):
        b,c=invented();b['comparisons'][0]['sides']['PF']['normalization']['log_B_margin']='-.1'
        self.assertEqual(run(b,c)['explanation']['status'],'UNRESOLVED')
    def test_saved_upper_not_step_b(self):
        b,c=invented();n=b['comparisons'][0]['sides']['PR']['normalization']
        n.update(upper_route='SAVED_UPPER_BINARY64',normalization_upper_saved=n['step_b'])
        self.assertEqual(run(b,c)['explanation']['sides']['PR']['reason'],'SAVED_UPPER_LOG_BINDING')
    def test_exp_route_rejects_unexpected_binary64(self):
        b,c=invented();b['comparisons'][0]['sides']['PF']['normalization']['normalization_upper_saved']='1'
        self.assertEqual(run(b,c)['explanation']['status'],'UNRESOLVED')
    def test_nonfinite_is_not_missing_zero(self):
        b,c=invented();b['comparisons'][0]['sides']['PR']['normalization']['log_B_margin']='NaN'
        self.assertEqual(run(b,c)['explanation']['status'],'UNRESOLVED')
    def test_headroom_mismatch(self):
        b,c=invented();b['comparisons'][0]['sides']['PR']['headroom']['sine']='.01'
        self.assertEqual(run(b,c)['explanation']['sides']['PR']['reason'],'HEADROOM_MISMATCH')
    def test_rounding_mismatch(self):
        b,c=invented();b['comparisons'][0]['sides']['PR']['shots']['sine']+=1
        self.assertEqual(run(b,c)['explanation']['sides']['PR']['reason'],'SAVED_ROUNDING_CONSISTENCY')
    def test_null_and_ineligible(self):
        for value,status in [(None,'UNRESOLVED'),(False,'INELIGIBLE')]:
            b,c=invented();b['comparisons'][0]['sides']['PR']['eligible']=value
            self.assertEqual(run(b,c)['explanation']['status'],status)
    def test_missing_preserved(self):
        b,c=invented();b['comparisons'][0]['sides']['PR']['normalization']=None
        self.assertEqual(run(b,c)['explanation']['status'],'EXPLANATORY_INPUT_MISSING')
    def test_missing_does_not_hide_pf_numeric_failure(self):
        b,c=invented();row=b['comparisons'][0];row['sides']['PR']['normalization']=None
        row['sides']['PF']['headroom']['sine']='.01'
        self.assertEqual(run(b,c)['explanation']['status'],'UNRESOLVED')
    def test_axis_binding(self):
        b,c=invented();b['comparisons'][0]['sides']['PR']['allowance']={'real':'.001','imag':'.001'}
        self.assertEqual(run(b,c)['explanation']['sides']['PR']['reason'],'EXPLANATION_AXIS_SCHEMA')
    def test_candidate_binding(self):
        b,c=invented();b['comparisons'][0]['sides']['PR']['identity']['candidate_id']='OTHER'
        with self.assertRaisesRegex(m.AnalysisError,'IDENTITY_BINDING'):run(b,c)
    def test_cohort_binding(self):
        b,c=invented();b['comparisons'][0]['PR_cohort']='INDEPENDENT_FIXED_CONFIRMATION_N48'
        with self.assertRaisesRegex(m.AnalysisError,'COHORT'):run(b,c)
    def test_science_binding(self):
        b,c=invented();b['comparisons'][0]['sides']['PF']['identity']['binding']={'system':'H4'}
        with self.assertRaisesRegex(m.AnalysisError,'IDENTITY_BINDING'):run(b,c)
    def test_core_never_recalculated(self):
        b,c=invented();before=copy.deepcopy((b,c))
        with patch.object(old,'analyze_bundle',side_effect=AssertionError('CORE CALLED')),patch.object(old,'paired_costs',side_effect=AssertionError('AVERAGE CALLED')):
            out=run(b,c)
        self.assertTrue(out['core_reused_without_recalculation']);self.assertEqual((b,c),before)
    def test_saved_core_not_complete(self):
        b,c=invented();c[0]['core_status']='UNRESOLVED'
        with self.assertRaisesRegex(m.AnalysisError,'CORE_REVIEW_REQUIRED'):run(b,c)
    def test_core_target_join(self):
        b,c=invented();c[0]['contrast_id']='OTHER'
        with self.assertRaisesRegex(m.AnalysisError,'FIXED_CORE_TARGET_JOIN'):run(b,c)
    def test_scientific_kind_rejected_by_artificial_permit(self):
        b,c=invented();b['kind']='SAVED_REAL_VALUES'
        with self.assertRaisesRegex(m.AnalysisError,'DATA_KIND'):run(b,c)
    def test_log_domain_overflow_path(self):
        b,c=invented();s=b['comparisons'][0]['sides']['PR'];s['normalization'].update(step_b=repr(math.exp(700)),step_count=1,log_B_nominal='700',log_B_margin='10')
        h=float(s['headroom']['cosine']);n=math.log(2)+2*710-2*math.log(h)+math.log(math.log(2/.025))
        s.update(shots={a:None for a in m.AXES},integer_status={a:'LOG_DOMAIN_ONLY' for a in m.AXES},log_shot_bound={a:repr(n) for a in m.AXES})
        out=run(b,c)['explanation']['sides']['PR'];self.assertEqual(out['status'],'EXPLANATION_COMPLETE');self.assertIsNone(out['normalization']['B_upper']);self.assertIsNone(out['axes']['cosine']['Nstar'])
    def test_underflow_retains_log(self):self.assertIsNone(m.exp_safe(-10000))
    def test_wrong_integer_route(self):
        b,c=invented();b['comparisons'][0]['sides']['PR']['integer_status']['cosine']='LOG_DOMAIN_ONLY'
        self.assertEqual(run(b,c)['explanation']['sides']['PR']['reason'],'INTEGER_ROUTE_MISMATCH')


class AuthorizationAndIO(unittest.TestCase):
    def output_case(self,root):
        b,c=invented();cli.write_new(root/'saved_core',c)
        cli.save_outputs(root,m.analyze_bundle(b,c,m.artificial_permit()),c)
        return dict(core_results=str(root/'saved_core')),dict(target_ids=b['target_ids'],allowed_missing_side_ids=[])
    def test_parent_accepts_exact_core_copy(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);binding,contract=self.output_case(root);cli.accept_outputs(root,binding,contract)
    def test_parent_rejects_changed_core_copy(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);binding,contract=self.output_case(root)
            p=root/'joined_core_reference.json';data=cli.read(p);data['saved_core'][0]['saved_SE']='CHANGED';p.write_text(json.dumps(data))
            with self.assertRaisesRegex(m.AnalysisError,'CORE_COPY'):cli.accept_outputs(root,binding,contract)
    def test_parent_rejects_unresolved_explanation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);binding,contract=self.output_case(root)
            p=root/'explanations.json';data=cli.read(p);data[0]['explanation']['sides']['PR']['status']='UNRESOLVED';p.write_text(json.dumps(data))
            with self.assertRaisesRegex(m.AnalysisError,'NUMERIC_MISMATCH'):cli.accept_outputs(root,binding,contract)
    def test_registered_missing_does_not_allow_other_side(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);binding,contract=self.output_case(root);contract['allowed_missing_side_ids']=['INVENTED_1.PR']
            p=root/'explanations.json';data=cli.read(p);data[0]['explanation']['sides']['PF']['status']='EXPLANATORY_INPUT_MISSING';p.write_text(json.dumps(data))
            with self.assertRaisesRegex(m.AnalysisError,'UNEXPECTED_MISSING'):cli.accept_outputs(root,binding,contract)
    def test_saved_signal_container_schemas(self):
        spec=importlib.util.spec_from_file_location('explanation_preparer_test',ROOT/'scripts/prepare_mechanism_explanation.py')
        prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)
        signal={'cell':{'id':'INVENTED'}}
        for value in ([signal],{'signals':[signal]},signal):self.assertEqual(prep.records(value),[signal])
    def test_unknown_signal_container_rejected(self):
        spec=importlib.util.spec_from_file_location('explanation_preparer_test',ROOT/'scripts/prepare_mechanism_explanation.py')
        prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)
        with self.assertRaisesRegex(ValueError,'UNSUPPORTED_SIGNAL_CONTAINER'):prep.records({'wrong':[]})
    def test_unapproved_before_input_read(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'grant';p.write_text(json.dumps(dict(approved=False)))
            with self.assertRaisesRegex(m.AnalysisError,'AUTHORIZATION'):cli.gate(p,Path(d)/'DO_NOT_READ')
    def test_old_scope_not_reusable(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'grant';p.write_text(json.dumps(dict(approved=True,approved_by_user=True,runtime_authorization=True,one_shot=True,retry=False,next_stage_authorized=False,kind='SAVED_REAL_VALUES',scope='TRACK_A_MECHANISM_PRIMARY32')))
            with self.assertRaisesRegex(m.AnalysisError,'APPROVAL_SCOPE'):cli.gate(p,Path(d)/'DO_NOT_READ')
    def test_old_stop_write_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);oldroot=p/'old';oldroot.mkdir();stop=oldroot/'STOP.json';stop.write_text('PRESERVED')
            with self.assertRaises(m.AnalysisError):cli.prepare_output(oldroot/'new',oldroot,[str(oldroot)])
            self.assertEqual(stop.read_text(),'PRESERVED')
    def test_artificial_native_cli_and_core_copy(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);b,c=invented();b['saved_core_artificial']=c;(p/'fixture').write_text(json.dumps(b))
            proc=subprocess.run([sys.executable,'-I','-B',str(ROOT/'scripts/run_mechanism_explanation.py'),'--artificial-fixture',str(p/'fixture'),'--run-root',str(p/'run')],capture_output=True,text=True)
            self.assertEqual(proc.returncode,0,proc.stderr)
            self.assertEqual(cli.read(p/'run/joined_core_reference.json')['saved_core'],c)
            for name,h in cli.read(p/'run/terminal.json')['hashes'].items():self.assertEqual(cli.sha(p/'run'/name),h)
    def test_artificial_entry_refuses_real_kind(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);b,c=invented();b['kind']='SAVED_REAL_VALUES';(p/'fixture').write_text(json.dumps(b))
            proc=subprocess.run([sys.executable,'-I','-B',str(ROOT/'scripts/run_mechanism_explanation.py'),'--artificial-fixture',str(p/'fixture'),'--run-root',str(p/'run')],capture_output=True,text=True)
            self.assertNotEqual(proc.returncode,0);self.assertFalse((p/'run').exists())
    def test_preparation_has_no_numeric_imports(self):
        tree=ast.parse((ROOT/'scripts/prepare_mechanism_explanation.py').read_text())
        modules={a.name for node in ast.walk(tree) if isinstance(node,ast.Import) for a in node.names}
        self.assertFalse({'math','mechanism_explanation','statistics','fractions','numpy','qiskit'} & modules)


if __name__=='__main__':unittest.main(verbosity=2)
