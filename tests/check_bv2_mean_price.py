"""C 1db4d6e tests, adapted only for B's absent stream/helper files.

The provider SOURCE is unchanged; the inventory helper is read from pinned Git
and an absent stream entry receives a test-only raising sentinel.
"""
import ast
import builtins
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from trottertracks import mean_compiled_price as C
from trottertracks import bv2_mean_price as V
from trottertracks import pf_pr_rpe_pipeline as p
import trottertracks


def blob(path):return subprocess.check_output(['git','show',V.B_RELEASE+':'+path],cwd=ROOT)


def fixed(name,path):
    module=types.ModuleType(name);module.__file__=str(ROOT/path);module.__package__='trottertracks'
    sys.modules[name]=module
    if name.startswith("trottertracks."):setattr(trottertracks,name.rsplit(".",1)[-1],module)
    exec(compile(blob(path),V.B_RELEASE+':'+path,'exec'),module.__dict__)
    return module


# B has newer exact-scalar helpers. Load its fixed accounting/pipeline in this
# test process, preserving C's already imported original module objects.
# No B worktree or scientific package initializer is accessed.
assert (ROOT/'src/trottertracks/c1_stage3_accuracy.py').read_bytes()==blob('src/trottertracks/c1_stage3_accuracy.py')
fixed('trottertracks.h4_df10_rpe_connection','src/trottertracks/h4_df10_rpe_connection.py')
fixed('trottertracks.pf_pr_rpe_pipeline','src/trottertracks/pf_pr_rpe_pipeline.py')
A=fixed('trottertracks.pr_rpe_rmse','src/trottertracks/pr_rpe_rmse.py')
B=fixed('trottertracks.pr_rpe_rmse_controller','src/trottertracks/pr_rpe_rmse_controller.py')
I=fixed('trottertracks.pr_rpe_rmse_integration','src/trottertracks/pr_rpe_rmse_integration.py')
p=I.p
# Reuse the fixed B fixture functions; none of its test/cost/demo entries run.
fixed('demo_pr_rpe_rmse','scripts/demo_pr_rpe_rmse.py')
fixtures=fixed('_c_bv2_existing_fixtures','tests/check_pr_rpe_rmse_integration.py')
controller_fixtures=fixed('_c_bv2_controller_fixtures','tests/check_pr_rpe_rmse_controller.py')


def config(M=0,system='H4',method='PR',kappa='1'):
    return B.generate(I.a_spec(fixtures.inputs(M=M,system=system,kappa=kappa),method))


def records(config,model=False):
    output=[]
    for row in config['rounds']:
        for axis in p.AXES:
            r=dict(**p.measurement(row,axis,saved_sampler=True),identity=deepcopy(row['identity']),
                law_binding=deepcopy(row['law_binding']),axis=axis,quantity=V.QUANTITY,
                classification='MODEL_PREDICTION' if model else 'DIRECT_COMPILED_FINITE_SAMPLE',
                expectation='FINITE_COHORT_ONLY',cohort='ARTIFICIAL_COHORT',n_paths=3,
                mean=dict(rz_count=7),source=dict(fixture='ARTIFICIAL_COMPILED_PRICE_NOT_REAL'),
                normalization_is_price_multiplier=False,uncertainty=dict(joint_covariance=None,total_CI=None,systematic=None),model=None)
            if model:r['model']=dict(qualified_for_candidate=False,domain=deepcopy(row['candidate']),
                binding=deepcopy(config['spec']['binding']),compiler=config['spec']['profile']['compiler'])
            output.append(r)
    return output


class Tests(unittest.TestCase):
    def test_existing_B_adapter_preserves_new_schema_identity_law_source_and_cohort(self):
        c=config();given=records(c);before=deepcopy(given)
        out=V.V2MeanPriceLookup(c,given,integration=I)()
        self.assertEqual(given,before)
        for a,b in zip(out,given):
            for field in ('identity','law_binding','axis','source','cohort','n_paths','mean','quantity','uncertainty'):
                self.assertEqual(a[field],b[field])
            self.assertEqual(a['expectation'],'FINITE_COHORT_ONLY')
            self.assertEqual(a['price_status'],'SAVED_DIRECT_FINITE_COHORT')
            self.assertFalse(a['full_RPE_price_adopted'])

    def test_structural_single_path_local_and_omitted_mean_unit_are_rejected(self):
        c=config()
        for quantity in ['SINGLE_PATH_STRUCTURAL_PROXY','FULL_WRAPPER_STRUCTURAL_RZ_SINGLE_PATH','LOCAL_C_MINUS_S',None]:
            r=records(c);r[0]['quantity']=quantity
            with self.assertRaises(ValueError):V.V2MeanPriceLookup(c,r,integration=I)

    def test_model_remains_unqualified_and_unknown_uncertainty_null(self):
        c=config();given=records(c,model=True)
        out=V.V2MeanPriceLookup(c,given,integration=I)()
        self.assertTrue(all(r['model']['qualified_for_candidate'] is False for r in out))
        self.assertTrue(all(r['price_status']=='UNQUALIFIED_MODEL' for r in out))
        self.assertTrue(all(r['uncertainty']['joint_covariance'] is None for r in out))
        given[0]['model']['qualified_for_candidate']=True
        with self.assertRaises(ValueError):V.V2MeanPriceLookup(c,given,integration=I)()

    def test_old_C1_id_is_never_relabelled_to_B_V2(self):
        c=config();r=records(c)
        for x in r:x['identity']['candidate']['id']='14bd81a375400c9b0305'
        out=V.V2MeanPriceLookup(c,r,integration=I)()
        self.assertTrue(all(x['mean']['rz_count'] is None for x in out))
        self.assertTrue(all(x['identity']==c['rounds'][0]['identity'] for x in out))

    def test_foreign_law_identity_compiler_and_missing_source_are_rejected(self):
        c=config()
        mutations=[lambda r:r['law_binding'].update(candidate_id='FOREIGN'),
                   lambda r:r['identity']['binding'].update(state='FOREIGN'),
                   lambda r:r['identity']['profile'].update(compiler='FOREIGN'),
                   lambda r:r.update(source=None)]
        for change in mutations:
            r=records(c);change(r[0])
            with self.assertRaises(ValueError):V.V2MeanPriceLookup(c,r,integration=I)()

    def test_M_kappa_and_H_or_PF_retained_scope_are_distinct(self):
        a=config(M=1);b=config(M=2);c=config(M=1,kappa='2')
        for other in [b,c]:
            self.assertNotEqual(a['id'],other['id'])
            self.assertTrue(all(x['mean']['rz_count'] is None for x in V.V2MeanPriceLookup(other,records(a),integration=I)()))
        full=config(method='PF');args=fixtures.inputs();args['design']['PF']=dict(method='B0',prefix=1)
        cut=B.generate(I.a_spec(args,'PF'))
        self.assertNotEqual(full['id'],cut['id'])
        for cfg,method in [(full,'FULL_PF'),(cut,'TRUNCATED_PF')]:
            lookup=V.V2MeanPriceLookup(cfg,integration=I)
            self.assertEqual(lookup.method,method)
            self.assertTrue(all(x['price_status']=='OUTSIDE_DOMAIN' and x['mean']['rz_count'] is None for x in lookup()))

    def test_H6_early_prefix5_remains_missing_regular11piece_preserved(self):
        c=config(M=4,system='H6');out=V.V2MeanPriceLookup(c,records(c),integration=I)()
        self.assertTrue(all(x['mean']['rz_count'] is None for x in out if x['identity']['m']<4))
        regular=[x for x in out if x['identity']['m']==4]
        self.assertTrue(all(len(x['identity']['structure']['allocation'])==11 for x in regular))
        self.assertTrue(all(x['mean']['rz_count']==7 for x in regular))

    def test_Full_H_RTE_event_count_route_missing_not_PR_or_B3(self):
        c=B.generate(controller_fixtures.spec('FULL_H_RTE',M=0))
        out=V.V2MeanPriceLookup(c,integration=I)()
        self.assertTrue(all(x['mean']['rz_count'] is None and x['model'] is None for x in out))
        self.assertTrue(all(x['law_context']['price_law'] is None and x['law_context']['ideal_sampler_distance_qualified'] is None for x in out))
        self.assertIsNone(c['rounds'][0]['q_PF']);self.assertIsNone(c['rounds'][0]['R_outer'])
        bad=controller_fixtures.spec('FULL_H_RTE',M=0);bad['split']['original_PF_method']='B3'
        with self.assertRaises(ValueError):B.generate(bad)

    def test_finite_cohort_is_not_qualified_expectation_and_no_G_is_called(self):
        c=config();out=V.V2MeanPriceLookup(c,records(c),integration=I)()
        cohorts={(r['identity']['candidate']['id'],r['axis']):r['cohort'] for r in out}
        with patch.object(B,'cost',side_effect=AssertionError('No G')),patch.object(B,'select',side_effect=AssertionError('No selection')):
            result=B.assess(c,None,out,cohorts=cohorts)
        self.assertIn('UNQUALIFIED_C_EXPECTATION_OR_FORCED_COHORT',result['reasons'])
        self.assertIsNone(result['cost'])

    def test_cohort_choice_paired_axes_and_no_pooling(self):
        c=config();r=records(c);second=deepcopy(r)
        for x in second:x['cohort']='SECOND';x['mean']['rz_count']=9
        none=V.V2MeanPriceLookup(c,r+second,integration=I)()
        self.assertTrue(all(x['price_status']=='SELECTION_REQUIRED' for x in none))
        selected={(x['identity']['candidate']['id'],x['axis']):'SECOND' for x in r}
        out=V.V2MeanPriceLookup(c,r+second,integration=I,selections=selected)()
        self.assertTrue(all(x['mean']['rz_count']==9 and x['cohort']=='SECOND' for x in out))

    def test_saved_118_candidate_inventory_projects_no_price_or_N_values(self):
        inv=types.ModuleType('_c_bv2_inventory');inv.__file__=str(ROOT/'scripts/inventory_bv2_mean_prices.py')
        helper=subprocess.check_output(['git','show','1db4d6e7ebce872c069e337b76632046f60d8b56:scripts/inventory_bv2_mean_prices.py'],cwd=ROOT)
        exec(compile(helper,inv.__file__,'exec'),inv.__dict__)
        with patch.object(B,'generate',side_effect=AssertionError('No new physical configuration')):
            out=inv.inventory(ROOT)
        self.assertEqual(len(out['recipes']),118);self.assertEqual(len(out['rows']),236)
        self.assertFalse(out['actual_S_C_N_values_copied'])
        self.assertTrue(all(not r['eligible_for_RMSE_comparison'] for r in out['rows']))
        self.assertTrue(all('mean' not in r and 'S_stream' not in r for r in out['rows']))
        self.assertEqual(len(out['C1_cells']),8)

    def test_lookup_never_calls_scientific_or_mean_or_accounting_entries(self):
        c=config()
        with patch.object(C,'full_wrapper_mean',side_effect=AssertionError('No mean')),\
             patch.object(C,'stream_structure',side_effect=AssertionError('No S'),create=True),\
             patch.object(C,'combine_mean',side_effect=AssertionError('No prediction')),\
             patch.object(A,'evaluate',side_effect=AssertionError('No N')),\
             patch.object(B,'cost',side_effect=AssertionError('No G')):
            V.V2MeanPriceLookup(c,records(c),integration=I)()


if __name__=='__main__':unittest.main(verbosity=2)
