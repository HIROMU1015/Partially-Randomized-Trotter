"""Fixed published A metadata + reused B/C API; no scientific quantities."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
# Reuse the already tested immutable B loader; no old test cases are rerun.
from check_bv2_mean_price import I,B,C,V
from trottertracks import h4_s0_price_plan as m


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixed=m.read(ROOT,m.A_RELEASE,m.A_PATH)
        cls.plan=m.prepare(ROOT,integration=I)
        cls.config=B.generate(I.a_spec(cls.fixed['inputs'],'PR'))

    def test_fixed_physical_requests_keep_exact_kappa_and_old_ids_separate(self):
        p=self.plan
        self.assertEqual(p['kappa_paper_exact'],self.fixed['conditions']['kappa_paper_exact'])
        self.assertEqual(p['M'],12);self.assertEqual(p['saved_B_schema'],'PR_RPE_RMSE_V1')
        self.assertNotEqual(p['saved_B_id'],p['B_V2_configuration_id'])
        for r in p['focus_requests']:
            c=r['physical_candidate'];self.assertEqual((c['id'],c['q'],c['R_outer'],r['identity']['structure']['event_count']),m.FOCUS[r['m']])
            self.assertEqual(r['identity']['candidate']['source_candidate_id'],c['id'])
            self.assertIsNone(r['price']);self.assertFalse(r['model_qualified_for_candidate'])

    def test_changed_kappa_binding_law_compiler_and_axis_recipe_metadata_rejected(self):
        changes=[lambda d:d['inputs'].update(kappa_paper=str(d['conditions']['kappa_paper_approx'])),
                 lambda d:d['inputs']['design']['binding'].update(state='FOREIGN'),
                 lambda d:d['inputs']['design']['profile'].update(compiler='FOREIGN'),
                 lambda d:d['A_schedule']['rows'][0]['identity']['structure'].update(scalar_policy='TWICE')]
        for change in changes:
            d=deepcopy(self.fixed);change(d)
            with self.assertRaises(ValueError):m.prepare(ROOT,integration=I,configuration=d)

    def test_B_future_publication_must_match_whole_schedule_and_two_price_requests(self):
        self.assertEqual(m.verify_B_publication(self.plan,self.config,integration=I)['status'],'EXACT_METADATA_MATCH_ONLY')
        wrong=B.generate(I.a_spec(dict(self.fixed['inputs'],M=11),'PR'))
        with self.assertRaises(ValueError):m.verify_B_publication(self.plan,wrong,integration=I)
        wrong=deepcopy(self.config);wrong['rounds'][0]['law_binding']['definition']['source']={'foreign':True}
        with self.assertRaises(ValueError):m.verify_B_publication(self.plan,wrong,integration=I)

    def test_missing_prices_all13_no_small_q_or_R_fallback(self):
        self.assertEqual(len(self.plan['round_price_coverage']),13)
        self.assertTrue(all(x['price'] is None and not x['qualified_mean_compiled_price'] for x in self.plan['round_price_coverage']))
        focus=[x for x in self.plan['round_price_coverage'] if x['m'] in m.FOCUS]
        self.assertTrue(all(not x['saved_direct_metadata_present'] and not x['saved_model_metadata_present'] for x in focus))
        self.assertIsNone(self.plan['full_rpe_resources'])

    def test_two_axes_full_wrapper_law_source_nulls_and_no_S0a_promotion(self):
        records=V.V2MeanPriceLookup(self.config,integration=I)()
        self.assertEqual(len(records),26)
        for record in records:
            self.assertIsNone(record['mean']['rz_count'])
            self.assertIsNone(record['uncertainty']['joint_covariance'])
            self.assertFalse(record['full_RPE_price_adopted'])
        wrong=dict(records[0],quantity='SINGLE_PATH_STRUCTURAL_PROXY')
        with self.assertRaises(ValueError):V.V2MeanPriceLookup(self.config,[wrong],integration=I)

    def test_no_A_N_or_C_science_or_cost_calls_from_fixed_request_planning(self):
        with patch.object(I.A,'evaluate',side_effect=AssertionError('No N')),\
             patch.object(C,'stream_structure',side_effect=AssertionError('No S')),\
             patch.object(C,'full_wrapper_mean',side_effect=AssertionError('No mean')),\
             patch.object(C,'combine_mean',side_effect=AssertionError('No C prediction')),\
             patch.object(B,'cost',side_effect=AssertionError('No G')),\
             patch.object(B,'select',side_effect=AssertionError('No selection')):
            m.prepare(ROOT,integration=I)

    def test_latest_published_B_uses_same_controller_and_price_adapter(self):
        for path in ('src/trottertracks/pr_rpe_rmse_controller.py','src/trottertracks/pr_rpe_rmse_integration.py'):
            a=subprocess.check_output(['git','show',m.B_RELEASE+':'+path],cwd=ROOT)
            b=subprocess.check_output(['git','show',m.B_V2+':'+path],cwd=ROOT)
            self.assertEqual(a,b)


if __name__=='__main__':unittest.main(verbosity=2)
