"""Metadata, fixed-price arithmetic and reuse tests; no saved-signal risk call."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('_p7_h_budget_test', ROOT/'scripts/run_p7_s2_h_budget_q3_fixed.py')
C = importlib.util.module_from_spec(spec); sys.modules[spec.name] = C; spec.loader.exec_module(C)


class WidthBudgetTests(unittest.TestCase):
    def test_fixed_inputs_prices_caps_and_no_risk(self):
        with patch.object(C.T,'evaluate',side_effect=AssertionError('No saved H4 risk')) as risk:
            x = C.build_inputs()
        self.assertEqual(risk.call_count,0)
        self.assertEqual(x['actual_new_call_limit'],2)
        self.assertEqual(x['caps_N0'],[59,55,51,47,43,39,35,31,27,23,19,15,11])
        expected = {
            'A_SHAPE': ([21,19,18,17,15,14,12,11,9,8,6,5,3],421323290),
            'B_PRICE': ([59,55,51,47,43,39,35,31,27,16,7,2,1],434203864)}
        for case in x['cases']:
            vector,cost = expected[case['rule']]
            self.assertEqual([r['N_X'] for r in case['allocation']['shots']],vector)
            self.assertEqual([r['N_Y'] for r in case['allocation']['shots']],vector)
            self.assertEqual(case['conditional_cost'],cost)
            self.assertGreater(C.F.T.B.a.exact_scalar(case['generation']['next_breakpoint_cost']),x['budget_RZ'])
            self.assertEqual(case['configuration_id'],C.C.CID)
            self.assertEqual(len(case['price_terms']),26)
            self.assertEqual(sum(t['origin']=='DIRECT_COMPILED' for t in case['price_terms']),10)
            self.assertEqual(sum(t['origin']=='ADOPTED_RULE_EXTRAPOLATION' for t in case['price_terms']),16)
            self.assertEqual(sum(t['N_times_C'] for t in case['price_terms']),cost)
            self.assertTrue(all(1<=r['N_X']<=x['caps_N0'][r['m']] for r in case['allocation']['shots']))
            self.assertLessEqual(case['finite_scope_preflight']['transition_pairs_upper'],141559036)

    def test_price_identity_and_origin_mismatch_rejected_before_risk(self):
        prior = C.F.document(C.BASE,C.C.INPUT);base = prior['cases'][0]
        binding,view = C.F.metadata(base)
        table = C.F.document(C.PRICE,C.PRICE_PATH);before = deepcopy((table,base,view))
        for field,value,index in [('physical_id','FOREIGN_PHYSICS',0),('configuration_id','COARSE_ID',0),
                ('origin','DIRECT_COMPILED',25)]:
            bad = deepcopy(table);bad['rows'][index][field] = value
            with self.assertRaises(ValueError):C.joined_prices(bad,base,binding,view)
        self.assertEqual((table,base,view),before)

    def test_identical_rule_vectors_share_one_scientific_call(self):
        original = C.F.generate
        def same(caps,prices,budget,rule): return original(caps,prices,budget,'A_SHAPE')
        with patch.object(C.F,'generate',side_effect=same),patch.object(C.T,'evaluate',side_effect=AssertionError('No risk')):
            x = C.build_inputs()
        self.assertEqual(len(x['cases']),1)
        self.assertEqual(x['proposals'][1]['disposition'],'SAME_VECTOR_AS_OTHER_RULE')
        self.assertEqual(x['proposals'][1]['reused_result'],x['cases'][0]['label'])


if __name__=='__main__':unittest.main(verbosity=2)
