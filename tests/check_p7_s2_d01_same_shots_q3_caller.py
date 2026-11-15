"""Metadata and toy interval checks only; never evaluate saved H4 signals."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_p7_d01_summary_test',ROOT/'scripts/summarize_p7_s2_d01_same_shots_q3_fixed.py')
S=importlib.util.module_from_spec(spec);sys.modules[spec.name]=S;spec.loader.exec_module(S)
C=S.C


class FineWidthTests(unittest.TestCase):
    def test_actual_metadata_route_keeps_fine_schedule_and_no_price(self):
        with patch.object(C.T,'evaluate',side_effect=AssertionError('No H4 risk')) as risk:
            x=C.build_inputs()
        self.assertEqual(risk.call_count,0);self.assertEqual(x['maximum_scientific_calls'],1)
        c=x['cases'][0]
        self.assertEqual(c['configuration_id'],C.CID)
        self.assertEqual([r['q_PF'] for r in c['physical_rows']],[1<<m for m in range(13)])
        self.assertEqual([r['N_X'] for r in c['allocation']['shots']],C.VECTOR)
        self.assertEqual([r['N_Y'] for r in c['allocation']['shots']],C.VECTOR)
        self.assertNotIn('price_terms',c);self.assertIsNone(x['conditional_cost'])
        self.assertEqual(x['comparison']['coarse_cost_applies_to'],'S2_16h_over_7_ONLY')
        self.assertEqual(x['options']['maximum_verification_pairs'],55043173)

    def test_toy_interval_difference_and_margin_sign(self):
        x=S.intervals([2,3],[5,7],10)
        self.assertEqual([S.exact(v) for v in x['MSE_difference_fine_minus_coarse']],[-5,-2])
        self.assertEqual([S.exact(v) for v in x['fine_margin']],[7,8])
        self.assertEqual([S.exact(v) for v in x['coarse_margin']],[3,5])
        self.assertEqual([S.exact(v) for v in x['margin_gain']],[2,5])
        self.assertEqual(x['margin_change'],'INCREASED')
        self.assertEqual(S.intervals([4,6],[5,7],10)['margin_change'],'INTERVALS_OVERLAP')
        with self.assertRaises(ValueError):S.intervals([3,2],[5,7],10)

    def test_coarse_packet_or_false_original_id_rejected_before_risk(self):
        x=C.build_inputs();c=x['cases'][0];c['configuration_id']='COARSE_IS_NOT_THE_FINE_CONFIGURATION'
        with patch.object(C.T,'evaluate',side_effect=AssertionError('No risk')) as risk:
            with self.assertRaises(ValueError):C.F.metadata(c)
        self.assertEqual(risk.call_count,0)


if __name__=='__main__':unittest.main(verbosity=2)
