"""One-case metadata/cache regression. No real H4 risk evaluation."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_m12_two_test',ROOT/'scripts/run_p7_s2_m12_two_fixed.py')
C=importlib.util.module_from_spec(spec);sys.modules[spec.name]=C;spec.loader.exec_module(C)


class FixedTwoTests(unittest.TestCase):
    def test_only_fixed_second_candidate_no_search_or_science(self):
        with patch.object(C.T,'evaluate',side_effect=AssertionError('No real risk')) as risk,\
                patch.object(C.C,'search',side_effect=AssertionError('No search')):
            inputs=C.build_inputs()
        self.assertEqual(risk.call_count,0);self.assertEqual(len(inputs['cases']),1)
        case=inputs['cases'][0]
        self.assertEqual([v['N_X'] for v in case['allocation']['shots']],C.VECTOR)
        self.assertEqual([v['N_Y'] for v in case['allocation']['shots']],C.VECTOR)
        self.assertEqual(case['conditional_cost'],395962678)
        self.assertEqual(inputs['maximum_scientific_calls'],1)
        self.assertIsNone(C.C.cached_record(case,inputs))
        self.assertEqual(inputs['previous_achieved_cost_RZ'],415721884)

    def test_foreign_configuration_and_law_rejected(self):
        inputs=C.build_inputs();three=deepcopy(inputs['saved_cache'][-1]['case'])
        three['configuration_id']='FOREIGN'
        with self.assertRaises(ValueError):C.C.cached_record(three,inputs)
        bad=deepcopy(inputs);bad['options']['probability_rule']='FOREIGN'
        with self.assertRaises(ValueError):C.C.cached_record(inputs['saved_cache'][-1]['case'],bad)


if __name__=='__main__':unittest.main(verbosity=2)
