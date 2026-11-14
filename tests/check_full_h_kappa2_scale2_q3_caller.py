"""Fixed scale2 metadata only; real H4 risk must never be called in tests."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_scale2_q3_caller',ROOT/'scripts/run_full_h_kappa2_scale2_q3_fixed.py')
M=importlib.util.module_from_spec(spec);sys.modules[spec.name]=M;spec.loader.exec_module(M)


class Scale2Tests(unittest.TestCase):
    def test_metadata_and_integer_scope_without_risk(self):
        with patch.object(M.M.T,'evaluate',side_effect=AssertionError('No real H4 science'))as call:
            inputs=M.build_inputs(); M.validate_inputs(inputs)
        self.assertEqual(call.call_count,0)
        self.assertEqual(len(inputs['cases']),1);case=inputs['cases'][0]
        self.assertEqual(case['label'],'FULL_H_KAPPA2')
        for axis in ('X','Y'):
            self.assertEqual([r['N_'+axis]for r in case['allocation']['shots']],M.VECTOR)
        self.assertEqual(inputs['maximum_scientific_calls'],1)
        self.assertNotIn('first_round_full_support_states',inputs['finite_scope_preflight'])
        self.assertEqual(inputs['finite_scope_preflight']['transition_pairs_upper'],620478721)
        self.assertEqual(inputs['options']['maximum_round_outcomes'],16641)
        self.assertIsNone(inputs['sufficient_N']);self.assertIsNone(inputs['G_qualified'])

    def test_different_vector_scope_or_physical_ID_rejected_without_risk(self):
        with patch.object(M.M.T,'evaluate',side_effect=AssertionError('No real H4 science'))as call:
            inputs=M.build_inputs()
            changes=[('vector',lambda x:x['cases'][0]['allocation']['shots'][0].update(N_X=64)),
                     ('scope',lambda x:x['options']['total_state_refinement'].update(maximum_transition_pairs=141559036)),
                     ('ID',lambda x:x['cases'][0].update(total_configuration_id='FOREIGN'))]
            for label,change in changes:
                bad=deepcopy(inputs);change(bad)
                with self.subTest(label=label),self.assertRaisesRegex(ValueError,'one fixed proposal'):
                    M.validate_inputs(bad)
        self.assertEqual(call.call_count,0)


if __name__=='__main__':unittest.main(verbosity=2)
