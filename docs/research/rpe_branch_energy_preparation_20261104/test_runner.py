"""Runner integration with an artificial 18-round fixture, no H4 evaluation."""
import contextlib
import io
import json
import unittest
from unittest.mock import patch

from flint import ctx
import adapter as a
import run_limited_validation as run


def fixture():
    rows = [dict(m=m, q=1 << m, mu0={'real': '1', 'imag': '0'},
                 axes={axis: dict(N='100000', alpha='0.001', epsilon='0.01')
                       for axis in ('cosine', 'sine')}) for m in range(18)]
    constants = dict(energy_reference_binary64_decimal='0', delta_binary64_decimal='0.125',
                     alpha_total='0.05', beta_rpe='0.4', target_energy_precision_ha='0.01')
    return rows, constants


class RunnerTests(unittest.TestCase):
    def test_toy_end_to_end_is_ideal_conditional_only(self):
        with ctx.workprec(160):
            result = run.evaluate(*fixture())
        self.assertEqual(result['mathematical_literal_reference_decision'], 'CONDITIONAL_IDEAL_DECODER_SUFFICIENT')
        self.assertEqual(result['implementation_decoder_decision'], 'UNKNOWN/STOP')
        self.assertEqual(result['physical_ground_energy_decision'], 'UNKNOWN/STOP')
        self.assertIsNone(result['energy']['physical_energy_upper'])
        self.assertFalse(result['RPE_tolerance_certified'])
        self.assertTrue(all(x == 0 for x in result['prohibited_calls'].values()))

    def test_fixed_alpha_or_target_failure_is_not_retuned(self):
        rows, constants = fixture()
        for change in (dict(alpha_total='0.03'), dict(target_energy_precision_ha='1e-12')):
            with self.subTest(change=change), ctx.workprec(160):
                result = run.evaluate(rows, dict(constants, **change))
                self.assertEqual(result['mathematical_literal_reference_decision'], 'UNKNOWN/STOP')
                self.assertEqual(result['execution_status'], 'COMPLETED_MANDATORY_STOP')

    def test_unresolved_round_geometry_stops_propagation(self):
        rows, constants = fixture()
        rows[5]['axes']['cosine']['epsilon'] = '1'
        with ctx.workprec(160):
            result = run.evaluate(rows, constants)
        self.assertIsNone(result['branch'])
        self.assertIsNone(result['energy'])
        self.assertEqual(result['mathematical_literal_reference_decision'], 'UNKNOWN/STOP')

    def test_default_mode_cannot_call_numeric_evaluation(self):
        rows, _ = fixture()
        with patch.object(run, 'metadata', return_value=({'precision_bits': 160}, rows)), \
             patch.object(run, 'evaluate', side_effect=AssertionError('science forbidden')), \
             patch.object(a, 'round_geometry', side_effect=AssertionError('science forbidden')), \
             patch('sys.argv', ['run_limited_validation.py']), contextlib.redirect_stdout(io.StringIO()) as out:
            run.main()
        result = json.loads(out.getvalue())
        self.assertEqual(result['new_H4_numeric_calls'], 0)
        self.assertEqual(result['preparation_status'], 'PREPARED_NOT_SCIENTIFICALLY_EXECUTED')


if __name__ == '__main__':
    unittest.main()
