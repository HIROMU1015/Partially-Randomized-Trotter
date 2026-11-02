"""Artificial rational inputs only. Never read or evaluate H4 input bindings."""
from decimal import getcontext, localcontext, ROUND_CEILING
from fractions import Fraction as F
from pathlib import Path
import importlib
import json
import runpy
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

pkg = types.ModuleType('h4_pr_price_test_private')
pkg.__path__ = [str(Path(__file__).resolve().parents[1] / 'src/trotterlib')]
sys.modules['h4_pr_price_test_private'] = pkg
from h4_pr_price_test_private import h4_pr_mean_cost_scalar as M


class ScalarPreparationTests(unittest.TestCase):
    def test_sqrt_exact_square_inclusion_at_difficult_rationals(self):
        values = [F(0), F(1, 9), F(2), F(1, 7), F(10**300), F(1, 10**300),
                  F(10**120+1, 10**120), F(10**120-1, 10**120)]
        for value in values:
            with self.subTest(value=value):
                lo, hi = M.sqrt_interval_bounds(value, value)
                self.assertGreaterEqual(lo, 0)
                self.assertLessEqual(lo*lo, value)
                self.assertGreaterEqual(hi*hi, value)
                self.assertLessEqual(lo, hi)

    def test_interval_and_ambient_context(self):
        with localcontext() as ctx:
            ctx.prec, ctx.rounding = 5, ROUND_CEILING
            before = (getcontext().prec, getcontext().rounding)
            lo, hi = M.sqrt_interval_bounds(F(1, 3), F(2, 3))
            self.assertEqual((getcontext().prec, getcontext().rounding), before)
        self.assertLessEqual(lo*lo, F(1, 3))
        self.assertGreaterEqual(hi*hi, F(2, 3))

    def test_invalid_or_non_enclosing_sqrt_stops(self):
        for lower, upper in [(F(-1), F(2)), (F(2), F(1))]:
            with self.assertRaises(ValueError):
                M.sqrt_interval_bounds(lower, upper)
        with self.assertRaises(TypeError):
            M.sqrt_interval_bounds(0.1, F(1))
        with patch.object(M, '_sqrt_endpoint', return_value=F(5)):
            with self.assertRaisesRegex(ArithmeticError, 'SQRT_ENCLOSURE'):
                M.sqrt_interval_bounds(F(2), F(2))

    def test_strict_separation_equality_and_ambiguous_band(self):
        self.assertEqual(M.classify_separation(3, F(4), F(5)),
                         'SUFFICIENT_INEQUALITY_SEPARATED')
        self.assertEqual(M.classify_separation(4, F(4), F(5)),
                         'NUMERICALLY_INCONCLUSIVE')
        self.assertEqual(M.classify_separation(5, F(4), F(5)),
                         'SUFFICIENT_BOUND_NOT_SEPARATED')
        self.assertEqual(M.classify_separation(4, F(4), F(4)),
                         'SUFFICIENT_BOUND_NOT_SEPARATED')

    def test_artificial_batch_reuses_log_and_sqrt_once(self):
        toy = dict(n=1, L=0, q=1, r=1, K=0, n_samples=3,
                   c_min=dict(cosine=10000, sine=10000),
                   delta_axis=dict(cosine='0.2', sine='0.2'), delta_cost='0.4')
        with patch.object(M, 'trans_bounds', wraps=M.trans_bounds) as log:
            with patch.object(M, 'sqrt_interval_bounds', wraps=M.sqrt_interval_bounds) as sqrt:
                result = M.evaluate_scalar_batch(toy)
        log.assert_called_once_with('ln', F(10))
        self.assertEqual(sqrt.call_count, 1)
        self.assertEqual(result['numerical_separation_status'], 'BOTH_AXES_SEPARATED')
        axes = result['axes']
        self.assertEqual(axes['sine']['Pauli_uniform_upper']-axes['cosine']['Pauli_uniform_upper'], 1)
        self.assertEqual(axes['sine']['DF_mean_lower'], axes['cosine']['DF_mean_lower'])
        self.assertFalse(result['unconditional_full_law_guarantee'])
        self.assertIn('UNPROVED', result['count_premise_status'])

    def test_invalid_confidence_stops_before_log(self):
        toy = dict(n=1, L=0, q=1, r=1, K=0, n_samples=3,
                   c_min=dict(cosine=5, sine=5),
                   delta_axis=dict(cosine='0.2', sine='0.3'), delta_cost='0.4')
        with patch.object(M, 'trans_bounds') as log:
            with self.assertRaises(ValueError):
                M.evaluate_scalar_batch(toy)
            log.assert_not_called()

    def test_source_mismatch_stops_in_binding_only_code(self):
        runner = Path(__file__).resolve().parents[1] / 'scripts/run_h4_pr_mean_cost_scalar.py'
        check = runpy.run_path(str(runner))['checked_inputs']
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'toy_source.txt').write_text('artificial source')
            (root / 'input_bindings.json').write_text(json.dumps({
                'sha256': {'toy_source.txt': 'deliberately wrong checksum'}}))
            check.__globals__.update(ROOT=root, PREPARATION=root)
            with self.assertRaisesRegex(ValueError, 'SOURCE_OR_INPUT_MISMATCH'):
                check()

    def test_runner_artificial_output_and_no_overwrite(self):
        root = Path(__file__).resolve().parents[1]
        runner = root / 'scripts/run_h4_pr_mean_cost_scalar.py'
        main = runpy.run_path(str(runner))['main']
        toy = dict(n=1, L=0, q=1, r=1, K=0, n_samples=3,
                   c_min=dict(cosine=10000, sine=10000),
                   delta_axis=dict(cosine='0.2', sine='0.2'), delta_cost='0.4')
        pkg = types.ModuleType('h4_pr_price_private')
        pkg.__path__ = [str(root / 'src/trotterlib')]
        sys.modules['h4_pr_price_private'] = pkg
        kernel = importlib.import_module('h4_pr_price_private.h4_pr_mean_cost_scalar')
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'artificial_result.json'
            with patch.dict(main.__globals__, {'checked_inputs': lambda: (toy, {'sha256': {}})}):
                with patch.object(sys, 'argv', [str(runner), '--output', str(output)]):
                    with patch.object(kernel, 'evaluate_scalar_batch', wraps=kernel.evaluate_scalar_batch) as batch:
                        main()
                        before = output.read_bytes()
                        with self.assertRaises(FileExistsError):
                            main()
                        self.assertEqual(batch.call_count, 1)
                        self.assertEqual(output.read_bytes(), before)
                self.assertEqual(json.loads(before)['results']['numerical_separation_status'],
                                 'BOTH_AXES_SEPARATED')
                failed = Path(folder) / 'artificial_failure.json'
                with patch.object(sys, 'argv', [str(runner), '--output', str(failed)]):
                    with patch.object(kernel, 'evaluate_scalar_batch', side_effect=ArithmeticError('artificial')):
                        with self.assertRaises(ArithmeticError):
                            main()
                error = json.loads(failed.read_text())
                self.assertEqual(error['numerical_separation_status'], 'STOPPED_ON_EXCEPTION')
                self.assertEqual(error['population_claim_scope'], 'NO_NEW_POPULATION_CLAIM')


if __name__ == '__main__':
    unittest.main(verbosity=2)
