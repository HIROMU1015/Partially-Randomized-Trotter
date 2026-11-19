"""Exact finite mixed-method request translation, with synthetic metadata only."""
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests')]
from test_rpe_local0_search import fixture_config
from trottertracks.rpe_local0_search import RequestError, resolve, load, write


class DiscreteEntry(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.config = fixture_config(Path(self.temporary.name))
        self.forms = ['2nd', '4th', '4th(m5_best)']
        problem = load(self.config['problem'])
        problem['methods'] = {
            'FULL_PF': dict(family='PF', formulas=self.forms),
            'TRUNCATED_PF': dict(family='PF', formulas=self.forms),
            'PR_DF': dict(family='PR', generation=dict(formulas=['2nd'], supported_formulas=['2nd']))}
        write(self.config['problem'], problem)
        spec = load(self.config['round'])
        spec.update(methods=list(problem['methods']), method_overrides={
            'TRUNCATED_PF': dict(configuration_ids=['old-restricted-id'])},
            max_new_q3_per_method={m: 0 for m in problem['methods']},
            search_overrides=dict(schedule_wide={}, requested_widths={'PR_DF': [0.2]}),
            executor=dict(prior_run_dirs=['preserved-prior']))
        write(self.config['round'], spec)
        self.config.update(entry='schedule', absolute_kappa_schedule=True,
                           combined_width_search=True,
                           absolute_kappa_defaults=dict(h=dict(numerator=3602879701896397,
                                                              denominator=36028797018963968)))
        self.request = dict(action='preview', scope=dict(methods=list(problem['methods']), per_method={
            'FULL_PF': dict(formulas=self.forms, delta_over_h_values=['2', '4', '8', '16']),
            'TRUNCATED_PF': dict(formulas=self.forms, p=[6, 7, 8, 9], delta_over_h_values=['2', '4', '8', '16']),
            'PR_DF': dict(formulas=['2nd'], L_D=[5, 6, 7, 8, 9], delta_over_h_values=['1/2', '1'],
                          kappa=dict(mode='absolute', values=['1', '2', '4', '8', '16', '32', '64']))},
            stop=dict(max_new_q3_per_method=0)))

    def test_mixed_grid_keeps_exact_values_and_expands_old_pf_allowlist(self):
        before = {k: Path(self.config[k]).read_bytes() for k in ('problem', 'round')}
        problem, spec, canonical = resolve(self.config, self.request)
        for method in ('FULL_PF', 'TRUNCATED_PF'):
            self.assertEqual(spec['method_overrides'][method]['delta_over_h_values'], ['2', '4', '8', '16'])
            self.assertIsNone(spec['method_overrides'][method]['configuration_ids'])
        self.assertEqual(spec['method_overrides']['FULL_PF']['p_values'], [10])
        self.assertEqual(spec['method_overrides']['TRUNCATED_PF']['p_values'], [6, 7, 8, 9])
        values = ['1', '2', '4', '8', '16', '32', '64']
        self.assertEqual(spec['method_overrides']['PR_DF']['absolute_kappa_values'], values)
        self.assertEqual(spec['search_overrides']['schedule_wide_by_method']['PR_DF']['kappa_values'], values)
        self.assertEqual(canonical['per_method']['PR_DF']['kappa'], dict(mode='absolute', values=values))
        self.assertEqual(spec['method_overrides']['PR_DF']['generation']['formulas'], ['2nd'])
        widths = spec['search_overrides']['requested_widths']['PR_DF']
        fractions = [Fraction(v['numerator'], v['denominator']) for v in widths]
        h = Fraction(3602879701896397, 36028797018963968)
        self.assertEqual(fractions, [h / 2, h])
        self.assertEqual(spec['executor']['prior_run_dirs'], ['preserved-prior'])
        self.assertFalse(spec['science_approved'])
        self.assertEqual(before, {k: Path(self.config[k]).read_bytes() for k in before})

    def test_rejects_inexact_grid_and_conflicting_representations(self):
        for edit in ('float-width', 'float-kappa', 'both-widths', 'both-kappas'):
            value = deepcopy(self.request)
            pf = value['scope']['per_method']['FULL_PF']
            pr = value['scope']['per_method']['PR_DF']
            if edit == 'float-width': pf['delta_over_h_values'] = [0.2]
            elif edit == 'float-kappa': pr['kappa']['values'] = [1.0]
            elif edit == 'both-widths': pf['delta_over_h'] = ['2', '16']
            else: pr['kappa']['min'] = '1'
            with self.subTest(edit=edit), self.assertRaises(RequestError):
                resolve(self.config, value)

    def test_start_retains_explicit_zero_budget(self):
        value = deepcopy(self.request)
        value['action'] = 'start'
        _, spec, _ = resolve(self.config, value)
        self.assertTrue(spec['science_approved'])
        self.assertEqual(spec['max_new_q3_per_method'], {m: 0 for m in value['scope']['methods']})


if __name__ == '__main__':
    unittest.main()
