"""Artificial S_STATS fixtures only; no real saved statistics are loaded."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import h4_s_stats as s


def fixture():
    cells = [dict(id='artificial_A', prefix=2, q=1, r=4, R=4, K=2, method='B2', formula='2nd'),
             dict(id='artificial_B', prefix=3, q=2, r=4, R=8, K=2, method='B2', formula='2nd'),
             dict(id='artificial_B_K4', prefix=3, q=2, r=4, R=8, K=4, method='B2', formula='2nd')]
    assignments = [dict(cell_id=cid, epsilon=eps, role=role, cohort='ARTIFICIAL') for cid, eps, role in
        [('artificial_A', .05, 'PRIMARY'), ('artificial_A', .01, 'PRIMARY'),
         ('artificial_B', .05, 'PRIMARY'), ('artificial_B_K4', .05, 'BOUNDARY_K_PARTNER_DIAGNOSTIC')]]
    stats = [dict(x, saved_mean_exact='80', saved_paired_SE=(.1 if x['cell_id'].endswith('K4') else
        2. if x['epsilon'] == .01 else 1.), saved_n=4, saved_shots=dict(cosine=2, sine=3)) for x in assignments]
    pf = {str(e): dict(best=dict(cell_id='artificial_PF', G_exact=str(cost), shots=dict(cosine=2, sine=3)))
          for e, cost in [(.05, 100), (.01, 200)]}
    return dict(candidates=cells, assignments=assignments, saved_statistics=stats), pf


class NumericalDesign(unittest.TestCase):
    def test_known_normal_quantiles_and_integer_ceiling(self):
        reg, pf = fixture(); result = s.analyze(reg, pf)
        rows = [x for x in result['normal_scenarios'] if x['cell_id'] == 'artificial_A'
                and x['epsilon'] == .05 and x['eta'] == .02]
        self.assertEqual([x['n'] for x in rows], [12, 47, 187])
        self.assertEqual(result['recommended']['n_fixed']['artificial_A'], 187)

    def test_same_candidate_multi_epsilon_not_double_counted(self):
        reg, pf = fixture(); result = s.analyze(reg, pf)
        self.assertEqual(len(result['normal_scenarios']), 24)
        self.assertEqual(len(result['reference_directions']), 4)
        calls = result['recommended']['proposed_P_calls']
        self.assertEqual(calls['trajectory'], 561)
        self.assertEqual(calls['wrapper_compile'], 1122)
        self.assertEqual(calls['occurrence'], 935)
        self.assertEqual(calls['RTE_event'], 3740)
        self.assertEqual(calls['seed_slots'], 1309)

    def test_diagnostic_preserves_K2_sample_floor(self):
        reg, pf = fixture(); result = s.analyze(reg, pf)
        diag = next(x for x in result['recommended']['candidates'] if x['role'] != 'PRIMARY')
        self.assertEqual(diag['own_endpoint_max_n'], 2)
        self.assertEqual(diag['K2_partner'], 'artificial_B')
        self.assertEqual(diag['recommended_n'], 187)

    def test_exact_fraction_direction_is_not_rounded_to_tie(self):
        reg, pf = fixture()
        reg['saved_statistics'][0]['saved_mean_exact'] = '99999999999999999999/1000000000000000000'
        reg['saved_statistics'][1]['saved_mean_exact'] = '201'
        reg['saved_statistics'][2]['saved_mean_exact'] = '100'
        directions = s.analyze(reg, pf)['reference_directions']
        self.assertEqual([x['expected_direction'] for x in directions], [1, -1, 0, 1])
        self.assertEqual(directions[0]['PF_minus_saved_PR_exact'], '1/1000000000000000000')

    def test_no_inputs_mutated_and_no_tail_probability_invented(self):
        reg, pf = fixture(); before = deepcopy((reg, pf)); result = s.analyze(reg, pf)
        self.assertEqual((reg, pf), before)
        self.assertIsNone(result['unknown_tail_cost'])
        self.assertTrue(all(v['pi4'] is None for v in result['law_visibility'].values()))
        self.assertFalse(result['population_mean_certified'])

    def test_nonfinite_SE_rejected(self):
        reg, pf = fixture(); reg['saved_statistics'][0]['saved_paired_SE'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'POSITIVE_SAVED_PAIRED_SE'):s.analyze(reg, pf)

    def test_duplicate_endpoint_rejected(self):
        reg, pf = fixture(); reg['assignments'].append(reg['assignments'][0])
        with self.assertRaisesRegex(ValueError, 'FIXED_SUMMARY_COVERAGE'):s.analyze(reg, pf)

    def test_all_sensitivity_scenarios_preserve_n_cap_none(self):
        reg, pf = fixture(); result = s.analyze(reg, pf)
        self.assertEqual(len(result['sensitivity']), 6)
        self.assertIsNone(result['settings']['n_cap'])
        self.assertTrue(all(x['n'] >= 2 for x in result['normal_scenarios']))


class ApprovalAndOneShot(unittest.TestCase):
    def test_preparation_never_calls_numerical_design(self):
        import ast
        path=Path(__file__).resolve().parents[1]/'scripts/prepare_h4_s_stats.py'
        tree=ast.parse(path.read_text())
        prohibited={'analyze', 'normal_n_formula', 'evaluate_pairs', 'bootstrap', 'sample', 'transpile'}
        called={n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id
                for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, (ast.Name, ast.Attribute))}
        self.assertTrue(called.isdisjoint(prohibited))

    def test_P_grant_cannot_authorize_stats(self):
        with patch.object(s, 'verify_bytes', side_effect=AssertionError('NO_INPUT_READ')):
            with self.assertRaisesRegex(ValueError, 'S_STATS_ONLY'):
                s.authorize(dict(stage='S_STATS'), dict(stage='P'))

    def test_DRAFT_rejected_before_input_or_output(self):
        with patch.object(s, 'verify_bytes', side_effect=AssertionError('NO_INPUT_READ')):
            with self.assertRaisesRegex(ValueError, 'DRAFT_UNAUTHORIZED'):
                s.authorize(dict(stage='S_STATS', status='DRAFT_UNAUTHORIZED'), dict(stage='S_STATS'))

    def test_input_corruption_rejected_before_Git_or_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'artificial.json';path.write_text('{}')
            spec = dict(inputs=[dict(path=str(path), sha256='0'*64, role='artificial')])
            with self.assertRaisesRegex(ValueError, 'INPUT_CHANGED'):s.verify_bytes(spec)

    def test_one_shot_cannot_compute_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = dict(SOURCE='a'*40, output=str(root/'output'), lock=str(root/'ONE_SHOT.json'),
                        inputs=[dict(role='D_result', sha256='b'*64)])
            result = dict(normal_scenarios=[{}]*84, reference_directions=[{}]*14,
                          recommended=dict(n_fixed={str(i): 2 for i in range(9)}))
            with patch.object(s, 'authorize', return_value=fixture()), patch.object(s, 'verify_bytes'), \
                 patch.object(s, 'analyze', return_value=result) as analyze:
                s.execute(spec, {})
                with self.assertRaises(FileExistsError):s.execute(spec, {})
                self.assertEqual(analyze.call_count, 1)
            self.assertEqual(json.loads((root/'output/terminal.json').read_text())['status'], 'COMPLETE_MANDATORY_STOP')

    def test_failure_preserves_reserved_pass_and_STOP(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = dict(SOURCE='a'*40, output=str(root/'output'), lock=str(root/'ONE_SHOT.json'))
            with patch.object(s, 'authorize', return_value=fixture()), \
                 patch.object(s, 'analyze', side_effect=ValueError('ARTIFICIAL_FAILURE')):
                with self.assertRaisesRegex(ValueError, 'ARTIFICIAL_FAILURE'):s.execute(spec, {})
            self.assertTrue((root/'ONE_SHOT.json').exists())
            self.assertEqual(json.loads((root/'output/calls_ledger.json').read_text())['numeric_design_pass'], 1)
            self.assertFalse(json.loads((root/'output/STOP.json').read_text())['retry'])
            self.assertFalse((root/'output/result.json').exists())


if __name__ == '__main__':unittest.main()
