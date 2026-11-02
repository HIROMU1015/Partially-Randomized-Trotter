"""Deterministic artificial algebra tests plus fixed-result consistency checks."""
import ast
import csv
from decimal import Decimal, localcontext
import json
import math
from pathlib import Path
import unittest
import analyze_fixed_plan as r1

HERE=Path(__file__).resolve().parent


class ArtificialAlgebra(unittest.TestCase):
    def test_shared_anchor_coefficients_equal_direct_sum(self):
        q=[1,2,8];n=[2,3,5];c1,c2=3,5
        direct=sum(nn*((2-qq)*c1+(qq-1)*c2) for nn,qq in zip(n,q))
        a1=sum(nn*(2-qq) for nn,qq in zip(n,q));a2=sum(nn*(qq-1) for nn,qq in zip(n,q))
        self.assertEqual((a1,a2),(-28,38));self.assertEqual(direct,106)
        self.assertEqual(a1*c1+a2*c2,direct)

    def test_shared_observation_is_not_independent_repetition(self):
        shared_variance=(2+3)**2*4
        mistaken_independent_round_variance=(2**2+3**2)*4
        self.assertEqual(shared_variance,100)
        self.assertGreater(shared_variance,mistaken_independent_round_variance)

    def test_psd_envelope_two_terms_is_achievable(self):
        self.assertEqual(r1.covariance_envelope([2,-3]),(1,5))
        self.assertEqual(r1.covariance_envelope([1,1,1]),(0,3))
        self.assertEqual(r1.covariance_envelope([0,0]),(0,0))

    def test_pairing_and_anchor_signs(self):
        values=[-2,-2,3,3]
        self.assertEqual(r1.correlated_block_variance(values,0,0),26)
        self.assertEqual(r1.correlated_block_variance(values,0,1),52)
        self.assertEqual(r1.correlated_block_variance(values,-1,1),100)
        self.assertEqual(r1.correlated_block_variance(values,1,1),4)
        self.assertEqual(r1.correlated_block_variance(values,0,-1),0)
        with self.assertRaises(ValueError):r1.correlated_block_variance(values,1.01,0)

    def test_verdict_distinguishes_tie_and_separation(self):
        self.assertEqual(r1.difference_band(-4,3)['verdict'],'CONDITIONAL_SEPARATION')
        self.assertEqual(r1.difference_band(-4,4)['verdict'],'UNDETERMINED')
        self.assertEqual(r1.difference_band(4,3)['separated_candidate'],'Full')


class SavedResultConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a=json.loads((HERE/'DEPENDENCE_AND_SENSITIVITY.json').read_text())
        cls.v=json.loads((HERE/'VALIDATION.json').read_text())
        with (HERE/'R1_FIXED_PLAN_RESULTS.csv').open() as f:
            cls.rows=list(csv.DictReader(f))

    def test_fixed_totals_and_independent_decimal_sum(self):
        for label in ['Partial','Full']:
            rows=[r for r in self.rows if r['candidate']==label]
            decimal=sum((Decimal(r['coefficient_q1'])*Decimal(r['calibration_q1_mean'])+
                Decimal(r['coefficient_q2'])*Decimal(r['calibration_q2_mean']) for r in rows),Decimal(0))
            self.assertEqual(decimal,Decimal(self.a['reproduction']['decimal_exact_from_saved_decimal_means'][label]))
            self.assertEqual(float(decimal),self.a['reproduction']['saved_G'][label])
            self.assertEqual(float(decimal),self.a['reproduction']['aggregate_group_costs'][label])
        self.assertEqual(self.a['reproduction']['max_unit_price_abs_error'],0)
        self.assertEqual(self.a['reproduction']['max_unit_SE_abs_error'],0)

    def test_all_axes_and_coefficients(self):
        self.assertEqual(len(self.rows),72)
        self.assertEqual(len({(r['candidate'],r['round_index'],r['axis']) for r in self.rows}),72)
        groups={g['group_id']:g for g in self.a['shared_calibration_groups']}
        self.assertEqual(len(groups),14)
        for gid,g in groups.items():
            rows=[r for r in self.rows if r['group_id']==gid]
            for field,key in [('coefficient_q1','A1'),('coefficient_q2','A2')]:
                self.assertEqual(sum(int(r[field]) for r in rows),g[key])

    def test_independent_variance_and_psd_range_decimal_check(self):
        gs=self.a['shared_calibration_groups']
        with localcontext() as ctx:
            ctx.prec=50
            pieces=[Decimal(w)*Decimal(str(se)) for g in gs for w,se in zip(g['signed_D_anchor_weights'],g['anchor_SE'])]
            expected=sum((x*x for x in pieces),Decimal(0)).sqrt()
            actual=self.a['sampling']['shared_rounds_all_calibration_anchors_independent_REFERENCE']['SE']
            self.assertTrue(math.isclose(float(expected),actual,rel_tol=3e-16))
        upper=self.a['sampling']['unknown_covariance_PSD_diagonal_envelope']['SE_maximum']
        self.assertGreaterEqual(upper,actual)
        for s in self.a['sampling']['hypothetical_pairing_and_anchor_correlation_scenarios']:
            self.assertLessEqual(s['SE'],upper*(1+1e-15))

    def test_original_scenarios_reproduced(self):
        rec=self.a['reproduction'];new=rec['source_comparison_rebuilt'];old=rec['original_comparison']
        for key in ['sampling_uncertainty','measured_model_discrepancy_half_width','scenarios']:
            self.assertEqual(new[key],old[key])

    def test_break_even_signs(self):
        d=self.a['reproduction']['D']
        for t in self.a['break_even_sensitivity']:
            eps=t['point_reversal_fraction'];slope=t['adverse_D_slope']
            self.assertAlmostEqual(d+eps*slope,0,delta=2e-5)
            self.assertLess(d+.999*eps*slope,0)
            self.assertGreater(d+1.001*eps*slope,0)

    def test_unknowns_not_zero_and_no_optimistic_claim(self):
        self.assertIsNone(self.a['unknown_long_q_systematic']['finite_empirical_upper_bound'])
        self.assertEqual(self.a['unknown_long_q_systematic']['overall_cost_superiority'],'UNDETERMINED')
        self.assertEqual(self.a['sampling']['unknown_covariance_PSD_diagonal_envelope']['verdict'],'UNDETERMINED')
        for t in self.a['break_even_sensitivity']:
            self.assertIsNone(t['max_symmetric_fraction_for_unknown_PSD_sampling'])
        for s in self.a['systematics']:
            self.assertFalse(s['empirical_uniform_long_q_bound'])
            self.assertEqual(s['variants']['plus_saved_source_sampling']['verdict'],'UNDETERMINED')

    def test_fresh_replacement_and_A0_r_mismatch(self):
        for g in self.a['shared_calibration_groups']:
            if g['candidate']=='Partial':
                self.assertEqual(g['anchor_sample_counts'],[32,32] if g['r'] in (8,16,32) else [8,8])
        for row in self.rows:
            if row['candidate']=='Partial' and row['q'] in ('16','32'):
                self.assertEqual(row['r'],'1');self.assertEqual(row['holdout_q'],'8')

    def test_source_reuse_has_no_scientific_import_or_runner(self):
        names={x['name'] for x in self.v['reused_unchanged_definitions']}
        self.assertIn('_comparison',names);self.assertIn('AffineAxisCostModel',names)
        self.assertNotIn('_grid_search',names);self.assertNotIn('prepare_df_partial_s2',names)
        self.assertEqual(self.v['source_module_imports'],0)
        self.assertTrue(all(v==0 for v in self.v['prohibited_calls'].values()))
        imports=[n for n in ast.walk(ast.parse((HERE/'analyze_fixed_plan.py').read_text())) if isinstance(n,(ast.Import,ast.ImportFrom))]
        self.assertFalse(any('trotterlib' in ast.unparse(n) or 'qiskit' in ast.unparse(n) for n in imports))


if __name__=='__main__':unittest.main(verbosity=2)
