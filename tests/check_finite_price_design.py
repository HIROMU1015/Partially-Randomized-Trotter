"""Artificial checks: plan-time identifiability and admission to C's unchanged price_model_validation.

Fictional prices only (exact Fractions from declared coefficients). No compile, sampling,
real price, signal or fit on real data. C's module is executed from its c628e9e8 blob.
"""
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src')]
from trottertracks import finite_price_design as D  # noqa: E402

M = D.c_model(ROOT)
PROFILE = dict(compiler='1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b', P=0)
BASE = dict(route='FULL_H_RTE', quantity='FULL_WRAPPER_MEAN_COMPILED_PRICE', kind='DIRECT_COMPILED_FINITE_SAMPLE',
            law=dict(model='ideal_finite_RTE_law', K=2, price_law='PCG64_numpy_float64_CDF_saved_cohort'),
            implementation=dict(lowering_id='CONTROLLED_ZZ_CX_CRZ_CX_V1', native='H46_LONG_TIME_NATIVE_COMPOSE',
                                scalar_quantity='HAMILTONIAN_CONSTANT_PLUS_EXTRACTED_FULL_H_IDENTITY'),
            profile=PROFILE, expectation='FINITE_COHORT_ONLY')
SOURCE = dict(fixture='ARTIFICIAL')
COEF = dict(one=Fraction(1234), event_count=Fraction(321, 2), outer_boundary_count=Fraction(57))


def cell(cid, q, R, n=2, role='CALIBRATION', **kw):
    return dict(BASE, cell_id=cid, role=role, q=q, R=R, n_paths=n, **kw)


def plan(cells, fm='FULL_H_GROUP_V1'):
    return dict(feature_map=fm, cells=cells, design_source=dict(fixture='ARTIFICIAL_DESIGN'))


def price(c, fm='FULL_H_GROUP_V1', coef=COEF):
    f = D.features(c, fm)
    return sum(coef[k] * v for k, v in f.items())


def record(c, axis='X', fm='FULL_H_GROUP_V1', **over):
    kw = dict(cell_id=c['cell_id'], role=c['role'], route=c['route'], axis=axis, features=D.features(c, fm),
              rz_count=price(c, fm), kind=c['kind'], source=SOURCE, implementation=c['implementation'],
              profile=c['profile'], n_paths=c['n_paths'], law=c['law'])
    kw.update(over)
    return M.record(**kw)


Q1 = [cell('G566', 1, 566), cell('G4528', 1, 4528), cell('G36223', 1, 36223)]
HOLD = [cell('M5_ROUND', 32, 283, role='HOLDOUT'), cell('M6_ROUND', 64, 566, n=1, role='HOLDOUT')]
D7 = Q1 + [cell('Q8R566', 8, 566), cell('P2R566', 2, 566, n=1, composed_of=['G566#p0#g0', 'G566#p1#g0'])]


class Tests(unittest.TestCase):
    def test_q1_only_is_not_identified_and_c_fit_also_refuses(self):
        ident = D.identifiability(Q1, 'FULL_H_GROUP_V1')
        self.assertEqual((ident['rank'], ident['reason']), (2, 'ALL_Q_EQUAL_ONE_BOUNDARY_OR_STEP_COLUMN_NOT_SEPARABLE'))
        x = D.design_matrix(Q1, 'FULL_H_GROUP_V1')
        null = [Fraction(v) for v in ident['null_direction']]
        self.assertTrue(all(sum(a * b for a, b in zip(row, null)) == 0 for row in x))
        with self.assertRaisesRegex(ValueError, 'Unidentified calibration design'):
            D.check_plan(plan(Q1 + HOLD))
        records = [record(c) for c in Q1]
        with self.assertRaisesRegex(ValueError, 'singular'):         # existing C fit, after prices exist
            M.fit(records, features=['one', 'event_count', 'outer_boundary_count'], design_source=SOURCE)
        with self.assertRaisesRegex(ValueError, 'Unidentified calibration design'):  # our gate, before C's fit
            D.fit_identified(ROOT, records, plan=plan(Q1 + HOLD), axis='X')

    def test_dependent_q_designs_are_rejected(self):
        fixed_r = [cell('G566', 1, 566), cell('Q2', 2, 566), cell('Q4', 4, 566)]
        self.assertEqual(D.identifiability(fixed_r, 'FULL_H_GROUP_V1')['reason'], 'FIXED_R_EVENT_COUNT_IS_AFFINE_IN_Q')
        line = [cell('A', 1, 12), cell('B', 2, 9), cell('C', 3, 8), cell('E', 6, 7)]   # s = 12 + 6(q-1)
        ident = D.identifiability(line, 'FULL_H_GROUP_V1')
        self.assertEqual((ident['rank'], ident['reason']), (2, 'CELLS_COLLINEAR_IN_FEATURE_PLANE'))
        for cells in (fixed_r, line):
            with self.assertRaisesRegex(ValueError, 'Unidentified'):
                D.check_plan(plan(cells))
        one_q = [cell('A', 2, 10), cell('B', 2, 20), cell('C', 2, 30)]
        self.assertEqual(D.identifiability(one_q, 'FULL_H_GROUP_V1')['reason'], 'SINGLE_Q_VALUE_COLLINEAR_WITH_INTERCEPT')

    def test_identified_design_reaches_unchanged_c_fit_exactly(self):
        p = plan(D7 + HOLD)
        summary = D.check_plan(p)
        self.assertEqual((summary['identifiability']['rank'], summary['new_compiles']), (3, (2 + 2 + 2 + 2 + 1 + 2 + 1) * 2))
        out = D.fit_identified(ROOT, [record(c) for c in D7], plan=p, axis='X')
        model = out['model']
        got = {k: Fraction(v['numerator'], v['denominator']) for k, v in model['coefficients'].items()}
        self.assertEqual(got, COEF)
        self.assertEqual((model['status'], model['qualified_for_candidate']), ('CALIBRATED_UNVALIDATED', False))
        holds = [record(c) for c in HOLD]
        checked = M.validate(model, holds, tolerance=dict(kind='RELATIVE', value=Fraction(1, 100), source=SOURCE))
        self.assertEqual((checked['status'], checked['qualified_for_candidate']), ('HOLDOUT_WITHIN_DECLARED_TOLERANCE', False))

    def test_law_lowering_scalar_profile_are_never_mixed(self):
        changes = dict(law=dict(BASE['law'], K=4),
                       implementation=dict(BASE['implementation'], lowering_id='CONTROLLED_ZZ_GENERIC_RZZ_CONTROL1_PROVIDER_C31EF424'),
                       profile=dict(PROFILE, compiler='0' * 64))
        scalar = dict(BASE['implementation'], scalar_quantity='HAMILTONIAN_CONSTANT_PLUS_B0I_TAIL_IDENTITY')
        for key, value in list(changes.items()) + [('implementation', scalar)]:
            cells = deepcopy(D7); cells[3][key] = value
            with self.assertRaisesRegex(ValueError, 'One route/law/implementation'):
                D.check_plan(plan(cells + HOLD))
            hold = deepcopy(HOLD); hold[0][key] = value
            with self.assertRaisesRegex(ValueError, 'One route/law/implementation'):
                D.check_plan(plan(D7 + hold))
        p = plan(D7 + HOLD); records = [record(c) for c in D7]
        bad = deepcopy(records); bad[3]['implementation'] = changes['implementation']
        with self.assertRaisesRegex(ValueError, 'route/law/implementation/profile differs'):
            D.fit_identified(ROOT, bad, plan=p, axis='X')
        with self.assertRaisesRegex(ValueError, 'One route'):           # C's own domain check agrees
            M.fit(bad, features=['one', 'event_count', 'outer_boundary_count'], design_source=SOURCE)

    def test_calibration_and_holdout_never_mix(self):
        with self.assertRaisesRegex(ValueError, 'both calibration and holdout'):
            D.check_plan(plan(D7 + HOLD + [cell('G566_AGAIN', 1, 566, role='HOLDOUT')]))
        leaky = cell('H2', 2, 566, n=1, role='HOLDOUT', cohort='HOLDOUT_COHORT', composed_of=['G566#p0#g0', 'G566#p1#g0'])
        with self.assertRaisesRegex(ValueError, 'not independent'):
            D.check_plan(plan(D7 + HOLD + [leaky]))
        p = plan(D7 + HOLD); records = [record(c) for c in D7]
        with self.assertRaisesRegex(ValueError, 'Holdout or unknown-role'):
            D.fit_identified(ROOT, records[:-1] + [record(HOLD[0])], plan=p, axis='X')
        with self.assertRaisesRegex(ValueError, 'exactly the planned calibration cells'):
            D.fit_identified(ROOT, records[:-1], plan=p, axis='X')
        with self.assertRaisesRegex(ValueError, 'exactly the planned calibration cells'):
            D.fit_identified(ROOT, records[:-1] + [record(D7[-1], axis='Y')], plan=p, axis='X')

    def test_p1_single_path_is_not_adopted_as_expectation(self):
        p1 = cell('P1_M3', 8, 71, n=1, cohort=D.P1_WIRING_COHORT)
        with self.assertRaisesRegex(ValueError, 'wiring cohort'):
            D.check_plan(plan(D7 + [p1] + HOLD))
        with self.assertRaisesRegex(ValueError, 'never declares a law expectation'):
            D.check_plan(plan(D7 + HOLD[:1] + [dict(HOLD[1], expectation='QUALIFIED_EXPECTATION')]))
        p = plan(D7 + HOLD); records = [record(c) for c in D7]
        with self.assertRaisesRegex(ValueError, 'exactly the planned calibration cells'):
            D.fit_identified(ROOT, records + [record(p1)], plan=p, axis='X')
        # The published P-1 records (a04470c0a) are B single-path records, not C calibration records.
        prices = json.loads(subprocess.check_output(['git', 'show', 'a04470c0ad489eea1798856e648f863f9092cfb1:'
            'docs/research/d_full_h_kappa2_p1_results_20261114/run02_revalidation01/PRICES.json'], cwd=ROOT))
        for r in prices:
            cls = D.classify_saved_price(r)
            self.assertEqual((cls['evidence'], cls['law_expectation'], cls['calibration_record']),
                             ('SINGLE_PATH_DIRECT_FINITE_SAMPLE', False, False))
            self.assertIn('E[C]', cls['forbidden'])
        with self.assertRaisesRegex(ValueError, 'C PRICE_CALIBRATION_RECORD_V1'):
            D.fit_identified(ROOT, prices, plan=p, axis='X')
        claim = dict(prices[0], law_expectation=True)
        with self.assertRaises(ValueError):
            D.classify_saved_price(claim)
        with self.assertRaisesRegex(ValueError, 'n_paths'):        # a planned n=2 cell cannot take a 1-path record
            D.fit_identified(ROOT, [record(c) for c in D7[:-1]] + [record(D7[-1], n_paths=2)], plan=p, axis='X')

    def test_paired_composition_rules_and_covariance(self):
        bad = [dict(D7[-1], composed_of=['G566#p0#g0']),                       # q=2 needs two group paths
               dict(D7[-1], composed_of=['G566#p0#g0', 'G566#p0#g0']),         # distinct paths
               dict(D7[-1], composed_of=['G4528#p0#g0', 'G4528#p1#g0']),       # same R as the composed cell
               dict(D7[-1], n_paths=2)]
        for c in bad:
            with self.assertRaisesRegex(ValueError, 'paired composition'):
                D.check_plan(plan(D7[:-1] + [c] + HOLD))
        import numpy as np
        cov = D.coefficient_covariance(D7, 'FULL_H_GROUP_V1')
        self.assertTrue(np.allclose(cov, cov.T) and np.all(np.linalg.eigvalsh(cov) > 0))
        independent = D.coefficient_covariance(D7[:-1] + [dict(D7[-1], composed_of=None)], 'FULL_H_GROUP_V1')
        self.assertFalse(np.allclose(cov, independent))                         # shared group paths are used

    def test_pauli_two_path_m0_is_not_a_price_model(self):
        pbase = dict(BASE, route='PR_PAULI_TAIL', implementation=dict(lowering_id='PAULI_D2_CENTRAL_ROTATION_CONTROL_9C4006BF'))
        m0 = [dict(pbase, cell_id='M0', role='CALIBRATION', q=1, R=3, n_paths=2)]
        ident = D.identifiability(m0, 'PR_STEP_V1')
        self.assertEqual((ident['rank'], ident['reason']), (1, 'FEWER_DISTINCT_CELLS_THAN_COEFFICIENTS'))
        steps = [dict(pbase, cell_id='S%d' % R, role='CALIBRATION', q=1, R=R, n_paths=2) for R in (171, 1361, 10885)]
        self.assertEqual(D.identifiability(steps, 'PR_STEP_V1')['reason'],
                         'ALL_Q_EQUAL_ONE_BOUNDARY_OR_STEP_COLUMN_NOT_SEPARABLE')
        paired = dict(pbase, cell_id='P2R171', role='CALIBRATION', q=2, R=171, n_paths=1, composed_of=['S171#p0#g0', 'S171#p1#g0'])
        self.assertEqual(D.check_plan(plan(steps + [paired], 'PR_STEP_V1'))['identifiability']['rank'], 3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
