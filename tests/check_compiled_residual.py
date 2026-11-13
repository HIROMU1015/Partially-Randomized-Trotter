"""D = C - S residual records: pinned metadata plus artificial/mock values only.

No sampling, circuit build, compile, structural recount, fit, N/RMSE or G. Every
numerical S/C/D below is an artificial fixture; the fixed B configuration and the
C1/S0a/C-plan files are read as published metadata. Passing these tests says
nothing about the scientific validity of any H4 residual model.
"""
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tests')]
from trottertracks import compiled_residual as R
from trottertracks import abc_s0_published_intake as S0
from trottertracks import rpe_diagnostic_intake as K

p = R.p; B = S0.B
COMPILER = p.PROFILE['compiler']
SCIENCE_MODULES = ('qiskit', 'pyscf', 'h46_long_time_native', 'h46_long_time_signal', 'h46_new2_native', 'r_price_phase1_science')


def result(index, C, S, *, digest=None, conditional=False, candidate='CAND', cohort='DEV', compiler=COMPILER):
    """Artificial C1 Backend.price_requests-shaped result (values are fixtures)."""
    return dict(cell_id='cell', candidate_id=candidate, index=index, cohort=cohort, conditional=conditional, C=list(C), S=list(S),
                event_digest=digest or ('digest%d' % index), compiler=compiler, fingerprints=['fx%d' % index, 'fy%d' % index],
                source=dict(fixture='SYNTHETIC_ONLY'))


def context(role='DEVELOPMENT', **kw):
    base = dict(candidate_id='CAND', role=role, compiler=COMPILER, S_scope=R.FULL, C_scope=R.FULL,
                law_binding=dict(fixture='LAW'), structural_source=dict(fixture='S'), compiled_source=dict(fixture='C'))
    base.update(kw)
    return base


class ResidualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = p.SavedArtifacts(ROOT)
        cls.config = cls.store.read(dict(commit=S0.B_RELEASE, path=S0.BDIR + 'CONFIGURATION_V2.json'))
        cls.inventory = R.saved_paired_inventory(cls.store)
        cls.rounds = R.target_rounds(cls.store)

    # -- envelopes on the fixed B configuration (values artificial)
    def structural_row(self, m, axis, mean_S, law='IDEAL_FINITE_RTE_LAW'):
        row = self.config['rounds'][m]
        return dict(m=m, axis=axis, identity=deepcopy(row['identity']), law_binding=deepcopy(row['law_binding']),
                    physical_candidate_id=row['physical_candidate']['id'], profile=deepcopy(self.config['spec']['profile']),
                    source=dict(fixture='SYNTHETIC_K1'), law_context=dict(law=law), checks=dict(K0='PASS_SYNTHETIC'),
                    uncertainty=dict(numerical_interval=None), mean_S=mean_S)

    def residual(self, m, axis, mean_D, mode='PAIRED_DIRECT_FINITE_COHORT', law='IDEAL_FINITE_RTE_LAW', **kw):
        row = self.config['rounds'][m]
        out = dict(quantity=R.MEAN_D, target='C_MINUS_S', mode=mode, coverage=R.FULL, m=m, axis=axis,
                   identity=deepcopy(row['identity']), law_binding=deepcopy(row['law_binding']),
                   physical_candidate_id=row['physical_candidate']['id'], profile=deepcopy(self.config['spec']['profile']),
                   law_context=dict(law=law), mean_D=mean_D, role='DEVELOPMENT', source=dict(fixture='SYNTHETIC_D'),
                   uncertainty=dict(descriptive_variance=None),
                   expectation='FINITE_COHORT_ONLY' if mode == 'PAIRED_DIRECT_FINITE_COHORT' else 'SUPPLIED_MODEL_EXPECTATION',
                   n_paths=2 if mode == 'PAIRED_DIRECT_FINITE_COHORT' else None)
        out.update(kw)
        return out

    def test_01_S_and_C_must_come_from_the_same_full_wrapper_path(self):
        records = R.paired_residuals(result(0, (120, 120), (100, 100)), context=context())
        self.assertEqual([(r['axis'], r['D']) for r in records], [('X', 20), ('Y', 20)])
        self.assertEqual(records[0]['path'], records[1]['path'])
        for bad, message in ((context(S_scope='INTERNAL_RTE_ONLY'), 'full wrapper'), (context(C_scope='TIME_EVOLUTION_ONLY'), 'full wrapper'),
                             (context(candidate_id='OTHER'), 'candidate'), (context(role='POOLED'), 'role')):
            with self.assertRaisesRegex(ValueError, message):
                R.paired_residuals(result(0, (1, 1), (1, 1)), context=bad)
        no_digest = result(0, (1, 1), (1, 1)); no_digest['event_digest'] = None
        with self.assertRaisesRegex(ValueError, 'event digest'):
            R.paired_residuals(no_digest, context=context())

    def test_02_X_and_Y_are_separate_records_of_the_same_paths(self):
        recs = [r for i, (c, s) in enumerate([((130, 125), (100, 100)), ((90, 95), (100, 100)), ((110, 112), (100, 100))])
                for r in R.paired_residuals(result(i, c, s), context=context())]
        out = R.cohort_summary(recs)
        self.assertEqual(out['n_paths'], 3)                         # paths, not 6 axis samples
        self.assertEqual((out['X']['mean_D'], out['Y']['mean_D']), (Fraction(10), Fraction(32, 3)))
        self.assertIsNotNone(out['XY_covariance_D'])
        with self.assertRaisesRegex(ValueError, 'paired X and Y'):
            R.cohort_summary([r for r in recs if not (r['path']['index'] == 2 and r['axis'] == 'Y')])
        partial = deepcopy(recs); partial[5]['D'] = None              # path 2, Y missing
        out = R.cohort_summary(partial)
        self.assertIsNone(out['Y']['mean_D']); self.assertEqual(out['X']['mean_D'], Fraction(10))

    def test_03_residual_sign_is_free_and_addition_is_exact(self):
        down = R.paired_residuals(result(0, (95, 97), (100, 100)), context=context())
        self.assertEqual([r['D'] for r in down], [-5, -3])           # compiler may give fewer RZ than S
        S, D = self.structural_row(0, 'X', 1000), self.residual(0, 'X', Fraction(-7, 2))
        out = R.combine(S, D, config=self.config)
        self.assertEqual(out['mean'], Fraction(1993, 2)); self.assertEqual(out['quantity'], R.MEAN_C)
        with self.assertRaisesRegex(ValueError, 'no clamping'):
            R.combine(self.structural_row(0, 'X', 3), self.residual(0, 'X', -4), config=self.config)

    def test_04_no_double_counting_of_S_or_stacked_residuals(self):
        S = self.structural_row(0, 'X', 1000)
        with self.assertRaisesRegex(ValueError, 'one residual'):
            R.combine(S, [self.residual(0, 'X', 5), self.residual(0, 'X', 6, mode='C1_AFFINE')], config=self.config)
        with self.assertRaisesRegex(ValueError, 'double count'):
            R.combine(S, self.residual(0, 'X', 5, quantity=R.MEAN_C), config=self.config)
        with self.assertRaisesRegex(ValueError, 'double count'):
            R.combine(S, self.residual(0, 'X', 5, target='RAW_COMPILED_CLUSTER'), config=self.config)
        local = R.combine(S, self.residual(0, 'X', 5, mode='LOCAL_C_MINUS_S', coverage='INTERNAL_RTE_ONLY'), config=self.config)
        self.assertIsNone(local['mean']); self.assertIn('RESIDUAL_NOT_FULL_WRAPPER', local['reasons'])
        # The existing C internal-only local form stays a component, never a full price. Its lazy import of the
        # old cluster module loads the trotterlib package (pyscf/qiskit), so it runs in a child process.
        code = ("import itertools,json,sys;sys.path[:0]=['src'];from trottertracks import mean_compiled_price as M;"
                "keys=['k%d:%s'%(n,','.join(map(str,pt))) for n in (1,2,3) for pt in itertools.product((0,2),repeat=n)];"
                "r=M.local_residual_mean(4,{0:.99,2:.01},{k:1. for k in keys},target='C_MINUS_S');"
                "print(json.dumps([r['full_wrapper_mean'],r['boundary_residual'],r['qualification']]))")
        out = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(out.stdout), [None, None, 'TRUNCATED_LOCAL_RESIDUAL_COMPONENT_ONLY'])

    def test_05_missing_values_stay_null(self):
        out = R.combine(self.structural_row(6, 'Y', None), self.residual(6, 'Y', 12), config=self.config)
        self.assertIsNone(out['mean']); self.assertIn('MEAN_S_MISSING', out['reasons'])
        record = R.b_price_record(out, config=self.config, cohort='C_MINUS_S:TEST', source=dict(fixture='T'))
        self.assertEqual((record['classification'], record['price_status']), ('UNAVAILABLE', 'MISSING'))
        self.assertIsNone(record['mean']['rz_count'])
        out = R.combine(self.structural_row(6, 'Y', 900), self.residual(6, 'Y', None), config=self.config)
        self.assertIsNone(out['mean']); self.assertIn('MEAN_D_MISSING', out['reasons'])
        self.assertTrue(all(v is None for v in (out['uncertainty']['total_CI'], out['uncertainty']['S_D_cross_covariance'])))

    def test_06_DIRECT_finite_cohort_and_MODEL_residuals_stay_distinct(self):
        S = self.structural_row(0, 'X', 1000)
        direct = R.combine(S, self.residual(0, 'X', 20), config=self.config)
        model = R.combine(S, self.residual(0, 'X', 20., mode='C1_AFFINE'), config=self.config)
        self.assertEqual(direct['expectation'], 'LAW_EXPECTATION_S_PLUS_FINITE_COHORT_D')
        self.assertEqual(model['expectation'], 'LAW_EXPECTATION_S_PLUS_MODEL_D')
        self.assertFalse(direct['qualified_for_candidate'] or model['qualified_for_candidate'])
        with self.assertRaisesRegex(ValueError, 'finite cohort'):
            R.combine(S, self.residual(0, 'X', 20, expectation='QUALIFIED_EXPECTATION'), config=self.config)
        for out in (direct, model):
            record = R.b_price_record(out, config=self.config, cohort='C_MINUS_S:TEST', source=dict(fixture='T'))
            self.assertEqual((record['classification'], record['price_status']), ('MODEL_PREDICTION', 'UNQUALIFIED_MODEL'))
            self.assertFalse(record['model']['qualified_for_candidate'])
            self.assertIsNone(record['uncertainty']['all_law_expectation'])

    def test_07_calibration_holdout_and_forced_strata_are_not_pooled(self):
        cal = R.paired_residuals(result(0, (5, 5), (1, 1)), context=context('CALIBRATION'))
        hold = R.paired_residuals(result(1, (6, 6), (1, 1)), context=context('HOLDOUT'))
        with self.assertRaisesRegex(ValueError, 'roles or forced'):
            R.cohort_summary(cal + hold)
        forced = R.paired_residuals(result(2, (9, 9), (1, 1), conditional=True), context=context('CALIBRATION'))
        self.assertEqual({r['stratum'] for r in forced}, {'FORCED_CONDITIONAL'})
        with self.assertRaisesRegex(ValueError, 'roles or forced'):
            R.cohort_summary(cal + forced)
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            R.identifiability([(8, 4), (8, 8), (16, 4)], ['intercept', 'q', 'qR'], holdout=[(8, 8)])

    def test_08_saved_coefficients_are_refused_outside_their_domain(self):
        h4 = self.inventory['frozen_models']['H4']
        cells = {c['cell_id']: c for c in self.inventory['cells']}
        c1 = dict(system='H4', series='H4_TOL_ONLY_DF10', prefix=1, delta=.1, K=2, formula='2nd', control='symmetric_directional',
                  compiler=COMPILER, unit=R.FULL, quantity=R.MEAN_D, qualified_for_candidate=h4['qualified_for_candidate'],
                  evaluated_points=[(cells[c]['q_PF'], cells[c]['R_outer']) for c in h4['point_roles']])
        for r in (self.rounds[0], self.rounds[6]):
            target = dict(c1, q_PF=r['q_PF'], R_outer=r['R_outer'])
            reasons = R.legacy_domain(c1, target)
            self.assertIn('UNOBSERVED_Q_R_POINT', reasons); self.assertIn('MODEL_NOT_QUALIFIED', reasons)
        old_cluster = dict(c1, series='H4_DF12', prefix=3, control='uncontrolled', unit='UNCONTROLLED_LOCAL_WINDOW',
                           quantity='RAW_COMPILED_CLUSTER_KAPPA')
        reasons = R.legacy_domain(old_cluster, dict(c1, q_PF=1, R_outer=2))
        for reason in ('DOMAIN_SERIES_DIFFERS', 'DOMAIN_CONTROL_DIFFERS', 'UNIT_NOT_CONTROLLED_FULL_WRAPPER',
                       'MODEL_IS_NOT_A_C_MINUS_S_RESIDUAL'):
            self.assertIn(reason, reasons)

    def test_09_compiler_profile_and_identity_mismatches_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'compiler'):
            R.paired_residuals(result(0, (1, 1), (1, 1), compiler='OTHER_COMPILER'), context=context())
        other = self.residual(0, 'X', 1); other['profile'] = dict(other['profile'], compiler='OTHER_COMPILER')
        with self.assertRaisesRegex(ValueError, 'profile'):
            R.combine(self.structural_row(0, 'X', 10), other, config=self.config)
        with self.assertRaisesRegex(ValueError, 'identity'):
            R.combine(self.structural_row(0, 'X', 10), dict(self.residual(6, 'X', 1), m=0), config=self.config)
        with self.assertRaisesRegex(ValueError, 'same round and axis'):
            R.combine(self.structural_row(0, 'X', 10), self.residual(0, 'Y', 1), config=self.config)
        law = R.combine(self.structural_row(0, 'X', 10), self.residual(0, 'X', 1, law='PCG64_CDF_SAMPLER'), config=self.config)
        self.assertIn('S_AND_D_LAW_DIFFER_OR_UNDECLARED', law['reasons'])

    def test_10_B_rejects_an_unqualified_E_C_record_and_structural_S(self):
        packet = dict(configuration_id=self.config['id'], source=dict(fixture='SYNTHETIC_K1'), kernel_source=dict(fixture='K'),
                      stage='K1', K0_status='PASS', quantity=K.STRUCTURAL, qualified_for_candidate=False,
                      mean_compiled_C=None, mean_C_minus_S=None,
                      rows=[self.structural_row(m, a, 1000 + m) for m in (0, 6) for a in p.AXES])
        for r in packet['rows']:
            r.pop('m'); r['m'] = r['identity']['m']
        accepted = K.accept_structural_packet(self.config, packet)
        self.assertEqual(accepted['compiled_price_records'], [])
        prices = []; cohorts = {}
        for r in packet['rows']:
            out = R.combine(r, self.residual(r['m'], r['axis'], 25), config=self.config)
            prices.append(R.b_price_record(out, config=self.config, cohort='C_MINUS_S:TEST', source=dict(fixture='T')))
            cohorts[(self.config['rounds'][r['m']]['candidate']['id'], r['axis'])] = 'C_MINUS_S:TEST'
        with patch.object(B, 'cost', side_effect=AssertionError('No G')):
            result_ = B.assess(self.config, None, prices, cohorts=cohorts)
        self.assertFalse(result_['eligible']); self.assertIsNone(result_['cost'])
        for reason in ('C_UNQUALIFIED_MODEL', 'UNQUALIFIED_C_EXPECTATION_OR_FORCED_COHORT'):
            self.assertIn(reason, result_['reasons'])
        structural_price = dict(prices[0], quantity=K.STRUCTURAL, price_status=None, classification='MODEL_PREDICTION')
        self.assertIn('C_STRUCTURAL_OR_LOCAL_QUANTITY_IS_NOT_MEAN_COMPILED_PRICE',
                      B.assess(self.config, None, [structural_price] + prices[1:], cohorts=cohorts)['reasons'])

    def test_11_q_and_R_dependence_needs_an_identifiable_design(self):
        focus = [(r['q_PF'], r['R_outer']) for r in (self.rounds[0], self.rounds[6])]
        self.assertEqual(focus, [(1, 2), (64, 128)])
        two = R.identifiability(focus, ['intercept', 'q', 'qR'])
        self.assertEqual((two['rank'], two['identifiable'], two['R_over_q_constant']), (2, False, True))
        path = R.identifiability([(r['q_PF'], r['R_outer']) for r in self.rounds[:7]], ['intercept', 'q', 'R', 'qR'])
        self.assertFalse(path['identifiable'])                       # R = 2q on m0..m6: q and R inseparable
        on_line = R.identifiability([(1, 2), (8, 16), (64, 128)], ['intercept', 'q', 'qR'])
        self.assertTrue(on_line['identifiable'] and on_line['R_over_q_constant'] and on_line['lack_of_fit_dof'] == 0)
        c1 = R.identifiability([(8, 4), (8, 8), (16, 4)], ['intercept', 'q', 'qR'], holdout=[(16, 8), (8, 16), (32, 4), (64, 128)])
        self.assertTrue(c1['identifiable'])
        self.assertEqual([h['transfer'] for h in c1['holdout']], ['R_ONLY', 'CROSS_OF_SEEN_LEVELS', 'Q_ONLY', 'JOINT_NEW'])

    def test_12_saved_inventory_is_copied_and_targets_have_no_saved_price(self):
        cells = {c['cell_id']: c for c in self.inventory['cells']}
        self.assertEqual({k: (v['saved_paths'], v['paired_complete_paths']) for k, v in cells.items() if k.startswith('H4')},
                         {'H4_q8_R4': (9, 9), 'H4_q8_R8': (8, 8), 'H4_q16_R4': (33, 33), 'H4_q16_R8': (8, 8)})
        self.assertTrue(all(not c['residual_mean_saved'] and c['compilers'] == [COMPILER] for c in cells.values()))
        self.assertEqual(cells['H6_q2_R16']['event_digests'], 0)
        self.assertEqual((self.inventory['new_residual_statistics'], self.inventory['compile']), (0, 0))
        self.assertEqual(len(self.inventory['forced_rare_pairs']), 4)
        self.assertTrue(all(v['parity'] for v in self.inventory['s0a_parity']))
        self.assertEqual(len(self.rounds), 13)
        self.assertTrue(all(r['price'] is None and not r['saved_direct'] for r in self.rounds))
        c1_points = {(c['q_PF'], c['R_outer']) for c in cells.values() if c['system'] == 'H4'}
        self.assertFalse(c1_points & {(r['q_PF'], r['R_outer']) for r in self.rounds})

    def test_13_no_science_is_started(self):
        self.assertFalse([m for m in sys.modules if m.split('.')[0] in SCIENCE_MODULES])


if __name__ == '__main__':
    unittest.main(verbosity=2)
