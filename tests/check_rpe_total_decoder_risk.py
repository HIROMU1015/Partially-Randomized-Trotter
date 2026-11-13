"""Artificial laws only: total estimator Z1+T1 and its Q3 risk. No MC, no saved raw."""
from contextlib import ExitStack
from copy import deepcopy
from fractions import Fraction
from itertools import product
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tests')]
import numpy as np
from check_rpe_deterministic_risk import case, cost_fixture
from check_rpe_signal_statistics_caller import MOCK
from trottertracks import rpe_total_decoder as T
from trottertracks import rpe_total_risk as R

D, S, K, F, a = R.D, R.S, R.K, R.F, R.a
BIG = dict(source=MOCK, maximum_frontier_states=10**6, maximum_transition_pairs=10**8)
CASES = {   # W0 SYNTHETIC_CHECKS laws (B fixture: h=10, artificial loss family [0,20])
    'B_fixture_like': dict(vector=(2, 2), raws=[(.25, -.5)]*2),
    'zero_tie_rich': dict(vector=(2, 2, 2), raws=[(0., 0.)]*3),
    'mixed_parity': dict(vector=(1, 3, 2), raws=[(.6, .1), (.3, -.2), (.5, .4)]),
    'four_rounds': dict(vector=(2, 2, 2, 2), raws=[(.5, .2)]*4),
    'five_rounds_N1': dict(vector=(1, 1, 1, 1, 1), raws=[(.4, .1)]*5),
    'near_principal_edge': dict(vector=(2, 2, 2), target=-.3,
        raws=[(.6*math.cos(3*2**m), .6*math.sin(3*2**m)) for m in range(3)]),
    'odd_only': dict(vector=(3, 1, 3), raws=[(0., 0.)]*3),
    'principal_edge_wrap': dict(vector=(1, 1, 1, 1, 1), target=-(math.pi-.02)/10,
        raws=[(.9*math.cos((math.pi-.02)*2**m), .9*math.sin((math.pi-.02)*2**m)) for m in range(5)]),
}


def make(name=None, *, vector=None, raws=None, target=0., total=BIG, partial=BIG):
    spec = CASES[name] if name else dict(vector=vector, raws=raws, target=target)
    args = case(M=len(spec['vector'])-1, vectors=[tuple(spec['vector'])])
    for mean, raw in zip(args['signal_packet']['means'], spec['raws']):
        mean['raw'] = dict(real=raw[0], imag=raw[1])
    args['evaluation_target']['energy_ha'] = spec.get('target', 0.)
    args['options']['estimator'] = T.ESTIMATOR
    if partial is not None:
        args['options']['state_refinement'] = deepcopy(partial)
    if total is not None:
        args['options']['total_state_refinement'] = deepcopy(total)
    return args


def evaluate(args):
    return R.evaluate(**args)[0]['diagnostic']


def oracle(args):
    """Independent comb/Fraction law over every count history; both decoders."""
    h = float(a.exact_scalar(args['config']['spec']['h'])); E = a.exact_scalar(args['evaluation_target']['energy_ha'])
    support = []
    for m, (mean, shots) in enumerate(zip(args['signal_packet']['means'], args['allocations'][0]['shots'])):
        axes = []
        for axis, component in (('X', 'real'), ('Y', 'imag')):
            n = shots['N_'+axis]; p = (1+a.exact_scalar(mean['raw'][component]))/2
            axes.append([(k, Fraction(math.comb(n, k))*p**k*(1-p)**(n-k)) for k in range(n+1)])
        support.append([(dict(m=m, X=dict(plus=kx, shots=shots['N_X']), Y=dict(plus=ky, shots=shots['N_Y'])), px*py)
                        for (kx, px), (ky, py) in product(*axes) if px*py])
    M = len(support)-1
    out = dict(mse=Fraction(0), m1=Fraction(0), m2=Fraction(0), eta=Fraction(0), unresolved=Fraction(0),
               zero_any=Fraction(0), identical=True, histories=0, decoded=0, zero_round=[Fraction(0)]*(M+1),
               tie_round=[Fraction(0)]*(M+1), statuses_match=True, path_excess=None)
    for history in product(*support):
        w = math.prod(v[1] for v in history); rows = [v[0] for v in history]
        old = S.decode_counts(rows, h=h, M=M, prior=[-1e3, 1e3])
        new = T.decode_counts_total(rows, h=h, M=M, prior=[-1e3, 1e3])
        e = a.exact_scalar(new['energy_estimate_ha']); loss = (e-E)**2
        out['histories'] += 1; out['mse'] += w*loss; out['m1'] += w*e; out['m2'] += w*e*e
        out['statuses_match'] &= old['status'] == new['partial_decoder_status']
        if old['status'] == S.DECODED:
            out['decoded'] += 1; out['eta'] += w*loss
            out['identical'] &= old['energy_estimate_ha'].hex() == new['energy_estimate_ha'].hex()
        else:
            out['unresolved'] += w
        if new['zero_hold_rounds']:
            out['zero_any'] += w
        for m in new['zero_hold_rounds']: out['zero_round'][m] += w
        for m in new['tie_argmin_rounds']: out['tie_round'][m] += w
        path = new['phase_path']
        for m in range(1, M+1):
            excess = R.circular_upper(path[m], path[m-1]) - R.PI_LO/2**m
            out['path_excess'] = excess if out['path_excess'] is None else max(out['path_excess'], excess)
    out['variance'] = out['m2']-out['m1']**2; out['bias'] = out['m1']-E
    return out


def X(value):
    return a.exact_scalar(value)


class TotalDecoderRiskTests(unittest.TestCase):
    def setUp(self):
        stack = ExitStack(); self.addCleanup(stack.close)
        for module, name in ((S, 'simulate'), (S, 'simulate_prepared'), (K, 'diagnose'), (D.p, 'run_missing'),
                             (D.p.SavedArtifacts, 'read')):
            stack.enter_context(patch.object(module, name, side_effect=AssertionError('MC/saved/physical provider forbidden')))

    @classmethod
    def setUpClass(cls):
        cls.oracles = {name: oracle(make(name)) for name in CASES}

    def test_01_bit_identical_on_original_resolved_histories(self):
        for name, o in self.oracles.items():
            self.assertTrue(o['identical'], name); self.assertTrue(o['statuses_match'], name)
        self.assertGreater(sum(o['decoded'] for o in self.oracles.values()), 7000)
        rng = np.random.default_rng(20261114)
        for M in range(13):
            n = rng.choice([1, 3, 5, 7, 63], size=(400, M+1, 2))
            k = rng.integers(0, n+1)
            xy = np.vectorize(lambda kk, nn: S.count_mean(int(kk), int(nn))['mean'])(k, n)
            old = F.decode_coordinates_batch(xy, h=.1); new = T.decode_coordinates_batch_total(xy, h=.1)
            for x, y in zip(old, new):
                self.assertEqual(x['status'], y['partial_decoder_status'])
                if x['status'] == F.DECODED:
                    self.assertEqual(x['energy_estimate_ha'].hex(), y['energy_estimate_ha'].hex())
                    self.assertEqual(x['decoded']['final_phase_estimate'], y['final_phase_estimate'])
                self.assertTrue(math.isfinite(y['energy_estimate_ha']))

    def test_02_zero_and_tie_always_give_an_estimate_with_history(self):
        zero = T.decode_supplied_coordinates_total([(0., 0.), (.5, .5), (0., 0.)], h=.1)
        self.assertEqual(zero['phase_path'][0].hex(), (0.0).hex())            # +0.0, not -0.0
        self.assertEqual(zero['phase_path'][2], zero['phase_path'][1])           # held
        self.assertEqual(zero['zero_hold_rounds'], [0, 2]); self.assertEqual(zero['status'], T.DECODED_TOTAL)
        self.assertEqual(zero['partial_decoder_status'], 'UNRESOLVED_ZERO_COORDINATES')
        self.assertEqual(T.decode_supplied_coordinates_total([(0., 0.)], h=.1)['energy_estimate_ha'], 0.)
        # Exact computed tie (rpe_decoder_fast test geometry): argmin lowest ID is kept.
        q = 32; previous = 2.650718801466388
        value, tie = T.step(np.array([0.]), np.array([previous]), q, np.array([False]))
        candidates = F._wrap((0.+2*math.pi*np.arange(q))/q)
        distances = np.abs(np.angle(np.exp(1j*(candidates-previous))))
        self.assertTrue(bool(tie[0])); self.assertEqual(float(value[0]), float(candidates[int(np.argmin(distances))]))
        self.assertGreater(self.oracles['zero_tie_rich']['zero_any'], 0)
        self.assertGreater(sum(self.oracles['zero_tie_rich']['tie_round']), 0)

    def test_03_complete_total_DP_equals_exhaustive_enumeration(self):
        for name, o in self.oracles.items():
            out = evaluate(make(name)); total = out['deterministic_risk']; dp = total['bounds']['total_state_dp']
            self.assertEqual(dp['status'], 'COMPLETE_TOTAL_STATE_DP', name)
            self.assertEqual(X(dp['MSE_exact']), o['mse'], name)
            mom = dp['moments']; self.assertEqual(mom['status'], 'EXACT')
            self.assertEqual(X(mom['mean_energy'][0]), o['m1']); self.assertEqual(X(mom['second_moment'][0]), o['m2'])
            self.assertEqual(X(mom['variance'][0]), o['variance']); self.assertEqual(X(mom['bias'][0]), o['bias'])
            self.assertEqual(X(mom['bias_squared'][0]) + X(mom['variance'][0]), o['mse'])
            self.assertEqual(X(dp['partial_decoder']['resolved_MSE_contribution']), o['eta'])
            self.assertEqual(X(dp['partial_decoder']['unresolved_probability']), o['unresolved'])
            self.assertEqual(X(dp['any_zero']), o['zero_any'])
            for m, row in enumerate(dp['rounds']):
                self.assertEqual(X(row['zero_hold_probability_tracked']), o['zero_round'][m], (name, m))
                self.assertEqual(X(row['tie_argmin_probability_tracked']), o['tie_round'][m], (name, m))
            refined = out['partial_decoder_risk']['state_refinement']   # B's DP: identical resolved part
            self.assertEqual(X(refined['resolved_MSE_contribution']), o['eta'])
            self.assertEqual(X(refined['unresolved_probability']), o['unresolved'])
            self.assertEqual(X(total['MSE_upper']), o['mse']); self.assertEqual(X(total['MSE_lower']), o['mse'])

    def test_04_partial_scopes_keep_valid_residual_and_moment_intervals(self):
        for name in ('zero_tie_rich', 'four_rounds', 'five_rounds_N1', 'mixed_parity', 'near_principal_edge'):
            o = self.oracles[name]
            for states, pairs in ((0, 0), (0, 10**6), (1, 10**6), (3, 10**6), (10**6, 7), (2, 40)):
                dp = evaluate(make(name, total=dict(BIG, maximum_frontier_states=states,
                    maximum_transition_pairs=pairs)))['deterministic_risk']['bounds']['total_state_dp']
                if dp['status'] == 'COMPLETE_TOTAL_STATE_DP':
                    self.assertEqual(X(dp['MSE_exact']), o['mse']); continue
                self.assertEqual(dp['status'], 'PARTIAL_TOTAL_STATE_DP_WITH_STATE_RESIDUAL_BOUND', (name, states, pairs))
                self.assertGreater(X(dp['residual_probability']), 0)
                self.assertLessEqual(X(dp['MSE_lower']), o['mse']); self.assertGreaterEqual(X(dp['MSE_upper']), o['mse'])
                mom = dp['moments']
                for key, value in (('mean_energy', o['m1']), ('second_moment', o['m2']), ('bias', o['bias']),
                                   ('bias_squared', o['bias']**2), ('variance', o['variance'])):
                    self.assertLessEqual(X(mom[key][0]), value, (name, key)); self.assertGreaterEqual(X(mom[key][1]), value, (name, key))
                lo, hi = dp['partial_decoder']['unresolved_probability_bounds']
                self.assertLessEqual(X(lo), o['unresolved']); self.assertGreaterEqual(X(hi), o['unresolved'])
                self.assertLessEqual(X(dp['partial_decoder']['resolved_MSE_contribution_lower']), o['eta'])

    def test_05_update_width_lemma_guard_and_first_unsafe_caps(self):
        for name, o in self.oracles.items():
            self.assertLessEqual(o['path_excess'], R.TRANSITION_ALLOWANCE, name)
            total = evaluate(make(name, total=None))['deterministic_risk']; lemma = total['bounds']['lemma']
            self.assertEqual(lemma['status'], 'TOTAL_ESTIMATOR_FIRST_UNSAFE_LEMMA_BOUND', name)
            self.assertGreaterEqual(X(lemma['MSE_upper']), o['mse'], name)
            for cap in lemma['caps'][:2]:
                self.assertEqual(X(cap['loss_cap']), X(lemma['loss_upper']))
        # Negative controls: dropping the principal guard under-bounds near +-pi.
        unguarded = lambda radius, *, h, theta, loss_upper: (min(loss_upper, (radius/h+R.division_slack(h))**2), True)
        args = make('principal_edge_wrap', total=dict(BIG, maximum_frontier_states=1))
        o = self.oracles['principal_edge_wrap']
        with patch.object(R, 'energy_loss_cap', side_effect=unguarded):
            bad = evaluate(args)['deterministic_risk']['bounds']
        self.assertLess(X(bad['lemma']['MSE_upper']), o['mse'])
        self.assertLess(X(bad['total_state_dp']['MSE_upper']), o['mse'])
        good = evaluate(args)['deterministic_risk']['bounds']
        self.assertGreaterEqual(X(good['lemma']['MSE_upper']), o['mse'])
        self.assertGreaterEqual(X(good['total_state_dp']['MSE_upper']), o['mse'])
        # Large-q transitions incl. computed midpoints: width <= pi/2^m + allowance.
        rng = np.random.default_rng(7); worst = -1.
        for m in range(1, 13):
            q = 1 << m; n = 20000
            principal = F._wrap(rng.uniform(-math.pi, math.pi, n)); estimate = F._wrap(rng.uniform(-math.pi, math.pi, n))
            k = rng.integers(0, q, n//10)
            estimate = np.concatenate([estimate, F._wrap((principal[:n//10]+2*math.pi*k)/q+math.pi/q)])
            principal = np.concatenate([principal, principal[:n//10]])
            zero = rng.random(len(principal)) < .05
            value, _ = T.step(principal, estimate, q, zero)
            self.assertTrue(np.all(value[zero] == estimate[zero]))
            move = np.abs(value-estimate); move = np.minimum(move, 2*math.pi-move)
            worst = max(worst, float(np.max(move-math.pi/q)))
        self.assertLess(worst, float(R.TRANSITION_ALLOWANCE))

    def test_06_existing_B_bounds_are_consistency_checks_for_the_total_MSE(self):
        for name, o in self.oracles.items():
            out = evaluate(make(name)); partial = out['partial_decoder_risk']; refined = partial['state_refinement']
            self.assertGreaterEqual(X(partial['coarse_MSE_upper']), o['mse'], name)
            self.assertLessEqual(X(refined['resolved_MSE_contribution']), o['mse'], name)
            self.assertGreaterEqual(X(refined['MSE_upper']), o['mse'], name)
            self.assertTrue(out['deterministic_risk']['bounds']['partial_state_dp_valid_for_total'])
        # A family upper below L_D is NOT a valid bound for the total estimator.
        args = make('zero_tie_rich'); args['statistical_contract']['summary_options']['unresolved_loss']['upper'] = 0
        out = evaluate(args); bounds = out['deterministic_risk']['bounds']
        self.assertFalse(bounds['partial_state_dp_valid_for_total'])
        self.assertNotIn('partial_state_dp', bounds['upper_candidates'])
        self.assertLess(X(out['partial_decoder_risk']['state_refinement']['MSE_upper']), self.oracles['zero_tie_rich']['mse'])
        self.assertEqual(X(out['deterministic_risk']['MSE_upper']), self.oracles['zero_tie_rich']['mse'])

    def test_07_parity_pi_and_binary64_boundaries(self):
        o = self.oracles['odd_only']
        self.assertEqual(o['zero_any'], 0); self.assertTrue(all(v == 0 for v in o['zero_round']))
        self.assertTrue(all(S.count_mean(k, n)['mean'] != 0 for n in range(1, 101, 2) for k in range(n+1)))
        dp = evaluate(make('odd_only'))['deterministic_risk']['bounds']['total_state_dp']
        self.assertEqual(X(dp['any_zero']), 0)
        for target in (0., math.nextafter(math.pi/10, -math.inf), math.nextafter(math.pi/10, math.inf), -math.pi/10):
            args = make(vector=(2, 2), raws=[(-1., 0.), (-1., 0.)], target=target)
            total = evaluate(args)['deterministic_risk']
            self.assertEqual(X(total['bounds']['total_state_dp']['MSE_exact']), oracle(args)['mse'], target)
        self.assertEqual(float(F._wrap(math.atan2(0., -1.))), -math.pi)
        self.assertEqual(float(F._wrap(math.nextafter(-math.pi, -math.inf))), math.pi)   # endpoint +pi_f
        table = dict(status='ENUMERATED', m=0, q=1, denominator=2,
            outcomes=[dict(principal=v, zero=False, weight=1) for v in (-0., 0.)])
        dp = R.total_state_dp([table], h=1., target=0., scope=dict(BIG))
        self.assertEqual(dp['rounds'][0]['retained_states'], 2)      # signed zeros stay distinct states
        self.assertEqual(X(dp['MSE_exact']), 0)

    def test_08_numerically_unqualified_cases_are_UNKNOWN(self):
        with self.assertRaises(ValueError):
            args = make('B_fixture_like'); args['options'].pop('estimator'); R.evaluate(**args)
        args = make('B_fixture_like'); args['options']['phase_radii'] = [2.6]*2      # closure fails
        total = evaluate(args)['deterministic_risk']
        self.assertEqual(total['bounds']['lemma']['reason'], 'GOOD_CLOSURE_CERTIFICATE_REQUIRED')
        self.assertEqual(total['MSE_upper_method'], 'MIN_OF_COMPLETED_VALID_UPPER_BOUNDS:total_state_dp')
        args = make('B_fixture_like', total=None); args['options']['phase_radii'] = [2.6]*2
        self.assertEqual(evaluate(args)['deterministic_risk']['reason'], 'NO_VALID_TOTAL_ESTIMATOR_UPPER_BOUND')
        args = make('B_fixture_like'); args['options']['model_loss_contract']['adopted'] = False
        self.assertEqual(evaluate(args)['deterministic_risk']['reason'], 'ALL_EVENT_LOSS_NOT_ADOPTED')
        args = make('B_fixture_like'); args['options']['maximum_round_outcomes'] = 1
        self.assertEqual(evaluate(args)['deterministic_risk']['status'], 'UNKNOWN')
        def nonfinite(principal, estimate, q, zero):
            return np.full(len(principal), np.nan), np.zeros(len(principal), dtype=bool)
        with patch.object(T, 'step', side_effect=nonfinite):
            total = evaluate(make('B_fixture_like'))['deterministic_risk']
        self.assertEqual(total['bounds']['total_state_dp']['reason'], 'NONFINITE_TOTAL_DECODER_TRANSITION')
        self.assertNotIn('total_state_dp', total['bounds']['upper_candidates'])
        with patch.object(R, 'TRANSITION_ALLOWANCE', Fraction(0)):
            dp = evaluate(make('four_rounds'))['deterministic_risk']['bounds']['total_state_dp']
        self.assertEqual(dp['reason'], 'DECLARED_TRANSITION_ALLOWANCE_EXCEEDED')
        tables = R.tables_for(K.prepare(**{k: v for k, v in make('B_fixture_like').items()
                                          if k not in ('source', 'options')})[0], make('B_fixture_like')['options'])
        tables[0]['outcomes'][0]['weight'] += 1
        self.assertEqual(R.total_state_dp(tables, h=10., target=0., scope=BIG)['reason'], 'COUNT_MASS_NOT_CONSERVED')
        for scope in ({}, dict(BIG, maximum_frontier_states=-1), dict(BIG, maximum_transition_pairs=True)):
            with self.assertRaises(ValueError): R.total_state_dp(tables, h=10., target=0., scope=scope)
        args = make(vector=(1,)*14, raws=[(1., 0.)]*14)
        self.assertEqual(evaluate(args)['deterministic_risk']['reason'], 'OUTSIDE_VALIDATED_R3_SCALAR_EQUIVALENCE_DOMAIN')

    def test_09_min_only_combination_bindings_and_unchanged_partial_output(self):
        for name in CASES:
            args = make(name); before = deepcopy(args)
            records = R.evaluate(**args); total = records[0]['diagnostic']['deterministic_risk']
            candidates = {k: X(v) for k, v in total['bounds']['upper_candidates'].items()}
            self.assertEqual(X(total['MSE_upper']), min(candidates.values()))
            self.assertIn(total['MSE_upper_method'].split(':')[1], candidates)
            self.assertEqual(records[0]['deterministic_input']['options']['estimator'], T.ESTIMATOR)
            legacy = deepcopy(before); legacy['options'] = {k: v for k, v in legacy['options'].items()}
            self.assertEqual(records[0]['diagnostic']['partial_decoder_risk'],
                             D.evaluate(**legacy)[0]['diagnostic']['deterministic_risk'])
            self.assertEqual(args, before)
            self.assertEqual(args['config']['spec']['rmse_policy']['error_contract'],
                             before['config']['spec']['rmse_policy']['error_contract'])
        args = case(vectors=[(1, 5), (8, 1)]); args['options'].update(estimator=T.ESTIMATOR, total_state_refinement=BIG)
        records = R.evaluate(**args); prices, qualifications, options = cost_fixture(args, records)
        out = D.compare(args['config'], records, prices, qualifications=qualifications,
            model_limit=dict(MSE_upper=1, source=MOCK, scope='MODEL_MSE_FILTER_ONLY'), **options)
        self.assertEqual(out['comparison']['point_minimum']['allocation_id'], 'v1')
        qualifications['v0']['rmse_evidence']['deterministic_input']['options'].pop('estimator')
        with self.assertRaises(ValueError):
            D.compare(args['config'], records, prices, qualifications=qualifications,
                model_limit=dict(MSE_upper=1, source=MOCK, scope='MODEL_MSE_FILTER_ONLY'), **options)

    def test_10_q3_roles_and_no_delta_E_double_count(self):
        o = self.oracles['four_rounds']; args = make('four_rounds')
        args['options']['q3_limit'] = dict(MSE_upper=a.exact_value(o['mse']), source=MOCK, scope='MODEL_MSE_FILTER_ONLY')
        q3 = evaluate(args)['deterministic_risk']['q3']
        self.assertTrue(q3['within_limit']); self.assertFalse(q3['physical_total_error_guarantee'])
        self.assertFalse(q3['minkowski_reference']['delta_E_added'])
        self.assertEqual(set(q3['transfer_terms'].values()), {'UNKNOWN'})
        d = q3['design_diagnostics']
        self.assertEqual(X(d['bias_squared'][0]) + X(d['variance'][0]), X(q3['model_MSE_upper']))
        args['options']['q3_limit']['MSE_upper'] = a.exact_value(o['mse'] - Fraction(1, 10**30))
        self.assertFalse(evaluate(args)['deterministic_risk']['q3']['within_limit'])
        args['options']['q3_limit'].pop('source')
        with self.assertRaises(ValueError): evaluate(args)

    def test_11_original_partial_decoder_still_stops_on_zero_and_tie(self):
        rows = [dict(m=0, X=dict(plus=1, shots=2), Y=dict(plus=1, shots=2))]       # X=Y=0
        self.assertEqual(S.decode_counts(rows, h=10., M=0, prior=[-1, 1])['status'], 'UNRESOLVED_ZERO_COORDINATES')
        self.assertIsNone(S.decode_counts(rows, h=10., M=0, prior=[-1, 1])['energy_estimate_ha'])
        self.assertEqual(T.decode_counts_total(rows, h=10., M=0, prior=[-1, 1])['energy_estimate_ha'], -0.)
        self.assertNotIn(T.ESTIMATOR, (S.DECODER,)); self.assertEqual(T.ORIGINAL_ESTIMATOR, S.DECODER)
        o = self.oracles['zero_tie_rich']
        self.assertGreater(o['unresolved'], o['zero_any'])                     # original tie stops still exist
        args = make('zero_tie_rich'); args['options'] = {k: v for k, v in args['options'].items()
                                                       if k not in ('estimator', 'total_state_refinement')}
        legacy = D.evaluate(**args)[0]['diagnostic']['deterministic_risk']
        self.assertEqual(X(legacy['state_refinement']['unresolved_probability']), o['unresolved'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
