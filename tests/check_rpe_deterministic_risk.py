"""Artificial count laws/prices only. Every RPE MC entry is forbidden."""
from contextlib import ExitStack
from copy import deepcopy
from fractions import Fraction
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tests')]
from check_rpe_signal_statistics_caller import synthetic, I, MOCK
from check_pr_rpe_rmse_controller import data
from trottertracks import rpe_deterministic_risk as D

B, S, K, a = D.B, D.S, D.K, D.a


def case(M=1, vectors=None):
    config, packet, _, target, contract = synthetic(M=M)
    design = deepcopy(config['spec']); design['h'] = 10
    config = B.generate(design); packet['configuration_id'] = config['id']; target['energy_ha'] = 0.
    for mean, row in zip(packet['means'], config['rounds']):
        mean.update(identity=deepcopy(row['identity']), law_binding=deepcopy(row['law_binding']),
            raw=dict(real=1., imag=0.), corrected=dict(real=1., imag=0.), normalization=dict(log_B=0.))
    vectors = vectors or [tuple([1]*(M+1))]
    allocations = [dict(allocation_id='v'+str(i), source=MOCK, shot_policy='USER_SUPPLIED',
        shots=[dict(m=m, N_X=n, N_Y=n) for m,n in enumerate(vector)]) for i,vector in enumerate(vectors)]
    contract['summary_options'] = dict(unresolved_loss=dict(kind='BOUNDED_INTERVAL', lower=0, upper=20, source=MOCK))
    options = dict(source=MOCK, phase_radii=[1.]*(M+1), probability_rule=D.EXACT_AFFINE,
        maximum_round_outcomes=10000, maximum_verification_pairs=1000000,
        model_loss_contract=dict(adopted=True, source=MOCK))
    return dict(config=config, signal_packet=packet, allocations=allocations, evaluation_target=target,
        statistical_contract=contract, integration=I, source=MOCK, options=options)


def cost_fixture(args, records):
    config = args['config']; qualifications = {}
    for record in records:
        packet, prices, cohorts = data(config)
        for row, shots in zip(packet['rounds'], record['allocation']['shots']):
            row.update(N_X=shots['N_X'], N_Y=shots['N_Y'])
        packet['rmse_evidence'].update(statistical_contract=deepcopy(args['statistical_contract']),
            deterministic_input=deepcopy(record['deterministic_input']))
        qualifications[record['allocation']['allocation_id']] = packet
    extension = dict(lowering_id='MOCK_LOWERING', source=MOCK)
    coverage = dict(id='MOCK_BOX', source=MOCK, configuration_id=config['id'],
        simultaneous=True, scope='ALL_SELECTED_ROUNDS_AXES')
    bindings = []
    for price in prices:
        value = 1 if price['identity']['m'] == 0 else 100
        price['mean']['rz_count'] = value
        bindings.append(dict(identity=deepcopy(price['identity']), law_binding=deepcopy(price['law_binding']),
            axis=price['axis'], cohort=price['cohort'], source=MOCK, price_source=deepcopy(price['source']),
            profile_extension=deepcopy(extension), mean_interval=dict(quantity='FULL_WRAPPER_MEAN_COMPILED_RZ_INTERVAL',
                lower=value, upper=value, source=MOCK, qualification='QUALIFIED',
                evidence_kind='DERIVED_PATHWISE_BOUND', domain=deepcopy(price['identity']['candidate']), coverage=coverage)))
    return prices, qualifications, dict(cohorts=cohorts, statistical_contract=args['statistical_contract'],
        price_extension=extension, price_bindings=bindings, interval_contract=coverage)


class DeterministicRiskTests(unittest.TestCase):
    def setUp(self):
        self.no_mc = ExitStack()
        for module, name in ((S,'simulate'), (S,'simulate_prepared'), (K,'diagnose'), (D.p,'run_missing')):
            self.no_mc.enter_context(patch.object(module, name, side_effect=AssertionError('MC/science provider forbidden')))
        self.addCleanup(self.no_mc.close)

    def test_01_exact_binomial_mass_tiny_probability_and_huge_integer_endpoint(self):
        axis = D.binomial_axis(0., 2000, probability_rule=D.EXACT_AFFINE)
        self.assertEqual(sum(axis['numerators']), axis['denominator'])
        tiny = Fraction(axis['numerators'][0], axis['denominator'])
        self.assertGreater(tiny, 0); self.assertEqual(float(tiny), 0.)
        report = D.probability_report(tiny)
        self.assertFalse(report['zero_exact']); self.assertLess(report['log_probability'], -1000)
        self.assertEqual(a.exact_scalar(report['exact']), Fraction(1,2**2000))
        n = 2**1200+3
        certain = D.binomial_axis(1., n, probability_rule=D.EXACT_AFFINE)
        self.assertEqual(certain['counts'], [n]); self.assertEqual(certain['numerators'], [1])
        args = case(M=0, vectors=[(n,)])
        args['signal_packet']['means'][0]['raw'] = dict(real=1., imag=1.)
        record = D.evaluate(**args)[0]
        self.assertEqual(record['allocation']['shots'][0]['N_X'], n)
        self.assertIsNotNone(record['diagnostic']['deterministic_risk']['MSE_upper'])
        for bad in (True, 1., 0, -1):
            with self.assertRaises(ValueError): D.binomial_axis(0., bad, probability_rule=D.EXACT_AFFINE)

    def test_02_probability_rules_are_explicit_and_legacy_rounding_not_hidden(self):
        raw = -2**-54
        exact = D.binomial_axis(raw, 1, probability_rule=D.EXACT_AFFINE)
        old = D.binomial_axis(raw, 1, probability_rule=D.LEGACY_AFFINE)
        self.assertNotEqual(exact['probability'], old['probability'])
        self.assertEqual(a.exact_scalar(old['probability']), Fraction(1,2))
        row = dict(m=0, raw=dict(real=1-2**-53, imag=1.), N_X=2**1200+3, N_Y=1)
        exact_table = D.round_distribution(row, h=1., target_energy=0., radius=1.,
            probability_rule=D.EXACT_AFFINE, maximum_outcomes=1)
        legacy_table = D.round_distribution(row, h=1., target_energy=0., radius=1.,
            probability_rule=D.LEGACY_AFFINE, maximum_outcomes=1)
        self.assertEqual(exact_table['status'], 'UNKNOWN')
        self.assertEqual(legacy_table['cells'], 1)
        with self.assertRaises(ValueError): D.binomial_axis(0., 1, probability_rule='IMPLICIT')

    def test_03_round_zero_tie_wrong_branch_and_count_dependent_phase(self):
        row = dict(m=1, raw=dict(real=0., imag=0.), N_X=2, N_Y=2)
        table = D.round_distribution(row, h=1., target_energy=0., radius=1.,
            probability_rule=D.EXACT_AFFINE, maximum_outcomes=9)
        zero = sum(Fraction(v['weight'],table['denominator']) for v in table['outcomes'] if v['zero'])
        self.assertEqual(zero, Fraction(1,4))
        transitions = D.transition_distribution(table, 0.)
        tied = next(v for v in transitions if (v['X_plus'],v['Y_plus']) == (0,1))
        self.assertEqual(tied['status'], 'UNRESOLVED_BRANCH_TIE'); self.assertIsNone(tied['estimate'])
        origin = next(v for v in transitions if (v['X_plus'],v['Y_plus']) == (1,1))
        self.assertEqual(origin['status'], 'UNRESOLVED_ZERO_COORDINATES')
        wrong = D.transition_distribution(table, math.pi*.75)
        self.assertTrue(any(v['oracle_mismatch'] for v in wrong))
        self.assertGreater(len({v['delta'] for v in table['outcomes']}), 2)

    def test_04_small_full_count_enumeration_matches_when_all_events_are_good(self):
        args = case(M=2, vectors=[(1,2,3)])
        record = D.evaluate(**args)[0]; risk = record['diagnostic']['deterministic_risk']
        rows = [dict(m=m, raw=mean['raw'], N_X=n['N_X'], N_Y=n['N_Y']) for m,(mean,n) in
            enumerate(zip(args['signal_packet']['means'], args['allocations'][0]['shots']))]
        full = S.exact_counts(rows, h=10, M=2, prior=[-1,1],
            policy=args['statistical_contract']['simulation_policy'], maximum_states=1000)
        reference = S.summarize(full['trials'], target_energy_ha=0.,
            unresolved_loss=args['statistical_contract']['summary_options']['unresolved_loss'])
        self.assertEqual(a.exact_scalar(risk['all_good']['exact']), 1)
        self.assertAlmostEqual(float(a.exact_scalar(risk['MSE_upper'])), reference['all_trial_MSE_interval'][1], places=15)
        self.assertEqual(risk['decoder_certificate']['status'], 'VERIFIED_FINITE_BINARY64_SUPPORT')

    def test_05_bound_covers_full_enumeration_including_unsafe_and_unresolved(self):
        for vector in ((1,1), (2,1), (2,2)):
            args = case(vectors=[vector])
            for mean in args['signal_packet']['means']: mean['raw'] = dict(real=.25, imag=-.5)
            risk = D.evaluate(**args)[0]['diagnostic']['deterministic_risk']
            rows = [dict(m=m, raw=mean['raw'], N_X=n, N_Y=n) for m,(mean,n) in
                enumerate(zip(args['signal_packet']['means'], vector))]
            full = S.exact_counts(rows, h=10, M=1, prior=[-1,1],
                policy=args['statistical_contract']['simulation_policy'], maximum_states=1000)
            reference = S.summarize(full['trials'], target_energy_ha=0.,
                unresolved_loss=args['statistical_contract']['summary_options']['unresolved_loss'])
            self.assertIsNotNone(risk['MSE_upper'])
            self.assertGreaterEqual(float(a.exact_scalar(risk['MSE_upper']))+1e-12, reference['all_trial_MSE_interval'][1])
            bounds = risk['unresolved_probability_bounds']
            self.assertLessEqual(float(a.exact_scalar(bounds['lower'])), reference['unresolved_probability']+1e-12)
            self.assertGreaterEqual(float(a.exact_scalar(bounds['upper']))+1e-12, reference['unresolved_probability'])
            first = sum(a.exact_scalar(v['exact']) for v in risk['first_unsafe'])
            self.assertEqual(first+a.exact_scalar(risk['all_good']['exact']), 1)

    def test_06_pi_over_three_is_not_automatically_assumed_and_bad_guard_is_UNKNOWN(self):
        args = case(); args['options']['phase_radii'] = [math.pi/3]*2
        safe = D.evaluate(**args)[0]['diagnostic']['deterministic_risk']
        self.assertIsNotNone(safe['MSE_upper'])  # finite actual angles have positive margins
        args['options']['phase_radii'] = [2.6]*2
        for mean in args['signal_packet']['means']: mean['raw'] = dict(real=0., imag=0.)
        unsafe = D.evaluate(**args)[0]['diagnostic']['deterministic_risk']
        self.assertEqual(unsafe['status'], 'UNKNOWN')
        self.assertEqual(unsafe['reason'], 'GOOD_SET_NOT_CLOSED_UNDER_ORIGINAL_DECODER')
        self.assertIsNone(unsafe['MSE_upper'])

    def test_07_explicit_count_pair_scope_and_unadopted_loss_are_UNKNOWN(self):
        for changed in (dict(maximum_round_outcomes=1), dict(maximum_verification_pairs=1),
                        dict(model_loss_contract=dict(adopted=False, source=MOCK))):
            args = case(); args['options'].update(changed)
            self.assertEqual(D.evaluate(**args)[0]['diagnostic']['deterministic_risk']['status'], 'UNKNOWN')
        args = case(); args['statistical_contract']['summary_options']['unresolved_loss'] = dict(kind='UNDEFINED', source=MOCK)
        self.assertIsNone(D.evaluate(**args)[0]['diagnostic']['deterministic_risk']['MSE_upper'])
        args = case(); args['signal_packet']['means'][0]['raw'] = None
        self.assertEqual(D.evaluate(**args)[0]['diagnostic']['deterministic_risk']['reason'], 'MISSING_STATISTICAL_INPUT')

    def test_08_all_13_rounds_separate_axes_original_inputs_and_qualification_preserved(self):
        args = case(M=12); args['allocations'][0]['shots'][5]['N_Y'] = 2
        before = deepcopy(args)
        record = D.evaluate(**args)[0]; risk = record['diagnostic']['deterministic_risk']
        self.assertEqual(len(risk['rounds']), 13); self.assertIsNotNone(risk['MSE_upper'])
        self.assertEqual(record['allocation']['shots'][5]['N_Y'], 2)
        self.assertEqual(record['signal_input'], I.signal_input(args['signal_packet']))
        self.assertEqual(risk['scientific_energy_qualification'], 'UNKNOWN')
        for key in ('n_packet','sufficient_N','global_RMSE_bound','G'): self.assertIsNone(record[key])
        self.assertEqual(args, before); self.assertEqual(risk['MC_calls'], 0)

    def test_09_no_normalization_or_corrected_mean_double_counting(self):
        args = case(); expected = D.evaluate(**args)[0]['diagnostic']['deterministic_risk']['MSE_upper']
        for mean in args['signal_packet']['means']:
            mean['corrected'] = dict(real=100., imag=200.); mean['normalization']['log_B'] = 1000.
        self.assertEqual(D.evaluate(**args)[0]['diagnostic']['deterministic_risk']['MSE_upper'], expected)

    def test_10_energy_alias_loss_is_not_circularly_removed_and_risk_is_exact(self):
        args = case(); args['evaluation_target']['energy_ha'] = 2*math.pi/10
        risk = D.evaluate(**args)[0]['diagnostic']['deterministic_risk']
        self.assertGreater(float(a.exact_scalar(risk['MSE_upper'])), .1)
        self.assertEqual(risk['drift_refinement'], 'UNQUALIFIED_NOT_USED')
        tiny = Fraction(1, 10**500); root = D._sqrt_display_upper(tiny)
        self.assertGreater(root, 0); self.assertGreaterEqual(a.exact_scalar(root)**2, tiny)

    def test_11_cost_minimum_is_not_shot_minimum_existing_exact_accounting_used(self):
        args = case(vectors=[(1,5),(8,1)]); records = D.evaluate(**args)
        prices, qualifications, options = cost_fixture(args, records)
        result = D.compare(args['config'], records, prices, qualifications=qualifications,
            model_limit=dict(MSE_upper=dict(numerator=1,denominator=100), source=MOCK, scope='MODEL_MSE_FILTER_ONLY'), **options)
        compared = result['comparison']
        self.assertEqual([sum(s['N_X']+s['N_Y'] for s in r['allocation']['shots']) for r in records], [12,18])
        self.assertEqual([a.exact_scalar(v['G']) for v in compared['allocations']], [1002,216])
        self.assertEqual(compared['point_minimum']['allocation_id'], 'v1')
        self.assertEqual(compared['N_evidence_kind'], 'SYNTHETIC_ONLY')
        # No qualification was manufactured by the deterministic evaluator.
        self.assertTrue(all(r['n_packet'] is None for r in records))

    def test_12_missing_N_price_and_model_filter_never_filled_with_zero(self):
        args = case(vectors=[(1,5),(8,1)]); records = D.evaluate(**args)
        prices, qualifications, options = cost_fixture(args, records)
        limit = dict(MSE_upper=0, source=MOCK, scope='MODEL_MSE_FILTER_ONLY')
        with patch.object(B, 'cost', side_effect=AssertionError('unqualified G forbidden')):
            out = D.compare(args['config'], records, prices, qualifications=qualifications, model_limit=limit, **options)
            self.assertIsNone(out['comparison']['point_minimum']['allocation_id'])
            limit['MSE_upper'] = 1
            no_n = D.compare(args['config'], records, prices, qualifications={}, model_limit=limit, **options)
            self.assertIsNone(no_n['comparison']['point_minimum']['allocation_id'])
            no_c = D.compare(args['config'], records, [], qualifications=qualifications, model_limit=limit, **dict(options,cohorts={}))
            self.assertTrue(all(v['G'] is None for v in no_c['comparison']['allocations']))

    def test_13_source_candidate_law_and_external_qualification_mix_rejected(self):
        for change in ('source','candidate','law'):
            args = case()
            if change == 'source': args['signal_packet']['source'] = dict(fixture='OTHER_LD')
            if change == 'candidate': args['signal_packet']['configuration_id'] = 'OTHER_LD'
            if change == 'law': args['signal_packet']['means'][0]['law_binding']['definition']['model'] = 'OTHER_LAW'
            with patch.object(D,'binomial_axis',side_effect=AssertionError('invalid input was evaluated')):
                with self.assertRaises(ValueError): D.evaluate(**args)
        args = case(); records = D.evaluate(**args); prices, quals, options = cost_fixture(args, records)
        quals['v0']['rmse_evidence']['deterministic_input']['signal_input']['source'] = dict(fixture='OTHER_LD')
        with self.assertRaises(ValueError):
            D.compare(args['config'], records, prices, qualifications=quals,
                model_limit=dict(MSE_upper=1, source=MOCK, scope='MODEL_MSE_FILTER_ONLY'), **options)


if __name__ == '__main__': unittest.main()
