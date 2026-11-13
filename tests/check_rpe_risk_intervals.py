"""Artificial finite distributions/MC only; no saved H4 numerical analysis."""
from copy import deepcopy
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tests')]
from check_rpe_signal_statistics_caller import synthetic, prepare, I, MOCK, count
from trottertracks import rpe_risk_intervals as R
from trottertracks import rpe_signal_statistics_caller as K
from trottertracks import rpe_decoder_fast as F

S, B = K.S, K.B
SUPPORT = dict(upper=16., source=MOCK, scope='ALL_DECODED_OUTCOMES')


def setup(n=8, *, model=False):
    args = synthetic(M=0)
    args[4]['summary_options']['confidence'] = None
    if model:
        args[4]['summary_options']['unresolved_loss'] = dict(kind='BOUNDED_INTERVAL', lower=1., upper=4., source=MOCK)
    entry = prepare(args)[0]
    failures = dict(resolved_probability=dict(numerator=1, denominator=20),
        resolved_contribution=dict(numerator=1, denominator=20))
    if model: failures['model_risk'] = dict(numerator=1, denominator=20)
    loss = None if not model else dict(adopted=True, source=MOCK,
            loss_support=dict(upper=4., source=MOCK, scope='ALL_DECODER_OUTCOMES'))
    family = R.confidence_family([dict(id='toy', entry=entry, repetitions=n, seed=7,
        source=MOCK, failures=failures, conditional_loss_support=deepcopy(SUPPORT), model_loss_contract=loss)],
        failure_probability=dict(numerator=1, denominator=5), source=MOCK)
    options = dict(family=family, member_id='toy')
    return entry, options


def revised(options, **changes):
    # Alternate artificial pre-run plans; never data-driven real-run updates.
    result = deepcopy(options)
    result['family']['members'][0].update(changes)
    return result


def decoded(loss=1.):
    return dict(status=S.DECODED, energy_estimate_ha=-.2+math.sqrt(loss), estimate_in_supplied_prior=True)


def unresolved():
    return dict(status='UNRESOLVED_BRANCH_TIE', energy_estimate_ha=None)


def evaluate(entry, options, trials):
    # Exact artificial trial ensembles: no RNG/decoder execution in these tests.
    with patch.object(S, 'simulate_prepared', return_value=dict(trials=deepcopy(trials), repetitions=len(trials), seed=7)):
        return K.diagnose(entry, repetitions=len(trials), seed=7, source=MOCK, risk_options=options)


class RiskIntervalTests(unittest.TestCase):
    def test_01_ratio_and_zero_denominator_conditions(self):
        out = R.conditional_ratio_interval(probability_interval=[.2,.8], contribution_interval=[.1,.3], loss_upper=2.)
        self.assertAlmostEqual(out['MSE'][0], .125); self.assertAlmostEqual(out['MSE'][1], 1.5)
        self.assertEqual(out['RMSE'], [math.sqrt(x) for x in out['MSE']])
        out = R.conditional_ratio_interval(probability_interval=[0,.8], contribution_interval=[0,.3], loss_upper=2.)
        self.assertEqual(out['status'], 'UNKNOWN'); self.assertIsNone(out['MSE'])
        clipped = R.conditional_ratio_interval(probability_interval=[.2,.8], contribution_interval=[0,3.], loss_upper=2.)
        self.assertEqual(clipped['MSE'][1], 2.)
        with self.assertRaises(ValueError):
            R.conditional_ratio_interval(probability_interval=[.9,.8], contribution_interval=[0,.3], loss_upper=2.)

    def test_02_all_resolved_preserves_point_and_does_not_adopt_all_event_loss(self):
        entry, opts = setup()
        opts = revised(opts, conditional_loss_support=dict(SUPPORT, upper=2.))
        before = deepcopy(entry)
        result = evaluate(entry, opts, [decoded()]*8)
        old = result['diagnostic']['summary']; risk = result['risk_intervals']
        self.assertEqual(risk['conditional_empirical_RMSE'], old['conditional_on_decoded_RMSE'])
        interval = risk['conditional_interval']
        radius = 2*math.sqrt(math.log(40)/(2*8))
        self.assertAlmostEqual(interval['contribution_interval'][0], max(0.,1-radius))
        self.assertAlmostEqual(interval['contribution_interval'][1], min(2.,1+radius))
        self.assertLessEqual(interval['MSE'][0], 1.); self.assertGreaterEqual(interval['MSE'][1], 1.)
        self.assertGreater(risk['unresolved']['probability_interval'][1], 0.)
        self.assertEqual(risk['model_risk']['status'], 'UNKNOWN')
        self.assertEqual(risk['scientific_energy_contract']['status'], 'UNKNOWN')
        self.assertEqual(entry, before)
        for key in ('n_packet', 'sufficient_N', 'global_RMSE_bound', 'G', 'winner'):
            self.assertIsNone(result[key])

    def test_03_unresolved_events_contribute_to_p_and_v_without_zero_all_event_loss(self):
        entry, opts = setup()
        result = evaluate(entry, opts, [decoded()]*4+[unresolved()]*4)
        old = result['diagnostic']['summary']; risk = result['risk_intervals']
        self.assertEqual(old['resolved_MSE_contribution'], .5)
        self.assertEqual(risk['conditional_empirical_RMSE'], 1.)
        self.assertEqual(risk['unresolved']['empirical_probability'], .5)
        self.assertIsNone(old['all_observed_trials_RMSE'])
        self.assertIsNone(risk['model_risk']['MSE'])
        self.assertLessEqual(risk['conditional_interval']['MSE'][0], 1.)
        self.assertGreaterEqual(risk['conditional_interval']['MSE'][1], 1.)

    def test_04_all_unresolved_p_lower_zero_is_UNKNOWN(self):
        entry, opts = setup()
        result = evaluate(entry, opts, [unresolved()]*8)['risk_intervals']
        self.assertIsNone(result['conditional_empirical_RMSE'])
        self.assertEqual(result['conditional_interval']['probability_interval'][0], 0.)
        self.assertEqual(result['conditional_interval']['reason'], 'RESOLVED_PROBABILITY_LOWER_BOUND_ZERO')
        self.assertIsNone(result['risk_upper_input']['conditional_MSE_upper'])

    def test_05_support_is_external_uniform_scope_not_sample_or_prior_range(self):
        entry, opts = setup()
        for support in (None, dict(upper=4.), dict(SUPPORT, scope='SUPPLIED_PRIOR_ONLY')):
            out = evaluate(entry, revised(opts, conditional_loss_support=support), [decoded()]*8)['risk_intervals']
            self.assertEqual(out['conditional_interval']['status'], 'UNKNOWN')
            self.assertIsNotNone(out['unresolved']['probability_interval'])
        with self.assertRaisesRegex(ValueError, 'no clipping'):
            evaluate(entry, revised(opts, conditional_loss_support=dict(SUPPORT, upper=.1)), [decoded()]*8)

    def test_06_original_zero_tie_and_decoded_wrong_branch_loss_are_retained(self):
        zero = S.decode_counts([count((1,2),(1,2))], h=1, M=0, prior=[-2,2])
        tie = S.decode_counts([count((2,2),(1,2)),count((0,2),(1,2),1)], h=1, M=1, prior=[-2,2])
        wrong = dict(decoded(100.), estimate_in_supplied_prior=False, artificial_wrong_branch=True)
        entry, opts = setup(n=3)
        opts = revised(opts, conditional_loss_support=dict(SUPPORT, upper=100.))
        out = evaluate(entry, opts, [zero, tie, wrong])
        self.assertEqual(out['diagnostic']['simulation']['trials'], [zero, tie, wrong])
        self.assertAlmostEqual(out['risk_intervals']['conditional_empirical_RMSE'], 10.)
        self.assertEqual(out['diagnostic']['summary']['estimate_outside_prior_count'], 1)
        self.assertIsNone(out['diagnostic']['summary']['branch_error_probability'])
        self.assertEqual(out['risk_intervals']['unresolved']['count'], 2)

    def test_07_model_envelope_reuses_existing_loss_CI_but_never_qualifies_energy_or_N(self):
        entry, opts = setup(model=True)
        trials = [decoded()]*4+[unresolved()]*4
        out = evaluate(entry, opts, trials)['risk_intervals']
        expected = S.summarize(trials, target_energy_ha=-.2,
            unresolved_loss=entry['statistical_contract']['summary_options']['unresolved_loss'],
            iid_repetitions=True, confidence=R._confidence(opts['family']['members'][0]['failures']['model_risk']),
            loss_support=opts['family']['members'][0]['model_loss_contract']['loss_support'])
        self.assertEqual(out['model_risk']['existing_envelope'], expected['population_RMSE_CI'])
        self.assertIsNotNone(out['risk_upper_input']['all_event_model_MSE_upper'])
        self.assertIsNone(out['risk_upper_input']['scientific_energy_RMSE_upper'])
        self.assertEqual(out['scientific_energy_contract']['status'], 'UNKNOWN')
        model = opts['family']['members'][0]['model_loss_contract']
        for alternative in (dict(model, adopted=False), dict(model, loss_support=None)):
            unknown = evaluate(entry, revised(opts, model_loss_contract=alternative), trials)['risk_intervals']
            self.assertEqual(unknown['model_risk']['status'], 'UNKNOWN')

    def test_08_finite_artificial_distribution_coverage(self):
        # Exhaust all d=0..n outcomes of IID D~Bernoulli(p), loss|D=1.
        # No MC; verifies simultaneous p,v and conditional coverage for tiny laws.
        n = 12; entry, opts = setup(n=n)
        opts = revised(opts, conditional_loss_support=dict(SUPPORT, upper=1.))
        by_count = [evaluate(entry, opts, [decoded()]*d+[unresolved()]*(n-d))['risk_intervals'] for d in range(n+1)]
        for probability in (0., .05, .3, .8, 1.):
            joint = conditional = 0.
            for d, out in enumerate(by_count):
                mass = math.comb(n,d)*probability**d*(1-probability)**(n-d)
                ci = out['conditional_interval']; pl, pu = ci['probability_interval']; vl, vu = ci['contribution_interval']
                if pl <= probability <= pu and vl <= probability <= vu: joint += mass
                # UNKNOWN imposes no claim, hence counts as not excluding truth.
                if ci['MSE'] is None or ci['MSE'][0] <= 1. <= ci['MSE'][1]: conditional += mass
            self.assertGreaterEqual(joint, .9-1e-12)
            if probability > 0: self.assertGreaterEqual(conditional, .9-1e-12)

    def test_09_joint_failure_budget_and_unplanned_MC_are_rejected_before_draws(self):
        entry, opts = setup()
        bad = deepcopy(opts); bad['family']['failure_probability'] = dict(numerator=1, denominator=100)
        for options in (bad, dict(opts, member_id='not-planned')):
            with patch.object(S, 'simulate_prepared', side_effect=AssertionError('MC forbidden')):
                with self.assertRaises(ValueError): K.diagnose(entry, repetitions=8, seed=7, source=MOCK, risk_options=options)
        for kwargs in (dict(repetitions=9, seed=7), dict(repetitions=8, seed=8)):
            with patch.object(S, 'simulate_prepared', side_effect=AssertionError('MC forbidden')):
                with self.assertRaises(ValueError): K.diagnose(entry, source=MOCK, risk_options=opts, **kwargs)
        self.assertGreaterEqual(R._confidence(dict(numerator=1, denominator=20)), .95)
        with self.assertRaises(ValueError): R._confidence(dict(numerator=1, denominator=10**100))

    def test_10_LD_SOURCE_law_allocation_and_target_slot_mixing_rejected(self):
        entry, opts = setup()
        for field in ('configuration_id', 'source', 'law', 'target', 'allocation'):
            other = deepcopy(entry)
            if field == 'configuration_id': other['configuration_id'] = 'OTHER_LD_CONFIGURATION'
            if field == 'source': other['signal_input']['source'] = dict(fixture='OTHER_SOURCE')
            if field == 'law': other['signal_input']['means'][0]['law_binding']['definition']['model'] = 'OTHER_LAW'
            if field == 'target': other['evaluation_target']['kind'] = 'OTHER_TARGET_MEANING'
            if field == 'allocation': other['allocation']['shots'][0]['N_X'] += 1
            with patch.object(S, 'simulate_prepared', side_effect=AssertionError('MC forbidden')):
                with self.assertRaises(ValueError):
                    K.diagnose(other, repetitions=8, seed=7, source=MOCK, risk_options=opts)

    def test_11_two_configurations_two_allocations_13_rounds_use_existing_caller(self):
        entries = []
        for ld in (1,2):
            args = synthetic(M=12); design = deepcopy(args[0]['spec'])
            design['L_D'] = ld; design['split']['L_D'] = ld
            config = B.generate(design); args[1]['configuration_id'] = config['id']
            for row, mean in zip(config['rounds'], args[1]['means']):
                mean.update(identity=deepcopy(row['identity']), law_binding=deepcopy(row['law_binding']))
            args = (config, *args[1:])
            for multiplier in (1,2):
                values = deepcopy(args); allocation = values[2][0]
                allocation['allocation_id'] = 'vector'+str(multiplier)
                for row in allocation['shots']:
                    row['N_X'] *= multiplier; row['N_Y'] *= multiplier
                entries.append(prepare(values)[0])
        failures = dict(resolved_probability=dict(numerator=1,denominator=100),
            resolved_contribution=dict(numerator=1,denominator=100))
        family = R.confidence_family([dict(id=str(i), entry=e, repetitions=3, seed=i, source=MOCK,
            failures=failures, conditional_loss_support=SUPPORT) for i,e in enumerate(entries)],
            failure_probability=.1, source=MOCK)
        for i, entry in enumerate(entries):
            old = K.diagnose(entry, repetitions=3, seed=i, source=MOCK, batch_decoder=F.decode_sampled_counts)
            new = K.diagnose(entry, repetitions=3, seed=i, source=MOCK, batch_decoder=F.decode_sampled_counts,
                risk_options=dict(family=family, member_id=str(i)))
            self.assertEqual(old['diagnostic'], new['diagnostic'])
            self.assertEqual(new['signal_input'], I.signal_input(dict(entry['signal_input'])))
            self.assertEqual(len(new['allocation']['shots']), 13)
            self.assertIsNone(new['G']); self.assertIsNone(new['n_packet'])
        self.assertNotEqual(entries[0]['configuration_id'], entries[2]['configuration_id'])

    def test_12_missing_inputs_remain_UNKNOWN_and_no_scientific_provider_runs(self):
        entry, opts = setup()
        simulation = dict(status='NOT_EVALUATED_MISSING_STATISTICAL_INPUT', trials=None)
        with patch.object(S, 'simulate_prepared', return_value=simulation), \
             patch.object(B, 'cost', side_effect=AssertionError('no G')), \
             patch.object(S.A, 'nominal_axis_shots', side_effect=AssertionError('no N producer')):
            out = K.diagnose(entry, repetitions=8, seed=7, source=MOCK, risk_options=opts)
        self.assertEqual(out['risk_intervals']['conditional_interval']['status'], 'UNKNOWN')
        self.assertIsNone(out['risk_intervals']['risk_upper_input']['conditional_MSE_upper'])
        self.assertIsNone(out['n_packet']); self.assertIsNone(out['G'])


if __name__ == '__main__': unittest.main()
