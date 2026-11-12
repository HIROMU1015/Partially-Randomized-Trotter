"""Final A/B/C synthetic integration. No molecular inputs, compile or real G."""
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from trottertracks import pr_rpe_rmse_integration as I
from trottertracks import mean_compiled_price as C
from demo_pr_rpe_rmse import artificial_design, mock_provider

A = I.A; B = I.B; p = I.p
SOURCE = dict(fixture='SYNTHETIC_ONLY_NO_MOLECULAR_DATA')


def inputs(M=1, kappa='1', system='H4'):
    d = artificial_design(); d.update(system=system)
    d['binding']['system'] = system
    d['profile'] = dict(p.PROFILE, compiler='MOCK_COMPILER')
    d['source'] = deepcopy(SOURCE)
    if system == 'H6':
        d.update(delta=1.6, formula='4th(m5_best)', prefix=7, PF=dict(method='FULL', prefix=19))
        early = []
        for m in range(4):
            for arm in ('PF', 'PR'):
                c = dict(system='H6', method='FULL' if arm == 'PF' else 'PR', formula=d['formula'],
                    prefix=19 if arm == 'PF' else 5, T=.1*(1 << m), q=1, delta=.1*(1 << m),
                    R_outer=None if arm == 'PF' else 32, K=None if arm == 'PF' else 2)
                c['id'] = p.candidate_id(c)
                early.append(dict(m=m, q_RPE=1 << m, q_PF=1, candidate=c, structure=p.structure(c)))
        d['early_rounds'] = early
    policy = A.rmse_policy(target_rmse='20', qpe_rmse='10', evolution_rmse='1', error_rule='paper_quadrature',
                           normalization_rule='finite_B_diagnostic', source=SOURCE)
    tail = dict(lambda_R='1/2', binding=deepcopy(d['binding']), prefix=d['prefix'], source=deepcopy(SOURCE))
    return dict(M=M, kappa_paper=kappa, design=d, tail_norm=tail, policy=policy)


def prepare(args, method='PR'):
    schedule = A.schedule(**args)
    evaluation = A.evaluate(schedule, mean_provider=mock_provider)  # artificial signals only
    handoff = A.b_handoff(schedule, evaluation, SOURCE, evidence_domain='SYNTHETIC_ONLY')
    packet = next(v for v in handoff['packets'] if v['spec']['method'] == method)
    config = B.generate(I.a_spec(args, method))
    normalized = I.n_from_a(config, packet, inputs=args)
    return config, normalized, packet, schedule


def mock_qualified_n(normalized, *, shots=3):
    """Separate given synthetic N, never taken from A.nominal_N_X/Y."""
    n = deepcopy(normalized)
    n.update(status='RMSE_SCHEDULE_EVALUATED', evidence_kind='SYNTHETIC_ONLY',
             source=dict(fixture='GIVEN_MOCK_SUFFICIENT_N_NOT_NOMINAL'))
    for row in n['rounds']:
        row.update(N_X=shots, N_Y=shots)
    n['rmse_evidence'].update(source=deepcopy(n['source']), status='QUALIFIED', rmse_bound_ha=1,
        covers_state_target_alias_prior_decoder_all_rounds=True, finite_K_status='QUALIFIED', signed_piece_kappa_status='QUALIFIED')
    return n


def toy_prices(config, targets=None):
    """One-state CLOSED ARTIFICIAL reward kernel; no physical closure assertion."""
    records = []; selection = {}
    states = ('MOCK_STATE',); identity = C.MeanKernel.identity(states)
    for row in config['rounds']:
        if config['spec']['method'] != 'PR':
            continue
        E = row['structure']['event_count']
        reward = 1. if targets is None else targets[row['m']] / E
        event = C.MeanKernel(states, [[1.]], [reward])
        pieces = [C.MeanPiece(w, r, identity, event, identity, SOURCE)
                  for w, r in zip(row['structure']['signed_weights'], row['structure']['allocation'])]
        mean = C.full_wrapper_mean(row['q_PF'], pieces, identity, identity, [1.], [0.])['mean']
        for axis in p.AXES:
            record = dict(**p.measurement(row, axis, saved_sampler=True), identity=deepcopy(row['identity']),
                law_binding=deepcopy(row['law_binding']), axis=axis, classification='DIRECT_COMPILED_FINITE_SAMPLE',
                cohort='SYNTHETIC_COHORT', n_paths=999, expectation='FINITE_COHORT_ONLY', mean=dict(rz_count=mean),
                source=deepcopy(SOURCE), model=None, normalization_is_price_multiplier=False,
                uncertainty=dict(joint_covariance=None, R_q_systematic=None, total_CI=None, all_law_expectation=None))
            records.append(record); selection[row['candidate']['id'], axis] = record['cohort']
    return records, selection


def run_case(args, targets=None):
    config, normalized, packet, schedule = prepare(args)
    n = mock_qualified_n(normalized, shots=1)
    records, cohorts = toy_prices(config, targets)
    lookup = I.MeanPriceLookup(config, records, selections=cohorts)
    prices = lookup()
    result = B.cost(config, n, prices, cohorts=cohorts, price_policy='finite_cohort_scenario')
    return config, result


class IntegrationTests(unittest.TestCase):
    def test_A_B_C_final_end_to_end_with_separate_mock_N(self):
        config, normalized, packet, _ = prepare(inputs())
        before = deepcopy(packet); records, cohorts = toy_prices(config)
        prices = I.MeanPriceLookup(config, records, selections=cohorts)()
        blocked = B.cost(config, normalized, prices, cohorts=cohorts, price_policy='finite_cohort_scenario')
        self.assertFalse(blocked['eligible']); self.assertIsNone(blocked['cost'])
        self.assertTrue(all(v['N_X'] is None and v['nominal_N_X'] is not None for v in normalized['rounds']))
        n = mock_qualified_n(normalized)
        eligible = B.assess(config, n, prices, cohorts=cohorts, price_policy='finite_cohort_scenario')
        self.assertTrue(eligible['eligible']); self.assertIsNone(eligible['cost'])
        result = B.cost(config, n, prices, cohorts=cohorts, price_policy='finite_cohort_scenario')
        self.assertEqual(B.a.exact_scalar(result['cost']), sum(3*B.a.exact_scalar(v['mean']['rz_count']) for v in prices))
        self.assertEqual(packet, before); self.assertFalse(normalized['nominal_promoted_to_sufficient'])
        self.assertFalse(result['full_RPE_qualified'])

    def test_A_fixed_grid_differs_from_B_ceil_and_is_not_relaxed(self):
        args = inputs(); args['design']['delta'] = .15
        self.assertEqual(B._ceil(Fraction(.1)/Fraction(.15)), 1)
        with self.assertRaisesRegex(ValueError, 'Nonintegral PF grid'):
            I.a_spec(args)

    def test_H6_eleven_pieces_and_m0_3_exceptions_preserved(self):
        args = inputs(M=5, system='H6'); config, normalized, _, schedule = prepare(args)
        arows = [v for v in schedule['rows'] if p.arm(v['identity']['candidate']) == 'PR']
        for row, request in zip(config['rounds'], arows):
            c = request['identity']['candidate']
            self.assertEqual(row['physical_candidate'], c)
            self.assertEqual(row['structure'], request['identity']['structure'])
            self.assertEqual(len(row['structure']['allocation']), 11)
            self.assertEqual(row['mapping']['kappa_applied'], row['m'] >= 4)
            self.assertEqual(row['candidate']['prefix'], 5 if row['m'] < 4 else 7)
            self.assertEqual(B.a.exact_scalar(row['actual_delta_pf']), Fraction(c['delta']))
        self.assertTrue(all(r['candidate']['R_outer'] == 32 for r in config['rounds'][:4]))
        records, cohorts = toy_prices(config)
        prices = I.MeanPriceLookup(config, records, selections=cohorts)()
        self.assertTrue(all(v['price_status'] == 'OUTSIDE_DOMAIN' and v['mean']['rz_count'] is None for v in prices[:8]))
        self.assertTrue(all(v['price_status'] == 'SAVED_DIRECT_FINITE_COHORT' for v in prices[8:]))

    def test_A_and_B_common_event_counts_match_on_regular_rounds(self):
        config, _, _, _ = prepare(inputs(M=4, kappa='12'))
        for row in config['rounds']:
            mapped = A.paper_event_mapper(kappa_paper='12', lambda_R='1/2', T=Fraction(row['physical_candidate']['T']),
                q_PF=row['q_PF'], signed_weights=row['structure']['signed_weights'])
            self.assertEqual(mapped['event_count'], row['structure']['event_count'])
            self.assertEqual(mapped['allocation'], row['structure']['allocation'])
        self.assertEqual(config['spec']['kappa']['kind'], A.KAPPA_DEFINITION)

    def test_M_and_kappa_bind_entire_schedule_PF_has_no_kappa(self):
        c1, _, _, _ = prepare(inputs(M=1, kappa='1'))
        c2, _, _, _ = prepare(inputs(M=2, kappa='1'))
        c3, _, _, _ = prepare(inputs(M=1, kappa='2'))
        self.assertEqual(len({c1['id'], c2['id'], c3['id']}), 3)
        self.assertNotEqual(c1['rounds'][0]['identity'], c2['rounds'][0]['identity'])
        p1 = prepare(inputs(kappa='1'), 'PF')[0]; p2 = prepare(inputs(kappa='2'), 'PF')[0]
        self.assertEqual(p1['id'], p2['id']); self.assertIsNone(p1['spec']['kappa'])
        self.assertNotEqual(p1['id'], c1['id'])
        self.assertTrue(all(v['mean']['rz_count'] is None for v in I.MeanPriceLookup(p1, [])()))
        b0_args = inputs(); b0_args['design']['PF'] = dict(method='B0', prefix=7)
        b0, n0, _, _ = prepare(b0_args, 'PF')
        self.assertEqual(b0['rounds'][0]['candidate']['physical_method'], 'B0')
        self.assertIn('UNQUALIFIED_A_PF_BASELINE_SCOPE', B.assess(b0, mock_qualified_n(n0), [], cohorts={})['reasons'])
        with self.assertRaisesRegex(ValueError, 'FULL_H_RTE is not B3'):
            I.a_spec(inputs(), 'FULL_H_RTE')

    def test_no_roundwise_hybrid_and_M_kappa_menu_selection(self):
        args1 = inputs(kappa='1'); args1['tail_norm']['lambda_R'] = '10'
        args2 = inputs(kappa='4'); args2['tail_norm']['lambda_R'] = '10'
        c1, r1 = run_case(args1, [1, 100])
        c2, r2 = run_case(args2, [9, 1])
        selected = B.select([c1, c2], [r1, r2])
        self.assertEqual(selected['winner']['configuration_id'], c2['id'])
        self.assertEqual(B.a.exact_scalar(selected['winner']['cost']), 20)
        self.assertNotEqual(B.a.exact_scalar(selected['winner']['cost']), 4)
        args3 = inputs(M=2, kappa='4'); args3['tail_norm']['lambda_R'] = '10'
        c3, r3 = run_case(args3)
        self.assertTrue(B.select([c1, c2, c3], [r1, r2, r3])['menu_complete'])
        foreign = deepcopy(args3); foreign['tail_norm']['lambda_R'] = '1'
        c4, r4 = run_case(foreign)
        with self.assertRaisesRegex(ValueError, 'Split/SOURCE'):
            B.select([c1, c4], [r1, r4])

    def test_changed_A_candidate_policy_law_and_nominal_promotion_rejected(self):
        args = inputs(); config, _, packet, _ = prepare(args)
        for mutate in (lambda v: v['N_packet']['rounds'][0].update(N_X=17),
                       lambda v: v['N_packet']['rounds'][0]['identity']['candidate'].update(R_outer=999),
                       lambda v: v['N_packet']['rounds'][0]['law_binding'].update(K=4),
                       lambda v: v['N_packet']['policy'].update(family='HIGH_PROBABILITY_RPE')):
            bad = deepcopy(packet); mutate(bad)
            with self.assertRaises(ValueError): I.n_from_a(config, bad, inputs=args)

    def test_C_foreign_identity_law_compiler_and_old_schedule_not_relabelled(self):
        config, _, _, _ = prepare(inputs()); records, cohorts = toy_prices(config)
        for mutate in (lambda v: v['identity']['profile'].update(compiler='FOREIGN'),
                       lambda v: v['identity']['candidate'].update(R_outer=999),
                       lambda v: v['law_binding'].update(configuration_id='FOREIGN')):
            bad = deepcopy(records); mutate(bad[0])
            with self.assertRaises(ValueError): I.MeanPriceLookup(config, bad, selections=cohorts)
        other = prepare(inputs(kappa='2'))[0]
        missing = I.MeanPriceLookup(other, records)()
        self.assertTrue(all(v['mean']['rz_count'] is None for v in missing))

    def test_C_model_false_and_missing_remain_unqualified(self):
        config, normalized, _, _ = prepare(inputs()); records, cohorts = toy_prices(config)
        for v in records:
            row = next(r for r in config['rounds'] if r['identity'] == v['identity'])
            v['classification'] = 'MODEL_PREDICTION'
            v['model'] = dict(qualified_for_candidate=False, q_variable='q_PF', q=row['q_PF'],
                domain=deepcopy(row['candidate']), binding=deepcopy(config['spec']['binding']),
                compiler=config['spec']['profile']['compiler'], source=SOURCE, validation_source=SOURCE, coverage='MOCK_ONLY')
        before = deepcopy(records); prices = I.MeanPriceLookup(config, records, selections=cohorts)()
        self.assertTrue(all(v['model']['qualified_for_candidate'] is False for v in prices))
        self.assertEqual(records, before)
        n = mock_qualified_n(normalized)
        self.assertIsNone(B.cost(config, n, prices, cohorts=cohorts, price_policy='finite_cohort_scenario')['cost'])
        missing = I.MeanPriceLookup(config, [])()
        self.assertIsNone(B.cost(config, n, missing, cohorts=cohorts, price_policy='finite_cohort_scenario')['cost'])

    def test_passive_C_provider_no_price_calculation_or_new_signal_N(self):
        args = inputs(); config, normalized, packet, _ = prepare(args); records, cohorts = toy_prices(config)
        with (patch.object(C, 'full_wrapper_mean', side_effect=AssertionError('No new mean')),
              patch.object(C, 'local_residual_mean', side_effect=AssertionError('No residual evaluation')),
              patch.object(C, 'combine_mean', side_effect=AssertionError('No model prediction')),
              patch.object(A, 'nominal_axis_shots', side_effect=AssertionError('No N reanalysis')),
              patch.object(A, 'existing_finite_mean', side_effect=AssertionError('No physical signal'))):
            same = I.n_from_a(config, packet, inputs=args)
            prices = I.MeanPriceLookup(config, records, selections=cohorts)()
        self.assertEqual(same, normalized)
        self.assertTrue(all(v['schedule_mapping']['context']['schedule_id'] == config['id'] for v in prices))
        for row, value in zip(records, prices):
            for field in ('mean', 'source', 'cohort', 'identity', 'law_binding'):
                self.assertEqual(row[field], value[field])

    def test_scalar_factors_not_counted_twice_and_huge_mock_N_exact(self):
        config, normalized, _, _ = prepare(inputs(M=0)); records, cohorts = toy_prices(config)
        prices = I.MeanPriceLookup(config, records, selections=cohorts)()
        N = 2**1200 + 3; n = mock_qualified_n(normalized, shots=N)
        result = B.cost(config, n, prices, cohorts=cohorts, price_policy='finite_cohort_scenario')
        self.assertEqual(B.a.exact_scalar(result['cost']), 2*N*B.a.exact_scalar(prices[0]['mean']['rz_count']))
        self.assertFalse(result['normalization_is_price_multiplier'])
        self.assertFalse(result['classical_samples_are_shot_multipliers'])

    def test_source_A_C_implementations_byte_preserved(self):
        sources = [(I.A_RELEASE, 'src/trottertracks/pr_rpe_rmse.py'),
                   (I.C_SOURCE, 'src/trottertracks/mean_compiled_price.py')]
        for commit, path in sources:
            self.assertEqual((ROOT/path).read_bytes(), subprocess.check_output(['git', 'show', commit+':'+path], cwd=ROOT))


if __name__ == '__main__':
    unittest.main(verbosity=2)
