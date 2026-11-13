"""Synthetic whole-allocation integration only; no saved molecular numerics."""
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tests')]
from check_rpe_decoder_statistics import toy, POLICY, MOCK, UNDEFINED, count
from check_pr_rpe_rmse_controller import data
from trottertracks import rpe_shot_allocation as W
from trottertracks import rpe_decoder_fast as F

B, S = W.B, W.S


def fixture(vectors=((1, 5), (8, 1)), *, finite=False, model=False):
    config, means, _, target = toy(M=len(vectors[0])-1)
    contract = dict(id='MOCK_STATISTICAL_CONTRACT', source=MOCK, claim_level='SYNTHETIC_ONLY',
        simulation_policy=POLICY, allow_conditional_unqualified=True,
        summary_options=dict(unresolved_loss=UNDEFINED, iid_repetitions=True, confidence=.9))
    allocations = [dict(allocation_id=key, source=dict(fixture='ALLOCATION_'+key), shot_policy='USER_SUPPLIED',
        shots=[dict(m=m, N_X=n, N_Y=n) for m, n in enumerate(vector)])
        for key, vector in zip(('a', 'b'), vectors)]
    evaluations = []
    for allocation in allocations:
        packet, prices, cohorts = data(config, finite=finite, model=model)
        for row, shots in zip(packet['rounds'], allocation['shots']):
            row.update(N_X=shots['N_X'], N_Y=shots['N_Y'])
        packet['rmse_evidence']['statistical_contract'] = deepcopy(contract)
        evaluations.append(dict(configuration_id=config['id'], allocation=deepcopy(allocation),
            statistical_contract=deepcopy(contract), source=MOCK, n_packet=packet))
    extension = dict(lowering_id='MOCK_LOWERING', source=MOCK)
    coverage = dict(id='MOCK_SIMULTANEOUS_BOX', source=MOCK, configuration_id=config['id'],
        simultaneous=True, scope='ALL_SELECTED_ROUNDS_AXES', confidence=.9)
    bindings = []
    for price in prices:
        value = 1 if price['identity']['m'] == 0 else 100
        price['mean']['rz_count'] = value
        bindings.append(dict(identity=deepcopy(price['identity']), law_binding=deepcopy(price['law_binding']),
            axis=price['axis'], cohort=price['cohort'], price_source=deepcopy(price['source']), source=MOCK,
            profile_extension=deepcopy(extension), mean_interval=dict(
                quantity='FULL_WRAPPER_MEAN_COMPILED_RZ_INTERVAL', lower=value, upper=value,
                source=MOCK, qualification='QUALIFIED', evidence_kind='SIMULTANEOUS_MEAN_INTERVAL',
                domain=deepcopy(price['identity']['candidate']), coverage=deepcopy(coverage))))
    kwargs = dict(cohorts=cohorts, statistical_contract=contract, price_extension=extension,
        price_bindings=bindings, interval_contract=coverage)
    return config, means, target, allocations, evaluations, prices, kwargs


class AllocationTests(unittest.TestCase):
    def test_01_same_physical_ID_point_cost_and_outer_controller(self):
        c, _, _, allocations, evidence, prices, options = fixture()
        before = deepcopy((c, allocations, evidence, prices, options))
        out = W.compare(c, allocations, evidence, prices, **options)
        self.assertEqual([sum(r['N_X']+r['N_Y'] for r in v['shots']) for v in allocations], [12, 18])
        self.assertEqual([B.a.exact_scalar(v['G']) for v in out['allocations']], [1002, 216])
        self.assertEqual(out['point_minimum']['allocation_id'], 'b')
        self.assertEqual(out['strict_interval_winner'], 'b')
        self.assertEqual(out['outer_evaluation']['allocation_id'], 'b')
        self.assertEqual(out['outer_evaluation']['allocation'], allocations[1])
        self.assertEqual(out['outer_evaluation']['statistical_contract'], options['statistical_contract'])
        self.assertEqual(out['outer_selection']['winner']['configuration_id'], c['id'])
        self.assertTrue(out['menu_complete'])
        for v, ev in zip(out['allocations'], evidence):
            expected = B.cost(c, ev['n_packet'], prices, cohorts=options['cohorts'])
            self.assertEqual(v['evaluation']['cost'], expected['cost'])
            self.assertEqual(v['configuration_id'], c['id'])
        self.assertEqual(before, (c, allocations, evidence, prices, options))

    def test_02_prepare_MC_batch_and_diagnostic_not_qualified_N(self):
        c, means, target, allocations, _, prices, options = fixture()
        prepared = W.prepare(c, means, allocations, target=target,
            statistical_contract=options['statistical_contract'])
        with patch.object(B.p, 'run_missing', side_effect=AssertionError('no signal/compile provider')):
            fast = [W.diagnose(v, repetitions=13, seed=91, source=MOCK,
                batch_decoder=F.decode_sampled_counts) for v in prepared]
            scalar = W.diagnose(prepared[0], repetitions=13, seed=91, source=MOCK)
            accelerated = fast[0]['diagnostic']['simulation']['trials']
            self.assertTrue(all(v.get('decoder_acceleration_source') for v in accelerated))
            self.assertEqual([{k:v for k,v in row.items() if k != 'decoder_acceleration_source'}
                              for row in accelerated], scalar['diagnostic']['simulation']['trials'])
            self.assertEqual(fast[0]['diagnostic']['summary'], scalar['diagnostic']['summary'])
            out = W.compare(c, allocations, fast, prices, **options)
        for entry in out['allocations']:
            self.assertIsNone(entry['G']); self.assertIsNone(entry['diagnostic']['summary']['sufficient_N'])
        self.assertIsNone(out['outer_evaluation']); self.assertIsNone(out['point_minimum']['allocation_id'])

    def test_03_original_decoder_is_sequential_and_alias_loss_is_not_erased(self):
        # The identical late-round count does not determine the final energy.
        last = count((1, 1), (1, 1), 1)
        left = S.decode_counts([count((1, 1), (1, 1)), last], h=1, M=1, prior=[-4, 4])
        right = S.decode_counts([count((0, 1), (0, 1)), last], h=1, M=1, prior=[-4, 4])
        self.assertEqual(left['status'], S.DECODED); self.assertEqual(right['status'], S.DECODED)
        self.assertNotEqual(left['decoded']['rounds'][-1]['selected_branch'], right['decoded']['rounds'][-1]['selected_branch'])
        self.assertNotEqual(left['energy_estimate_ha'], right['energy_estimate_ha'])
        summary = S.summarize([right], target_energy_ha=left['energy_estimate_ha'], unresolved_loss=UNDEFINED)
        self.assertGreater(summary['all_observed_trials_RMSE'], 3)

    def test_04_unqualified_RMSE_and_missing_price_do_not_call_cost(self):
        c, _, _, allocations, evidence, prices, options = fixture()
        for ev in evidence: ev['n_packet']['rmse_evidence']['status'] = 'UNQUALIFIED'
        with patch.object(B, 'cost', side_effect=AssertionError('unqualified arithmetic')):
            out = W.compare(c, allocations, evidence, prices, **options)
        self.assertTrue(all(v['G_status'] == 'UNKNOWN' for v in out['allocations']))
        c, _, _, allocations, evidence, prices, options = fixture()
        with patch.object(B, 'cost', side_effect=AssertionError('missing-price arithmetic')):
            out = W.compare(c, allocations, evidence, prices[:-1], **options)
        self.assertIsNone(out['strict_interval_winner']); self.assertIsNone(out['outer_evaluation'])
        with patch.object(B, 'cost', side_effect=AssertionError('unknown-price arithmetic')):
            out = W.compare(c, allocations, evidence, [], cohorts={},
                statistical_contract=options['statistical_contract'])
        self.assertTrue(all(v['G_status'] == 'UNKNOWN' for v in out['allocations']))
        c, _, _, allocations, evidence, prices, options = fixture()
        for allocation, ev in zip(allocations, evidence):
            allocation['shot_policy'] = 'NOMINAL'; ev['allocation'] = deepcopy(allocation)
        with patch.object(B, 'cost', side_effect=AssertionError('nominal arithmetic')):
            out = W.compare(c, allocations, evidence, prices, **options)
        self.assertTrue(all(v['G'] is None for v in out['allocations']))

    def test_05_huge_integer_rational_cost_and_one_shot_gap(self):
        n = 2**1200+3
        c, _, _, allocations, evidence, prices, options = fixture(vectors=((n,), (n+1,)))
        third = dict(numerator=1, denominator=3)
        for price in prices: price['mean']['rz_count'] = third
        for binding in options['price_bindings']: binding['mean_interval'].update(lower=third, upper=third)
        out = W.compare(c, allocations, evidence, prices, **options)
        self.assertEqual(B.a.exact_scalar(out['point_minimum']['cost']), Fraction(2*n, 3))
        self.assertEqual(B.a.exact_scalar(out['allocations'][1]['G'])-B.a.exact_scalar(out['allocations'][0]['G']), Fraction(2, 3))
        self.assertEqual(out['strict_interval_winner'], 'a')

    def test_06_evaluation_allocation_source_contract_or_vector_mixing_rejected(self):
        for change in (
            lambda e: e.update(configuration_id='FOREIGN'),
            lambda e: e['allocation'].update(source={'fixture':'FOREIGN'}),
            lambda e: e.update(source={'fixture':'FOREIGN'}),
            lambda e: e['statistical_contract'].update(claim_level='OTHER'),
            lambda e: e['n_packet']['rmse_evidence']['statistical_contract'].update(id='OTHER'),
            lambda e: e['n_packet']['rounds'][0].update(N_X=123),
        ):
            c, _, _, allocations, evidence, prices, options = fixture()
            change(evidence[0])
            with self.assertRaises(ValueError): W.compare(c, allocations, evidence, prices, **options)
        for field in ('identity', 'law_binding'):
            c, _, _, allocations, evidence, prices, options = fixture()
            evidence[0]['n_packet']['rounds'][0][field] = {'fixture':'FOREIGN'}
            out = W.compare(c, allocations, evidence, prices, **options)
            self.assertIsNone(out['allocations'][0]['G'])
            self.assertIsNone(out['outer_evaluation'])
        c, _, _, allocations, evidence, prices, options = fixture()
        evidence[1]['n_packet']['evidence_kind'] = 'SCIENTIFIC'
        evidence[1]['n_packet']['rmse_evidence']['status'] = 'INFEASIBLE'
        with self.assertRaisesRegex(ValueError, 'synthetic and scientific'):
            W.compare(c, allocations, evidence, prices, **options)

    def test_07_D_sidecar_lowering_law_SOURCE_and_quantity(self):
        for field, value in (('price_source', {'fixture':'FOREIGN'}), ('law_binding', {}),
                             ('profile_extension', {'lowering_id':'OLD'})):
            c, _, _, allocations, evidence, prices, options = fixture()
            options['price_bindings'][0][field] = value
            with patch.object(B, 'cost', side_effect=AssertionError('unbound price arithmetic')):
                out = W.compare(c, allocations, evidence, prices, **options)
            self.assertTrue(all(v['G'] is None for v in out['allocations']))
        c, _, _, allocations, evidence, prices, options = fixture()
        prices[0]['quantity'] = 'FULL_WRAPPER_MEAN_STRUCTURAL_RZ_COUNT'
        out = W.compare(c, allocations, evidence, prices, **options)
        self.assertTrue(all(v['G'] is None for v in out['allocations']))

    def test_08_unknown_intervals_and_shared_price_cancellation(self):
        c, _, _, allocations, evidence, prices, options = fixture()
        for sidecar in options['price_bindings']: sidecar['mean_interval'] = None
        out = W.compare(c, allocations, evidence, prices, **options)
        self.assertEqual(out['point_minimum']['allocation_id'], 'b')
        self.assertIsNone(out['strict_interval_winner'])
        self.assertEqual(out['allocations'][0]['G_interval'], [None, None])
        c, _, _, allocations, evidence, prices, options = fixture(vectors=((1000, 1), (1000, 2)))
        for sidecar in options['price_bindings'][:2]:
            sidecar['mean_interval'].update(lower=0, upper=None)
        out = W.compare(c, allocations, evidence, prices, **options)
        self.assertTrue(all(v['G_interval'][1] is None for v in out['allocations']))
        self.assertEqual(out['strict_interval_winner'], 'a')
        gap = next(v for v in out['pairwise_shared_price_difference_intervals'] if v['left'] == 'a')
        self.assertEqual([B.a.exact_scalar(v) for v in gap['interval']], [-200, -200])
        self.assertIsNone(out['covariance']); self.assertFalse(out['missing_covariance_is_zero'])

    def test_09_unqualified_model_cohort_and_interval_coverage_remain_distinct(self):
        c, _, _, allocations, evidence, prices, options = fixture(model=True)
        prices[0]['model']['qualified_for_candidate'] = False
        out = W.compare(c, allocations, evidence, prices, **options)
        self.assertTrue(all(v['G'] is None for v in out['allocations']))
        c, _, _, allocations, evidence, prices, options = fixture(finite=True)
        self.assertIsNone(W.compare(c, allocations, evidence, prices, **options)['point_minimum']['cost'])
        out = W.compare(c, allocations, evidence, prices, price_policy='finite_cohort_scenario', **options)
        self.assertEqual(out['allocations'][0]['G_status'], 'FINITE_COHORT_CONDITIONAL_SCENARIO')
        self.assertIsNone(out['strict_interval_winner'])
        c, _, _, allocations, evidence, prices, options = fixture()
        options['price_bindings'][0]['mean_interval']['coverage']['simultaneous'] = False
        out = W.compare(c, allocations, evidence, prices, **options)
        self.assertIsNone(out['strict_interval_winner'])

    def test_10_missing_partial_inputs_ties_and_duplicate_allocations(self):
        c, means, target, allocations, evidence, prices, options = fixture()
        out = W.compare(c, allocations, evidence[:1], prices, **options)
        self.assertEqual(out['pending_allocation_ids'], ['b'])
        self.assertEqual(out['point_minimum']['allocation_id'], 'a')
        self.assertIsNone(out['outer_evaluation'])
        prepared = W.prepare(c, means[:-1], allocations, target=target,
            statistical_contract=options['statistical_contract'])
        with patch.object(S, 'simulate', side_effect=AssertionError('missing-input MC')):
            diagnostic = W.diagnose(prepared[0], repetitions=1, seed=9, source=MOCK)
        self.assertIsNone(diagnostic['diagnostic']['summary'])
        self.assertIsNone(diagnostic['n_packet'])
        c, _, _, allocations, evidence, prices, options = fixture(vectors=((1, 1), (1, 1)))
        out = W.compare(c, allocations, evidence, prices, **options)
        self.assertEqual(out['point_minimum']['ties'], ['a', 'b'])
        self.assertIsNone(out['strict_interval_winner'])
        with self.assertRaisesRegex(ValueError, 'unique allocation'):
            W.compare(c, [allocations[0]]*2, evidence, prices, **options)


if __name__ == '__main__': unittest.main(verbosity=2)
