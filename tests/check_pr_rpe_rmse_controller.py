"""Artificial H/lambda/N/C only. No molecular numerical evaluations."""
from copy import deepcopy
from fractions import Fraction
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks import pr_rpe_rmse_controller as m


def spec(method='PR', M=1, kappa=1, L_D=1, delta=1):
    source = dict(fixture='SYNTHETIC_ONLY')
    profile = deepcopy(m.p.PROFILE); profile['compiler'] = 'MOCK_COMPILER'
    policy = dict(family='PR_PAPER_RMSE', id='MOCK_RMSE_CONTRACT', source=source,
                  target_rmse_ha=dict(numerator=1, denominator=10), error_contract={'fixture': 'MOCK_GLOBAL_RMSE'},
                  scientific_status='PROPOSED_NOT_ADOPTED')
    s = dict(binding=dict(system='MOCK', hamiltonian='MOCK_H', state='MOCK_STATE', snapshot='MOCK_SNAPSHOT',
             decomposition='MOCK_DECOMPOSITION', source=source),
        split=dict(L_D=L_D, source=source, randomized_scope='TAIL_AFTER_PREFIX', lambda_R=1, lambda_source=source,
                   one_body_policy='DETERMINISTIC', deterministic_scope='PREFIX'),
        method=method, L_D=L_D, delta_pf=delta, formula='2nd', h=1, M=M,
        kappa=dict(kind='PAPER_TOTAL_ROTATIONS', value=kappa, definition_source=source),
        event_mapping=dict(kind=m.DRAFT_MAPPER, source=source), finite_K=2,
        law=dict(model='ideal_finite_RTE_law', sampling='MOCK_IDEAL', source=source), rmse_policy=policy, profile=profile)
    if method == 'PF':
        s.update(kappa=None, finite_K=None, event_mapping=dict(kind='NONE', source=source))
        s['split'].update(deterministic_scope='FULL_H', randomized_scope='NONE')
        s['law']['model'] = 'deterministic'
    if method == 'FULL_H_RTE':
        s.update(L_D=0, delta_pf=None, formula=None)
        s['split'].update(L_D=0, randomized_scope='FULL_H_INCLUDING_ONE_BODY', one_body_policy='RANDOMIZED', deterministic_scope='NONE')
    return s


def data(config, *, shots=1, price=1, finite=False, model=False):
    source = dict(fixture='SYNTHETIC_ONLY')
    packet = dict(configuration_id=config['id'], policy=deepcopy(config['spec']['rmse_policy']), source=source,
        status='RMSE_SCHEDULE_EVALUATED', evidence_kind='SYNTHETIC_ONLY', rounds=[],
        rmse_evidence=dict(source=source, status='QUALIFIED', rmse_bound_ha=dict(numerator=1, denominator=20),
            error_contract=deepcopy(config['spec']['rmse_policy']['error_contract']),
            covers_state_target_alias_prior_decoder_all_rounds=True, finite_K_status='QUALIFIED', signed_piece_kappa_status='QUALIFIED'))
    prices = []; cohorts = {}
    for row in config['rounds']:
        packet['rounds'].append(dict(m=row['m'], identity=deepcopy(row['identity']), law_binding=deepcopy(row['law_binding']),
                                    N_X=shots, N_Y=shots))
        for axis in m.p.AXES:
            c = dict(identity=deepcopy(row['identity']), law_binding=deepcopy(row['law_binding']), axis=axis,
                **m.p.measurement(row, axis), cohort='MOCK_COHORT', n_paths=999,
                classification='MODEL_PREDICTION' if model else 'DIRECT_COMPILED_DETERMINISTIC' if config['spec']['method'] == 'PF' else 'DIRECT_COMPILED_FINITE_SAMPLE',
                expectation='FINITE_COHORT_ONLY' if finite else 'DETERMINISTIC' if config['spec']['method'] == 'PF' else 'QUALIFIED_EXPECTATION',
                mean=dict(rz_count=price), source=source, model=None,
                uncertainty=dict(shared_calibration_group='MOCK_SHARED', joint_covariance=None, total_CI=None))
            if model:
                var = 'event_count' if config['spec']['method'] == 'FULL_H_RTE' else 'q_PF'
                c['model'] = dict(qualified_for_candidate=True, q_variable=var,
                    q=row['structure']['event_count'] if var == 'event_count' else row['q_PF'], source=source,
                    validation_source=source, coverage='MOCK_ONLY', domain=deepcopy(row['candidate']),
                    binding=deepcopy(config['spec']['binding']), compiler=config['spec']['profile']['compiler'])
            prices.append(c); cohorts[row['candidate']['id'], axis] = c['cohort']
    return packet, prices, cohorts


def evaluate(c, values, policy='qualified_expectation'):
    n, prices, cohorts = values
    return m.cost(c, n, prices, cohorts=cohorts, price_policy=policy)


class ControllerTests(unittest.TestCase):
    def test_shared_kappa_exact_schedule_integer_rounding(self):
        c = m.generate(spec(M=2, delta=dict(numerator=3, denominator=4)))
        self.assertEqual([m.a.exact_scalar(v['T']) for v in c['rounds']], [1, 2, 4])
        self.assertEqual([v['q_PF'] for v in c['rounds']], [2, 3, 6])
        self.assertEqual([v['R_outer'] for v in c['rounds']], [1, 2, 3])
        self.assertEqual([v['structure']['event_count'] for v in c['rounds']], [2, 6, 18])
        self.assertTrue(all(v['mapping']['kappa'] == c['spec']['kappa'] for v in c['rounds']))
        self.assertFalse(c['scientific_definition_adopted'])
        c['rounds'][1]['R_outer'] = 7
        with self.assertRaisesRegex(ValueError, 'mutated'): m.assess(c, None, [], cohorts={})

    def test_M_changes_all_round_identities_and_N_not_reused(self):
        c1 = m.generate(spec(M=1)); c2 = m.generate(spec(M=2))
        self.assertEqual(len(c2['rounds']), 3); self.assertNotEqual(c1['id'], c2['id'])
        self.assertNotEqual(c1['rounds'][0]['identity'], c2['rounds'][0]['identity'])
        old = data(c1); new = data(c2)
        result = m.assess(c2, old[0], new[1], cohorts=new[2])
        self.assertFalse(result['eligible']); self.assertIn('N_CONFIGURATION_OR_POLICY_MISMATCH', result['reasons'])

    def test_signed_schedule_and_existing_allocation_are_reused(self):
        s = spec(M=1, kappa=32); s['formula'] = '4th(m5_best)'
        c = m.generate(s)
        for row in c['rounds']:
            expected = m.p.structure(dict(formula=s['formula'], method='PR', q=row['q_PF'], R_outer=row['R_outer']))
            self.assertEqual(row['structure'], expected)
            self.assertTrue(any(v < 0 for v in expected['signed_weights']))
            self.assertEqual(sum(expected['allocation']), row['R_outer'])
            self.assertEqual(expected['event_count'], row['q_PF'] * row['R_outer'])

    def test_unknown_mapper_remains_unresolved_and_custom_mapper_echo_checked(self):
        s = spec(); s['event_mapping']['kind'] = 'MOCK_A_DEFINED_MAPPING'
        c = m.generate(s)
        self.assertTrue(all(r['resolution'] == 'NOT_EVALUATED' for r in c['rounds']))
        self.assertIn('UNRESOLVED_KAPPA_MAPPING', m.assess(c, None, [], cohorts={})['reasons'])
        def resolver(s, row):
            return dict(kind=s['event_mapping']['kind'], source=s['event_mapping']['source'], kappa=s['kappa'],
                        R_outer=3, event_count=row['q_PF'] * 3)
        c = m.generate(s, event_mapper=resolver)
        self.assertTrue(evaluate(c, data(c))['eligible'])
        def bad(s, row):
            out = resolver(s, row); out['kappa'] = {'kind': 'OTHER'}; return out
        with self.assertRaisesRegex(ValueError, 'common kappa'): m.generate(s, event_mapper=bad)

    def test_generation_qualification_and_aggregation_are_separate(self):
        c = m.generate(spec()); n, prices, cohorts = data(c)
        status = m.assess(c, n, prices, cohorts=cohorts)
        self.assertTrue(status['eligible']); self.assertIsNone(status['cost'])
        selected = m.select([c], [status])
        self.assertFalse(selected['menu_complete']); self.assertIsNone(selected['winner']['cost'])
        for change in ('finite_K_status', 'signed_piece_kappa_status', 'covers_state_target_alias_prior_decoder_all_rounds'):
            bad = deepcopy(n); bad['rmse_evidence'][change] = None
            self.assertIsNone(m.cost(c, bad, prices, cohorts=cohorts)['cost'])

    def test_supplied_RMSE_bound_and_unstable_infeasible_statuses(self):
        c = m.generate(spec()); n, prices, cohorts = data(c)
        n['rmse_evidence']['rmse_bound_ha'] = 1
        result = m.cost(c, n, prices, cohorts=cohorts)
        self.assertEqual(result['status'], 'INFEASIBLE'); self.assertIsNone(result['cost'])
        bad = deepcopy(n); bad['configuration_id'] = 'FOREIGN'
        invalid = m.assess(c, bad, prices, cohorts=cohorts)
        self.assertNotEqual(invalid['status'], 'INFEASIBLE')
        self.assertFalse(m.select([c], [invalid])['menu_complete'])
        duplicate = deepcopy(n); duplicate['rounds'].append(deepcopy(duplicate['rounds'][0]))
        self.assertNotEqual(m.assess(c, duplicate, prices, cohorts=cohorts)['status'], 'INFEASIBLE')
        n['rmse_evidence']['status'] = 'NUMERICAL_UNSTABLE'
        self.assertEqual(m.assess(c, n, prices, cohorts=cohorts)['status'], 'NUMERICAL_UNSTABLE')

    def test_high_probability_N_and_unadopted_scientific_contract_not_reinterpreted(self):
        c = m.generate(spec()); n, prices, cohorts = data(c)
        bad = deepcopy(n); bad['policy']['family'] = 'HIGH_PROBABILITY_RPE'
        self.assertFalse(m.assess(c, bad, prices, cohorts=cohorts)['eligible'])
        bad = deepcopy(n); bad['status'] = 'CONDITIONAL_ROUND_PASS'
        self.assertFalse(m.assess(c, bad, prices, cohorts=cohorts)['eligible'])
        bad = deepcopy(n); bad['evidence_kind'] = 'SCIENTIFIC'
        self.assertIn('RMSE_ERROR_CONTRACT_NOT_SCIENTIFICALLY_ADOPTED', m.assess(c, bad, prices, cohorts=cohorts)['reasons'])

    def test_missing_null_duplicate_price_never_filled_or_multiplied(self):
        c = m.generate(spec()); n, prices, cohorts = data(c)
        for bad in ([], prices[:-1], prices + [deepcopy(prices[0])]):
            self.assertIsNone(m.cost(c, n, bad, cohorts=cohorts)['cost'])
        bad = deepcopy(prices); bad[0]['mean']['rz_count'] = None
        self.assertIsNone(m.cost(c, n, bad, cohorts=cohorts)['cost'])
        bad[0]['mean']['rz_count'] = float('inf')
        self.assertEqual(m.cost(c, n, bad, cohorts=cohorts)['status'], 'NUMERICAL_UNSTABLE')
        result = evaluate(c, (n, prices, cohorts))
        self.assertEqual(m.a.exact_scalar(result['cost']), 4)  # no n_paths=999 or B multiplier
        self.assertFalse(result['normalization_is_price_multiplier'])
        for field in ('normalization_is_price_multiplier', 'classical_samples_are_shot_multipliers'):
            bad = deepcopy(prices); bad[0][field] = True
            self.assertFalse(m.assess(c, n, bad, cohorts=cohorts)['eligible'])

    def test_full_identity_law_source_compiler_cohort_measurement_mismatch(self):
        c = m.generate(spec()); n, prices, cohorts = data(c)
        changes = [lambda v: v['identity']['binding'].update(snapshot='FOREIGN'),
            lambda v: v['identity']['configuration']['spec']['kappa'].update(value=99),
            lambda v: v['identity']['profile'].update(compiler='FOREIGN'),
            lambda v: v['law_binding'].update(configuration_id='FOREIGN'),
            lambda v: v.update(source=None), lambda v: v.update(outcome_convention='OPPOSITE'),
            lambda v: v['units'].update(rz_count='WRONG'), lambda v: v.update(cohort='UNSELECTED')]
        for change in changes:
            bad = deepcopy(prices); change(bad[0])
            self.assertFalse(m.assess(c, n, bad, cohorts=cohorts)['eligible'])
        bad = deepcopy(n); bad['rounds'][0]['N_X'] = 1.
        self.assertFalse(m.assess(c, bad, prices, cohorts=cohorts)['eligible'])
        bad = deepcopy(n); bad['rounds'].append(deepcopy(bad['rounds'][0]))
        self.assertFalse(m.assess(c, bad, prices, cohorts=cohorts)['eligible'])

    def test_MODEL_domain_and_C1_false_remain_disqualified(self):
        c = m.generate(spec()); n, prices, cohorts = data(c, model=True)
        self.assertTrue(m.assess(c, n, prices, cohorts=cohorts)['eligible'])
        for change in (lambda v: v['model'].update(qualified_for_candidate=False),
                       lambda v: v['model']['domain'].update(R_outer=999),
                       lambda v: v['model'].update(validation_source=None),
                       lambda v: v['model'].update(q_variable='q_RPE')):
            bad = deepcopy(prices); change(bad[0])
            self.assertIsNone(m.cost(c, n, bad, cohorts=cohorts)['cost'])

    def test_finite_cohort_is_explicit_conditional_and_unknown_law_distance_stays_false(self):
        c = m.generate(spec()); n, prices, cohorts = data(c, finite=True)
        for value in prices:
            value['law_context'].update(price_law='MOCK_FINITE_SAMPLER', ideal_sampler_distance_qualified=False)
        self.assertFalse(m.assess(c, n, prices, cohorts=cohorts)['eligible'])
        result = m.cost(c, n, prices, cohorts=cohorts, price_policy='finite_cohort_scenario')
        self.assertTrue(result['eligible']); self.assertEqual(result['cost_scope'], 'FINITE_COHORT_CONDITIONAL_SCENARIO')
        self.assertFalse(result['full_RPE_qualified']); self.assertIsNone(result['uncertainty']['total_CI'])
        self.assertFalse(prices[0]['law_context']['ideal_sampler_distance_qualified'])

    def test_huge_integer_N_and_binary64_price_exactness(self):
        c = m.generate(spec(M=0)); N = 2**1200 + 3
        result = evaluate(c, data(c, shots=N, price=1.1))
        self.assertEqual(m.a.exact_scalar(result['cost']), 2*N*Fraction.from_float(1.1))
        self.assertEqual(result['terms'][0]['N'], N)
        c2 = m.generate(spec(M=0, kappa=2)); v1 = data(c, shots=N); v2 = data(c2, shots=N)
        v2[0]['rounds'][0]['N_Y'] = N - 1
        selected = m.select([c, c2], [evaluate(c, v1), evaluate(c2, v2)])
        self.assertEqual(selected['winner']['configuration_id'], c2['id'])
        self.assertEqual(m.a.exact_scalar(selected['winner']['cost']), 2*N - 1)

    def test_common_kappa_selects_whole_schedule_not_roundwise_hybrid(self):
        c1 = m.generate(spec(kappa=1)); c2 = m.generate(spec(kappa=4))
        v1 = data(c1); v2 = data(c2)
        for value in v1[1]: value['mean']['rz_count'] = 1 if value['identity']['m'] == 0 else 100
        for value in v2[1]: value['mean']['rz_count'] = 9 if value['identity']['m'] == 0 else 1
        results = [evaluate(c1, v1), evaluate(c2, v2)]
        out = m.select([c1, c2], results)
        self.assertEqual(out['winner']['configuration_id'], c2['id'])
        self.assertEqual(m.a.exact_scalar(out['winner']['cost']), 20)
        self.assertNotEqual(m.a.exact_scalar(out['winner']['cost']), 4)
        self.assertFalse(out['scientific_optimum_adopted'])

    def test_M_kappa_outer_groups_missing_infeasible_ties_and_scope(self):
        configs = [m.generate(spec(M=M, kappa=k, L_D=L, delta=d))
                   for L, d in [(1, 1), (2, 2)] for M in (0, 1) for k in (1, 2)]
        results = [evaluate(c, data(c, price=1)) for c in configs]
        selected = m.select(configs, results)
        self.assertEqual(len(selected['inner_M_kappa_minima']), 2)
        self.assertEqual(len(selected['winner']['ties']), 4)
        self.assertTrue(selected['menu_complete']); self.assertFalse(selected['fair_method_comparison_complete'])
        partial = m.select(configs, results[:-1])
        self.assertEqual(partial['selection_scope'], 'AVAILABLE_QUALIFIED_SUBSET')
        self.assertEqual(partial['unexplored_scope'], 'OUTSIDE_EXPLICIT_FINITE_MENU')
        self.assertFalse(partial['strict_cost_dominance_qualified'])

    def test_PF_PR_full_H_RTE_and_B3_are_distinct(self):
        configs = [m.generate(spec(method=method, M=0)) for method in m.METHODS]
        self.assertIsNone(configs[2]['rounds'][0]['q_PF'])
        self.assertIsNone(configs[2]['rounds'][0]['actual_delta_pf'])
        self.assertEqual(len({v['id'] for v in configs}), 3)
        results = [evaluate(c, data(c, price=price)) for c, price in zip(configs, (4, 2, 3))]
        out = m.select(configs, results)
        self.assertFalse(out['fair_method_comparison_complete'])  # truncated PF is absent
        self.assertTrue(m.select(configs, results, required_methods=m.METHODS)['fair_method_comparison_complete'])
        self.assertEqual(out['winner']['configuration_id'], configs[1]['id'])
        self.assertFalse(out['cross_method_advantage_qualified'])
        self.assertTrue(evaluate(configs[2], data(configs[2], model=True))['eligible'])
        bad = spec('FULL_H_RTE'); bad['split']['one_body_policy'] = 'RETAINED_B3'
        with self.assertRaisesRegex(ValueError, 'B3'): m.generate(bad)
        with self.assertRaisesRegex(ValueError, 'B3'): m.generate(spec('B3'))

    def test_fixed_h_formula_K_policy_and_price_scope_not_hidden_search_variables(self):
        c = m.generate(spec()); result = evaluate(c, data(c))
        for field, value in [('h', 2), ('formula', '4th(m5_best)'), ('finite_K', 4)]:
            s = spec(); s[field] = value; foreign = m.generate(s)
            with self.assertRaises(ValueError): m.select([c, foreign], [result, evaluate(foreign, data(foreign))])
        c2 = m.generate(spec(kappa=2)); other = evaluate(c2, data(c2, finite=True), 'finite_cohort_scenario')
        with self.assertRaisesRegex(ValueError, 'price policy'): m.select([c, c2], [result, other])

    def test_no_backend_or_scientific_import_in_metadata_and_synthetic_entries(self):
        with (patch.object(m.p, 'run_missing', side_effect=AssertionError('No provider')),
              patch.object(m.p, 'evaluate', side_effect=AssertionError('No legacy HP accounting')),
              patch.object(m.a, 'aggregate_supplied_axes', side_effect=AssertionError('Do not fake HP qualification'))):
            c = m.generate(spec()); result = evaluate(c, data(c)); m.select([c], [result])
        env = os.environ.copy(); env['PYTHONPATH'] = 'src'
        code = "import sys; from trottertracks.pr_rpe_rmse_controller import generate; assert not any(x in sys.modules for x in ('numpy','scipy','qiskit'))"
        subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, env=env, check=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
