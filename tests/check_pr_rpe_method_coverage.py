"""Artificial supplied N/C only; saved fixtures are inspected for metadata."""
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts'), str(ROOT / 'tests')]
from check_pr_rpe_rmse_controller import spec, data, evaluate, m as B
from check_pr_rpe_rmse_integration import inputs, I


def truncated():
    s = spec('PF', M=0)
    s['split'].update(deterministic_scope='TRUNCATED_H', retained_hamiltonian=dict(
        construction='MOCK_B0_KEEP_ONE_BODY_AND_PREFIX', prefix=s['L_D'],
        target_hamiltonian=s['binding']['hamiltonian'], source={'fixture': 'SYNTHETIC_ONLY'}))
    return B.generate(s)


def qualified_truncated(c):
    values = data(c)
    values[0]['rmse_evidence'].update(truncation_status='QUALIFIED', covers_truncation_error=True,
        target_hamiltonian=deepcopy(c['spec']['binding']['hamiltonian']),
        retained_hamiltonian=deepcopy(c['spec']['split']['retained_hamiltonian']))
    return values


class MethodCoverageTests(unittest.TestCase):
    def test_full_and_truncated_PF_have_separate_target_retained_identity(self):
        full = B.generate(spec('PF', M=0)); cut = truncated()
        self.assertNotEqual(full['id'], cut['id'])
        self.assertEqual(B.comparison_method(full['spec']), 'FULL_PF')
        self.assertEqual(B.comparison_method(cut['spec']), 'TRUNCATED_PF')
        self.assertEqual(full['spec']['binding'], cut['spec']['binding'])
        self.assertNotEqual(full['rounds'][0]['candidate']['retained_hamiltonian'],
                            cut['rounds'][0]['candidate']['retained_hamiltonian'])
        self.assertEqual(full['schema'], 'PR_RPE_RMSE_V2')
        mislabeled = deepcopy(cut['spec']); mislabeled['split']['deterministic_scope'] = 'FULL_H'
        with self.assertRaisesRegex(ValueError, 'Full PF'): B.generate(mislabeled)
        legacy = deepcopy(full); legacy['schema'] = 'PR_RPE_RMSE_V1'
        with self.assertRaisesRegex(ValueError, 'mutated'): B.assess(legacy, None, [], cohorts={})
        for c in (full, cut):
            self.assertIsNone(c['spec']['kappa'])
            s = deepcopy(c['spec']); s['kappa'] = dict(value=1)
            with self.assertRaisesRegex(ValueError, 'kappa'): B.generate(s)

    def test_truncated_requires_original_H_error_not_just_retained_H_PASS(self):
        c = truncated(); values = data(c)
        self.assertIn('UNQUALIFIED_TRUNCATION_ERROR_AGAINST_ORIGINAL_H', evaluate(c, values)['reasons'])
        values = qualified_truncated(c)
        self.assertTrue(evaluate(c, values)['eligible'])
        values[0]['rmse_evidence']['target_hamiltonian'] = 'ONLY_TRUNCATED_H'
        self.assertIsNone(evaluate(c, values)['cost'])
        values[0]['rmse_evidence']['status'] = 'INFEASIBLE'
        self.assertNotEqual(evaluate(c, values)['status'], 'INFEASIBLE')  # foreign contract cannot prune

    def test_A_H4_B0_is_truncated_PF_and_H6_early_recipe_is_preserved(self):
        args = inputs(); args['design']['PF'] = dict(method='B0', prefix=7)
        a = B.generate(I.a_spec(args, 'PF'))
        self.assertEqual(B.comparison_method(a['spec']), 'TRUNCATED_PF')
        self.assertTrue(all(r['candidate']['physical_method'] == 'B0' for r in a['rounds']))
        args['kappa_paper'] = '12'
        self.assertEqual(a['id'], B.generate(I.a_spec(args, 'PF'))['id'])
        h6 = B.generate(I.a_spec(inputs(M=4, system='H6')))
        self.assertEqual([r['candidate']['prefix'] for r in h6['rounds']], [5, 5, 5, 5, 7])
        self.assertEqual([r['R_outer'] for r in h6['rounds'][:4]], [32]*4)

    def test_pure_RTE_rejects_B3_and_deterministic_one_body(self):
        pure = B.generate(spec('FULL_H_RTE', M=0)); pr = B.generate(spec(M=0))
        self.assertNotEqual(pure['id'], pr['id'])
        self.assertIsNone(pure['rounds'][0]['q_PF'])
        for changes in (dict(one_body_policy='DETERMINISTIC'), dict(deterministic_scope='ONE_BODY'),
                        dict(original_PF_method='B3')):
            s = spec('FULL_H_RTE'); s['split'].update(changes)
            with self.assertRaisesRegex(ValueError, 'B3'): B.generate(s)

    def test_all_four_methods_use_existing_exact_cost_and_whole_schedule_selection(self):
        pure_spec = spec('FULL_H_RTE', M=0)
        pure_spec['kappa']['kind'] = 'MOCK_PURE_ROTATIONS'
        pure_spec['event_mapping']['kind'] = 'SUPPLIED_MOCK_EVENT_COUNTS'
        def supplied_events(s, row):
            return dict(kind=s['event_mapping']['kind'], source=s['event_mapping']['source'],
                        kappa=s['kappa'], R_outer=None, event_count=3)
        configs = [B.generate(spec('PF', M=0)), truncated(), B.generate(spec(M=0)),
                   B.generate(pure_spec, event_mapper=supplied_events)]
        values = [data(configs[0], price=5), qualified_truncated(configs[1]),
                  data(configs[2], price=3), data(configs[3], price=4)]
        # n_paths and diagnostics are never quantum-shot multipliers.
        for _, prices, _ in values:
            for p in prices: p['normalization_diagnostic'] = {'B': 99999}; p['n_paths'] = 700
        results = [evaluate(c, v) for c, v in zip(configs, values)]
        self.assertEqual([B.a.exact_scalar(v['cost']) for v in results], [10, 2, 6, 8])
        out = B.select(configs, results)
        self.assertTrue(out['fair_method_comparison_complete'])
        self.assertEqual(out['winner']['configuration_id'], configs[1]['id'])
        self.assertIsNone(out['PF_baseline_optimality']); self.assertFalse(out['cross_method_advantage_qualified'])

    def test_per_axis_coverage_separates_saved_presence_from_qualification(self):
        c = B.generate(spec()); n, prices, cohorts = data(c)
        prices[0].update(price_status='OUTSIDE_DOMAIN')
        out = B.assess(c, n, prices, cohorts=cohorts)
        self.assertIsNone(out['cost']); self.assertFalse(out['eligible'])
        first = out['coverage'][0]
        self.assertEqual(first['C']['availability'], 'AVAILABLE')
        self.assertEqual(first['C']['qualification'], 'OUTSIDE_DOMAIN')
        self.assertEqual(first['N']['qualification'], 'AVAILABLE')
        self.assertEqual(first['N']['source'], n['source'])
        self.assertIsNone(first['C']['uncertainty']['joint_covariance'])
        self.assertEqual(out['coverage'][1]['C']['qualification'], 'AVAILABLE')
        for status in ('UNQUALIFIED', 'NOT_EVALUATED', 'INFEASIBLE', 'FAILED'):
            prices[0]['price_status'] = status
            self.assertIsNone(evaluate(c, (n, prices, cohorts))['cost'])

    def test_missing_null_and_unqualified_MODEL_do_not_select(self):
        c = B.generate(spec()); n, prices, cohorts = data(c, model=True)
        prices[0]['model']['qualified_for_candidate'] = False
        prices[1]['mean']['rz_count'] = None
        result = evaluate(c, (n, prices, cohorts))
        self.assertTrue(all(v['C']['qualification'] == 'UNQUALIFIED' for v in result['coverage'][:2]))
        self.assertEqual(result['coverage'][1]['C']['value_status'], 'MISSING')
        self.assertIsNone(B.select([c], [result])['winner']['configuration_id'])
        missing = B.assess(c, None, [], cohorts={})
        self.assertTrue(all(v['N']['availability'] == v['C']['availability'] == 'MISSING' for v in missing['coverage']))

    def test_foreign_candidate_law_compiler_and_high_probability_N_are_rejected(self):
        c = B.generate(spec()); n, prices, cohorts = data(c)
        for change in (lambda x: x['identity']['profile'].update(compiler='FOREIGN'),
                       lambda x: x['law_binding'].update(kappa={'value': 999})):
            bad = deepcopy(prices); change(bad[0])
            self.assertEqual(evaluate(c, (n, bad, cohorts))['coverage'][0]['C']['qualification'], 'UNQUALIFIED')
        n['policy']['family'] = 'HIGH_PROBABILITY_RPE'
        self.assertIsNone(evaluate(c, (n, prices, cohorts))['cost'])

    def test_two_round_pilot_and_S0a_do_not_become_global_N_or_mean_C(self):
        c = B.generate(spec(M=2)); n, prices, cohorts = data(c)
        n['rmse_evidence'].update(covers_state_target_alias_prior_decoder_all_rounds=False)
        self.assertIsNone(evaluate(c, (n, prices, cohorts))['cost'])
        n = data(c)[0]
        prices[0]['quantity'] = 'SINGLE_PATH_STRUCTURAL_PROXY'
        result = evaluate(c, (n, prices, cohorts))
        self.assertIn('C_STRUCTURAL_OR_LOCAL_QUANTITY_IS_NOT_MEAN_COMPILED_PRICE', result['reasons'])
        self.assertIsNone(result['cost'])

    def test_explicit_formula_menu_allows_different_method_design_freedoms(self):
        full = B.generate(spec('PF', M=0)); s = spec(M=0); s['formula'] = '4th(m5_best)'
        pr = B.generate(s); results = [evaluate(c, data(c)) for c in (full, pr)]
        with self.assertRaisesRegex(ValueError, 'formula'): B.select([full, pr], results)
        menu = dict(source={'fixture': 'UNADOPTED_MOCK_MENU'}, by_method={'FULL_PF': ['2nd'], 'PR': ['4th(m5_best)']})
        out = B.select([full, pr], results, formula_menu=menu)
        self.assertTrue(out['menu_complete']); self.assertFalse(out['search_ranges_scientifically_adopted'])
        foreign = deepcopy(s); foreign['binding']['geometry'] = 'FOREIGN_GEOMETRY'
        with self.assertRaisesRegex(ValueError, 'Common'): B.select([full, B.generate(foreign)], [], formula_menu=menu)

    def test_mixed_kappa_rounds_cannot_become_a_configuration(self):
        c = B.generate(spec(M=2)); other = B.generate(spec(M=2, kappa=2))
        c['rounds'][1] = other['rounds'][1]
        with self.assertRaisesRegex(ValueError, 'mutated'): B.assess(c, None, [], cohorts={})

    def test_scientific_record_requires_explicit_prior_decoder_and_molecular_context(self):
        s = spec(); s['rmse_policy']['scientific_status'] = 'EXTERNALLY_QUALIFIED'
        c = B.generate(s); n, prices, cohorts = data(c); n['evidence_kind'] = 'SCIENTIFIC'
        result = B.assess(c, n, prices, cohorts=cohorts)  # mock numbers, no cost call
        self.assertIn('MISSING_SCIENTIFIC_COMPARISON_CONTEXT', result['reasons'])
        n['rmse_evidence']['status'] = 'INFEASIBLE'
        self.assertNotEqual(B.assess(c, n, prices, cohorts=cohorts)['status'], 'INFEASIBLE')

    def test_no_provider_compile_fit_signal_or_cost_call_from_metadata(self):
        with (patch.object(B, 'cost', side_effect=AssertionError('No G')),
              patch.object(B, 'select', side_effect=AssertionError('No optimizer')),
              patch.object(B.p, 'run_missing', side_effect=AssertionError('No provider')),
              patch.object(I.A, 'evaluate', side_effect=AssertionError('No scientific N'))):
            for method in B.METHODS:
                c = B.generate(spec(method)); B.assess(c, None, [], cohorts={})
            B.generate(I.a_spec(inputs()))
        env = dict(os.environ, PYTHONPATH='src')
        subprocess.run([sys.executable, '-B', '-c',
            "import sys; from trottertracks import pr_rpe_rmse_controller; "
            "assert not any(x in sys.modules for x in ('numpy','scipy','qiskit'))"], cwd=ROOT, env=env, check=True)

    def test_saved_coverage_is_metadata_not_new_RMSE_N_C_or_G(self):
        from inventory_pr_rpe_method_coverage import inventory
        with (patch.object(B, 'generate', side_effect=AssertionError('No new real schedule')),
              patch.object(B, 'cost', side_effect=AssertionError('No real G')),
              patch.object(B.p, 'run_missing', side_effect=AssertionError('No real providers'))):
            saved = inventory()
        self.assertEqual(len(saved['recipes']), 118)
        self.assertEqual(len(saved['rows']), 236)
        self.assertEqual(sum(r['N']['P1_reference'] is not None for r in saved['rows']), 24)
        self.assertTrue(all(r['G'] is None and not r['eligible_for_RMSE_comparison'] for r in saved['rows']))
        self.assertTrue(all(not v['current_RMSE_identity_available'] for v in saved['recipes'].values()))
        self.assertTrue(all(v['method'] == 'TRUNCATED_PF' for v in saved['recipes'].values()
                            if v['candidate']['system'] == 'H4' and v['candidate']['method'] == 'B0'))
        self.assertTrue(all(p['model_qualified_for_candidate'] is False for r in saved['rows']
                            for p in r['C']['C1_sources'] if p['classification'] == 'MODEL_PREDICTION'))
        self.assertIsNone(saved['winner']); self.assertFalse(saved['saved_signal_N_C_G_values_copied'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
