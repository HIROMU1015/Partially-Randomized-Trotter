"""Published metadata intake + artificial D layouts; no scientific evaluations."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tests')]
from trottertracks import abc_s0_published_intake as intake
from trottertracks import full_baseline_v2_requests as D
from trottertracks import pr_rpe_rmse_evidence as E
from check_pr_rpe_rmse_controller import spec, data
from check_pr_rpe_method_coverage import truncated

B = D.B


def explicit(M=1, layouts=None):
    """Toy lambda/H/state/normalization; no saved molecular numerical values."""
    s = spec(M=M); s['binding'].update(system='H4', rank=10)
    prep = dict(representation='DF', source=dict(D.SOURCES['full_h_prepare'], function='prepare_full_h'),
        one_body_policy='RANDOMIZED', one_body_components=1, deterministic_blocks=0,
        randomized_fragments=list(range(10)), all_fragments=10, lambda_R=3, one_body_lambda_R=1,
        lambda_source={'fixture': 'SYNTHETIC_ONLY'}, normalization_policy='full_h_df_plus_one_body_exact_Z',
        scalar=dict(constant=0, extracted_identity=0, application='ONCE_AT_TOTAL_T', identity_components_randomized=0),
        preparation_hash='MOCK_PREPARATION', binding=deepcopy(s['binding']))
    design = D.full_h_rte_spec(s, M=M, finite_K=2, kappa=None, preparation=prep)
    return D.generate(design, layout_by_round=layouts if layouts is not None else {m: dict(q=1, R_outer=2) for m in range(M+1)})


class PublishedIntakeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs, cls.sources = intake.load_published()

    def test_01_official_A_C_intake_and_nulls_without_S0_producers(self):
        with patch.object(intake.H, 'prepare', side_effect=AssertionError('No S0 intake rerun')), \
             patch.object(intake.H, 'canonical', side_effect=AssertionError('No S0 canonical rerun')), \
             patch.object(intake.C, 'prepare', side_effect=AssertionError('No C reconstruction')), \
             patch.object(intake.H.I.A, 'evaluate', side_effect=AssertionError('No N')), \
             patch.object(B, 'cost', side_effect=AssertionError('No real G')):
            result = intake.accept(**self.inputs)
        self.assertTrue(result['A_packet_accepted'])
        self.assertEqual(result['C_formal_axis_requests'], 26)
        self.assertEqual(result['C_round_metadata_matches'], 13)
        self.assertEqual(result['nominal_N_XY'], {'0':[64,64], '6':[38,38]})
        self.assertEqual(len(result['unevaluated_rounds']), 11)
        self.assertIsNone(result['G']); self.assertIsNone(result['winner'])
        self.assertIsNone(result['global_RMSE_bound'])
        self.assertFalse(result['planner']['comparison_complete'])
        self.assertEqual(result['planner']['method_coverage']['PR']['present'], True)

    def test_02_A_source_law_identity_and_diagnostics_rejected(self):
        mutations = [lambda v: v['packet']['adapter_source'].update(S0='OTHER_SOURCE'),
            lambda v: v['packet']['source'].update(commit='OTHER_SOURCE'),
            lambda v: v['packet']['rounds'][0]['law_binding'].update(kappa=None),
            lambda v: v['packet']['rounds'][0]['identity']['profile'].update(compiler='OTHER_COMPILER'),
            lambda v: v['packet']['rounds'][0].update(nominal_N_X=65),
            lambda v: v['packet']['rounds'][1].update(N_X=1),
            lambda v: v['packet']['rmse_evidence'].update(rmse_bound_ha=0)]
        for mutate in mutations:
            values = deepcopy(self.inputs); mutate(values)
            with self.assertRaises(ValueError): intake.accept(**values)

    def test_03_C_focus_all_round_axis_and_source_mismatches_rejected(self):
        mutations = [lambda v: v['c_plan']['focus_requests'][0]['identity']['profile'].update(compiler='OTHER'),
            lambda v: v['c_plan']['focus_requests'][1]['law_binding'].update(kappa=None),
            lambda v: v['c_plan']['round_price_coverage'][12].update(R_outer=1),
            lambda v: v['c_plan']['round_price_coverage'][1]['source'].update(commit='OTHER'),
            lambda v: v['b_requests'][25]['identity']['binding'].update(snapshot='OTHER'),
            lambda v: v['b_requests'][0].update(requested_mean=0),
            lambda v: v['c_plan']['focus_requests'][0].update(model_qualified_for_candidate=True)]
        for mutate in mutations:
            values = deepcopy(self.inputs); mutate(values)
            with self.assertRaises(ValueError): intake.accept(**values)


class LayoutExtensionTests(unittest.TestCase):
    def test_04_Yoshida4_and_m5_signed_formulas_full_truncated_PF(self):
        configs = []
        for make in (lambda: B.generate(spec('PF')), truncated):
            for formula in ('4th', '4th(m5_best)'):
                s = deepcopy(make()['spec']); s['formula'] = formula
                c = B.generate(s); configs.append(c)
                self.assertIsNone(c['spec']['kappa']); self.assertIsNone(c['spec']['finite_K'])
                for row in c['rounds']:
                    self.assertEqual(row['structure']['signed_weights'], list(D.p._schedule().weights(formula)))
                    self.assertLess(min(row['structure']['signed_weights']), 0)
                    self.assertTrue(all(n == 0 for n in row['structure']['allocation']))
        self.assertEqual(len({c['id'] for c in configs}), 4)
        self.assertEqual({B.comparison_method(c['spec']) for c in configs}, {'FULL_PF','TRUNCATED_PF'})
        bad = spec(); bad['formula'] = '4th'
        with self.assertRaises(ValueError): B.generate(bad)

    def test_05_explicit_no_kappa_keeps_schedule_and_full_context(self):
        c = explicit(M=2); B._validate(c)
        self.assertIsNone(c['spec']['kappa'])
        self.assertEqual(c['spec']['event_mapping']['kind'], B.EXPLICIT_LAYOUT)
        self.assertEqual([B.a.exact_scalar(r['T']) for r in c['rounds']], [1,2,4])
        requests = D.requests(c); self.assertTrue(D.validate_requests(c, requests))
        for row in requests['rounds']:
            for price in row['C']:
                for key in ('configuration_id','physical_candidate_id','identity','law_binding','physical_schedule_mapping'):
                    self.assertEqual(row['A'][key], price[key])
                self.assertFalse(price['physical_schedule_mapping']['physical_layout']['priced_equivalence_claim'])

    def test_06_kappa_schedule_is_distinct_and_not_invented(self):
        explicit_c = explicit()
        old = deepcopy(explicit_c['spec']); old['kappa'] = dict(kind='MOCK_KAPPA', value=1, definition_source={'fixture':'MOCK'})
        with self.assertRaisesRegex(ValueError, 'kappa=None'): B.generate(old, event_mapper=D.layout_mapper({0:dict(q=1,R_outer=2)}))
        old['event_mapping']['kind'] = D.LAYOUT_KIND
        kappa_c = D.generate(old, layout_by_round={m:dict(q=1,R_outer=2) for m in range(2)})
        self.assertNotEqual(explicit_c['id'], kappa_c['id'])
        selection = B.select([explicit_c, kappa_c], [], required_methods=('FULL_H_RTE',))
        self.assertEqual({v['parameterization'] for v in selection['schedule_menu']}, {'EXPLICIT_LAYOUT','COMMON_KAPPA'})
        self.assertEqual(len(selection['inner_M_kappa_minima']), 2)
        self.assertIsNone(selection['winner']['cost'])
        old['kappa'] = None
        with self.assertRaisesRegex(ValueError, 'common kappa'): D.generate(old, layout_by_round={})

    def test_07_equal_events_different_grouping_and_source_distinct(self):
        one = explicit(M=0, layouts={0:dict(q=1,R_outer=8)})
        two = explicit(M=0, layouts={0:dict(q=2,R_outer=4)})
        self.assertEqual(one['rounds'][0]['structure']['event_count'], two['rounds'][0]['structure']['event_count'])
        self.assertNotEqual(one['id'], two['id'])
        self.assertNotEqual(D.requests(one)['rounds'][0]['A']['physical_candidate_id'], D.requests(two)['rounds'][0]['A']['physical_candidate_id'])

    def test_08_full_H_one_body_B3_scalar_and_source_guards(self):
        mutations = [lambda s: s['split'].update(original_PF_method='B3'),
            lambda s: s['split']['preparation'].update(one_body_policy='DETERMINISTIC'),
            lambda s: s['split']['preparation'].update(all_fragments=9, randomized_fragments=list(range(9))),
            lambda s: s['split']['preparation']['scalar'].update(application='PER_EVENT'),
            lambda s: s['split'].update(lambda_R=1),
            lambda s: s['law'].update(preparation_source={'fixture':'WRONG_SOURCE'})]
        for change in mutations:
            s = deepcopy(explicit()['spec']); change(s)
            with self.assertRaises(ValueError): D.generate(s, layout_by_round={0:dict(q=1,R_outer=2)})

    def test_09_layout_integrity_and_missing_round_are_not_guessed(self):
        c = explicit(M=1, layouts={0:dict(q=2,R_outer=4)})
        result = B.assess(c, None, [], cohorts={})
        self.assertIn('UNRESOLVED_PHYSICAL_LAYOUT', result['reasons']); self.assertIsNone(result['cost'])
        self.assertIsNone(D.requests(c)['rounds'][1]['A']['physical_candidate_id'])
        for key, value in (('priced_equivalence_claim',True),('taylor_K',4),('scalar','PER_EVENT'),('q',3)):
            def bad(s, row):
                m = D.layout_mapper({0:dict(q=2,R_outer=4)})(s,row)
                if m: m['physical_layout'][key] = value
                return m
            with self.assertRaises(ValueError): B.generate(c['spec'], event_mapper=bad)
        with self.assertRaises(ValueError): D.generate(c['spec'], layout_by_round={2:dict(q=1,R_outer=1)})

    def test_10_external_bridge_rejects_identity_law_profile_and_physical_id_changes(self):
        c = explicit(M=0); packet = D.requests(c)
        for key in ('physical_candidate_id','v2_candidate_id','law_binding','identity'):
            bad = deepcopy(packet); bad['rounds'][0]['C'][0][key] = 'WRONG'
            with self.assertRaises(ValueError): D.validate_requests(c, bad)
        bad = deepcopy(packet); bad['profile']['compiler'] = 'OTHER'
        with self.assertRaises(ValueError): D.validate_requests(c, bad)

    def test_11_only_mock_values_sum_select_and_finite_cohort_MODEL_false_stays_unqualified(self):
        c = explicit(M=0); n, prices, cohorts = data(c, shots=10**40+1, price=3)
        with patch.object(B, 'cost', wraps=B.cost) as call:
            result = B.cost(c, n, prices, cohorts=cohorts)
        self.assertEqual(call.call_count, 1)
        self.assertEqual(B.a.exact_scalar(result['cost']), 6*(10**40+1))
        self.assertEqual(B.select([c],[result],required_methods=('FULL_H_RTE',))['winner']['configuration_id'], c['id'])
        n, prices, cohorts = data(c, finite=True)
        self.assertFalse(B.assess(c,n,prices,cohorts=cohorts)['eligible'])
        self.assertTrue(B.assess(c,n,prices,cohorts=cohorts,price_policy='finite_cohort_scenario')['eligible'])
        n, prices, cohorts = data(c, model=True)
        for record in prices: record['model']['qualified_for_candidate'] = False
        self.assertIsNone(B.cost(c,n,prices,cohorts=cohorts)['cost'])

    def test_12_planner_and_D_requests_never_run_science_or_real_cost(self):
        c = explicit()
        with patch.object(B, 'cost', side_effect=AssertionError('No cost in planner')):
            requests = D.requests(c); D.validate_requests(c, requests)
            result = E.plan([dict(configuration=c)], price_policy='qualified_expectation')
        self.assertIsNone(result['G'])
        self.assertFalse(result['comparison_complete'])
        forbidden = {'qiskit','pyscf','h46_long_time_signal','h46_long_time_native','h46_full_h_pure_rte','h46_full_h_t20_opt'}
        self.assertFalse(forbidden & {m.split('.')[0] for m in sys.modules})


if __name__ == '__main__':
    unittest.main(verbosity=2)
