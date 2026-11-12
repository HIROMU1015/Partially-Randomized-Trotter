"""D Full PF / DF Full-H RTE request adapter: metadata and artificial checks only.

Published facts (saved binding, lambda, selection IDs) are read from pinned Git
objects. No Hamiltonian/state load, signal, N, sampling, compile, price or G.
"""
from copy import deepcopy
import subprocess
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts'), str(ROOT/'tests')]
from trottertracks import full_baseline_v2_requests as D
from trottertracks import pr_rpe_rmse_evidence as E
from check_pr_rpe_rmse_controller import spec
from check_pr_rpe_method_coverage import truncated

B = D.B
TENTH = B.a.exact_value(0.1)
KAPPA = dict(kind='MOCK_SUPPLIED_KAPPA', value=1, definition_source={'fixture': 'SYNTHETIC_ONLY'})
A_S0 = '7da8a35f68e9fdc2b46c361dd2c68e2a799b1e75'
SCIENCE_MODULES = ('h46_long_time_signal', 'h46_long_time_native', 'h46_full_h_pure_rte', 'h46_full_h_t20_opt',
                   'h46_new2_native', 'h46_m5_tol_native', 'qiskit', 'pyscf')


def base(h=TENTH, M=3):
    s = spec(M=M); prep = D.saved_full_h_preparation('H4')
    s['binding'].update({k: prep['binding'][k] for k in D.BINDING_KEYS if k in prep['binding']}, rank=10)
    s['h'] = h
    return s, prep


def full_pf(formula='2nd', M=3, h=TENTH):
    s, _ = base(h, M)
    return D.generate(D.full_pf_spec(s, formula=formula, delta_pf=TENTH, M=M, rank=10))


def full_h(layout=None, M=3, K=4, prep_edit=None):
    s, prep = base(M=M)
    if prep_edit:
        prep_edit(prep)
    return D.generate(D.full_h_rte_spec(s, M=M, finite_K=K, kappa=KAPPA, preparation=prep),
                      layout_by_round=layout if layout is not None else {3: dict(q=16, R_outer=16)})


class FullBaselineTests(unittest.TestCase):
    def test_01_full_and_truncated_PF_are_distinct(self):
        full = full_pf(); cut = truncated()
        self.assertEqual(B.comparison_method(full['spec']), 'FULL_PF')
        self.assertEqual(B.comparison_method(cut['spec']), 'TRUNCATED_PF')
        retained = full['rounds'][0]['candidate']['retained_hamiltonian']
        self.assertEqual((retained['construction'], retained['prefix']), ('FULL_H', 10))
        self.assertEqual(retained['target_hamiltonian'], full['spec']['binding']['hamiltonian'])
        with self.assertRaisesRegex(ValueError, 'Full PF and Full-H RTE only'):
            D.requests(cut)

    def test_02_H4_B0_prefix7_is_not_full_PF(self):
        s, _ = base()
        with self.assertRaisesRegex(ValueError, 'full rank'):
            D.full_pf_spec(s, formula='2nd', delta_pf=TENTH, M=3, rank=7)
        bad = D.full_pf_spec(s, formula='2nd', delta_pf=TENTH, M=3, rank=10)
        bad['split']['retained_hamiltonian'].update(construction='B0', prefix=7)
        with self.assertRaisesRegex(ValueError, 'Full PF cannot carry a truncated'):
            B.generate(bad)
        physical = D.requests(full_pf())['rounds'][0]['A']['physical_candidate']
        self.assertEqual((physical['method'], physical['prefix']), ('FULL', 10))

    def test_03_full_H_RTE_keeps_one_body_randomized(self):
        for edit, message in [(lambda v: v.update(one_body_components=0), 'One-body'),
                              (lambda v: v.update(one_body_policy='DETERMINISTIC'), 'One-body'),
                              (lambda v: v.update(deterministic_blocks=1), 'One-body'),
                              (lambda v: v.update(randomized_fragments=list(range(9))), 'All DF fragments'),
                              (lambda v: v.update(tail_only_lambda_R=v['lambda_R']), 'Tail-only lambda'),
                              (lambda v: v.update(one_body_lambda_R=0.), 'one-body l1')]:
            with self.assertRaisesRegex(ValueError, message):
                full_h(prep_edit=edit)
        c = full_h()
        self.assertEqual(c['spec']['split']['randomized_scope'], 'FULL_H_INCLUDING_ONE_BODY')
        self.assertGreater(c['spec']['split']['lambda_R'], c['spec']['split']['preparation']['one_body_lambda_R'])
        s, prep = base(); bad = D.full_h_rte_spec(s, M=3, finite_K=4, kappa=KAPPA, preparation=prep)
        bad['split']['one_body_policy'] = 'DETERMINISTIC'
        with self.assertRaisesRegex(ValueError, 'randomize one-body'):
            D.generate(bad, layout_by_round={3: dict(q=1, R_outer=1)})

    def test_04_DF_B3_is_not_full_H_RTE(self):
        with self.assertRaisesRegex(ValueError, 'prepare_full_h'):
            full_h(prep_edit=lambda v: v['source'].update(path='scripts/h46_b3.py', function='prepare'))
        s, prep = base(); bad = D.full_h_rte_spec(s, M=3, finite_K=4, kappa=KAPPA, preparation=prep)
        bad['split']['original_PF_method'] = 'B3'
        with self.assertRaisesRegex(ValueError, 'B3'):
            D.generate(bad, layout_by_round={3: dict(q=1, R_outer=1)})

    def test_05_identity_scalar_is_applied_once(self):
        for edit in (lambda v: v['scalar'].update(identity_components_randomized=1),
                     lambda v: v['scalar'].update(application='PER_PF_STEP'),
                     lambda v: v['scalar'].pop('extracted_identity')):
            with self.assertRaisesRegex(ValueError, 'applied once'):
                full_h(prep_edit=edit)
        for c in (full_pf(), full_h()):
            self.assertTrue(all(r['structure']['scalar_policy'] == 'scalar_identity_once_at_total_T' for r in c['rounds']))
        layout = full_h()['rounds'][3]['mapping']['physical_layout']
        self.assertEqual(layout['scalar'], 'ONCE_AT_TOTAL_T')

    def test_06_formula_signed_coefficients_and_fragment_order_are_kept(self):
        weights = D.p._schedule().weights('4th(m5_best)')
        m5 = full_pf('4th(m5_best)')
        self.assertTrue(all(r['structure']['signed_weights'] == list(weights) for r in m5['rounds']))
        self.assertLess(min(weights), 0)
        catalog = D.catalog()['formulas']
        self.assertEqual(catalog['4th']['v2_status'], 'B_V2_FORMULA_EXTENSION_REQUIRED')
        self.assertNotEqual(catalog['4th']['signed_weights'], catalog['4th(m5_best)']['signed_weights'])
        self.assertEqual(catalog['8th(Morales)']['v2_status'], 'NO_DF_CONTROLLED_FULL_WRAPPER_PATH')
        for label in ('4th', '8th(Yoshida)'):
            with self.assertRaisesRegex(ValueError, 'B V2 formula extension'):
                full_pf(label)
        with self.assertRaisesRegex(ValueError, 'saved order'):
            full_h(prep_edit=lambda v: v.update(randomized_fragments=list(reversed(v['randomized_fragments']))))

    def test_07_same_event_count_different_grouping_is_not_the_same_request(self):
        one = full_h({3: dict(q=1, R_outer=256)}); two = full_h({3: dict(q=16, R_outer=16)})
        self.assertEqual(one['rounds'][3]['structure']['event_count'], two['rounds'][3]['structure']['event_count'])
        self.assertNotEqual(one['id'], two['id'])
        a, b = (D.requests(c)['rounds'][3]['A']['physical_candidate_id'] for c in (one, two))
        self.assertNotEqual(a, b)
        self.assertFalse(two['rounds'][3]['mapping']['physical_layout']['priced_equivalence_claim'])
        s, prep = base()
        with self.assertRaisesRegex(ValueError, 'explicit physical layout'):
            D.generate(D.full_h_rte_spec(s, M=3, finite_K=4, kappa=KAPPA, preparation=prep))
        with self.assertRaisesRegex(ValueError, 'integer q and R_outer'):
            full_h({3: dict(event_count=256)})

    def test_08_DF_and_Pauli_sources_are_not_mixed(self):
        with self.assertRaisesRegex(ValueError, 'DF Full-H RTE only'):
            full_h(prep_edit=lambda v: v.update(representation='PAULI_JW'))
        pauli = D.pauli_reference()
        self.assertEqual(pauli['status'], 'SOURCE_ONLY_NOT_CONNECTED_BY_D')
        self.assertFalse(pauli['same_numerical_H_state_as_DF'])
        df = full_h(); other = deepcopy(df['spec']); other['binding']['hamiltonian'] = 'PAULI_JW_H4_HAMILTONIAN'
        other_config = D.generate(other, layout_by_round={3: dict(q=16, R_outer=16)})
        with self.assertRaisesRegex(ValueError, 'Common H/state'):
            B.select([df, other_config], [])

    def test_09_saved_H_state_snapshot_match_the_V2_binding(self):
        import json
        a_spec = json.loads(D._blob(A_S0, 'docs/research/pr_rpe_rmse_s0_results_20261113/run01/CONFIGURATION.json'))['B_configuration']['spec']
        prep = D.saved_full_h_preparation('H4')
        self.assertTrue(D.check_binding(a_spec['binding'], prep['binding']))
        self.assertEqual(prep['binding']['rank'], D.saved_grid()['rank']['H4'])
        for key in ('state', 'snapshot', 'hamiltonian', 'system'):
            changed = dict(a_spec['binding'], **{key: 'OTHER'})
            with self.assertRaisesRegex(ValueError, 'binding differs'):
                D.check_binding(changed, prep['binding'])
        s, prep = base(); s['binding']['state'] = 'OTHER_STATE'
        with self.assertRaisesRegex(ValueError, 'binding differs'):
            D.full_h_rte_spec(s, M=3, finite_K=4, kappa=KAPPA, preparation=prep)
        for c in (full_pf(), full_h()):
            self.assertTrue(all(r['identity']['binding'] == c['spec']['binding'] for r in c['rounds']))

    def test_10_A_and_C_requests_point_to_one_physical_candidate(self):
        for c, saved in ((full_pf(), 'SAVED_LONG_TIME_GRID_POINT_CHECK_CARRIER'), (full_h(), 'SAVED_LIGHT_STUDY_CANDIDATE_FINITE_COHORT')):
            row = D.requests(c)['rounds'][3]
            self.assertEqual(len(row['C']), 2); self.assertEqual({v['axis'] for v in row['C']}, set(D.p.AXES))
            for price in row['C']:
                for key in ('physical_candidate_id', 'v2_candidate_id', 'identity', 'law_binding', 'physical_schedule_mapping'):
                    self.assertEqual(price[key], row['A'][key])
            self.assertEqual(row['A']['saved_asset_status'], saved)
            self.assertTrue(row['A']['saved_reference_time'])
        self.assertEqual(D.requests(full_h())['rounds'][3]['A']['physical_candidate_id'], 'FULL_H_8d6f23210529d68452dd')
        price = D.requests(full_h())['rounds'][3]['C'][0]
        self.assertFalse(price['law_context']['ideal_sampler_distance_qualified'])
        self.assertIsNotNone(price['saved_price_source'])
        self.assertEqual(D.requests(full_pf())['rounds'][3]['C'][0]['law_context']['price_law'], 'deterministic')

    def test_11_full_PF_never_carries_kappa_or_finite_K(self):
        c = full_pf()
        self.assertIsNone(c['spec']['kappa']); self.assertIsNone(c['spec']['finite_K'])
        self.assertTrue(all(r['candidate']['K'] is None and r['R_outer'] is None for r in c['rounds']))
        bad = deepcopy(c['spec']); bad['kappa'] = KAPPA
        with self.assertRaisesRegex(ValueError, 'without kappa'):
            B.generate(bad)
        with self.assertRaisesRegex(ValueError, 'no RTE layout'):
            D.generate(c['spec'], layout_by_round={0: dict(q=1, R_outer=1)})
        s, prep = base()
        with self.assertRaisesRegex(ValueError, 'source-bound kappa'):
            D.full_h_rte_spec(s, M=3, finite_K=4, kappa=None, preparation=prep)

    def test_12_missing_N_C_stay_null_and_B_planner_reports_next_work(self):
        configs = [full_pf(), full_h()]
        for c in configs:
            out = D.requests(c)
            self.assertIsNone(out['N_C_values']); self.assertIsNone(out['G'])
            for row in out['rounds']:
                self.assertIsNone(row['A']['N_X']); self.assertIsNone(row['A']['rmse_bound_ha'])
                self.assertTrue(all(v['mean']['rz_count'] is None and v['price_status'] == 'NOT_EVALUATED' for v in row['C']))
        unresolved = D.requests(configs[1])['rounds'][0]['A']
        self.assertIsNone(unresolved['physical_candidate_id']); self.assertEqual(unresolved['saved_asset_status'], 'LAYOUT_NOT_SUPPLIED')
        plan = E.plan([dict(configuration=c) for c in configs], price_policy='qualified_expectation',
                      required_methods=('FULL_PF', 'FULL_H_RTE'))
        self.assertIsNone(plan['G']); self.assertFalse(plan['evidence_menu_resolved'])
        owners = {w['owner'] for c in plan['configurations'] for r in c['rows'] for w in r['next_work']}
        self.assertEqual(owners, {'Codex A', 'Codex C'})
        self.assertIn('UNRESOLVED_KAPPA_MAPPING', plan['configurations'][1]['reasons'])

    def test_13_no_scientific_runner_or_compile_is_started(self):
        with patch.object(B, 'cost', side_effect=AssertionError('No G')):
            D.catalog(); [D.requests(c) for c in (full_pf(), full_h(), full_pf('4th(m5_best)'))]
        self.assertFalse([m for m in sys.modules if m.split('.')[0] in SCIENCE_MODULES])

    def test_14_pinned_sources_exist(self):
        for name, item in D.SOURCES.items():
            subprocess.check_call(['git', 'cat-file', '-e', item['commit'] + ':' + item['path']], cwd=ROOT)
        selection = D._blob(D.FULL_H, D.SOURCES['full_h_saved_selection']['path'])
        self.assertIn(b'FULL_H_8d6f23210529d68452dd', selection)


if __name__ == '__main__':
    unittest.main(verbosity=2)
