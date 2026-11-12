"""D Full PF / DF Full-H RTE request adapter: metadata and artificial checks only.

Published facts (saved binding, lambda, selection IDs, pinned coefficients) are
read from pinned Git objects. No Hamiltonian/state load, signal, N, sampling,
compile, price or G. Any kappa below is a test-only probe of the published B
interface and never appears in adapter output.
"""
import ast
from copy import deepcopy
import json
import math
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
A_S0 = '7da8a35f68e9fdc2b46c361dd2c68e2a799b1e75'
LIGHT_H4 = {3: dict(q=16, R_outer=16)}
SCIENCE_MODULES = ('h46_long_time_signal', 'h46_long_time_native', 'h46_full_h_pure_rte', 'h46_full_h_t20_opt',
                   'h46_new2_native', 'h46_m5_tol_native', 'qiskit', 'pyscf')


def base(h=TENTH, M=3):
    s = spec(M=M); prep = D.saved_full_h_preparation('H4')
    s['binding'].update({k: prep['binding'][k] for k in D.BINDING_KEYS if k in prep['binding']}, rank=10)
    s['h'] = h
    return s, prep


def pf_spec(formula='2nd', M=3):
    s, _ = base(M=M)
    return D.full_pf_spec(s, formula=formula, delta_pf=TENTH, M=M, rank=10)


def full_pf(formula='2nd', M=3):
    return D.bind(pf_spec(formula, M))


def fh_spec(K=4, M=3, prep_edit=None):
    s, prep = base(M=M)
    if prep_edit:
        prep_edit(prep)
    return D.full_h_rte_spec(s, M=M, finite_K=K, preparation=prep)


def full_h(layout=LIGHT_H4, **kw):
    return D.bind(fh_spec(**kw), layout_by_round=layout)


def pinned_weights(label):
    """Execute weights() from the pinned h46_m5_plan text (not the working copy)."""
    tree = ast.parse(D._blob(D.BASE, 'src/trotterlib/h46_m5_plan.py'))
    nodes = [n for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'TAIL']
    nodes += [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'weights']
    scope = {}; exec(compile(ast.Module(body=nodes, type_ignores=[]), 'pinned', 'exec'), scope)
    return list(scope['weights'](label))


class FullBaselineTests(unittest.TestCase):
    def test_01_standard_4th_and_m5_best_are_separate_formulas(self):
        catalog = D.catalog()['formulas']
        self.assertNotEqual(catalog['4th']['formula_id'], catalog['4th(m5_best)']['formula_id'])
        self.assertEqual(catalog['4th']['v2_status'], 'D_PHYSICAL_REQUEST_READY_' + D.PENDING_FORMULA)
        y4, m5 = full_pf('4th'), full_pf('4th(m5_best)')
        self.assertEqual((y4['v2_status'], m5['v2_status']), (D.PENDING_FORMULA, 'B_V2_GENERATED'))
        a = [r['physical_candidate_id'] for r in y4['physical']['rounds']]
        b = [r['physical_candidate_id'] for r in m5['physical']['rounds']]
        self.assertTrue(all(x != y for x, y in zip(a, b)))
        self.assertEqual({r['formula_id'] for r in y4['physical']['rounds']}, {'YOSHIDA4_STANDARD_H46_M5_PLAN_BINARY64'})

    def test_02_standard_4th_signed_weights_order_and_negative_time(self):
        weights = pinned_weights('4th')
        rows = full_pf('4th')['physical']['rounds']
        self.assertTrue(all(r['signed_weights'] == weights for r in rows))
        u = 2**(1/3)
        self.assertEqual(weights, [1/(2-u), -u/(2-u), 1/(2-u)])
        self.assertTrue(weights[1] < 0 and all(r['negative_time'] for r in rows))
        # Same numbers as the existing s3odr4 list (middle first there); D keeps the h46 piece order.
        tree = ast.parse(D._blob(D.BASE, 'src/trotterlib/product_formula.py'))   # text only; no package import
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'yoshida_4th_list')
        node.returns = None; scope = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'pinned', 'exec'), scope)
        middle, outer = scope['yoshida_4th_list']()
        self.assertEqual((weights[0], weights[1], weights[2]), (outer, middle, outer))
        self.assertEqual(rows[3]['piece_time'], 'T*w_j/q')

    def test_03_full_PF_keeps_one_body_and_all_fragments(self):
        for formula in ('2nd', '4th', '4th(m5_best)'):
            for row in full_pf(formula)['physical']['rounds']:
                self.assertEqual((row['prefix'], row['physical_candidate']['method']), (10, 'FULL'))
                self.assertEqual(row['retained'], 'FULL_H_ONE_BODY_PLUS_ALL_FRAGMENTS')
                self.assertEqual((row['scalar'], row['control']), ('ONCE_AT_TOTAL_T', 'symmetric_directional'))
        self.assertEqual(D.saved_grid()['rank'], {'H4': 10, 'H6': 19})
        cut = truncated()
        self.assertEqual(B.comparison_method(cut['spec']), 'TRUNCATED_PF')
        with self.assertRaisesRegex(ValueError, 'Full PF and Full-H RTE only'):
            D.requests(cut)

    def test_04_H4_B0_prefix7_is_not_full_PF(self):
        s, _ = base()
        with self.assertRaisesRegex(ValueError, 'full rank'):
            D.full_pf_spec(s, formula='4th', delta_pf=TENTH, M=3, rank=7)
        bad = pf_spec(); bad['split']['retained_hamiltonian'].update(construction='B0', prefix=7)
        with self.assertRaisesRegex(ValueError, 'Full PF cannot carry a truncated'):
            D.bind(bad)
        bad = pf_spec('4th'); bad['split']['deterministic_scope'] = 'TRUNCATED_H'
        with self.assertRaisesRegex(ValueError, 'Full PF and Full-H RTE only'):
            D.bind(bad)

    def test_05_full_PF_has_no_kappa_K_or_layout(self):
        for formula in ('2nd', '4th'):
            s = pf_spec(formula)
            self.assertIsNone(s['kappa']); self.assertIsNone(s['finite_K'])
            self.assertTrue(all(r.get('mapping') is None and r['physical_candidate']['K'] is None
                                for r in D.bind(s)['physical']['rounds']))
            with self.assertRaisesRegex(ValueError, 'no RTE layout'):
                D.bind(s, layout_by_round={0: dict(q=1, R_outer=1)})
        bad = pf_spec(); bad['kappa'] = dict(kind='X', value=1, definition_source={'fixture': 'X'})
        with self.assertRaisesRegex(ValueError, 'without kappa'):
            D.bind(bad)
        with self.assertRaisesRegex(ValueError, 'out of scope'):
            pf_spec('8th(Morales)')

    def test_06_full_H_RTE_randomizes_every_one_body_component(self):
        for edit, message in [(lambda v: v.update(one_body_components=0), 'One-body'),
                              (lambda v: v.update(one_body_policy='DETERMINISTIC'), 'One-body'),
                              (lambda v: v.update(deterministic_blocks=1), 'One-body'),
                              (lambda v: v.update(randomized_fragments=list(range(9))), 'All DF fragments'),
                              (lambda v: v.update(randomized_fragments=list(range(9, -1, -1))), 'saved order'),
                              (lambda v: v.update(tail_only_lambda_R=v['lambda_R']), 'Tail-only lambda'),
                              (lambda v: v.update(one_body_lambda_R=0.), 'one-body l1')]:
            with self.assertRaisesRegex(ValueError, message):
                fh_spec(prep_edit=edit)
        s = fh_spec()
        self.assertEqual((s['split']['randomized_scope'], s['split']['one_body_policy']), ('FULL_H_INCLUDING_ONE_BODY', 'RANDOMIZED'))
        self.assertGreater(s['split']['lambda_R'], s['split']['preparation']['one_body_lambda_R'])
        self.assertNotEqual(s['split']['lambda_R'], 2.8629570945298686)   # PR tail-only lambda of A S0

    def test_07_DF_B3_is_not_full_H_RTE(self):
        with self.assertRaisesRegex(ValueError, 'prepare_full_h'):
            fh_spec(prep_edit=lambda v: v['source'].update(path='scripts/h46_b3.py', function='prepare'))
        bad = fh_spec(); bad['split']['original_PF_method'] = 'B3'; bad['kappa'] = dict(kind='X', value=1, definition_source={'f': 1})
        with self.assertRaisesRegex(ValueError, 'B3'):
            B.generate(bad)

    def test_08_identity_scalar_is_not_double_counted(self):
        for edit in (lambda v: v['scalar'].update(identity_components_randomized=1),
                     lambda v: v['scalar'].update(application='PER_PF_STEP'),
                     lambda v: v['scalar'].pop('extracted_identity')):
            with self.assertRaisesRegex(ValueError, 'applied once'):
                fh_spec(prep_edit=edit)
        row = full_h()['physical']['rounds'][3]
        self.assertEqual((row['scalar'], row['mapping']['physical_layout']['scalar']), ('ONCE_AT_TOTAL_T', 'ONCE_AT_TOTAL_T'))
        prep = D.saved_full_h_preparation('H4')['scalar']
        self.assertTrue(math.isclose(prep['constant'] + prep['extracted_identity'], prep['total'], abs_tol=1e-12))

    def test_09_q_by_R_layout_is_kept_whole(self):
        packet = full_h({1: dict(q=2, R_outer=5), 3: dict(q=16, R_outer=16)})
        row = packet['physical']['rounds'][3]; layout = row['mapping']['physical_layout']
        self.assertEqual((layout['q'], layout['R_outer'], layout['allocation'], layout['event_positions']), (16, 16, [16], 256))
        self.assertEqual((row['event_count'], row['K'], layout['taylor_K']), (256, 4, 4))
        self.assertEqual(layout['event_tau'], 'lambda_R*(T/q)/R_outer')
        self.assertEqual(layout['component_law'], packet['physical']['spec']['law'])
        self.assertEqual(layout['compose_source'], D.SOURCES['pf_native'])
        self.assertIsNone(row['mapping']['kappa'])
        self.assertFalse(layout['paper_kappa_optimality_claim'])
        self.assertEqual(B.a.exact_scalar(row['outer_step_width']), B.a.exact_scalar(row['T']) / 16)
        self.assertEqual(packet['physical']['rounds'][1]['event_count'], 10)
        self.assertEqual(packet['physical']['rounds'][0]['saved_asset_status'], 'LAYOUT_NOT_SUPPLIED')
        for bad in ({3: dict(event_count=256)}, {3: dict(q=0, R_outer=16)}, {9: dict(q=1, R_outer=1)}):
            with self.assertRaisesRegex(ValueError, 'integer q and R_outer'):
                full_h(bad)
        with self.assertRaisesRegex(ValueError, 'no kappa'):
            s, prep = base(); D.full_h_rte_spec(s, M=3, finite_K=4, preparation=prep, kappa=dict(kind='K', value=1))

    def test_10_same_event_count_different_grouping_are_different_candidates(self):
        one = full_h({3: dict(q=1, R_outer=256)}); two = full_h(LIGHT_H4)
        r1, r2 = one['physical']['rounds'][3], two['physical']['rounds'][3]
        self.assertEqual(r1['event_count'], r2['event_count'])
        self.assertNotEqual(r1['physical_candidate_id'], r2['physical_candidate_id'])
        self.assertEqual(r2['physical_candidate_id'], 'FULL_H_8d6f23210529d68452dd')
        self.assertEqual(r2['saved_asset_status'], 'SAVED_LIGHT_STUDY_CANDIDATE_FINITE_COHORT')
        self.assertEqual(r1['saved_asset_status'], 'NO_SAVED_EXACT_CANDIDATE')
        self.assertFalse(r2['mapping']['physical_layout']['priced_equivalence_claim'])

    def test_11_A_and_C_requests_point_to_one_physical_candidate(self):
        for packet in (full_pf('2nd'), full_pf('4th'), full_h()):
            out = D.requests(packet)
            for row in out['rounds']:
                self.assertEqual({v['axis'] for v in row['C']}, set(D.p.AXES))
                for price in row['C']:
                    for key in ('physical_candidate_id', 'physical_round', 'v2_candidate_id', 'v2_configuration_id',
                                'identity', 'law_binding', 'binding', 'T', 'q_PF'):
                        self.assertEqual(price[key], row['A'][key])
                    self.assertEqual(price['profile']['wrapper'], 'full_Hadamard_with_measurement')
                    self.assertEqual(price['profile']['P'], 0)
        pf = D.requests(full_pf('2nd'))['rounds'][3]
        self.assertEqual(pf['A']['saved_asset_status'], 'SAVED_LONG_TIME_GRID_POINT_CHECK_CARRIER')
        self.assertEqual(pf['C'][0]['law_context']['price_law'], 'deterministic')
        fh = D.requests(full_h())['rounds'][3]
        self.assertFalse(fh['C'][0]['law_context']['ideal_sampler_distance_qualified'])
        self.assertIsNotNone(fh['C'][0]['saved_price_source'])
        self.assertIn(D.PENDING_LAYOUT, fh['A']['missing'])

    def test_12_consistency_with_the_published_B_V2_interface(self):
        # Supported formulas: B.generate reproduces the physical recipe and IDs map one to one.
        for formula in ('2nd', '4th(m5_best)'):
            packet = full_pf(formula); config = packet['v2_configuration']
            self.assertEqual([r['V2_candidate_id'] for r in packet['identity_map']], [r['candidate']['id'] for r in config['rounds']])
            self.assertEqual(D.requests(config), D.requests(packet))
            self.assertEqual(B.comparison_method(config['spec']), 'FULL_PF')
            tampered = deepcopy(config); tampered['rounds'][1]['structure']['signed_weights'] = [2.]
            with self.assertRaisesRegex(ValueError, 'differs from the physical recipe'):
                D._check_against_b(packet['physical'], tampered)
        # Standard 4th: the formula check is the only blocker in the published B.
        y4 = pf_spec('4th')
        with self.assertRaisesRegex(ValueError, 'supported fixed formula'):
            B.generate(y4)
        B.generate(dict(deepcopy(y4), formula='4th(m5_best)'))
        # Explicit Full-H layout: kappa is the only blocker; a test-only probe kappa passes the rest.
        s = fh_spec()
        with self.assertRaises((TypeError, AttributeError, ValueError)):
            B.generate(s)
        probe = deepcopy(s); probe['kappa'] = dict(kind='TEST_ONLY_PROBE', value=1, definition_source={'test': 'probe'})
        def mapper(spec_, row):
            if row['m'] != 3:
                return None
            v = D.layout_mapping(spec_, LIGHT_H4[3]); v['kappa'] = deepcopy(spec_['kappa'])
            return v
        config = B.generate(probe, event_mapper=mapper)
        self.assertEqual(config['rounds'][3]['structure']['event_count'], 256)
        self.assertEqual(config['rounds'][3]['mapping']['physical_layout'],
                         full_h()['physical']['rounds'][3]['mapping']['physical_layout'])
        self.assertNotIn('TEST_ONLY_PROBE', json.dumps(D.requests(full_h())))
        # Pending cases never invent V2 IDs.
        for packet in (full_pf('4th'), full_h()):
            self.assertIsNone(packet['v2_configuration']); self.assertFalse(packet['invented_v2_ids'])
            self.assertTrue(all(r['V2_candidate_id'] is None and r['V2_configuration_id'] is None for r in packet['identity_map']))
            self.assertFalse(any(packet['b_interface'][k] for k in ('formula_hook', 'kappa_free_explicit_layout', 'physical_candidate_hook')))
        with self.assertRaisesRegex(ValueError, D.PENDING_FORMULA):
            D.generate(pf_spec('4th'))
        # The published B planner accepts the generated Full PF and lists A/C work.
        plan = E.plan([dict(configuration=full_pf()['v2_configuration'])], price_policy='qualified_expectation',
                      required_methods=('FULL_PF',))
        self.assertIsNone(plan['G'])
        self.assertEqual({w['owner'] for r in plan['configurations'][0]['rows'] for w in r['next_work']}, {'Codex A', 'Codex C'})

    def test_13_no_scientific_runner_or_compile_is_started(self):
        with patch.object(B, 'cost', side_effect=AssertionError('No G')):
            D.catalog()
            for packet in (full_pf('2nd'), full_pf('4th'), full_pf('4th(m5_best)'), full_h()):
                D.requests(packet)
        self.assertFalse([m for m in sys.modules if m.split('.')[0] in SCIENCE_MODULES])

    def test_14_missing_N_C_stay_null(self):
        for packet in (full_pf('4th'), full_h()):
            out = D.requests(packet)
            self.assertIsNone(out['N_C_values']); self.assertIsNone(out['G'])
            for row in out['rounds']:
                self.assertIsNone(row['A']['N_X']); self.assertIsNone(row['A']['rmse_bound_ha'])
                self.assertTrue(all(v['mean']['rz_count'] is None and v['price_status'] == 'NOT_EVALUATED' for v in row['C']))
        self.assertIsNone(D.requests(full_h())['rounds'][0]['A']['physical_candidate_id'])

    def test_15_DF_Pauli_and_bindings_are_not_mixed(self):
        with self.assertRaisesRegex(ValueError, 'DF Full-H RTE only'):
            fh_spec(prep_edit=lambda v: v.update(representation='PAULI_JW'))
        self.assertFalse(D.pauli_reference()['same_numerical_H_state_as_DF'])
        a_spec = json.loads(D._blob(A_S0, 'docs/research/pr_rpe_rmse_s0_results_20261113/run01/CONFIGURATION.json'))['B_configuration']['spec']
        prep = D.saved_full_h_preparation('H4')
        self.assertTrue(D.check_binding(a_spec['binding'], prep['binding']))
        for key in ('state', 'snapshot', 'hamiltonian', 'system'):
            with self.assertRaisesRegex(ValueError, 'binding differs'):
                D.check_binding(dict(a_spec['binding'], **{key: 'OTHER'}), prep['binding'])

    def test_16_pinned_sources_exist(self):
        for item in D.SOURCES.values():
            subprocess.check_call(['git', 'cat-file', '-e', item['commit'] + ':' + item['path']], cwd=ROOT)
        self.assertIn(b'FULL_H_8d6f23210529d68452dd', D._blob(D.FULL_H, D.SOURCES['full_h_saved_selection']['path']))


if __name__ == '__main__':
    unittest.main(verbosity=2)
