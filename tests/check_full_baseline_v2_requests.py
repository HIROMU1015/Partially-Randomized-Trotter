"""D Full PF / DF Full-H RTE request adapter on the published B V2 (f0c4bb55).

Metadata and artificial checks only. Published facts (saved binding, lambda,
selection IDs, pinned coefficients, B's own module copy) are read from pinned
Git objects. No Hamiltonian/state load, signal, N, sampling, compile, price or
G. Any kappa below is a synthetic test fixture and never comes from D.
"""
import ast
from copy import deepcopy
import json
import math
import subprocess
import sys
from pathlib import Path
import types
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
LIGHT_H4 = {3: dict(q=16, R_outer=16)}       # saved light-study H4 q16/R16/K4 at T=0.8 (m=3 for h=0.1)
SYNTHETIC_KAPPA = dict(kind='SYNTHETIC_SUPPLIED_KAPPA', value=1, definition_source={'fixture': 'SYNTHETIC_ONLY'})
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


def fh_spec(K=4, M=3, prep_edit=None, kappa=None):
    s, prep = base(M=M)
    if prep_edit:
        prep_edit(prep)
    return D.full_h_rte_spec(s, M=M, finite_K=K, preparation=prep, kappa=kappa)


def full_h(layout=LIGHT_H4, **kw):
    return D.bind(fh_spec(**kw), layout_by_round=layout)


def pinned_weights(label):
    """Execute weights() from the pinned h46_m5_plan text (not the working copy)."""
    tree = ast.parse(D._blob(D.BASE, 'src/trotterlib/h46_m5_plan.py'))
    nodes = [n for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'TAIL']
    nodes += [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'weights']
    scope = {}; exec(compile(ast.Module(body=nodes, type_ignores=[]), 'pinned', 'exec'), scope)
    return list(scope['weights'](label))


def b_published_module():
    """B's own copy of this adapter at the fixed B commit (published interface reference)."""
    src = D._blob(D.B_LATEST, 'src/trottertracks/full_baseline_v2_requests.py').decode()
    mod = types.ModuleType('trottertracks._b_published_full_baseline'); mod.__package__ = 'trottertracks'; mod.__file__ = D.__file__
    exec(compile(src, 'B_PUBLISHED:' + D.B_LATEST[:8], 'exec'), mod.__dict__)
    return mod


class FullBaselineTests(unittest.TestCase):
    def test_01_standard_4th_maps_to_B_canonical_ID_and_stays_distinct_from_m5(self):
        catalog = D.catalog()['formulas']
        self.assertEqual({catalog[f]['v2_status'] for f in ('2nd', '4th', '4th(m5_best)')}, {'B_V2_SUPPORTED'})
        y4, m5 = full_pf('4th'), full_pf('4th(m5_best)')
        self.assertEqual({r['V2_formula_id'] for r in y4['identity_map']}, {'PF_STANDARD_YOSHIDA4_BINARY64_V1'})
        self.assertEqual({r['physical_formula_label'] for r in y4['identity_map']}, {'YOSHIDA4_STANDARD_H46_M5_PLAN_BINARY64'})
        self.assertEqual({r['V2_formula_id'] for r in m5['identity_map']}, {'PF_M5_BEST_20D64C2_BINARY64_V1'})
        # The canonical ID is the one B's pipeline assigns, not a D label.
        self.assertIn(b"'PF_STANDARD_YOSHIDA4_BINARY64_V1' if c['formula'] == '4th'",
                      D._blob(D.B_LATEST, D.SOURCES['b_formula_id']['path']))
        self.assertNotEqual(y4['v2_configuration']['id'], m5['v2_configuration']['id'])
        a = [r['physical_candidate_id'] for r in y4['physical']['rounds']]
        b = [r['physical_candidate_id'] for r in m5['physical']['rounds']]
        self.assertTrue(all(x != y for x, y in zip(a, b)))
        self.assertEqual(list(y4['resolved_pending']), [D.PENDING_FORMULA])
        self.assertEqual(y4['resolved_pending'][D.PENDING_FORMULA]['resolved_by'], D.B_LATEST)

    def test_02_standard_4th_signed_weights_order_and_negative_time(self):
        weights = pinned_weights('4th')
        packet = full_pf('4th')
        rows = packet['physical']['rounds']
        self.assertTrue(all(r['signed_weights'] == weights for r in rows))
        self.assertTrue(all(v['structure']['signed_weights'] == weights for v in packet['v2_configuration']['rounds']))
        u = 2**(1/3)
        self.assertEqual(weights, [1/(2-u), -u/(2-u), 1/(2-u)])
        self.assertTrue(weights[1] < 0 and all(r['negative_time'] for r in rows))
        tree = ast.parse(D._blob(D.BASE, 'src/trotterlib/product_formula.py'))   # text only; no package import
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'yoshida_4th_list')
        node.returns = None; scope = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'pinned', 'exec'), scope)
        middle, outer = scope['yoshida_4th_list']()
        self.assertEqual((weights[0], weights[1], weights[2]), (outer, middle, outer))
        self.assertEqual(rows[3]['piece_time'], 'T*w_j/q')
        # Same coefficient SOURCE bytes in D's pin, B's commit and the saved long-time carrier commit.
        blobs = {subprocess.check_output(['git', 'rev-parse', c + ':src/trotterlib/h46_m5_plan.py'], cwd=ROOT)
                 for c in (D.BASE, D.B_LATEST, '155529e20a8fac39e00786c40b57830b71203c4f')}
        self.assertEqual(len(blobs), 1)

    def test_03_full_PF_keeps_one_body_and_all_fragments(self):
        for formula in ('2nd', '4th', '4th(m5_best)'):
            packet = full_pf(formula)
            for row, v2 in zip(packet['physical']['rounds'], packet['v2_configuration']['rounds']):
                self.assertEqual((row['prefix'], row['physical_candidate']['method']), (10, 'FULL'))
                self.assertEqual(row['retained'], 'FULL_H_ONE_BODY_PLUS_ALL_FRAGMENTS')
                self.assertEqual((row['scalar'], row['control']), ('ONCE_AT_TOTAL_T', 'symmetric_directional'))
                self.assertEqual(v2['candidate']['retained_hamiltonian']['construction'], 'FULL_H')
            self.assertEqual(B.comparison_method(packet['v2_configuration']['spec']), 'FULL_PF')
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
            packet = D.bind(s)
            self.assertTrue(all(r.get('mapping') is None and r['physical_candidate']['K'] is None for r in packet['physical']['rounds']))
            self.assertTrue(all(v['candidate']['K'] is None and v['mapping'] is None for v in packet['v2_configuration']['rounds']))
            with self.assertRaisesRegex(ValueError, 'no RTE layout'):
                D.bind(s, layout_by_round={0: dict(q=1, R_outer=1)})
        bad = pf_spec(); bad['kappa'] = deepcopy(SYNTHETIC_KAPPA)
        with self.assertRaisesRegex(ValueError, 'without kappa'):
            D.bind(bad)

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
        # B re-checks the preparation inside generate (kappa-free kind).
        bad = deepcopy(s); bad['split']['preparation']['one_body_policy'] = 'DETERMINISTIC'
        with self.assertRaisesRegex(ValueError, 'One-body'):
            D.bind(bad, layout_by_round=LIGHT_H4)

    def test_07_DF_B3_is_not_full_H_RTE(self):
        with self.assertRaisesRegex(ValueError, 'prepare_full_h'):
            fh_spec(prep_edit=lambda v: v['source'].update(path='scripts/h46_b3.py', function='prepare'))
        for kappa in (None, SYNTHETIC_KAPPA):
            bad = fh_spec(kappa=deepcopy(kappa)); bad['split']['original_PF_method'] = 'B3'
            with self.assertRaisesRegex(ValueError, 'B3'):
                D.bind(bad, layout_by_round=LIGHT_H4)

    def test_08_identity_scalar_is_not_double_counted(self):
        for edit in (lambda v: v['scalar'].update(identity_components_randomized=1),
                     lambda v: v['scalar'].update(application='PER_PF_STEP'),
                     lambda v: v['scalar'].pop('extracted_identity')):
            with self.assertRaisesRegex(ValueError, 'applied once'):
                fh_spec(prep_edit=edit)
        packet = full_h(); row = packet['physical']['rounds'][3]
        self.assertEqual((row['scalar'], row['mapping']['physical_layout']['scalar']), ('ONCE_AT_TOTAL_T', 'ONCE_AT_TOTAL_T'))
        self.assertEqual(packet['v2_configuration']['rounds'][3]['structure']['scalar_policy'], 'scalar_identity_once_at_total_T')
        prep = D.saved_full_h_preparation('H4')['scalar']
        self.assertTrue(math.isclose(prep['constant'] + prep['extracted_identity'], prep['total'], abs_tol=1e-12))

    def test_09_kappa_free_layout_keeps_q_by_R_and_is_not_a_kappa_schedule(self):
        packet = full_h({1: dict(q=2, R_outer=5), 3: dict(q=16, R_outer=16)})
        config = packet['v2_configuration']
        self.assertEqual(config['spec']['event_mapping']['kind'], B.EXPLICIT_LAYOUT)
        self.assertIsNone(config['spec']['kappa'])
        row = packet['physical']['rounds'][3]; layout = row['mapping']['physical_layout']
        self.assertEqual((layout['q'], layout['R_outer'], layout['allocation'], layout['event_positions']), (16, 16, [16], 256))
        self.assertEqual((row['event_count'], row['K'], layout['taylor_K']), (256, 4, 4))
        self.assertEqual(config['rounds'][3]['structure']['event_count'], 256)
        self.assertEqual(config['rounds'][3]['mapping'], row['mapping'])
        self.assertEqual(layout['event_tau'], 'lambda_R*(T/q)/R_outer')
        self.assertEqual(layout['compose_source'], D.SOURCES['pf_native'])
        self.assertEqual((row['grouping'], row['parameterization']), ('S2_OUTER_Q_TIMES_R', 'EXPLICIT_LAYOUT_NO_KAPPA'))
        self.assertIsNone(row['mapping']['kappa']); self.assertIsNone(config['rounds'][3]['law_binding']['kappa'])
        self.assertFalse(row['paper_kappa_optimality_claim'])
        self.assertEqual(B.a.exact_scalar(row['outer_step_width']), B.a.exact_scalar(row['T']) / 16)
        self.assertEqual(packet['physical']['rounds'][1]['event_count'], 10)
        self.assertEqual(packet['physical']['rounds'][0]['saved_asset_status'], 'LAYOUT_NOT_SUPPLIED')
        self.assertEqual([r['V2_resolution'] for r in packet['identity_map']], ['NOT_EVALUATED', 'GENERATED', 'NOT_EVALUATED', 'GENERATED'])
        self.assertIn('UNRESOLVED_PHYSICAL_LAYOUT', B.assess(config, None, [], cohorts={})['reasons'])
        for bad in ({3: dict(event_count=256)}, {3: dict(q=0, R_outer=16)}, {9: dict(q=1, R_outer=1)}):
            with self.assertRaisesRegex(ValueError, 'integer q and R_outer'):
                full_h(bad)
        with self.assertRaisesRegex(ValueError, 'explicit physical layout'):
            D.bind(fh_spec())
        with self.assertRaisesRegex(ValueError, 'named source-bound'):
            fh_spec(kappa=dict(kind='K', value=1))
        # A supplied kappa is a different parameterization (legacy kind), never merged with the kappa-free one.
        legacy = full_h(kappa=deepcopy(SYNTHETIC_KAPPA))
        self.assertEqual(legacy['v2_configuration']['spec']['event_mapping']['kind'], D.LAYOUT_KIND)
        self.assertNotEqual(legacy['v2_configuration']['id'], full_h()['v2_configuration']['id'])
        self.assertEqual(legacy['physical']['rounds'][3]['physical_candidate_id'], full_h()['physical']['rounds'][3]['physical_candidate_id'])
        self.assertEqual(legacy['resolved_pending'], {})
        menu = B.select([full_h()['v2_configuration'], legacy['v2_configuration']], [], required_methods=('FULL_H_RTE',))
        self.assertEqual({v['parameterization'] for v in menu['schedule_menu']}, {'EXPLICIT_LAYOUT', 'COMMON_KAPPA'})

    def test_10_same_event_count_different_grouping_are_different_candidates(self):
        one = full_h({3: dict(q=1, R_outer=256)}); two = full_h(LIGHT_H4)
        r1, r2 = one['physical']['rounds'][3], two['physical']['rounds'][3]
        self.assertEqual(r1['event_count'], r2['event_count'])
        self.assertNotEqual(r1['physical_candidate_id'], r2['physical_candidate_id'])
        self.assertNotEqual(one['v2_configuration']['id'], two['v2_configuration']['id'])
        self.assertNotEqual(one['identity_map'][3]['V2_candidate_id'], two['identity_map'][3]['V2_candidate_id'])
        self.assertEqual(r2['physical_candidate_id'], 'FULL_H_8d6f23210529d68452dd')
        self.assertEqual(r2['saved_asset_status'], 'SAVED_LIGHT_STUDY_CANDIDATE_FINITE_COHORT')
        self.assertEqual(r1['saved_asset_status'], 'NO_SAVED_EXACT_CANDIDATE')
        self.assertFalse(r2['mapping']['physical_layout']['priced_equivalence_claim'])
        selection = json.loads(D._blob(D.FULL_H, D.SOURCES['full_h_saved_selection']['path']))
        saved = next(v for v in selection['candidates'] if v['id'] == r2['physical_candidate_id'])
        self.assertEqual((saved['q'], saved['R_outer'], saved['K'], saved['T'], saved['system']), (16, 16, 4, .8, 'H4'))

    def test_11_A_and_C_requests_point_to_one_physical_candidate(self):
        for packet in (full_pf('2nd'), full_pf('4th'), full_pf('4th(m5_best)'), full_h()):
            out = D.requests(packet); config = packet['v2_configuration']
            self.assertEqual(out['configuration_id'], config['id'])
            self.assertTrue(D.validate_requests(config, out))
            for row, v2 in zip(out['rounds'], config['rounds']):
                self.assertEqual(row['A']['v2_candidate_id'], v2['candidate']['id'])
                self.assertEqual({v['axis'] for v in row['C']}, set(D.p.AXES))
                for price in row['C']:
                    for key in ('configuration_id', 'physical_candidate_id', 'physical_round', 'v2_candidate_id', 'identity',
                                'law_binding', 'physical_schedule_mapping', 'binding', 'T', 'q_PF', 'b_commit'):
                        self.assertEqual(price[key], row['A'][key])
                    self.assertEqual(price['profile']['wrapper'], 'full_Hadamard_with_measurement')
                    self.assertEqual(price['profile']['P'], 0)
                self.assertNotIn(D.PENDING_FORMULA, row['A']['missing']); self.assertNotIn(D.PENDING_LAYOUT, row['A']['missing'])
        pf = D.requests(full_pf('2nd'))['rounds'][3]
        self.assertEqual(pf['A']['saved_asset_status'], 'SAVED_LONG_TIME_GRID_POINT_CHECK_CARRIER')
        self.assertEqual(pf['C'][0]['law_context']['price_law'], 'deterministic')
        fh = D.requests(full_h())['rounds'][3]
        self.assertFalse(fh['C'][0]['law_context']['ideal_sampler_distance_qualified'])
        self.assertIsNotNone(fh['C'][0]['saved_price_source'])
        self.assertEqual(fh['A']['physical_candidate_id'], 'FULL_H_8d6f23210529d68452dd')
        bad = D.requests(full_h()); bad['rounds'][3]['C'][1]['physical_candidate_id'] = 'FULL_H_OTHER'
        with self.assertRaisesRegex(ValueError, 'bridge mismatch'):
            D.validate_requests(full_h()['v2_configuration'], bad)

    def test_12_V2_IDs_are_those_of_the_published_B_interface(self):
        published = b_published_module()
        s, prep = base()
        for formula in ('2nd', '4th', '4th(m5_best)'):
            packet = full_pf(formula); config = packet['v2_configuration']
            self.assertEqual(config, published.generate(published.full_pf_spec(s, formula=formula, delta_pf=TENTH, M=3, rank=10)))
            self.assertEqual([r['V2_candidate_id'] for r in packet['identity_map']], [r['candidate']['id'] for r in config['rounds']])
            self.assertEqual(D.requests(config), D.requests(packet))
            tampered = deepcopy(config); tampered['rounds'][1]['structure']['signed_weights'] = [2.]
            with self.assertRaisesRegex(ValueError, 'differs from the physical recipe'):
                D._check_against_b(packet['physical'], tampered)
            tampered = deepcopy(config); tampered['rounds'][2]['structure']['formula_id'] = 'OTHER_FORMULA_ID'
            with self.assertRaisesRegex(ValueError, 'differs from the physical recipe'):
                D._check_against_b(packet['physical'], tampered)
        rte = full_h()
        self.assertEqual(rte['v2_configuration'], published.generate(
            published.full_h_rte_spec(s, M=3, finite_K=4, kappa=None, preparation=prep), layout_by_round=LIGHT_H4))
        self.assertEqual(list(rte['resolved_pending']), [D.PENDING_LAYOUT])
        self.assertEqual(published.requests(rte['v2_configuration'])['rounds'][3]['A']['physical_candidate_id'],
                         rte['physical']['rounds'][3]['physical_candidate_id'])
        for packet in (full_pf('4th'), rte):
            self.assertFalse(packet['invented_v2_ids'])
            self.assertTrue(all(r['V2_configuration_id'] == packet['v2_configuration']['id'] and r['b_commit'] == D.B_LATEST
                                for r in packet['identity_map']))
            self.assertTrue(packet['b_interface']['formula_hook'] and packet['b_interface']['kappa_free_explicit_layout'])
            self.assertFalse(packet['b_interface']['physical_candidate_hook'])
        # A relabelled configuration (not reproducible from its spec) is rejected.
        other = deepcopy(rte['v2_configuration']); other['rounds'][3]['candidate']['id'] = 'RMSE:INVENTED'
        with self.assertRaises(ValueError):
            D.requests(other)
        plan = E.plan([dict(configuration=full_pf('4th')['v2_configuration']), dict(configuration=rte['v2_configuration'])],
                      price_policy='qualified_expectation', required_methods=('FULL_PF', 'FULL_H_RTE'))
        self.assertIsNone(plan['G'])
        self.assertEqual({w['owner'] for c in plan['configurations'] for r in c['rows'] for w in r['next_work']}, {'Codex A', 'Codex C'})

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
        subprocess.check_call(['git', 'cat-file', '-e', D.B_LATEST + ':' + D.B_INTERFACE['path']], cwd=ROOT)

    def test_17_eighth_order_PF_is_not_connected(self):
        catalog = D.catalog()['formulas']
        for label in ('8th(Morales)', '8th(Yoshida)'):
            self.assertEqual(catalog[label]['v2_status'], 'OUT_OF_SCOPE_NO_DF_CONTROLLED_FULL_WRAPPER_PATH')
            with self.assertRaisesRegex(ValueError, 'out of scope'):
                pf_spec(label)
            with self.assertRaisesRegex(ValueError, 'supported fixed formula'):
                B.generate(dict(pf_spec('4th'), formula=label))


if __name__ == '__main__':
    unittest.main(verbosity=2)
