"""Artificial matrices/laws/schedules only; no saved H4 numerical evaluation."""
import copy
import csv
from fractions import Fraction as Q
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import round_adapter as a
import run_round_enclosures as run
from flint import acb, acb_mat, arb, ctx


def row(m=0, r=1):
    return {'ld': 3, 'm': m, 'q': 1 << m, 'r': r, 'K': 2, 'signal_tau64': '0.125',
            'mu0': {'real': '1.0', 'imag': '0.0'},
            'axes': {axis: {'N': '100000', 'alpha': '0.1', 'epsilon': '0.5'} for axis in ('cosine', 'sine')}}


def schema_fixture():
    fields = ['ld', 'round_index', 'q_m', 'r_m', 'K_m', 'delta_time', 'saved_complex_json',
              'axes_results_json', 'saved_beta_PF', 'saved_beta_RTE', 'saved_beta_stat']
    text = io.StringIO(); writer = csv.DictWriter(text, fields); writer.writeheader()
    points, models = [], []
    for m in range(18):
        r = 1 if m < 8 else 2 if m < 10 else 4 if m < 11 else 8 if m < 13 else 16 if m < 15 else 32
        fixed = row(m, r)
        writer.writerow({'ld': '3', 'round_index': str(m), 'q_m': str(1 << m), 'r_m': str(r), 'K_m': '2',
            'delta_time': '0.02', 'saved_complex_json': json.dumps(fixed['mu0']),
            'axes_results_json': json.dumps({axis: {'saved_N': x['N'], 'saved_alpha': x['alpha'],
                'saved_epsilon_coordinate': x['epsilon']} for axis, x in fixed['axes'].items()}),
            'saved_beta_PF': '0.01', 'saved_beta_RTE': '0.01', 'saved_beta_stat': '0.38'})
        point = {'round_index': str(m), 'q_m': str(1 << m), 'r_m': str(r), 'K_m': '2', 'tau_m': '0.125'}
        points.append({**point, 'state_results': [{'state_label': 'physical_df_ground_state',
                                                   'attenuated_event_mean_signal': fixed['mu0']}]})
        models.append({**point, 'epsilon_coordinate': '0.5', 'axes': {
            axis: {'shots': x['N'], 'alpha': x['alpha']} for axis, x in fixed['axes'].items()}})
    return text.getvalue(), {'points': points}, {'coherent_reoptimization': {'best_by_ld': {'3': {'best': {'rounds': models}}}}}


class ExtensionTests(unittest.TestCase):
    def setUp(self):
        self.prec, self.threads = ctx.prec, ctx.threads
        ctx.prec, ctx.threads = 96, 1

    def tearDown(self):
        ctx.prec, ctx.threads = self.prec, self.threads

    def test_18_schema_has_all_six_r_groups(self):
        rows = run.bind_rows(*schema_fixture())
        self.assertEqual([x['m'] for x in rows], list(range(18)))
        self.assertEqual(sorted({x['r'] for x in rows}), [1, 2, 4, 8, 16, 32])
        self.assertTrue(all(x['K'] == 2 for x in rows))

    def test_missing_duplicate_or_unsupported_input_stops(self):
        for kind in ('missing_point', 'duplicate_point', 'bad_K', 'missing_axis', 'wrong_mean', 'wrong_N'):
            csv_text, grid, opt = schema_fixture()
            if kind == 'missing_point': grid['points'].pop()
            if kind == 'duplicate_point': grid['points'].append(copy.deepcopy(grid['points'][0]))
            if kind == 'bad_K': grid['points'][0]['K_m'] = '4'
            if kind == 'missing_axis': del opt['coherent_reoptimization']['best_by_ld']['3']['best']['rounds'][0]['axes']['sine']
            if kind == 'wrong_mean': grid['points'][0]['state_results'][0]['attenuated_event_mean_signal']['real'] = '0.0'
            if kind == 'wrong_N': opt['coherent_reoptimization']['best_by_ld']['3']['best']['rounds'][0]['axes']['cosine']['shots'] = '99'
            with self.subTest(kind=kind), self.assertRaises((ValueError, KeyError)):
                run.bind_rows(csv_text, grid, opt)

    def test_signal_and_model_tau_never_substituted(self):
        text, grid, opt = schema_fixture()
        opt['coherent_reoptimization']['best_by_ld']['3']['best']['rounds'][0]['tau_m'] = '0.25'
        item = run.bind_rows(text, grid, opt)[0]
        self.assertEqual((item['signal_tau64'], item['model_tau64']), ('0.125', '0.25'))

    def test_decode_retains_saved_and_conversion_radii(self):
        saved = {'shape': [1, 1], 'center': [['1/3', '0']], 'spectral_radius_upper': '1/1000', 'target_norm_upper': '1'}
        value = a.decode(saved)
        self.assertGreater(value.radius, a.b.real('1/1000'))
        with self.assertRaises(ValueError): a.decode({**saved, 'spectral_radius_upper': '-1'})

    def test_saved_norm_reused_without_old_expectation(self):
        operator = a.b.unit(acb_mat([[acb(0, 1), 0], [0, 1]]))
        with patch.object(a.b, 'expectation', side_effect=AssertionError('no norm recomputation')):
            value = a.expectation(operator, [['2', '0'], ['0', '0']], {'lower': '4', 'upper': '4'})
        self.assertTrue(value.contains(acb(0, 1)))
        self.assertTrue(a.interval({'lower': '3999/1000', 'upper': '4001/1000'}).contains(4))
        for interval in ({'lower': '0', 'upper': '1'}, {'lower': '2', 'upper': '1'}):
            with self.assertRaises(ValueError): a.expectation(operator, [['2', '0'], ['0', '0']], interval)

    def test_general_even_K_and_varying_r_on_artificial_identity(self):
        # For theta=0 and all components I: E=(sum p_n*(-1)^(n/2)) I.
        unit = a.b.unit(a.b.identity(2))
        for orders, masses in [([0], ['1']), ([0, 2], ['3/4', '1/4']), ([0, 2, 4], ['1/2', '1/4', '1/4'])]:
            scalar = sum(Q(p)*(-1)**(n//2) for n, p in zip(orders, masses))
            for r in (1, 2, 4, 8, 16, 32):
                step = a.make_step(unit, unit, unit, r=r, orders=orders, angles=[0.]*len(orders), law={'grid_masses': masses})
                value = a.expectation(step, [['1', '0'], ['0', '0']], {'lower': '1', 'upper': '1'})
                self.assertTrue(value.contains(a.b.real(scalar**r)))

    def test_source_law_replay_different_K_synthetic_only(self):
        for k in (0, 2, 4):
            orders, angles, law = a.replay_order_law(0.125, k)
            self.assertEqual(orders, list(range(0, k+1, 2)))
            self.assertEqual(len(angles), len(orders))
            self.assertEqual(sum(Q(x) for x in law['grid_masses']), 1)

    def test_F_product_R_order_noncommuting(self):
        forward = a.b.unit(acb_mat([[0, 1], [1, 0]]))
        reverse = a.b.unit(acb_mat([[1, 0], [0, acb(0, 1)]]))
        average = a.b.unit(acb_mat([[1, 0], [0, -1]]))
        step = a.make_step(average, forward, reverse, r=1, orders=[0], angles=[.25], law={'grid_masses': ['1']})
        with ctx.workprec(224):
            rotation = acb_mat([[acb(0, a.b.real(-.25)).exp(), 0], [0, acb(0, a.b.real(.25)).exp()]])
            target = reverse.center*rotation*forward.center
            wrong = forward.center*rotation*reverse.center
        self.assertLess(a.b.frobenius(step.center-target), step.radius+a.b.real('1e-25'))
        self.assertGreater(a.b.frobenius(step.center-wrong), 1)

    def test_phase_transfer_once_no_extra_attenuation(self):
        unit = a.b.unit(a.b.identity(2))
        item = row(3, 4)
        with patch.object(a.mean, 'builder_phase_bound', return_value=a.b.upper(a.b.real('1/100'))) as phase:
            out = a.evaluate_round(unit, [['1', '0'], ['0', '0']], {'lower': '1', 'upper': '1'}, item)
        phase.assert_called_once_with(8, 4)
        self.assertTrue(a.interval(out['logical_signed_mean_interval']['real']).contains(1))
        self.assertLess(Q(out['axes']['cosine']['U_cat_upper']), Q(11, 1000))
        self.assertIsNone(out['axes']['cosine']['U_exec_upper'])

    def test_K4_and_m17_not_new_science(self):
        unit = a.b.unit(a.b.identity(2))
        for invalid in ({**row(), 'K': 4}, row(17, 32), {**row(), 'ld': 12}):
            with self.assertRaises(ValueError): a.evaluate_round(unit, [['1', '0'], ['0', '0']], {'lower': '1', 'upper': '1'}, invalid)

    def test_cached_path_encloses_independent_high_precision_reference(self):
        average = a.b.recenter(acb_mat([[a.b.real('1/2'), a.b.real('1/4')],
                                       [a.b.real('1/4'), a.b.real('-1/2')]]), norm=1)
        forward = a.b.unit(acb_mat([[0, 1], [1, 0]]))
        reverse = a.b.unit(acb_mat([[1, 0], [0, acb(0, 1)]]))
        orders, angles, law = a.replay_order_law(.125, 2)
        for r in (1, 2, 8, 32):
            step = a.make_step(average, forward, reverse, r=r, orders=orders, angles=angles, law=law)
            powered = a.b.power(step, 16)
            result = a.expectation(powered, [['1', '0'], ['0', '0']], {'lower': '1', 'upper': '1'})
            with ctx.workprec(224):
                matrix = acb_mat([[a.b.real('1/2'), a.b.real('1/4')], [a.b.real('1/4'), a.b.real('-1/2')]])
                event = acb_mat(2, 2)
                for n, theta, mass in zip(orders, angles, law['grid_masses']):
                    theta = a.b.real(theta)
                    event += a.b.real(mass)*(-1)**(n//2)*(theta.cos()*a.b.identity(2)-acb(0, 1)*theta.sin()*matrix)*(matrix**n)
                expected = ((reverse.center*(event**r)*forward.center)**16)[0, 0]
            self.assertTrue(result.contains(expected))

    def test_invalid_law_and_nonfinite_or_dimension_inputs_stop(self):
        unit = a.b.unit(a.b.identity(2))
        for r in (0, -1, True):
            with self.assertRaises(ValueError): a.make_step(unit, unit, unit, r=r, orders=[0], angles=[0.], law={'grid_masses': ['1']})
        with self.assertRaises(ValueError): a.make_step(unit, unit, unit, r=1, orders=[0], angles=[0.], law={'grid_masses': ['1/2']})
        with self.assertRaises(ValueError): a.expectation(unit, [['1', '0']], {'lower': '1', 'upper': '1'})
        with self.assertRaises(ValueError): a.decode({'shape': [1, 1], 'center': [['nan', '0']], 'spectral_radius_upper': '0', 'target_norm_upper': '1'})

    def test_strict_boundary_unknown_and_full_aggregation(self):
        yes = {'categorical_decision': 'SUFFICIENT_CONDITION_CONFIRMED'}
        rows = [{'m': m, 'axes': {'cosine': yes.copy(), 'sine': yes.copy()}} for m in range(18)]
        self.assertIn('CONFIRMED', a.aggregate(rows))
        rows[3]['axes']['sine']['categorical_decision'] = 'UNDETERMINED'
        self.assertEqual(a.aggregate(rows), 'UNDETERMINED')
        with self.assertRaises(ValueError): a.aggregate(rows[:-1])
        ambiguous = a.b.classify_interval(arb(0, 1), 0., 100, '.1', '.5')
        self.assertEqual(ambiguous['categorical_decision'], 'UNDETERMINED')

    def test_describe_never_enters_worker(self):
        files = {'M17RESULT': {'commit': 'artificial'}}
        with patch.object(run, 'metadata', return_value=({'files': files}, list(range(18)), {})), \
             patch.object(run, 'group_worker', side_effect=AssertionError('no science')):
            info = run.describe()
        self.assertEqual(info['new_H4_numeric_calls'], 0)
        self.assertFalse(info['RPE_tolerance_certified'])

    def test_stopped_worker_never_decodes_or_replays_law(self):
        import threading
        event = threading.Event(); event.set()
        run.initialize_worker(event)
        try:
            with patch.object(a, 'decode', side_effect=AssertionError('no new decoding')):
                output = run.group_worker([row()], {}, 1)
            self.assertTrue(output['stopped_before_law'])
            self.assertEqual(output['rounds'], [])
        finally:
            run.initialize_worker(None)

    def test_group_worker_reuses_cached_r32_step_and_stops_on_unknown(self):
        unit = a.b.unit(a.b.identity(2)); encoded = a.b.encode_matrix_ball(unit)
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, 'w') as z: z.writestr('96.json', json.dumps({'step': encoded}))
        raw = {'OPERATORS': archive.getvalue(), 'EXPORT': json.dumps({'state': [['1', '0'], ['0', '0']]}),
               'LAWS': json.dumps({'order': {'grid_masses': ['1']}}),
               'M17RESULT': json.dumps({'fixed_theta64_hex': [], 'enclosures': [{'state': {'saved_vector_norm_squared': {'lower': '1', 'upper': '1'}}}]})}
        unknown = {'m': 0, 'axes': {'cosine': {'categorical_decision': 'UNDETERMINED'}}}
        with patch.object(a, 'replay_order_law', side_effect=AssertionError('do not replay r32')), \
             patch.object(a, 'evaluate_round', return_value=unknown) as evaluate:
            output = run.group_worker([row(0, 32), row(1, 32)], raw, 1)
        self.assertEqual(evaluate.call_count, 1)
        self.assertEqual(output['law']['new_CDF_calls'], 0)

    def test_exception_writes_stop_result_fresh_output_only(self):
        class BrokenPool:
            def __init__(self, **kw): pass
            def __enter__(self): raise RuntimeError('artificial worker launch failure')
            def __exit__(self, *args): pass
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(run, 'metadata', return_value=({'files': {}}, [row(m) for m in range(18)], {})), \
             patch.object(run, 'frozen_source', return_value={'commit': 'artificial'}), \
             patch.object(run, 'ProcessPoolExecutor', BrokenPool):
            output = Path(directory)/'new'
            with self.assertRaises(RuntimeError): run.execute(output)
            result = json.loads((output/'RESULT.json').read_text())
            self.assertEqual(result['categorical_local_conditions'], 'UNDETERMINED')
            self.assertTrue(result['mandatory_STOP'])
            with self.assertRaises(FileExistsError): run.execute(output)

    def test_parallel_end_to_end_artificial_groups_with_m17_reuse(self):
        rows = run.bind_rows(*schema_fixture())
        for item in rows: item['signal_tau64'] = '0.0'
        encoded = a.b.encode_matrix_ball(a.b.unit(a.b.identity(2)))
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr('96.json', json.dumps({k: encoded for k in ('A', 'forward', 'reverse', 'step')}))
        old = {'axes': {axis: {'categorical_decision': 'SUFFICIENT_CONDITION_CONFIRMED'} for axis in ('cosine', 'sine')},
               'state': {'saved_vector_norm_squared': {'lower': '1', 'upper': '1'}}}
        raw = {'OPERATORS': archive.getvalue(), 'EXPORT': json.dumps({'state': [['1', '0'], ['0', '0']]}),
               'LAWS': json.dumps({'order': {'grid_masses': ['1']}}),
               'M17RESULT': json.dumps({'fixed_theta64_hex': [], 'enclosures': [old]})}
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(run, 'metadata', return_value=({'files': {}}, rows, raw)), \
             patch.object(run, 'frozen_source', return_value={'commit': 'artificial'}):
            output = Path(directory)/'synthetic'
            run.execute(output)
            record = json.loads((output/'RESULT.json').read_text())
            self.assertEqual(len(list(output.glob('group_r*.json'))), 6)
        self.assertEqual(len(record['groups']), 6)
        self.assertEqual(sum(len(g['rounds']) for g in record['groups']), 17)
        self.assertEqual(sum(g['law']['new_CDF_calls'] for g in record['groups']), 5)
        self.assertFalse(record['m17_recomputed'])
        self.assertEqual(record['categorical_local_conditions'], 'CONDITIONAL_IDEAL_CATEGORICAL_LOCAL_CONDITIONS_CONFIRMED')
        self.assertFalse(record['RPE_tolerance_certified'])
        self.assertIsNone(record['U_exec_upper'])


if __name__ == '__main__':
    unittest.main()
