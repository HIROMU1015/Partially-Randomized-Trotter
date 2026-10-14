"""Fabricated 92-cell registry/union6; real H8 saved values are never loaded."""
import copy
import importlib.util
from pathlib import Path
import statistics
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location('h8_d0_prototype',
    Path(__file__).resolve().parents[1]/'scripts/pre_review_h8_d0_adapter_v1.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def fixture():
    cells = a.registry()
    selected = [c['id'] for c in cells if c['method'] == 'B2'][:6]
    binding = dict(fixture_kind='ARTIFICIAL', analysis_source=a.D0_SOURCE, source='a'*40,
        seed_source='b'*40, result='c'*40, input='d'*64, compiler='e'*64,
        allowed_wrapper_sources=['a'*40])
    signals, wrappers = [], []
    for index, cell in enumerate(cells):
        s = dict(fixture_kind='ARTIFICIAL', cell=cell,
            error_decomposition=dict(total_signed=dict(real=0., imag=0.)),
            allowance=dict(real=1e-11, imag=1e-11, log_B_margin=1e-11,
                kind='EMPIRICAL', certified=False), log_B=0.,
            reference=dict(real=.7, imag=.2), signals=dict(corrected=dict(real=.7, imag=.2)))
        s['precision_rows'] = {str(e): {str(f): a.d0.accounting(s, e, f)
            for f in a.FACTORS} for e in a.ORIGINAL_EPSILONS}
        signals.append(s)
        phases = [('exploration', 8 if cell['method'] == 'B2' else 1)]
        if cell['id'] in selected:
            phases.append(('confirmation', 32))
        for phase, count in phases:
            for i in range(count):
                for axis in ('cosine', 'sine'):
                    # Deliberately correlated axis values, with different phase means.
                    v = 5+index+i if phase == 'exploration' else 50+index+2*i
                    if axis == 'sine':
                        v += 3*i
                    metric_values = {m: v for m in a.d0.METRICS}
                    if cell['id'] == 'B2_p7_q8_2nd_r4_K4':
                        metric_values['CX'] = 1  # Metric-specific optimum outside frozen union6.
                    wrappers.append(dict(cell_id=cell['id'], phase=phase, replica=i,
                        seed=a.task_seed(binding, cell['id'], phase, i), axis=axis,
                        source=binding['source'], compiler_hash=binding['compiler'],
                        control='symmetric_directional', event_digest=f'ARTIFICIAL/{phase}/{i}',
                        **metric_values))
    by = {s['cell']['id']: s for s in signals}
    selection = dict(selected_ids=selected, fixed_before_confirmation=True,
        exploration_confirmation_pooled=False, signal_digest=a.d0.digest(signals),
        registrations=[dict(task=dict(cell_id=cid, cell=by[cid]['cell'], phase='confirmation', replica=i,
            seed=a.task_seed(binding, cid, 'confirmation', i),
            expected_signal=by[cid]['signals']['corrected'])) for cid in selected for i in range(32)])
    binding['selection_digest'] = a.d0.digest(selection)
    old_maps = {}
    for scope, phase in [('H8_EXPLORATION92', 'exploration'), ('H8_CONFIRMATION_UNION6', 'confirmation')]:
        cfg = dict(primary_phase=phase, compiler_hashes=[binding['compiler']], point_rule='sum_axis_products')
        if phase == 'confirmation':
            cfg['confirmation_ids'] = selected
        rows = a.d0.analyze_scope(signals, [r for r in wrappers if r['phase'] == phase], cfg,
            a.ORIGINAL_EPSILONS, a.FACTORS, [r for r in wrappers if r['phase'] == 'exploration'])
        old_maps[scope] = dict(rows=[dict(cell_id=r['cell_id'], epsilon=r['epsilon'],
            allowance_factor=r['factor'], metric=m, G_point=r['metrics'][m]['G_point'],
            G_standard_error=r['metrics'][m]['G_standard_error'], N_total=r['N_total'])
            for r in rows if r['status'] == 'DERIVED_SAVED_SCALAR' for m in a.d0.METRICS])
    return signals, wrappers, selection, binding, old_maps


class H8D0AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = fixture()
        cls.result = a.analyze_artificial(*cls.data)

    def test_registry_and_scopes(self):
        self.assertEqual(len(a.registry()), 92)
        r = self.result['scopes']
        self.assertEqual(len(r['H8_EXPLORATION92']['records']), 92*5*2)
        self.assertEqual(len(r['H8_CONFIRMATION_UNION6']['records']), 26*5*2)

    def test_exploration8_confirmation32_no_pooling(self):
        cid = self.data[2]['selected_ids'][0]
        for scope, n in [('H8_EXPLORATION92', 8), ('H8_CONFIRMATION_UNION6', 32)]:
            row = next(r for r in self.result['scopes'][scope]['records'] if r['cell_id'] == cid)
            self.assertEqual(row['sample_count'], n)
        values = [next(r for r in self.result['scopes'][scope]['records'] if r['cell_id'] == cid)
            ['metrics']['RZ']['G_point'] for scope in ('H8_EXPLORATION92', 'H8_CONFIRMATION_UNION6')]
        self.assertNotEqual(*values)

    def test_paired_se_independent_oracle(self):
        row = next(r for r in self.result['scopes']['H8_CONFIRMATION_UNION6']['records'] if r['method'] == 'B2')
        pairs = a.d0.paired_rows([r for r in self.data[1] if r['phase'] == 'confirmation'],
            cell_id=row['cell_id'], phase='confirmation', compiler_hash=self.data[3]['compiler'])
        totals = [sum(row['accounting']['axes'][channel]['shots']*p[axis]['RZ']
            for axis, channel in a.d0.AXIS_CHANNEL.items()) for p in pairs]
        self.assertEqual(row['metrics']['RZ']['G_standard_error'], statistics.stdev(totals)/32**.5)
        self.assertEqual(set(row['metrics']), set(a.d0.METRICS))

    def test_both_old_scopes_pass_before_extension(self):
        seen = []
        verify = a.d0.verify_original
        accounting = a.d0.accounting
        def guard_accounting(record, epsilon, factor):
            if epsilon == .0005:
                self.assertEqual(seen, ['PASS', 'PASS'])
            return accounting(record, epsilon, factor)
        def guard_verify(*args, **kwargs):
            out = verify(*args, **kwargs); seen.append(out['status']); return out
        with mock.patch.object(a.d0, 'accounting', side_effect=guard_accounting), \
                mock.patch.object(a.d0, 'verify_original', side_effect=guard_verify):
            a.analyze_artificial(*self.data)

    def test_mismatch_stops_before_extension(self):
        data = copy.deepcopy(self.data)
        data[4]['H8_CONFIRMATION_UNION6']['rows'][0]['G_point'] += 1
        accounting = a.d0.accounting
        def no_extension(record, epsilon, factor):
            self.assertNotEqual(epsilon, .0005)
            return accounting(record, epsilon, factor)
        with mock.patch.object(a.d0, 'accounting', side_effect=no_extension):
            with self.assertRaisesRegex(ValueError, 'ORIGINAL_MAP_VALUE_MISMATCH'):
                a.analyze_artificial(*data)

    def test_truncated_old_map_rejected(self):
        data = copy.deepcopy(self.data); data[4]['H8_EXPLORATION92']['rows'].pop()
        with self.assertRaisesRegex(ValueError, 'FULL_ORIGINAL_MAP_COVERAGE'):
            a.analyze_artificial(*data)

    def test_source_selection_seed_and_duplicate_guards(self):
        for kind in ('input', 'source', 'selection', 'seed', 'duplicate'):
            data = copy.deepcopy(self.data)
            if kind == 'input': data[3]['input'] = '0'*64
            elif kind == 'source': data[1][0]['source'] = '0'*40
            elif kind == 'selection': data[2]['selected_ids'][0] = 'wrong'
            elif kind == 'seed': data[1][0]['seed'] += 1
            else: data[1].append(copy.deepcopy(data[1][0]))
            with self.assertRaises(ValueError): a.validate_inputs(*data[:4])

    def test_incomplete_paired_axes_rejected(self):
        data = copy.deepcopy(self.data); data[1].pop(0)
        with self.assertRaisesRegex(ValueError, 'INCOMPLETE_PAIRED_AXES'):
            a.analyze_artificial(*data)

    def test_missing_ineligible_numeric_unresolved_unobserved_separate(self):
        s = copy.deepcopy(self.data[0][0])
        cfg = dict(primary_phase='exploration', compiler_hashes=[self.data[3]['compiler']],
            point_rule='sum_axis_products')
        def status(): return a.d0.analyze_scope([s], [], cfg, [.0005], [1.])[0]['status']
        self.assertEqual(status(), 'MISSING')
        s['error_decomposition']['total_signed']['real'] = 1
        self.assertEqual(status(), 'INELIGIBLE')
        s['error_decomposition']['total_signed']['real'] = 0; s['log_B'] = 20
        self.assertEqual(status(), 'NUMERICALLY_UNRESOLVED')
        layer = a.unobserved_layer(4, 0, saved_probability=.001)
        self.assertEqual(layer['status'], 'UNOBSERVED')
        self.assertIsNone(layer['population_layer_cost'])

    def test_new_selection_outside_old_union_is_labelled(self):
        rows = self.result['confirmation_coverage'].values()
        self.assertTrue(any(r['status'] == 'OUTSIDE_ORIGINAL_UNION6' for r in rows))
        self.assertFalse(self.result['population_winner_certified'])

    def test_fixed_pair_and_envelope_are_separate(self):
        curve = next(iter(self.result['scopes']['H8_EXPLORATION92']['P'].values()))
        self.assertEqual(curve['lower_envelope']['domain'], 'P>=0')
        self.assertIn('fixed_RZ_selected_pairs', curve)
        self.assertTrue(curve['lower_envelope']['hypothetical_per_shot_only'])

    def test_real_analysis_entry_closed(self):
        data = copy.deepcopy(self.data); data[3].pop('fixture_kind')
        with self.assertRaisesRegex(ValueError, 'REAL_REACCOUNT_NOT_AUTHORIZED'):
            a.analyze_artificial(*data)


if __name__ == '__main__':
    unittest.main()
