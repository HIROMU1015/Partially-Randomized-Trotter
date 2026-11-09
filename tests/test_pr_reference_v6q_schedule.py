"""Schedule metadata and artificial/mock caller tests; no science circuits."""
import copy
from fractions import Fraction
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks import pr_reference_v6q_schedule as caller
from trottertracks import pr_reference_v6q as provider

# Reuse the existing Q1 artificial instruction/lane constructors and exact
# integer-grid oracle. Importing that test file does not run its old suite.
spec = importlib.util.spec_from_file_location('q1_artificial_helpers', ROOT/'tests/test_pr_reference_v6q.py')
q1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q1)


class EventSequence:
    """Represent an artificial length without allocating/iterating a trajectory."""
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        raise AssertionError('The adapter must not enumerate or resample events')


class ScheduleCallerTests(unittest.TestCase):
    def fixture(self, m=17):
        meta = caller.saved_schedule()[m]
        cfg = NS(rte_steps=meta.r, evolution_time=.02, finite_taylor_order=2,
                 dimensionless_step_time=.125)
        dist = NS(finite_taylor_order=2, orders=(0, 2), dimensionless_step_time=.125)
        f = q1.lane(.5, q1.instruction('rz', (.25,), (3,)))
        g = q1.lane(.25, q1.instruction('xx_plus_yy', (.5, .125), (2, 1)))
        f.num_qubits = g.num_qubits = 8
        assembly = dict(F=f, G=g, phase=-7.25, baseline=object())
        transfer = NS(assemble=Mock(return_value=assembly))

        def gadget(f, g, phase):
            # Artificial structure witness only; the real gadget is read-only.
            return NS(ops=(('CSWAP_FIRST',), ('F_SOURCE_INVERSE', id(f)),
                           ('G', id(g)), ('P', 16, phase), ('CSWAP_LAST',)), width=17)

        pilot = NS(reference_core=Mock(side_effect=gadget),
                   wrapper=Mock(side_effect=lambda core, axis:(axis, core)))
        return dict(meta=meta, cfg=cfg, dist=dist, events=EventSequence(meta.q*meta.r),
                    assembly=assembly, pilot=pilot, transfer=transfer)

    def invoke(self, fixture, mode='legacy', axes=False, **overrides):
        meta = fixture['meta']
        kwargs = dict(q=meta.q, r=meta.r, bits=meta.bits, cfg=fixture['cfg'],
                      dist=fixture['dist'], events=fixture['events'], mode=mode)
        kwargs.update(overrides)
        method = caller.reference_axes_for_saved_round if axes else caller.assemble_reference_for_saved_round
        with patch.object(caller, '_legacy_modules', return_value=(fixture['pilot'], fixture['transfer'])):
            return method(meta.m, **kwargs)

    def test_saved_schedule_metadata(self):
        schedule = caller.saved_schedule()
        self.assertEqual(len(schedule), 18)
        self.assertEqual([s.m for s in schedule], list(range(18)))
        self.assertEqual(schedule[17], caller.SavedRound(17, 131072, 32, 88))
        for row in schedule:
            self.assertEqual(row.q, 1 << row.m)
            self.assertEqual(caller.bind_round(row.m, q=row.q, r=row.r, bits=row.bits)[0], row)

    def test_opt_in_maps_only_m17(self):
        for row in caller.saved_schedule():
            _, selected = caller.bind_round(row.m, q=row.q, r=row.r, bits=row.bits,
                                            mode='exact_binary_m17')
            self.assertEqual(selected, 'exact_binary_m17' if row.m == 17 else 'legacy')

    def test_all_rounds_default_and_other17_opt_in_reuse_legacy_once(self):
        for meta in caller.saved_schedule():
            modes = ('legacy',) if meta.m == 17 else ('legacy', 'exact_binary_m17')
            for mode in modes:
                with self.subTest(m=meta.m, mode=mode):
                    fixture = self.fixture(meta.m)
                    with patch.object(caller, 'phase_for_round', side_effect=AssertionError('Duplicate phase evaluation')):
                        result = self.invoke(fixture, mode)
                    self.assertEqual(result['phase'], fixture['assembly']['phase'])
                    self.assertEqual(result['phase_mode'], 'legacy')
                    fixture['transfer'].assemble.assert_called_once_with(meta.q, meta.r,
                        fixture['cfg'], fixture['dist'], fixture['events'])
                    fixture['pilot'].reference_core.assert_called_once_with(
                        fixture['assembly']['F'], fixture['assembly']['G'], fixture['assembly']['phase'])

    def test_m17_phase_same_objects_and_one_provider_call(self):
        fixture = self.fixture()
        f, g = fixture['assembly']['F'], fixture['assembly']['G']
        before = copy.deepcopy((f, g))
        operation_ids = [id(i.operation) for lane in (f, g) for i in lane.data]
        with patch.object(caller, 'phase_for_round', wraps=provider.phase_for_round) as selection, \
             patch.object(provider, 'exact_binary_phase', wraps=provider.exact_binary_phase) as phase:
            result = self.invoke(fixture, 'exact_binary_m17')
        selection.assert_called_once_with(f, g, round_m=17, mode='exact_binary_m17')
        phase.assert_called_once_with(f, g)
        self.assertEqual(Fraction.from_float(result['phase']), q1.independent_exact(f, g))
        self.assertIs(result['assembly'], fixture['assembly'])
        self.assertEqual((f, g), before)
        self.assertEqual([id(i.operation) for lane in (f, g) for i in lane.data], operation_ids)
        fixture['pilot'].reference_core.assert_called_once_with(f, g, result['phase'])

    def test_only_relative_P_differs_in_synthetic_gadget(self):
        fixture = self.fixture()
        old = self.invoke(fixture)
        new = self.invoke(fixture, 'exact_binary_m17')
        for index, (left, right) in enumerate(zip(old['core'].ops, new['core'].ops, strict=True)):
            if index == 3:
                self.assertEqual(left[:2], right[:2])
                self.assertNotEqual(left[2], right[2])
            else:
                self.assertEqual(left, right)

    def test_X_Y_share_one_core_and_selected_phase(self):
        fixture = self.fixture()
        with patch.object(caller, 'phase_for_round', wraps=provider.phase_for_round) as selection:
            result = self.invoke(fixture, 'exact_binary_m17', axes=True)
        self.assertEqual(selection.call_count, 1)
        self.assertEqual(fixture['transfer'].assemble.call_count, 1)
        self.assertEqual(fixture['pilot'].reference_core.call_count, 1)
        self.assertEqual(fixture['pilot'].wrapper.call_args_list[0].args, (result['core'], 'X'))
        self.assertEqual(fixture['pilot'].wrapper.call_args_list[1].args, (result['core'], 'Y'))
        self.assertIs(result['axes']['X'][1], result['axes']['Y'][1])

    def test_caller_metadata_mismatches_rejected_before_build(self):
        for m in (0, 16, 17):
            fixture = self.fixture(m)
            for field in ('q', 'r', 'bits'):
                with self.subTest(m=m, field=field), self.assertRaises(ValueError):
                    self.invoke(fixture, 'exact_binary_m17', **{field:getattr(fixture['meta'], field)+1})
            fixture['transfer'].assemble.assert_not_called()

    def test_unknown_round_mode_and_types_rejected(self):
        for m in (-1, 18, None, '17', True):
            with self.subTest(m=m), self.assertRaises(ValueError):
                caller.bind_round(m, q=131072, r=32, bits=88)
        for mode in ('B1', '', None, True):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                caller.bind_round(17, q=131072, r=32, bits=88, mode=mode)
        for values in ((131072.,32,88), (131072,True,88), (131072,32,'88')):
            with self.subTest(values=values), self.assertRaises(ValueError):
                caller.bind_round(17, q=values[0], r=values[1], bits=values[2])

    def test_missing_duplicate_inconsistent_schedule_rejected(self):
        # Synthetic records are derived from metadata, with no hand-entered r/bits table.
        original = [vars(row).copy() for row in caller.saved_schedule()]
        defects = [original[:-1], original+[original[-1]], original[:3]+original[2:17]]
        for field, value in [('m',16), ('q',1), ('r',16), ('bits',87)]:
            rows = copy.deepcopy(original)
            rows[17][field] = value
            defects.append(rows)
        rows = copy.deepcopy(original); del rows[5]['bits']; defects.append(rows)
        for rows in defects:
            with self.subTest(rows=len(rows)), self.assertRaises(ValueError):
                caller._parse_schedule(dict(rows=rows))

    def test_RTE_metadata_or_length_mismatch_rejected_before_build(self):
        for field, value in [('rte_steps',1), ('evolution_time',.01), ('finite_taylor_order',4),
                             ('dimensionless_step_time',.25)]:
            fixture = self.fixture()
            setattr(fixture['cfg'], field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.invoke(fixture)
            fixture['transfer'].assemble.assert_not_called()
        fixture = self.fixture()
        fixture['events'] = EventSequence(1)
        with self.assertRaises(ValueError):
            self.invoke(fixture)
        fixture['transfer'].assemble.assert_not_called()

    def test_wrong_lane_width_rejected_before_core(self):
        fixture = self.fixture()
        fixture['assembly']['F'].num_qubits = 9
        with self.assertRaises(ValueError):
            self.invoke(fixture)
        fixture['pilot'].reference_core.assert_not_called()

    def test_fixed_scientific_source_unchanged(self):
        base = caller.SCHEDULE_COMMIT
        paths = ['src/trottertracks/pr_reference_transfer_qualification.py',
            'src/trottertracks/pr_reference_interference_pilot.py', 'src/trottertracks/track_c_compiler_a.py',
            'src/trottertracks/track_c_h4_finite_bits.py', 'src/trotterlib/df_partial_s2.py',
            'src/trotterlib/df_partial_s2_repeated.py', 'src/trotterlib/df_rte_qiskit.py']
        for path in paths:
            self.assertEqual((ROOT/path).read_bytes(), subprocess.check_output(['git','show',base+':'+path],cwd=ROOT))
        path = 'src/trottertracks/pr_reference_v6q.py'
        self.assertEqual((ROOT/path).read_bytes(), subprocess.check_output(['git','show',
            'eaf70d886d42364a01b7e949ebc21f3b026bbe72:'+path],cwd=ROOT))

    def test_new_module_imports_no_science_runtime(self):
        # This test file has imported only stdlib, the pure provider and Q1 mocks.
        self.assertFalse(any(n == 'qiskit' or n.startswith(('qiskit.', 'numpy', 'trotterlib')) for n in sys.modules))


if __name__ == '__main__':
    unittest.main()
