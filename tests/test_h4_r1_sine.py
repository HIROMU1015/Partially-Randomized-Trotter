"""Artificial sine/raw-ray/cache tests only; never read real H4 endpoints."""
import copy
from fractions import Fraction as F
import importlib.util
import io
import json
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
runner = runpy.run_path(str(ROOT / 'scripts/run_h4_r1_sine.py'))
A, R = runner['A'], runner['A'].R
import numpy as np
from qiskit import QuantumCircuit, qpy
from qiskit.quantum_info import Operator
from sine_private.h4_d2_pauli_control import central_control_body

spec = importlib.util.spec_from_file_location('sine_reused_R0_fixture', ROOT / 'tests/test_h4_r0_empirical.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)


def programs():
    source = old.toy_body()
    controlled = central_control_body(source)
    cosine, sine = [R.P.toy_wrapper(controlled.body, axis) for axis in ('cosine', 'sine')]
    return source, controlled.body, cosine, sine, controlled.sidecar


class SineTests(unittest.TestCase):
    def test_sine_positive_imag_convention_unclipped_nonunit_raw_ray(self):
        for n in (2, 3):
            source = old.toy_body(n)
            wrapper = R.P.toy_wrapper(central_control_body(source).body, 'sine')
            raw = np.arange(1, 1 + (1 << n)) + 1j * np.arange(1 << n)
            before = raw.copy()
            nu = R.raw_norm([[z.real, z.imag] for z in raw])
            _, receipt = R.propagate(wrapper, raw, nu)
            expected = (np.vdot(raw, Operator(source).data @ raw) / float(nu)).imag
            self.assertAlmostEqual(receipt['signal'], expected, places=12)
            self.assertTrue(np.array_equal(raw, before))
            self.assertGreater(receipt['p0_raw'] + receipt['p1_raw'], 1.)
            self.assertFalse(receipt['probability_API_used'])
            self.assertFalse(receipt['state_renormalized'])
            self.assertFalse(receipt['output_norm_division'])

    def test_QPY_sine_source_control_binding_without_evolution(self):
        source, control, cosine, sine, sidecar = programs()
        data = io.BytesIO()
        qpy.dump([source, control, cosine, sine], data)
        data.seek(0)
        source, control, cosine, sine = qpy.load(data)
        with patch('qiskit.quantum_info.Statevector.evolve', side_effect=AssertionError('STRUCTURAL_ONLY')):
            binding = A.validate_sine(source, control, cosine, sine, json.loads(json.dumps(sidecar)))
            R.validate_program(sine, compiled=False, n=2)
        self.assertEqual(binding['convention'], '+Im using Sdg before final H')
        self.assertEqual(binding['new_wrapper_build'], 0)

    def test_wrong_sine_phase_axis_or_wire_rejected_before_action(self):
        source, control, cosine, sine, sidecar = programs()
        for kind in ('S_sign', 'missing_sdg', 'wrong_wire', 'global_phase'):
            bad = sine.copy()
            if kind == 'S_sign': bad.data[-3] = bad.data[-3].replace(operation=bad.data[-3].operation.inverse())
            if kind == 'missing_sdg': bad.data.pop(-3)
            if kind == 'wrong_wire': bad.data[-3] = bad.data[-3].replace(qubits=(bad.qubits[0],))
            if kind == 'global_phase': bad.global_phase += .2
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                A.validate_sine(source, control, cosine, bad, sidecar)

    def test_PF_saved_imag_interval_exact_same_ray_conversion(self):
        saved = dict(implemented_law=dict(PF=dict(corrected_encoded_mean=dict(
            real=dict(lower='10', upper='20'), imag=dict(lower='1', upper='3')))), state_norm=dict(lower='2', upper='2'))
        value, record = A.saved_target('PF', saved, F(2), {})
        self.assertEqual(value, F(1))
        self.assertEqual(record['normalized_interval_exact'], ['1/2', '3/2'])
        self.assertIn('.imag', record['source'])
        self.assertFalse(record['compiled_signal_error_enclosed'])
        self.assertEqual(record['new_C_target_state_actions'], 0)

    def test_individual_PR_imag_cache_not_real_or_law_mean(self):
        cache = dict(PR_law_mean_used=False, normalization_B_applied_to_target=False,
            selected_trajectory_id='TOY_SELECTED', signal=.25, complex_signal=dict(real=.25, imag=-.375), numerical_error_upper=None)
        for key in ('PR0', 'PR1'):
            value, record = A.saved_target(key, {}, F(7), {key: cache})
            self.assertEqual(value, F(-3, 8))
            self.assertEqual(record['selected_trajectory_id'], 'TOY_SELECTED')
            self.assertEqual(record['new_C_target_state_actions'], 0)
            self.assertFalse(record['PR_law_mean_used'])

    def test_incompatible_cache_is_not_silently_substituted(self):
        cache = dict(PR_law_mean_used=False, normalization_B_applied_to_target=False,
            selected_trajectory_id='TOY', signal=.25, complex_signal=dict(real=.25, imag=.375), numerical_error_upper=None)
        for field, value in [('PR_law_mean_used', True), ('normalization_B_applied_to_target', True), ('numerical_error_upper', .1)]:
            bad = dict(cache, **{field: value})
            with self.assertRaises(ValueError): A.saved_target('PR0', {}, F(1), dict(PR0=bad))
        bad = copy.deepcopy(cache)
        bad['complex_signal']['real'] = .2
        with self.assertRaises(ValueError): A.saved_target('PR0', {}, F(1), dict(PR0=bad))

    def test_controller_three_sine_diagnostics_six_actions(self):
        env = runner['controller']()
        self.assertEqual(env['SCHEMA']['keys'], ['PF0_sine', 'PR0_sine', 'PR1_sine'])
        self.assertEqual(env['SCHEMA']['axis'], 'sine')
        self.assertEqual(env['SCHEMA']['logical_diagnostics'], 3)
        self.assertEqual(env['SCHEMA']['initial_maximum_state_actions'], 6)
        self.assertEqual(env['json'].loads(env['json'].dumps(env['SCHEMA'])), env['SCHEMA'])
        self.assertEqual(len(env['CODE']), 23)
        self.assertIn('sine', env['check'].__code__.co_consts)
        self.assertTrue(all(v == 0 for v in env['FORBIDDEN'].values()))

    def test_A_AXES_nested_and_A_GATE_flat_metadata(self):
        target = dict(SOURCE=A.P.AA, trajectory_id='TOY', trajectory_record_hash='TR', event_record_hash='EV')
        inp = dict(axis='sine', trajectory_id='TOY', case='PR0000')
        metrics = dict(axis='sine', case='PR0000', target=dict(trajectory_id='TOY', trajectory_record_hash='TR', event_record_hash='EV'))
        A.check_sine_metadata('PR0', target, inp, metrics)
        bad = copy.deepcopy(metrics)
        bad['target']['trajectory_id'] = 'WRONG'
        with self.assertRaises(ValueError): A.check_sine_metadata('PR0', target, inp, bad)
        A.check_sine_metadata('PR1', dict(target, SOURCE=A.P.AG), inp, dict(axis='sine', trajectory_id='TOY'))
        with self.assertRaises(ValueError): A.check_sine_metadata('PR1', dict(target, SOURCE=A.P.AG), inp, dict(axis='sine', trajectory_id='WRONG'))

    def test_six_nonrefundable_actions_no_new_ideal_or_cosine(self):
        self.assertEqual(sum(A.ACTION_LIMITS.values()), 6)
        with tempfile.TemporaryDirectory() as td:
            b = A.C.CallBudget(Path(td), A.LIMITS)
            for name in A.ACTION_LIMITS:
                b.take(name)
                with self.assertRaises(ValueError): b.take(name)
            for name in ['PR0_C_target', 'PR1_C_target', 'cosine', 'compile', 'Arb']:
                with self.assertRaises(ValueError): b.take(name)


if __name__ == '__main__':
    unittest.main(verbosity=2)
