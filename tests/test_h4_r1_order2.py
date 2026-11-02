"""Synthetic two-qubit order2 tests; no real H4 state action or compiler."""
import copy
from fractions import Fraction as F
import importlib.util
import io
import json
import math
from pathlib import Path
import runpy
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
runner = runpy.run_path(str(ROOT / 'scripts/run_h4_r1_order2.py'))
A = runner['A']
R = A.R
import numpy as np
from qiskit import QuantumCircuit, qpy
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Operator, SparsePauliOp
from r1_private.h4_d2_pauli_control import central_control_body

spec = importlib.util.spec_from_file_location('r0_artificial_fixture', ROOT / 'tests/test_h4_r0_empirical.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)


def fixture():
    model, join = old.selected_fixture()
    case = model['cases']['PR']
    case['components'] = [dict(label='XI', sign=1, coefficient_abs=.2),
                          dict(label='YZ', sign=-1, coefficient_abs=.3),
                          dict(label='ZZ', sign=-1, coefficient_abs=.4)]
    for i, event in enumerate(join['events']):
        order = 2 if i == 30 else 0
        ids = ['0', '1', '2'] if order else [str(i % 2)]
        event.update(event_position=i, taylor_order=order, product_component_ids=ids[:-1],
                     rotation_component_id=ids[-1], selected_component_ids=ids,
                     phase=dict(real=-1. if order else 1., imag=0.),
                     rotation_angle=case['law']['angles'][1 if order else 0])
        event['application_sequence'] = [dict(component_id=cid, application_index=k,
            role='rotation' if k == order else 'product',
            coefficient_sign=case['components'][int(cid)]['sign'],
            coefficient_abs=case['components'][int(cid)]['coefficient_abs']) for k, cid in enumerate(ids)]
    return model, join


def body(model, join):
    env = dict(np=np, math=math, PauliEvolutionGate=PauliEvolutionGate, SparsePauliOp=SparsePauliOp)
    R.extract('src/trotterlib/pauli_signal_adapter.py', 'append_pauli_evolution', env)
    append = R.extract('src/trotterlib/pauli_signal_adapter.py', 'append_pauli_event', env)
    case = model['cases']['PR']
    labels = {str(i): c['label'] for i, c in enumerate(case['components'])}
    out = QuantumCircuit(2)
    for step in range(8):
        for b in case['blocks']:
            env['append_pauli_evolution'](out, b['label'], b['coefficient'], .05)
        for record in join['events'][4 * step:4 * step + 4]:
            event = NS(application_sequence=[NS(**a) for a in record['application_sequence']],
                       phase=complex(record['phase']['real'], record['phase']['imag']),
                       rotation_angle=record['rotation_angle'])
            append(out, event, labels)
        for b in reversed(case['blocks']):
            env['append_pauli_evolution'](out, b['label'], b['coefficient'], .05)
        out.global_phase += .037
    return out


class Order2Tests(unittest.TestCase):
    def test_descriptor_sign_order_phase_guards_before_action(self):
        m, j = fixture()
        A.check_event_semantics(m, j)
        for kind in ('sign', 'phase', 'products', 'role', 'position', 'angle'):
            bad = copy.deepcopy(j)
            e = bad['events'][30]
            if kind == 'sign': e['application_sequence'][1]['coefficient_sign'] = 1
            if kind == 'phase': e['phase']['real'] = 1.
            if kind == 'products': e['product_component_ids'].reverse()
            if kind == 'role': e['application_sequence'][0]['role'] = 'rotation'
            if kind == 'position': e['event_position'] = 29
            if kind == 'angle': e['rotation_angle'] = .4
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                A.check_event_semantics(m, bad)

    def test_noncommuting_product_orientation_and_negative_sign(self):
        m, j = fixture()
        e = j['events'][30]
        a = R.api()
        ops = {str(i): c['sign'] * a.pauli(c['label']) for i, c in enumerate(m['cases']['PR']['components'])}
        event = NS(**{k: e[k] for k in ('rotation_component_id', 'product_component_ids', 'rotation_angle')}, phase=-1)
        v = math.cos(.07) * np.eye(4) - 1j * math.sin(.07) * ops['2']
        actual = a.event(event, ops)
        expected = -v @ ops['1'] @ ops['0']
        self.assertTrue(np.allclose(actual, expected, atol=1e-15, rtol=0))
        self.assertGreater(np.linalg.norm(actual - (-v @ ops['0'] @ ops['1'])), 1.)
        self.assertGreater(np.linalg.norm(actual - v @ ops['1'] @ ops['0']), 1.)

    def test_selected_full_schedule_against_original_append_body(self):
        m, j = fixture()
        raw = np.array([1, 2j, 3, .7j])
        nu = R.raw_norm([[z.real, z.imag] for z in raw])
        out, receipt = R.selected_C_action(m, j, raw, nu)
        original = Operator(body(m, j)).data @ raw
        self.assertTrue(np.allclose(out, original, atol=3e-12, rtol=0))
        self.assertAlmostEqual(receipt['signal'], (np.vdot(raw, original) / float(nu)).real, places=12)
        self.assertFalse(receipt['PR_law_mean_used'])
        self.assertIsNone(receipt['numerical_error_upper'])

    def test_order2_QPY_control_both_branches_and_raw_observable(self):
        m, j = fixture()
        source = body(m, j)
        control = central_control_body(source)
        wrapped = R.P.toy_wrapper(control.body, 'cosine')
        buf = io.BytesIO()
        qpy.dump([source, control.body, wrapped], buf)
        buf.seek(0)
        source, controlled, wrapped = qpy.load(buf)
        with patch('qiskit.quantum_info.Statevector.evolve', side_effect=AssertionError('STRUCTURAL_ONLY')):
            R.validate_controlled(source, controlled, wrapped, json.loads(json.dumps(control.sidecar)))
        U = Operator(source).data
        self.assertTrue(np.allclose(Operator(controlled).data, np.block([[np.eye(4), np.zeros((4, 4))], [np.zeros((4, 4)), U]]), atol=3e-12, rtol=0))
        raw = np.array([1, 2j, 3, .7j])
        nu = R.raw_norm([[z.real, z.imag] for z in raw])
        _, actual = R.propagate(wrapped, raw, nu)
        _, ideal = R.selected_C_action(m, j, raw, nu)
        self.assertAlmostEqual(actual['signal'], ideal['signal'], places=12)
        self.assertFalse(actual['state_renormalized'])
        self.assertFalse(actual['probability_API_used'])
        self.assertFalse(actual['output_norm_division'])

    def test_controller_reuses_R0_with_one_diagnostic_three_actions(self):
        env = runner['controller']()
        self.assertEqual(env['A'].KEYS, ('PR1_cosine',))
        self.assertEqual(env['SCHEMA']['logical_diagnostics'], 1)
        self.assertEqual(env['SCHEMA']['initial_maximum_state_actions'], 3)
        self.assertEqual(sum(env['A'].ACTION_LIMITS.values()), 3)
        self.assertEqual(len(env['CODE']), 21)
        self.assertEqual(env['json'].loads(env['json'].dumps(env['SCHEMA'])), env['SCHEMA'])
        self.assertIn('R1_ORDER2_BINARY64_EMPIRICAL_ACTIVE_V1', env['check'].__code__.co_consts)
        self.assertIn('f01c4018201de9335f20a8901ec5f40f9cf9e89b', env['prepare'].__code__.co_consts)
        self.assertTrue(all(v == 0 for v in env['FORBIDDEN'].values()))

    def test_three_nonrefundable_actions_reject_old_or_extra_scope(self):
        with tempfile.TemporaryDirectory() as td:
            b = A.C.CallBudget(Path(td), A.LIMITS)
            for k in A.ACTION_LIMITS:
                b.take(k)
                with self.assertRaises(ValueError): b.take(k)
            for k in ['PF_input', 'PR0_input', 'sine', 'compile', 'Arb']:
                with self.assertRaises(ValueError): b.take(k)

    def test_empirical_d_not_certificate(self):
        out = R.compare_signals(.1, .1, .1, 1.2873709223899619, F(1, 1000000))
        self.assertEqual(out['d'], 0.)
        self.assertFalse(out['LOCAL_BUDGET_MET'])
        self.assertIsNone(out['rho_upper'])
        self.assertIsNone(out['PR_law_rho'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
