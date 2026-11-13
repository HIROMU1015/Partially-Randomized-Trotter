"""Pauli-tail price connection: toy circuits + A's real dictionary/packet metadata.

Toy part: the native tests' 2-mode DF head blocks and independent Fock oracle,
a toy Pauli dictionary (anticommuting labels), existing sampler/D2/native code.
Real part: A's pinned LD1 dictionary and all-round packet are only read and
cross-checked; no H4 circuit is built, nothing is sampled for H4 or compiled.
"""
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks import pauli_tail_price_connection as P  # noqa: E402
from trottertracks.resource_applicability import ax2b_native_df_v5 as native  # noqa: E402

import numpy as np  # noqa: E402
from qiskit.quantum_info import Operator, Pauli, Statevector  # noqa: E402

spec = importlib.util.spec_from_file_location('toy_native_fixtures', ROOT / 'tests/tracks/resource_applicability/test_ax2b_native_df_v5.py')
F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)

PACKETS = dict(commit='714247619afb4714fd663eb15c0470ef32bdc7af',
               path='docs/research/df_head_pauli_tail_all_round_results_20261114/MEAN_HANDOFF.json')


def toy_dictionary():
    terms = []
    for label, c in (('YX', -.6), ('ZX', .2), ('YI', .25), ('XZ', -.15)):
        terms.append(dict(label_qiskit=label, coefficient=c, coefficient_hex=float(c).hex(), sign=1 if c > 0 else -1))
    lam = math.fsum(abs(t['coefficient']) for t in terms)
    for t in terms:
        t['probability'] = abs(t['coefficient']) / lam
    return dict(prefix=2, lambda_P=lam, saved_DF_scalar=.11, terms=terms)


def event_matrix(event, labels, n):
    result = np.eye(1 << n, dtype=complex)
    for app in event.application_sequence:
        p = Pauli(labels[app.component_id]).to_matrix()
        if app.role == 'product':
            op = app.coefficient_sign * p
        else:
            a = app.coefficient_sign * event.rotation_angle
            op = math.cos(a) * np.eye(1 << n) - 1j * math.sin(a) * p
        result = op @ result
    return event.phase * result


def toy_request(T=.4, q=2, R=3, K=2):
    d = toy_dictionary()
    return d, dict(m=0, T=T, q=q, R=R, K=K, prefix=2, candidate_id='RMSE:TOY', lambda_P=d['lambda_P'], step_time=T / q)


class Tests(unittest.TestCase):
    def build(self, policy='symmetric_directional', K=2):
        d, req = toy_request(K=K)
        comps, labels, n = P.components(d)
        steps, dist = P.sample_steps(comps, req, 'TOY_COHORT', 0)
        blocks = F.blocks()[:2]  # one-body + one squared DF fragment as the head
        scalar = P.total_scalar(d, .37)
        evo = P.compose(blocks, n, steps, labels, T=req['T'], scalar=scalar, control_policy=policy)
        expected = np.eye(4, dtype=complex)
        for step in steps:
            for b in blocks:
                expected = F.independent_term(b, step['signed_time'] / 2) @ expected
            for e in step['events']:
                expected = event_matrix(e, labels, n) @ expected
            for b in reversed(blocks):
                expected = F.independent_term(b, step['signed_time'] / 2) @ expected
        expected *= np.exp(-1j * scalar * req['T'])
        return d, req, steps, evo, expected, scalar, n

    def test_full_controlled_matrix_low_identity_and_order_phase(self):
        for policy in ('symmetric_directional', 'ordinary'):
            for K in (0, 2):
                d, req, steps, evo, expected, scalar, n = self.build(policy, K)
                self.assertEqual(sum(len(s['events']) for s in steps), req['q'] * req['R'])
                np.testing.assert_allclose(Operator(evo.circuit).data, F.ordinary(expected), atol=5e-13, rtol=0)

    def test_scalar_once_and_double_scalar_detected(self):
        d, req, steps, evo, expected, scalar, n = self.build()
        self.assertEqual(scalar, .37 + .11)
        last = evo.circuit.data[-1]
        self.assertEqual((last.operation.name, [evo.circuit.find_bit(q).index for q in last.qubits]), ('p', [n]))
        self.assertEqual(float(last.operation.params[0]), -scalar * req['T'])
        np.testing.assert_allclose(Operator(evo.circuit).data, F.ordinary(expected), atol=5e-13, rtol=0)
        # The per-step native calls carry scalar=0: no other ancilla phase has the total-T angle.
        same = [i for i in evo.circuit.data if i.operation.name == 'p' and float(i.operation.params[0]) == -scalar * req['T']]
        self.assertEqual(len(same), 1)
        wrong = evo.circuit.copy(); wrong.p(-scalar * req['T'], n)
        self.assertGreater(np.linalg.norm(Operator(wrong).data - F.ordinary(expected)), .01)

    def test_xy_measured_wrapper_sign(self):
        d, req, steps, evo, expected, scalar, n = self.build()
        for axis in ('cosine', 'sine'):
            w = native.build_native_hadamard_wrapper(evo, axis=axis, include_measurement=True,
                                                     max_instructions=len(evo.circuit.data) + 4)
            self.assertEqual(w.data[-1].operation.name, 'measure')
            p = Statevector.from_instruction(w.remove_final_measurements(inplace=False)).probabilities()
            target = expected[0, 0].real if axis == 'cosine' else expected[0, 0].imag
            self.assertAlmostEqual(float(sum(p[:4]) - sum(p[4:])), target, delta=4e-13)

    def test_seeds_are_reproducible_and_distinct_per_outer(self):
        d, req = toy_request()
        comps, labels, n = P.components(d)
        a, _ = P.sample_steps(comps, req, 'TOY_COHORT', 0); b, _ = P.sample_steps(comps, req, 'TOY_COHORT', 0)
        self.assertEqual([s['seed'] for s in a], [s['seed'] for s in b])
        self.assertEqual(len({s['seed'] for s in a}), req['q'])
        self.assertEqual([[e.selected_component_ids for e in s['events']] for s in a],
                         [[e.selected_component_ids for e in s['events']] for s in b])

    def test_invalid_dictionary_rejected(self):
        for mutate in (lambda d: d['terms'][0].update(label_qiskit='II'), lambda d: d['terms'][1].update(sign=-1),
                       lambda d: d.update(lambda_P=d['lambda_P'] * (1 + 1e-12)),
                       lambda d: d['terms'][2].update(coefficient_hex='0x1p-1')):
            d = toy_dictionary(); mutate(d)
            with self.assertRaises(ValueError):
                P.components(d)

    def test_real_LD1_dictionary_and_m0_packet_bind_without_circuits(self):
        handoff = json.loads(P._blob(ROOT, PACKETS['commit'], PACKETS['path']))
        packet = next(p for p in handoff['packets'] if p['prefix'] == 1)
        d = P.load_dictionary(ROOT, packet['dictionary_identity'])
        comps, labels, n = P.components(d)
        self.assertEqual((len(comps), n, d['prefix']), (184, 8, 1))
        self.assertTrue(all(c.df_fragment_id is None and c.basis_change_operations == () for c in comps))
        req = P.round_request(packet, 0, d)
        self.assertEqual((req['q'], req['R'], req['K'], req['prefix']), (1, 3, 2, 1))
        self.assertEqual(req['candidate_id'], packet['rounds'][0]['candidate_id'])
        self.assertEqual(P.total_scalar(d, 2.29310124732), 2.29310124732 + d['saved_DF_scalar'])
        bad = deepcopy(packet['dictionary_identity']); bad['dictionary_sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            P.load_dictionary(ROOT, bad)
        df = deepcopy(packet['dictionary_identity']); df['kind'] = 'DF_TAIL'
        with self.assertRaises(ValueError):
            P.load_dictionary(ROOT, df)
        self.assertEqual(handoff['C'], None)

    def test_ld1_m0_pilot_metadata(self):
        plan = P.pilot_plan(ROOT)
        self.assertEqual((plan['request']['q'], plan['request']['R'], plan['n'], plan['terms']), (1, 3, 8, 184))
        self.assertEqual(len(plan['seeds']), 1)
        self.assertEqual(plan['configuration_id'], 'd905d2197d4b01df3074b3844c0f4a2b990c21b9c3e1a0e3b6ea13a488b1fd27')
        self.assertIsNone(plan['execution_caller'])

    def test_c_guard_does_not_yet_know_the_pauli_route(self):
        from trottertracks import price_implementation_guard as G
        with self.assertRaises(ValueError):
            G._rte_source(str(ROOT), P.LOWERING_ID, 'a' * 40, P.MODULE)


if __name__ == '__main__':
    unittest.main(verbosity=2)
