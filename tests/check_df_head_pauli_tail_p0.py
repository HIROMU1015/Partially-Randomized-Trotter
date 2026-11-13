"""Nonmolecular P0 fixtures. Never load saved arrays or construct a circuit."""
from copy import deepcopy
import importlib
import math
from pathlib import Path
import sys
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
package = ModuleType('trotterlib'); package.__path__ = [str(ROOT/'src/trotterlib')]
sys.modules.setdefault('trotterlib', package)
from trottertracks import df_head_pauli_tail_p0 as p0
pauli = p0.pauli_api()


def fixture():
    # Three artificial fermion modes in the complete one-particle sector.
    # Mode 1 supplies a spectator exposing individual-label leakage.
    # Complex G and signed weights
    # exercise CAR/JW signs, endianness, identity and prefix differences.
    h = np.diag([.08, .12, -.14]).astype(complex)
    h[0, 2] = .02j; h[2, 0] = -.02j
    a = np.diag([.3, -.1, .4]).astype(complex)
    b = np.diag([.2, .1, -.3]).astype(complex)
    b[0, 2] = .13+.07j; b[2, 0] = .13-.07j
    c = np.diag([-.4, .11, .07]).astype(complex)
    c[0, 2] = -.21j; c[2, 0] = .21j
    ham = SimpleNamespace(n_qubits=3, n_blocks=3, constant=.17,
        one_body=h, lambdas=np.array([.9, -.2, .6]), g_matrices=(a, b, c))
    sector = SimpleNamespace(basis_indices=np.array([1, 2, 4]))
    state = np.array([1., .2, .3j]); state /= np.linalg.norm(state)
    full_state = np.zeros(8, complex); full_state[sector.basis_indices] = state
    # Source scalars in this synthetic fixture are independently fixed from
    # the known full trace formula Tr(A(g)^2)/2^n = ((Tr g)^2+Tr(g^2))/4.
    tails = {str(p): dict(scalar=math.fsum(float(w*((np.trace(g)**2+np.trace(g@g)).real)/4)
        for w, g in zip(ham.lambdas[p:], ham.g_matrices[p:], strict=True)),
        lambda_R=1., source='ARTIFICIAL_DESCRIPTOR') for p in (1, 2)}
    meta = dict(saved_tail_descriptors=tails, binding=dict(system='ARTIFICIAL_ONLY'))
    return ham, sector, full_state, state, meta


class P0Tests(unittest.TestCase):
    def setUp(self):
        for guard in (patch.object(np, 'load', side_effect=AssertionError('No saved arrays')),
            patch.object(pauli, 'QuantumCircuit', side_effect=AssertionError('No circuits'))):
            guard.start(); self.addCleanup(guard.stop)

    def test_two_prefixes_complex_signed_factors_and_common_scalar(self):
        args = fixture(); result = {}; products = {}
        p0.diagnose(ROOT, *args, result, lambda name, value: products.update({name:deepcopy(value)}))
        self.assertEqual([r['prefix'] for r in result['prefixes']], [1, 2])
        self.assertTrue(all(c['PASS'] for c in result['checks']))
        for p in (1, 2):
            record = result['prefixes'][p-1]
            self.assertIn('GATES_PASS', record['status'])
            self.assertGreater(record['individual_labels_leaving_sector'], 0)
            self.assertLess(record['identity_metrics']['Pauli_tail_full_fro'], 2e-14)
            self.assertIsNone(products['CHECKPOINT.json'].get('sufficient_N'))
        self.assertNotEqual(result['prefixes'][0]['dictionary_sha256'], result['prefixes'][1]['dictionary_sha256'])

    def test_basis_permutation_is_explicit_and_matches_CAR(self):
        ham, sector, full_state, state, meta = fixture()
        op = p0.square_factor_dictionary(ROOT, ham, 1)
        model = pauli.PauliHamiltonian.from_openfermion(op, num_qubits=3)
        little = model.sparse_operator().toarray()
        ctx = p0.car_api(ROOT)(ham, SimpleNamespace(basis_indices=np.arange(8)), full_state,
            meta['saved_tail_descriptors'])
        expected = sum(ctx.matrices[2:])
        self.assertGreater(np.linalg.norm(little-expected), .01)
        reverse = p0.reverse_bits(3)
        np.testing.assert_allclose(little[np.ix_(reverse, reverse)], expected, rtol=0, atol=2e-14)

    def test_label_sector_classification_exact_bit_images(self):
        self.assertEqual(p0.leakage('XIX', [1, 4])['escapes_from_sector'], 0)
        self.assertEqual(p0.leakage('IIX', [1, 4])['escaping_basis_transitions'], [(1, 5), (4, 0)])
        self.assertEqual(p0.leakage('IZZ', [1, 4])['saved_basis_flip_mask'], 0)

    def test_scalar_mismatch_stops_without_repair_or_second_prefix(self):
        args = fixture(); args[-1]['saved_tail_descriptors']['1']['scalar'] += .01
        result = {}; products = {}
        with self.assertRaisesRegex(ValueError, 'Pauli_tail_saved_scalar_fro'):
            p0.diagnose(ROOT, *args, result, lambda name, value: products.update({name:deepcopy(value)}))
        self.assertFalse(result['checks'][-1]['PASS'])
        self.assertEqual(len(result['prefixes']), 1)
        self.assertNotIn('LD2_DICTIONARY.json', products)

    def test_nonhermitian_or_state_lift_failure_is_not_projected_away(self):
        args = fixture(); args[0].one_body[0, 2] += .03
        with self.assertRaisesRegex(ValueError, 'block_0_hermiticity'):
            p0.diagnose(ROOT, *args, {}, lambda *values:None)
        args = fixture(); args[2][0] = .1
        with self.assertRaisesRegex(ValueError, 'saved_full_state_norm'):
            p0.diagnose(ROOT, *args, {}, lambda *values:None)


if __name__ == '__main__':
    unittest.main()
