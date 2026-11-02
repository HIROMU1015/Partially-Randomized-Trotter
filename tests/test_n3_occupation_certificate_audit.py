"""Small algebra and adversarial certificate fixtures, no transfer scan."""
import copy
from fractions import Fraction as F
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks.representation_exploration.n3_occupation_certificate_audit import (
    Audit, decode_case, exact_car_fixture, eye, rational, spectral_lower,
)


class AlgebraTests(unittest.TestCase):
    def test_car_ordering_and_external_density(self):
        for external in (0, 2, 3):
            for particles in (1, 2, 3):
                with self.subTest(external=external, particles=particles):
                    self.assertEqual(exact_car_fixture(external, particles), (True, True))

    def test_zero_pivot_valid_and_invalid(self):
        Audit().psd([[F(0), F(0)], [F(0), F(1)]])
        with self.assertRaisesRegex(ValueError, 'zero PSD pivot'):
            Audit().psd([[F(0), F(1)], [F(1), F(0)]])

    def test_singular_rank_one_and_negative(self):
        Audit().psd([[F(1), F(2)], [F(2), F(4)]])
        with self.assertRaisesRegex(ValueError, 'negative PSD pivot'):
            Audit().psd([[F(1), F(2)], [F(2), F(3)]])

    def test_nonsymmetric_and_ragged_rejected(self):
        for M in ([[F(1), F(0)], [F(1), F(1)]], [[F(1)], [F(1), F(1)]]):
            with self.assertRaises(ValueError):
                Audit().psd(M)

    def test_empty_and_full_particle_sums(self):
        G = [[F(-2), F(0)], [F(0), F(3)]]
        self.assertEqual(spectral_lower(G, 0, {'k': 0, 'lower': [0, 1]}, Audit()), 0)
        witness = {'k': 2, 'tau': [3, 1], 'Q': [[[5, 1], [0, 1]], [[0, 1], [0, 1]]], 'lower': [1, 1]}
        self.assertEqual(spectral_lower(G, 2, witness, Audit()), 1)
        self.assertGreater(F(1), 2 * F(-2))

    def test_particle_sum_bound_rejects_false_trace_and_psd(self):
        G = [[F(-2), F(0)], [F(0), F(3)]]
        w = {'k': 1, 'tau': [0, 1], 'Q': [[[2, 1], [0, 1]], [[0, 1], [0, 1]]], 'lower': [-2, 1]}
        self.assertEqual(spectral_lower(G, 1, w, Audit()), -2)
        false_trace = copy.deepcopy(w)
        false_trace['lower'] = [-1, 1]
        with self.assertRaisesRegex(ValueError, 'trace bound'):
            spectral_lower(G, 1, false_trace, Audit())
        false_psd = copy.deepcopy(w)
        false_psd['Q'][0][0] = [1, 1]
        with self.assertRaisesRegex(ValueError, 'negative PSD pivot'):
            spectral_lower(G, 1, false_psd, Audit())

    def test_invalid_particle_count(self):
        for k in (-1, 3, True):
            with self.assertRaisesRegex(ValueError, 'particle count'):
                spectral_lower(eye(2), k, {'k': k, 'lower': [0, 1]}, Audit())

    def test_rational_schema_and_bit_cap(self):
        for q in ([True, 1], [1, 0], [1, -1], [1.0, 1]):
            with self.assertRaisesRegex(ValueError, 'rational schema'):
                rational(q, Audit())
        with self.assertRaises(ArithmeticError):
            Audit(bit_cap=8).psd([[F(1, 2 ** 9)]])

    def test_class_schema_and_unordered_density(self):
        p = ROOT / 'artifacts/n3_occupation_redesign_preparation/2026-10-12/review_intake/posthoc_particle_certificate.json'
        original = json.loads(p.read_text())['reconstructed_input']
        decode_case(original, Audit())
        for mutate in ('oracle', 'negative', 'duplicate', 'partition'):
            bad = copy.deepcopy(original)
            if mutate == 'oracle':
                bad['true_gap'] = [1, 1]
            elif mutate == 'negative':
                bad['density'][0]['coefficient'] = [-1, 1]
            elif mutate == 'duplicate':
                row = copy.deepcopy(bad['density'][0])
                row['i'], row['j'] = row['j'], row['i']
                bad['density'].append(row)
            else:
                bad['active'] = bad['active'][::-1]
            with self.assertRaises(ValueError):
                decode_case(bad, Audit())


if __name__ == '__main__':
    unittest.main(verbosity=2)
