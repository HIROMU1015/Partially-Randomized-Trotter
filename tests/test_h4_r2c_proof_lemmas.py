"""Tiny synthetic algebra for R2-C. No H4 operands, compiler or saved states."""
from fractions import Fraction as Q
import math
from pathlib import Path
import runpy
import unittest

ROOT = Path(__file__).resolve().parents[1]
P = runpy.run_path(str(ROOT / 'scripts/inspect_h4_arb_pr0.py'))['P']
C = P.B._backend()


def assert_zero(test, matrix):
    for row in range(matrix.nrows()):
        for col in range(matrix.ncols()):
            test.assertTrue(matrix[row, col].contains(0))
            test.assertLess(abs(matrix[row, col]).upper(), C.real('1e-65'))


class R2CProofLemmas(unittest.TestCase):
    def test_native_h_phase_and_controlled_rotation_on_two_qubits(self):
        with C.ctx.workprec(256):
            I = C.identity(4)
            X, Z = C.pauli('IX'), C.pauli('IZ')
            imag = C.acb(0, 1)
            pi = C.arb.pi()
            rz = I * (pi / 4).cos() - Z * (imag * (pi / 4).sin())
            half = C.real(Q(1, 2))
            sx = I * C.acb(half, half) + X * C.acb(half, -half)
            H = (X + Z) * (1 / C.real(2).sqrt())
            phase = C.acb(0, pi / 4).exp()
            assert_zero(self, phase * rz * sx * rz - H)
            self.assertFalse((rz * sx * rz - H)[0, 0].contains(0))
            # ancilla wire 1 controls target wire 0; retain its relative phase.
            cx = C.acb_mat([[1, 0, 0, 0], [0, 1, 0, 0],
                            [0, 0, 0, 1], [0, 0, 1, 0]])
            theta = C.real(Q(3, 8))
            half_rz = I * (theta / 4).cos() - Z * (imag * (theta / 4).sin())
            inv_half_rz = I * (theta / 4).cos() + Z * (imag * (theta / 4).sin())
            expected = C.acb_mat([[1, 0, 0, 0], [0, 1, 0, 0],
                                 [0, 0, C.acb(0, -theta / 2).exp(), 0],
                                 [0, 0, 0, C.acb(0, theta / 2).exp()]])
            assert_zero(self, cx * inv_half_rz * cx * half_rz - expected)

    def test_winding_identity_including_negative_boundary(self):
        # A binary64 fmod-plus-adjust MODEL, not a certification of Rust/libm.
        # Every rounding residual is retained exactly as a dyadic rational.
        modulus = 2 * math.pi
        actual = 0.0
        ideal = Q(0)
        accumulated = Q(0)
        period = Q(44, 7)  # Deliberately distinct exact TOY period.
        for value in (math.pi, -8.0, 0.125, -2.0 ** -1074, 20.0):
            y = actual + value
            n = int(Q(y) / Q(modulus))  # fmod truncation quotient, exact here.
            f = math.fmod(y, modulus)
            if f < 0:
                n -= 1
                out = f + modulus
            else:
                out = f
            addition_error = Q(y) - Q(actual) - Q(value)
            remainder_error = Q(out) - (Q(y) - n * Q(modulus))
            accumulated += addition_error + remainder_error - n * (Q(modulus) - period)
            ideal += Q(value) - n * period
            self.assertEqual(Q(out) - ideal, accumulated)
            actual = out
        tiny = -2.0 ** -1074
        self.assertEqual(math.fmod(tiny, modulus) + modulus, modulus)
        # Thus a proof must allow the closed endpoint, not assume [0,p).

    def test_exact_euler_backward_lemma_near_both_singularities(self):
        # Evaluate the MATHEMATICAL Euler formula with Arb, never Qiskit
        # synthesis. Tensor with an idle qubit (same operator norm).
        with C.ctx.workprec(256):
            h = 1 / C.real(2).sqrt()
            units = [C.acb_mat([[1, 0], [0, 1]]),
                     C.acb_mat([[0, 1], [1, 0]]),
                     C.acb_mat([[h, h], [h, -h]]),
                     C.acb_mat([[C.acb(-1, 0), 0], [0, 1]])]
            t = C.real(Q(1, 2 ** 20))
            E = C.acb_mat([[t, C.acb(0, t)], [-t, t]])
            eps = 4 * t  # sum of entry magnitudes bounds ||E||_2.
            for U in units:
                M = U + E
                delta = (M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]).arg()
                a, b = abs(M[0, 0]), abs(M[1, 0])
                theta = 2 * C.arb.atan2(b, a)
                chi1, chi2 = M[1, 1].arg(), M[1, 0].arg()
                phi, lam = chi1 + chi2 - delta, chi1 - chi2
                def rz(angle):
                    return C.acb_mat([[C.acb(0, -angle / 2).exp(), 0],
                                      [0, C.acb(0, angle / 2).exp()]])
                ry = C.acb_mat([[(theta / 2).cos(), -(theta / 2).sin()],
                                [(theta / 2).sin(), (theta / 2).cos()]])
                V = rz(phi) * ry * rz(lam) * C.acb(0, delta / 2).exp()
                # Frobenius norm on the lifted two-qubit operator, divided by
                # sqrt(2), equals the one-qubit Frobenius norm and bounds its
                # operator norm. No angle condition number is used.
                error2 = sum((abs(V[i, j] - U[i, j]) ** 2
                              for i in range(2) for j in range(2)), C.real(0))
                self.assertLess(error2.sqrt().upper(), (24 * eps + 6 * eps ** 2).lower())


if __name__ == '__main__':
    unittest.main(verbosity=2)
