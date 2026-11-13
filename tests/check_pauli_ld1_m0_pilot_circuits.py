"""Artificial run of the Pauli pilot core in the real child import order (provider root first).

Toy 4-qubit DF Hamiltonian -> original prepare(ham, 1) native head; toy Pauli
dictionary; toy "P0" saved-order operators built by an INDEPENDENT Jordan-Wigner
construction in the saved mode-0-MSB basis. compile_cached is a counter.
No H4 input, transpile or price.
"""
from copy import deepcopy
from fractions import Fraction
import math
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import h46_long_time_signal as signal  # noqa: E402
signal.install()
import run_pauli_ld1_m0_pilot as PILOT  # noqa: E402
PILOT.register_pauli_modules()
P = PILOT.connection()

import numpy as np  # noqa: E402
from qiskit.quantum_info import Pauli  # noqa: E402


def jw_saved(n):
    """a_p in the saved basis: mode 0 = most significant tensor factor, Z string on modes < p."""
    I, Z = np.eye(2), np.diag([1., -1.]); lower = np.array([[0., 1.], [0., 0.]])  # |0><1|: empties an occupied mode
    ops = []
    for p in range(n):
        factors = [Z] * p + [lower] + [I] * (n - p - 1)
        a = factors[0]
        for f in factors[1:]:
            a = np.kron(a, f)
        ops.append(a.astype(complex))
    return ops


def one_body_saved(matrix, a):
    return sum(matrix[p, q] * a[p].conj().T @ a[q] for p in range(len(a)) for q in range(len(a)))


def toy():
    from trotterlib.df_hamiltonian import DFHamiltonian
    from h46_new2_native import OccupationOracle, prepare
    rng = np.random.default_rng(17)
    x = rng.normal(size=(4, 4)); g = []
    for _ in range(3):
        y = rng.normal(size=(4, 4)); g.append(.3 * (y + y.T))
    ham = DFHamiltonian(.41, .3 * (x + x.T), np.array([.4, .3, .2]), tuple(g), {})
    full = np.zeros(16, complex)
    for i in range(16):
        if bin(i).count('1') == 2:
            full[i] = rng.normal() + 1j * rng.normal()
    full /= np.linalg.norm(full)
    oracle = OccupationOracle(ham, full, prefix=ham.n_blocks)
    qstate = np.zeros(16, complex); qstate[oracle.qindices] = oracle.initial
    ctx = SimpleNamespace(ham=ham, oracle=oracle, qstate=qstate, preparations={1: prepare(ham, 1)})
    a = jw_saved(4)
    H1 = one_body_saved(ham.one_body, a); A0 = one_body_saved(g[0], a); F0 = ham.lambdas[0] * A0 @ A0
    sR = ctx.preparations[1].extracted_identity_coefficient
    terms = []
    for label, c in (('YXIZ', -.31), ('ZZII', .22), ('IXYI', .17), ('XIIX', -.12), ('IIZI', .09)):
        terms.append(dict(label_qiskit=label, label_in_saved_tensor_order=label[::-1], coefficient=c,
                          coefficient_hex=float(c).hex(), sign=1 if c > 0 else -1))
    lam = math.fsum(abs(t['coefficient']) for t in terms)
    for t in terms:
        t['probability'] = abs(t['coefficient']) / lam
    d = dict(prefix=1, lambda_P=lam, saved_DF_scalar=sR, pauli_identity=sR, terms=terms)
    A_s = P.pauli_sum(d, 'saved'); tail = A_s + sR * np.eye(16)
    operators = dict(head=H1 + F0, tail=tail, Pauli_nonidentity=A_s)
    blocks = np.array([H1, F0, tail])
    T = Fraction(1, 10)
    request = dict(m=0, T=T, q=1, R=3, K=2, prefix=1, candidate_id='RMSE:TOY_PAULI', lambda_P=lam, step_time=float(T))
    return ctx, d, operators, blocks, request


class Tests(unittest.TestCase):
    def setUp(self):
        import h46_long_time_native as native
        self.native = native
        self.tmp = Path(tempfile.mkdtemp(prefix='pauli_artificial_')); self.addCleanup(shutil.rmtree, self.tmp, True)
        native.PUBLIC, native.PRIVATE = self.tmp / 'public', self.tmp / 'private'
        self.ctx, self.d, self.ops, self.blocks, self.req = toy()
        self.calls = []

        def fake(wrapper, fingerprint, representative):
            self.calls.append(fingerprint)
            return dict(metrics={k: 1 for k in native.METRICS}), None, False
        self.fake = fake

    def run_core(self, **kw):
        return PILOT.pilot_core(self.native, self.ctx, kw.get('d', self.d), self.req, kw.get('ops', self.ops),
                                kw.get('blocks', self.blocks), cohort='TOY', compile_cached=self.fake)

    def test_pinned_modules_and_bit_reversal(self):
        self.assertTrue(PILOT.provider_checks())
        r = P.reverse_bits(3)
        self.assertEqual(list(r), [0, 4, 2, 6, 1, 5, 3, 7])
        op = np.arange(64.).reshape(8, 8)
        np.testing.assert_array_equal(P.qiskit_to_saved(P.saved_to_qiskit(op)), op)
        # A Qiskit-label Pauli converted from its saved-order (reversed-label) matrix:
        np.testing.assert_allclose(P.saved_to_qiskit(Pauli('XZI').to_matrix()), Pauli('IZX').to_matrix())

    def test_toy_pilot_low_high_xy_scalar_and_compile_path(self):
        out = self.run_core()
        self.assertEqual(out['status'], 'PASS')
        self.assertEqual({c['check'] for c in out['checks']}, {'low_control', 'high_control_leakage',
                         'independent_full_space_vector', 'Hadamard_cosine', 'Hadamard_sine'})
        self.assertEqual(len(self.calls), 2)
        self.assertTrue(out['bit_order']['pass'])
        self.assertGreater(out['bit_order']['unconverted_mismatch'], 1e-3)  # the conversion is not a no-op
        self.assertEqual(out['scalar'], .41 + self.d['saved_DF_scalar'])
        self.assertEqual(len(out['seeds']), 1)

    def test_dictionary_mismatch_rejected(self):
        bad = deepcopy(self.d); bad['terms'][0]['label_in_saved_tensor_order'] = bad['terms'][0]['label_qiskit']
        with self.assertRaises(ValueError):
            self.run_core(d=bad)
        wrong = deepcopy(self.ops); wrong['Pauli_nonidentity'] = P.pauli_sum(self.d, 'qiskit')
        with self.assertRaises(ValueError):
            self.run_core(ops=wrong)

    def test_missing_bit_conversion_or_wrong_head_fails_independent_oracle(self):
        original = P.saved_to_qiskit
        P.saved_to_qiskit = lambda op: np.asarray(op)
        try:
            with self.assertRaises(ValueError) as e:
                self.run_core()
        finally:
            P.saved_to_qiskit = original
        # The oracle itself is order-sensitive: feeding Qiskit-order blocks where saved order is expected
        # (i.e. converting twice) gives a different high-branch vector than the native circuit.
        out = self.run_core()
        comps, labels, n = P.components(self.d)
        steps, _ = P.sample_steps(comps, self.req, 'TOY', 0)
        kw = dict(scalar=out['scalar'], T=float(self.req['T']), state=self.ctx.qstate)
        right = P.full_space_expected(list(self.blocks[:2]), steps, labels, **kw)
        twice = P.full_space_expected([P.saved_to_qiskit(b) for b in self.blocks[:2]], steps, labels, **kw)
        self.assertGreater(np.linalg.norm(right - twice), 1e-3)
        blocks = self.blocks.copy(); blocks[1] = blocks[1] * 1.01
        with self.assertRaises(ValueError):
            self.run_core(blocks=blocks, ops=dict(self.ops, head=blocks[0] + blocks[1]))

    def test_scalar_once_and_sR_binding(self):
        bad = deepcopy(self.d); bad['saved_DF_scalar'] = bad['saved_DF_scalar'] + 1e-12
        with self.assertRaises(ValueError):
            self.run_core(d=bad)
        original = P.compose
        def doubled(*a, **k):
            evo = original(*a, **k)
            evo.circuit.p(-k['scalar'] * k['T'], evo.num_system_qubits)
            return evo
        P.compose = doubled
        try:
            with self.assertRaises(ValueError):
                self.run_core()
        finally:
            P.compose = original


if __name__ == '__main__':
    unittest.main(verbosity=2)
