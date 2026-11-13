"""Artificial circuits for P-1, in the same import order as the real child process.

The original install() puts the pinned provider root first (only its private
config path is read); then a toy 4-qubit DF Hamiltonian goes through the ORIGINAL
prepare_full_h/full_oracle (9b83 blob), the long-time native sample/compose/
cost_trial and the 01e068af swap. compile_cached is a counter: no H4 input,
transpile or price.
"""
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
import run_full_h_kappa2_p1 as P1  # noqa: E402  (no project package at import)

import numpy as np  # noqa: E402
from qiskit.quantum_info import Operator  # noqa: E402

TOY_ROW = dict(system='TOY', method='PR', formula='2nd', prefix=0, T=.8, q=2, R_outer=8, K=2, delta=.4, s=16,
               family='FULL_H_PURE_RTE', L_D=0, id='FULL_H_TOY_ARTIFICIAL_P1')


def toy_hamiltonian():
    from trotterlib.df_hamiltonian import DFHamiltonian
    rng = np.random.default_rng(11)
    a = rng.normal(size=(4, 4)); g = []
    for _ in range(2):
        b = rng.normal(size=(4, 4)); g.append(.25 * (b + b.T))
    return DFHamiltonian(.37, .3 * (a + a.T), np.array([.4, .3]), tuple(g), {})


def toy_state(n=4, particles=2):
    v = np.zeros(1 << n, complex); rng = np.random.default_rng(5)
    for i in range(1 << n):
        if bin(i).count('1') == particles:
            v[i] = rng.normal() + 1j * rng.normal()
    return v / np.linalg.norm(v)


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='p1_artificial_')); self.addCleanup(shutil.rmtree, self.tmp, True)
        self.full = P1.full_h_module(self.tmp / 'public', self.tmp / 'private')
        ham = toy_hamiltonian(); prep = self.full.prepare_full_h(ham)
        oracle = self.full.full_oracle(ham, toy_state())
        qstate = np.zeros(1 << ham.n_qubits, complex); qstate[oracle.qindices] = oracle.initial
        self.ctx = SimpleNamespace(system='TOY', ham=ham, preparations={0: prep}, oracle=oracle, qstate=qstate)
        self.full.context = lambda system: self.ctx
        self.compiles = []

        def fake_compile(wrapper, fingerprint, representative):
            self.compiles.append(fingerprint)
            return dict(metrics={k: 1 for k in ('rz_count', 'rz_depth', 'cx_count', 'cx_depth', 'total_depth', 'circuit_size')}), None, False
        self.fake_compile = fake_compile

    def evolutions(self):
        native = self.full.setup()
        prep = self.ctx.preparations[0]
        requests, _, _ = native.sample_requests(TOY_ROW, prep, P1.COHORT, 0)
        scalar = self.ctx.ham.constant + prep.extracted_identity_coefficient
        legacy = native.compose(TOY_ROW, prep, requests, scalar, lambda qc: 'x')
        provider = __import__('trotterlib.df_rte_qiskit', fromlist=['x'])
        with P1.zz_module().efficient_controlled_zz(provider):
            new = native.compose(TOY_ROW, prep, requests, scalar, lambda qc: 'x')
        return native, prep, requests, scalar, legacy, new

    def test_provider_root_modules_are_pinned(self):
        files = P1.provider_checks()
        self.assertTrue(all('h46-higher-pf-acquire' in v['file'] or 'provider' in v['file'] or 'src' in v['file']
                            for v in files.values()))

    def test_toy_pilot_end_to_end_without_compile(self):
        trial, checks, ok = P1.pilot(self.full, TOY_ROW, compile_cached=self.fake_compile)
        self.assertTrue(ok, checks)
        self.assertEqual(len(self.compiles), 2)
        self.assertEqual(set(checks['trial_gates']), {'low_control', 'high_control_leakage',
                         'independent_occupation_vector', 'Hadamard_cosine', 'Hadamard_sine'})
        self.assertGreater(checks['parity']['replaced_controlled_zz'], 0)
        self.assertEqual(checks['legacy_rzz_control_instructions'], 0)
        self.assertGreater(checks['one_body_applications'], 0)
        self.assertTrue(checks['scalar_once'])

    def test_controlled_action_low_identity_scalar_once(self):
        native, prep, requests, scalar, legacy, new = self.evolutions()
        U_old, U_new = Operator(legacy.circuit).data, Operator(new.circuit).data
        np.testing.assert_allclose(U_new, U_old, atol=1e-12, rtol=0)  # incl. global/control phase
        d = 1 << prep.num_system_qubits
        np.testing.assert_allclose(U_new[:d, :d], np.eye(d), atol=1e-12)  # ancilla 0 block
        self.assertLess(np.abs(U_new[:d, d:]).max(), 1e-12)
        np.testing.assert_allclose(U_new[d:, d:] @ self.ctx.qstate, self.ctx.oracle.trajectory(requests), atol=1e-10)
        n = prep.num_system_qubits
        self.assertTrue(P1.scalar_once(new.circuit, n, -scalar * TOY_ROW['T']))
        doubled = new.circuit.copy(); doubled.p(-scalar * TOY_ROW['T'], n)
        self.assertFalse(P1.scalar_once(doubled, n, -scalar * TOY_ROW['T']))
        self.assertGreater(np.abs(Operator(doubled).data - U_new).max(), 1e-3)
        self.assertEqual(P1.op_counts(new.circuit).get('crzz', 0), 0)
        self.assertGreater(P1.op_counts(legacy.circuit).get('crzz', 0), 0)
        self.assertEqual(P1.zz_module().instruction_parity(legacy.circuit, new.circuit)['status'], 'PASS')
        one = [c for c in prep.rte_preparation.symbolic_tail.components if c.df_fragment_id == 'full-h-one-body']
        self.assertTrue(one and all(len(c.diagonal_pauli_support) == 1 for c in one))

    def test_xy_measurement_sign(self):
        native = self.full.setup()
        prep = self.ctx.preparations[0]
        requests, _, _ = native.sample_requests(TOY_ROW, prep, P1.COHORT, 0)
        scalar = self.ctx.ham.constant + prep.extracted_identity_coefficient
        from trottertracks.resource_applicability import ax2b_native_df_v5 as nat
        from trottertracks.resource_applicability.h6_hash_reuse_v1 import use_fingerprint_reuse
        from trotterlib.df_trotter.circuit import simulate_statevector
        provider = __import__('trotterlib.df_rte_qiskit', fromlist=['x'])
        z = complex(np.vdot(self.ctx.qstate, self.ctx.oracle.trajectory(requests)))
        low = np.r_[self.ctx.qstate, np.zeros_like(self.ctx.qstate)]; d = len(self.ctx.qstate)
        with use_fingerprint_reuse() as fp, P1.zz_module().efficient_controlled_zz(provider):
            new = native.compose(TOY_ROW, prep, requests, scalar, fp.fingerprint)
            for axis, value in (('cosine', z.real), ('sine', z.imag)):
                w = nat.build_native_hadamard_wrapper(new, axis=axis, include_measurement=True,
                                                      max_instructions=len(new.circuit.data) + 4)
                self.assertEqual(w.data[-1].operation.name, 'measure')
                v = simulate_statevector(w.remove_final_measurements(inplace=False), low)
                self.assertAlmostEqual(float(np.vdot(v[:d], v[:d]).real - np.vdot(v[d:], v[d:]).real), value, delta=1e-10)


if __name__ == '__main__':
    unittest.main(verbosity=2)
