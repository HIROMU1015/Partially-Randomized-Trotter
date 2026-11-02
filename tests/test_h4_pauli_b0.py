"""Only artificial 2--3 system-qubit data; no molecular files or compiler."""
from dataclasses import replace
import importlib
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose
from qiskit.quantum_info import Operator, SparsePauliOp
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_h4_d1_adapter import A, toy  # reuse the existing private import path
from d1_private import h4_pauli_b0 as B
from d1_private.fixed_signal_evaluation import FixedSignalConfig
from d1_private.pauli_signal_adapter import PauliHamiltonian


def fixture(n=2):
    # Includes a negative retained term, Y, little-endian placement, a tail and
    # a scalar. The noncommuting retained terms exercise S2 order explicitly.
    labels = ("ZI", "YX", "IX") if n == 2 else ("ZII", "IYX", "XIZ")
    h = PauliHamiltonian.from_terms([(labels[0], .7), (labels[1], -.4),
                                    (labels[2], .1), ("I" * n, .37)])
    cfg = FixedSignalConfig("pauli_jw", method="PR", L_D=2, T=.4, q=2,
                            r=4, K=4, cost_samples=256, seed=17)
    return h, cfg


def oracle(h, cfg):
    n = h.num_qubits
    half = cfg.T / cfg.q / 2
    u = np.eye(2 ** n, dtype=complex)
    for _ in range(cfg.q):
        for label, c in (*h.terms[:cfg.L_D], *reversed(h.terms[:cfg.L_D])):
            p = SparsePauliOp.from_list([(label, c)]).to_matrix()
            u = expm(-1j * half * p) @ u
        u *= np.exp(-1j * h.constant * cfg.T / cfg.q)
    return u


class MatchedB0Tests(unittest.TestCase):
    def test_explicit_prefix_and_one_alias_candidate(self):
        _, pr = fixture()
        b0 = B.matched_config(pr)
        self.assertEqual(b0, B.matched_config(pr, alias="B0I"))
        self.assertEqual((b0.method, b0.L_D, b0.cost_samples), ("B0", 2, 1))
        for key in ("T", "q", "r", "K", "seed", "product_formula"):
            self.assertEqual(getattr(pr, key), getattr(b0, key))
        self.assertEqual(pr.cost_samples, 256)  # source config unchanged

    def test_missing_prefix_and_wrong_family_rejected(self):
        _, pr = fixture()
        for cfg in (replace(pr, L_D=None), replace(pr, L_D=0),
                    replace(pr, method="PF"), replace(pr, hamiltonian_type="df_native")):
            with self.assertRaises(ValueError):
                B.matched_config(cfg)
        with self.assertRaises(ValueError):
            B.matched_config(pr, alias="PF")

    def test_retained_order_constant_and_tail_exclusion(self):
        h, pr = fixture()
        cfg = B.matched_config(pr)
        body, adapter = B.build_body(h, cfg)
        assert_allclose(Operator(body).data, oracle(h, cfg), atol=3e-14, rtol=0)
        self.assertEqual(adapter.retained, h.terms[:2])
        self.assertEqual(adapter.tail, h.terms[2:])
        self.assertFalse(adapter.randomized)
        self.assertEqual(adapter.log_B_step, 0.)
        self.assertEqual(len(body.data), 2 * cfg.q * cfg.L_D)
        self.assertTrue(adapter.metadata()["B0I_equivalent_to_B0"])

    def test_both_axes_relative_phase_and_raw_ray(self):
        for n in (2, 3):
            h, pr = fixture(n); cfg = B.matched_config(pr)
            body, _ = B.build_body(h, cfg)
            before = Operator(body).data.copy()
            wrappers, controlled = B.central_wrappers(body)
            expected = np.eye(2 ** (n + 1), dtype=complex)
            expected[2 ** n:, 2 ** n:] = oracle(h, cfg)
            assert_allclose(Operator(controlled.body).data, expected, atol=4e-14, rtol=0)
            self.assertEqual(controlled.sidecar["ancilla"], n)
            raw = np.arange(1, 2 ** n + 1) + 1j * np.arange(2 ** n, 0, -1)
            # No state normalization in the bridge. Divide contractions by the
            # same norm only at the expectation boundary, as for the saved ray.
            norm = np.vdot(raw, raw).real
            mean = np.vdot(raw, oracle(h, cfg) @ raw) / norm
            for axis, wrapper in wrappers.items():
                unmeasured = wrapper.remove_final_measurements(inplace=False)
                initial = np.r_[raw, np.zeros_like(raw)]
                out = Operator(unmeasured).data @ initial
                observed = (np.vdot(out[:2 ** n], out[:2 ** n]).real
                            - np.vdot(out[2 ** n:], out[2 ** n:]).real) / norm
                self.assertAlmostEqual(observed, mean.real if axis == "cosine" else mean.imag,
                                       delta=4e-14)
                self.assertEqual(wrapper.num_clbits, 1)
                self.assertEqual(wrapper.find_bit(wrapper.data[-1].qubits[0]).index, n)
            assert_allclose(Operator(body).data, before, atol=0, rtol=0)

    def test_d1_retained_body_reuses_same_s2_recipe(self):
        model, join = toy("pauli_jw", "PR")
        retained = A.build(model, "PR", join, part="retained").body
        h = PauliHamiltonian.from_terms([("ZI", .7), ("XX", -.4), ("II", .37)])
        pr = FixedSignalConfig("pauli_jw", method="PR", L_D=1, T=.4, q=2,
                               r=1, K=4)
        direct, _ = B.build_body(h, B.matched_config(pr))
        assert_allclose(Operator(retained).data, Operator(direct).data, atol=0, rtol=0)
        wrappers, _ = B.central_wrappers(retained)
        self.assertEqual(set(wrappers), {"cosine", "sine"})

    def test_no_sampling_generic_control_or_compiler_in_new_path(self):
        p = importlib.import_module("d1_private.pauli_signal_adapter")
        f = importlib.import_module("d1_private.fixed_signal_evaluation")
        from qiskit.circuit import Gate
        from contextlib import ExitStack
        h, pr = fixture()
        with ExitStack() as stack:
            for module, name in ((p, "sample_rte_events"), (f, "evaluate_fixed_signal")):
                stack.enter_context(patch.object(module, name, side_effect=AssertionError(name)))
            stack.enter_context(patch.object(Gate, "control", side_effect=AssertionError("generic control")))
            stack.enter_context(patch("qiskit.transpile", side_effect=AssertionError("compile")))
            body, _ = B.build_body(h, B.matched_config(pr))
            B.central_wrappers(body)


if __name__ == "__main__":
    unittest.main()
