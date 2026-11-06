"""Synthetic technical regressions; no saved scientific input or compiler."""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trottertracks import track_d_b1_c1 as implementation
from trottertracks.c1_stage3_accuracy import _finite


class B1C1Tests(unittest.TestCase):
    def test_state_free_transport_counts_df_once_and_keeps_unknown_N(self):
        row = implementation.transport_round(
            dict(attenuation=.9, pf_operator_error=.02, rte_corrected_operator_error=.03),
            m=1, df_error=.01, base_time=.1,
            historical_target=dict(beta_total=.5, beta_stat=.2), finite=_finite)
        self.assertAlmostEqual(row["df_evolution_error"], .002)
        self.assertAlmostEqual(row["original_H_corrected_operator_error"], .052)
        self.assertAlmostEqual(row["observed_radius_lower"], .9 * .948)
        self.assertAlmostEqual(row["deterministic_phase_upper"], math.asin(.052))
        self.assertTrue(row["conditional_sufficient"])
        self.assertIsNone(row["N"])
        self.assertFalse(row["saved_own_DF_signal_used"])

    def test_insufficient_bound_and_unknown_premise_are_not_relaxed(self):
        args = dict(m=1, df_error=10., base_time=.1,
                    historical_target=dict(beta_total=.5, beta_stat=.2), finite=_finite)
        ref = dict(attenuation=.9, pf_operator_error=.02, rte_corrected_operator_error=.03)
        row = implementation.transport_round(ref, **args)
        self.assertEqual(row["df_evolution_error"], 2.)
        self.assertFalse(row["conditional_sufficient"])
        self.assertIsNone(row["deterministic_phase_upper"])
        with self.assertRaises(ValueError):
            implementation.transport_round({**ref, "pf_operator_error": None}, **args)

    def test_saved_raw_x_shell_removal_preserves_core_phase_and_rejects_other_shell(self):
        import numpy as np
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Operator
        core = QuantumCircuit(3); core.global_phase = .13
        core.crz(.27, 2, 0); core.cx(0, 1); core.crz(-.11, 2, 1)
        wrapper = QuantumCircuit(3, 1); wrapper.h(2)
        wrapper.compose(core, inplace=True); wrapper.h(2); wrapper.measure(2, 0)
        original_size = len(wrapper.data)
        extracted = implementation.unwrap_raw_x(wrapper)
        np.testing.assert_allclose(Operator(extracted).data, Operator(core).data, atol=1e-13, rtol=0)
        self.assertEqual(len(wrapper.data), original_size)
        wrapper.data[0] = wrapper.data[1]
        with self.assertRaisesRegex(ValueError, "WRAPPER_SHELL"):
            implementation.unwrap_raw_x(wrapper)
        self.assertEqual(implementation.PAIR, (0, 1))


if __name__ == "__main__":
    unittest.main()
