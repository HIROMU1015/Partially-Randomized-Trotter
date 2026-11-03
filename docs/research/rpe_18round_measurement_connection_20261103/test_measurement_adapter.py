"""Artificial fixtures and settings-only checks. No H4 signal/RNG/circuit calls."""
import ast
import cmath
from dataclasses import FrozenInstanceError, replace
import inspect
import json
import math
import sys
import unittest
from unittest.mock import Mock

import measurement_adapter as a


def fixture(m=0, ld=3, shots=4):
    return a.RoundSettings(
        ld, m, 2**m, 1 if ld == 3 else 0, 2 if ld == 3 else 0, .02,
        a.AxisSettings(shots, ".01"), a.AxisSettings(shots, ".01"),
        ".01", ".01" if ld == 3 else "0", ".38", ".1", a.PHYSICAL,
        "attenuated_event_mean_signal" if ld == 3 else "physical_PF_signal",
        "ARTIFICIAL_STATE_ONLY", "SYNTHETIC_ONLY",
    )


class MeasurementConnectionTests(unittest.TestCase):
    def test_fresh_factory_is_called_for_every_axis_shot(self):
        settings = fixture()
        factory = Mock(side_effect=lambda request: request.trajectory_key)
        seen = []
        def measure(request, trajectory):
            seen.append((request, trajectory))
            return 1 if request.shot < 3 else -1
        result = a.collect_virtual_round(settings, factory, measure)
        self.assertEqual(factory.call_count, 8)
        self.assertEqual(len({c.args[0].trajectory_key for c in factory.call_args_list}), 8)
        self.assertTrue(all(req.expected_event_count == 1 for req, _ in seen))
        self.assertEqual(result.signal, .5+.5j)
        self.assertEqual(result.trajectory_policy, "fresh_per_shot")
        self.assertIs(result.settings, settings)

    def test_reuse_is_two_factory_calls_and_explicitly_non_v2(self):
        factory = Mock(return_value="same_mock_trajectory")
        result = a.collect_virtual_round(fixture(), factory, lambda *_: 1,
                                         trajectory_policy="reuse_within_axis")
        self.assertEqual(factory.call_count, 2)
        decoded = a.decode_measurements([result.settings], [result])
        self.assertFalse(decoded["v2_fresh_call_policy_requested"])
        self.assertFalse(decoded["statistical_iid_verified"])

    def test_full_never_samples_an_rte_trajectory(self):
        factory = Mock(side_effect=AssertionError("Full must be tail-free"))
        def measure(request, trajectory):
            self.assertIsNone(trajectory)
            self.assertIsNone(request.trajectory_key)
            self.assertEqual(request.expected_event_count, 0)
            return 1
        result = a.collect_virtual_round(fixture(ld=12), factory, measure)
        factory.assert_not_called()
        self.assertEqual(result.trajectory_policy, "deterministic_full")

    def test_partial_full_control_and_corrected_laws_are_separated(self):
        for changes in ({"signal_kind":"corrected_finite_rte_signal"},
                        {"state_label":"effective_partial_s2_eigenstate_control"},
                        {"signal_kind":"physical_PF_signal"}, {"r":0}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(fixture(), **changes)
        with self.assertRaises(ValueError):
            replace(fixture(ld=12), r=1)
        with self.assertRaises(ValueError):
            a.decode_measurements([fixture()], [a.MeasuredMeans(fixture(ld=12), 1, 0)])

    def test_exp_minus_itH_sine_and_energy_sign(self):
        s = fixture()
        z = cmath.exp(-.3j)
        self.assertLess(a.marginal_coordinate(z, s, "sine"), 0)
        result = a.decode_measurements([s], [a.MeasuredMeans(s, z.real, z.imag)])
        self.assertAlmostEqual(result["energy"], .3 / s.delta)
        self.assertFalse(result["success_probability_evaluated"])

    def test_known_artificial_phase_reconstructs_18_rounds(self):
        rounds = [fixture(m) for m in range(18)]
        records = []
        for s in rounds:
            z = .4 * cmath.exp(.72j * s.q)
            records.append(a.MeasuredMeans(s, z.real, z.imag))
        result = a.decode_measurements(rounds, records)
        self.assertEqual(result["status"], "DECODED_CONDITIONAL")
        self.assertAlmostEqual(result["energy"], -.72 / .02)
        self.assertFalse(result["radius_lower_bound_verified"])
        self.assertFalse(result["alias_assumption_verified"])

    def test_low_nonzero_radius_is_not_a_success_certificate(self):
        s = fixture()
        result = a.decode_measurements([s], [a.MeasuredMeans(s, 1e-14, 0)])
        self.assertEqual(result["status"], "DECODED_CONDITIONAL")
        self.assertFalse(result["radius_lower_bound_verified"])
        self.assertFalse(result["success_probability_evaluated"])

    def test_zero_signal_does_not_decode_arg_zero(self):
        result = a.decode_measurements([fixture()], [a.MeasuredMeans(fixture(), 0, 0)])
        self.assertEqual(result["status"], "ZERO_SIGNAL")
        self.assertIsNone(result["energy"])

    def test_branch_tie_retains_original_choice_but_withholds_energy(self):
        rounds = [fixture(0), fixture(1)]
        result = a.decode_measurements(rounds, [a.MeasuredMeans(rounds[0], 1, 0),
                                               a.MeasuredMeans(rounds[1], -1, 0)])
        self.assertEqual(result["status"], "AMBIGUOUS_BRANCH")
        self.assertIsNone(result["energy"])
        self.assertIn("reconstruction", result)

    def test_alias_edges_are_flagged_and_wrap_neighbours_remain_conditional(self):
        s = fixture()
        result = a.decode_measurements([s], [a.MeasuredMeans(s, -1, 0)])
        self.assertEqual(result["status"], "ALIAS_BOUNDARY")
        for phase in (math.pi-.01, -math.pi+.01):
            z = cmath.exp(1j*phase)
            result = a.decode_measurements([s], [a.MeasuredMeans(s, z.real, z.imag)])
            self.assertEqual(result["status"], "DECODED_CONDITIONAL")
            self.assertFalse(result["alias_assumption_verified"])

    def test_invalid_means_outcomes_and_policy_are_rejected(self):
        for mean in (1.01, -1.01, math.nan, math.inf):
            with self.subTest(mean=mean), self.assertRaises(ValueError):
                a.MeasuredMeans(fixture(), mean, 0)
        for outcome in (0, True, 2, math.nan):
            with self.subTest(outcome=outcome), self.assertRaises(ValueError):
                a.collect_virtual_round(fixture(), Mock(), lambda *_, v=outcome: v)
        with self.assertRaises(ValueError):
            a.collect_virtual_round(fixture(), Mock(), Mock(), trajectory_policy="paired_cos_sin")
        with self.assertRaises(ValueError):
            a.marginal_coordinate(1.1+0j, fixture(), "cosine")

    def test_estimated_sample_radius_above_one_is_not_clamped(self):
        s = fixture()
        self.assertEqual(a.MeasuredMeans(s, 1, 1).signal, 1+1j)

    def test_settings_are_immutable_and_mismatches_rejected(self):
        s = fixture()
        with self.assertRaises(FrozenInstanceError):
            s.r = 2
        changed = replace(s, beta_stat=".3")
        with self.assertRaises(ValueError):
            a.decode_measurements([s], [a.MeasuredMeans(changed, 1, 0)])
        with self.assertRaises(ValueError):
            a.decode_measurements([fixture(0), fixture(2)],
                                  [a.MeasuredMeans(fixture(0), 1, 0), a.MeasuredMeans(fixture(2), 1, 0)])

    def test_decoder_has_no_E0_or_oracle_input_and_unchanged_definitions(self):
        self.assertEqual(tuple(inspect.signature(a.decode_measurements).parameters), ("rounds", "means"))
        decoder = a.original_decoder()
        self.assertEqual(tuple(inspect.signature(decoder).parameters), ("measured_round_phases", "q_values"))
        original = ast.parse(a.fixed_bytes(a.CARRIER, a.DECODER_PATH).decode())
        function, = [n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == "reconstruct_rpe_phase"]
        self.assertNotIn("E0", ast.unparse(function))
        self.assertNotIn("_oracle_branch", decoder.__globals__)
        self.assertNotIn("trotterlib", sys.modules)
        self.assertNotIn("qiskit", sys.modules)

    def test_saved_metadata_only_preserves_all_N_alpha_beta_and_tuples(self):
        original = json.loads(a.fixed_bytes(a.CARRIER, a.MODEL_PATH))
        for ld in (3, 12):
            rounds = a.load_fixed_rounds(ld)
            selected = original["coherent_reoptimization"]["best_by_ld"][str(ld)]["best"]
            self.assertEqual(len(rounds), 18)
            for s, row in zip(rounds, selected["rounds"]):
                self.assertEqual((s.q, s.r, s.k), tuple(row[k] for k in ("q_m", "r_m", "K_m")))
                self.assertEqual((s.beta_pf, s.beta_rte, s.beta_stat),
                                 tuple(str(selected[k]) for k in ("beta_pf_budget", "beta_rte_budget", "beta_stat_budget")))
                self.assertEqual(s.epsilon_coordinate, str(row["epsilon_coordinate"]))
                for axis in a.AXES:
                    self.assertEqual(getattr(s, axis), a.AxisSettings(row["axes"][axis]["shots"], str(row["axes"][axis]["alpha"])))
                self.assertFalse(hasattr(s, "E0"))
                self.assertFalse(hasattr(s, "saved_signal"))
            if ld == 12:
                self.assertTrue(all(s.physical_fingerprint == "NOT_RECORDED" for s in rounds))


if __name__ == "__main__":
    unittest.main()
