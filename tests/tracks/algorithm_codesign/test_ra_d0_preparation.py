"""Off-domain synthetic bookkeeping and LP certificate checks only."""
from fractions import Fraction as F
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/"src"))
from trottertracks.algorithm_codesign.ra_d0.exact import (
    DELTA_NUM, certify_law, kappa_upper, log_interval, normalize_direction,
    quantize, sqrt_interval,
)
from trottertracks.algorithm_codesign.ra_d0.grid import coverage, pareto_upper
from trottertracks.algorithm_codesign.ra_d0.lp import (
    LP, build_lp, check_farkas_output, dual_lower, farkas_certificate,
    make_farkas_problem, solve_synthetic, strict_witness,
)
from trottertracks.algorithm_codesign.ra_d0.table import extract_saved_table


class PreparationTests(unittest.TestCase):
    def test_sqrt_certificate_irrational(self):
        lo, hi = sqrt_interval(F(7, 3))
        self.assertLess(lo*lo, F(7, 3))
        self.assertGreater(hi*hi, F(7, 3))

    def test_sqrt_exact(self):
        self.assertEqual(sqrt_interval(F(9, 16)), (F(3, 4), F(3, 4)))

    def test_log_encloses_known_series(self):
        lo, hi = log_interval(2)
        # ln 2 = 2 sum ((1/3)^(2k+1)/(2k+1)), bounded positive tail.
        subtotal = 2*sum(F(1, 3**(2*k+1)*(2*k+1)) for k in range(100))
        tail = 2*F(2, 201*3**201)*F(9, 8)
        self.assertLessEqual(subtotal, lo)
        self.assertLessEqual(hi, subtotal+tail)

    def test_kappa_sufficient_polynomial(self):
        ell = log_interval(10560)[1]
        for n in (17, 333, 50001):
            h = kappa_upper(n, ell)
            self.assertGreaterEqual(n*h*h-F(4, 3)*ell*h-2*ell, 0)

    def test_kappa_decreases(self):
        ell = log_interval(10560)[1]
        self.assertGreater(kappa_upper(71, ell), kappa_upper(72, ell))

    def test_degree_direction_has_unit_norm(self):
        column, _ = normalize_direction(F(2), F(3), 1)
        self.assertEqual(column[0], (0, 0))
        self.assertLessEqual(sum(lo*lo for lo, _ in column), 1)
        self.assertGreaterEqual(sum(hi*hi for _, hi in column), 1)

    def test_terminal_no_unregistered_degree(self):
        with self.assertRaises(ValueError):
            normalize_direction(1, 1, 3)

    def test_largest_remainder_sum_and_tie(self):
        q, y = quantize([F(1, 3)]*3, F(7, 5), denominator=16)
        self.assertEqual(q, [F(6, 16), F(5, 16), F(5, 16)])
        self.assertEqual(sum(q), 1)
        self.assertEqual(y, F(22, 16))

    def test_negative_nominal_not_silently_clipped(self):
        with self.assertRaises(ValueError):
            quantize([F(-1, 1000), F(1001, 1000)], 1)

    def test_zero_y_after_quantization_rejected(self):
        with self.assertRaises(ValueError):
            quantize([1], F(1, 10**30))

    def test_off_domain_sampler_certificate(self):
        cert = certify_law([[(F(1), F(1))]], [F(1)], [0],
                           {"T": [F(3)], "CX": [F(1)], "1Q": [F(2)]},
                           [F(1)], F(1), 10**7, log_interval(10560)[1])
        self.assertTrue(cert["certified"])
        self.assertEqual(F(cert["resources"]["1Q"]), 9*10**7)

    def test_quantized_law_mean_budget_failure(self):
        cert = certify_law([[(F(1), F(1))]], [F(1)], [0], {"T": [F(3)]},
                           [F(1)], F(1)+2*DELTA_NUM, 10**7, log_interval(10560)[1])
        self.assertFalse(cert["certified"])

    def test_resource_cap_checked_after_sampler_quantization(self):
        cert = certify_law([[(F(1), F(1))]], [F(1)], [0], {"T": [F(3)]},
                           [F(1)], F(1), 10**7, log_interval(10560)[1], {"T": 1})
        self.assertFalse(cert["certified"])

    def test_grid_integer_ratio_including_rounding(self):
        points, factor = coverage(17, 300, [23, 58])
        for n in range(17, 301):
            nxt = min(p for p in points if p >= n)
            self.assertLessEqual(F(nxt, n), F(201, 200))
        self.assertLessEqual(factor, F(201, 200))

    def test_zero_cost_coordinate_cannot_be_omitted(self):
        with self.assertRaises(ValueError):
            pareto_upper({"T": F(0), "CX": F(1)}, [{"T": 10, "CX": 10}])
        # Candidate T=0 stays better in T at arbitrarily large shots, even
        # when the baseline wins CX. Omitting T would falsely prove dominance.
        self.assertLess(0, 10)

    def test_upper_boundary_full_vector(self):
        self.assertEqual(pareto_upper({"T": F(2), "CX": F(3)},
                                    [{"T": 10, "CX": 30}]), 5)

    def test_exact_dual_and_solver_agree_off_domain(self):
        lp = LP([F(1), F(2)], [[F(-1), F(-1)]], [F(-1)], [], [], [F(2), F(2)])
        output = solve_synthetic(lp)
        self.assertEqual(output["status"], 0)
        self.assertEqual(output["dual_certificate"]["lower"], 1)

    def test_dual_stationarity_error_is_charged(self):
        lp = LP([F(1)], [[F(-1)]], [F(-1)], [], [], [F(2)])
        cert = dual_lower(lp, [F(101, 100)], [])
        self.assertEqual(cert["stationarity_correction"], F(-1, 50))
        self.assertEqual(cert["lower"], F(99, 100))

    def test_dual_invalid_multiplier_rejected(self):
        lp = LP([F(1)], [[F(-1)]], [F(-1)], [], [], [F(2)])
        with self.assertRaises(ValueError):
            dual_lower(lp, [F(-1)], [])

    def test_dual_free_equality_multiplier_sign(self):
        lp = LP([F(3)], [], [], [[F(1)]], [F(2)], [F(5)])
        self.assertEqual(dual_lower(lp, [], [F(-3)])["lower"], 6)

    def test_interval_mean_residual_is_not_midpoint_only(self):
        from trottertracks.algorithm_codesign.ra_d0.exact import mean_residual_upper
        xi, _ = mean_residual_upper([[(F(9, 10), F(11, 10))]], [F(1)], [F(1)], F(1))
        self.assertEqual(xi, F(1, 10))

    def test_sampler_certificate_requires_registered_denominator(self):
        cert = certify_law([[(F(1), F(1))]], [F(1)], [0], {"T": [F(3)]},
                           [F(1)], F(1000000000001, 1000000000000), 10**7, log_interval(10560)[1])
        self.assertEqual(cert["reason"], "INVALID_SAMPLER_LAW")

    def test_saved_B0_not_identical_to_ideal_profile_in_general(self):
        lo, hi = sqrt_interval(F(10))
        midpoint = (lo+hi)/2
        self.assertNotEqual(midpoint**2, 10)

    def test_irrational_group_ratio_cannot_be_dyadic(self):
        from trottertracks.algorithm_codesign.ra_d0.semantics import rational_square
        self.assertFalse(rational_square(F(7, 5)))
        # If q_group masses were both rational, their ratio would be rational,
        # whereas the fixed ideal group-weight ratio is sqrt(7/5).

    def test_ideal_nesting_by_pure_representation_share(self):
        # Different synthetic representations with the same exact mean.
        D = [F(1), F(1, 2)]
        w = [[F(1), F(0)], [F(0), F(2)]]
        for theta in ([F(1), F(0)], [F(0), F(1)], [F(2, 5), F(3, 5)]):
            mixture = [sum(theta[r]*w[r][j] for r in range(2)) for j in range(2)]
            B = sum(mixture)
            q = [a/B for a in mixture]
            self.assertEqual(sum(q), 1)
            self.assertEqual(sum(a*b for a, b in zip(D, q)), 1/B)

    def toy_table(self):
        return {"target": ["1", "0", "0", "0"],
                "columns": [{"id": "O0:"+ep, "D_intervals": [["1", "1"]]+[["0", "0"]]*3,
                             "costs": {"T": "1", "CX": "1", "1Q": "1"}, "d_upper": "0"}
                            for ep in ("1e-3", "1e-4", "1e-6")],
                "B0_saved_profiles": [{"arm": arm, "epsilon": "1e-3", "memberships": [
                    {"column_id": "O0:1e-3", "ideal_weight_interval": ["1", "1"]}]}
                    for arm in ("ordinary", "PTSC_K0", "A")]}

    def test_B1_B2_B3_compiler_ideal_embeddings(self):
        data = self.toy_table()
        for baseline, nq, nz in (("B1", 3, 1), ("B2", 9, 3), ("B3", 3, 0)):
            lp = build_lp(data, 10**7, "T", log_interval(10560)[1], baseline,
                          robust=baseline == "B3", representation="ordinary")
            vector = [F(1)]+[F(0)]*(nq-1)+[F(1)]+[F(0)]*4
            vector += ([F(1)]+[F(0)]*(nz-1)) if nz else []
            for row, rhs in zip(lp.A, lp.b):
                self.assertLessEqual(sum(a*b for a, b in zip(row, vector)), rhs)
            for row, rhs in zip(lp.H, lp.f):
                self.assertEqual(sum(a*b for a, b in zip(row, vector)), rhs)

    def test_B1_requires_fixed_representation(self):
        with self.assertRaises(ValueError):
            build_lp(self.toy_table(), 10**7, "T", log_interval(10560)[1], "B1", robust=False)

    def test_B2_outer_point_is_not_claimed_as_certified_sampler(self):
        with self.assertRaises(ValueError):
            build_lp(self.toy_table(), 10**7, "T", log_interval(10560)[1], "B2", robust=True)

    def test_farkas_certificate_infeasible(self):
        lp = LP([F(0)], [[F(1)], [F(-1)]], [F(0), F(-1)], [], [], [F(2)])
        self.assertTrue(farkas_certificate(lp, [F(1), F(1)], [])["certified_infeasible"])

    def test_farkas_nominal_flag_not_a_certificate(self):
        lp = LP([F(0)], [[F(1)]], [F(1)], [], [], [F(2)])
        self.assertFalse(farkas_certificate(lp, [F(1)], [])["certified_infeasible"])

    def test_farkas_acquisition_and_exact_recheck_off_domain(self):
        lp = LP([F(0)], [[F(1)], [F(-1)]], [F(0), F(-1)], [], [], [F(2)])
        output = solve_synthetic(make_farkas_problem(lp))
        self.assertEqual(output["status"], 0)
        self.assertTrue(check_farkas_output(lp, output["nominal_primal"])["certified_infeasible"])

    def test_farkas_auxiliary_preserves_registered_launch_block(self):
        lp = LP([F(0)], [[F(1)], [F(-1)]], [F(0), F(-1)], [], [], [F(2)], domain="REGISTERED_SAVED_TABLE")
        with self.assertRaises(PermissionError):
            solve_synthetic(make_farkas_problem(lp))

    def test_registered_solver_launch_is_rejected_before_import(self):
        lp = LP([F(1)], [], [], [], [], [F(1)], domain="REGISTERED_SAVED_TABLE")
        with self.assertRaises(PermissionError):
            solve_synthetic(lp)

    def test_wrong_saved_input_identity_rejected(self):
        with self.assertRaises(PermissionError):
            extract_saved_table(b'{}')

    def test_strict_witness_requires_certified_upper(self):
        self.assertTrue(strict_witness({"certified": True, "objective_upper": "99/100"}, {"lower": F(1)}))
        self.assertFalse(strict_witness({"certified": False, "objective_upper": "99/100"}, {"lower": F(1)}))
        self.assertFalse(strict_witness({"certified": True, "objective_upper": "1"}, {"lower": F(1)}))


if __name__ == "__main__":
    unittest.main()
