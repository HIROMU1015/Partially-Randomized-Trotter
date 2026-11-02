"""Small artificial rational/phase controls; no SP-1 domain or saved data I/O."""
import cmath
from fractions import Fraction as F
from itertools import product
import math
import unittest

from trottertracks.algorithm_codesign.synthesis_placement.wrapper_accounting import (
    Gate, Path, population_profile, log_upper, axis_budget, total_t,
)


def enumerate_profile(paths):
    """Independent explicit outcome enumeration for tiny fixtures only."""
    out = dict(second_moment=F(0), range=F(0), expected_T_count=F(0),
               joint_weighted_T=F(0), synthesis_bias_upper=F(0))
    for path in paths:
        for indices in product(*(range(len(g.coefficients)) for g in path.gates)):
            p, weight = path.probability, path.outer_weight
            c, e = path.fixed_t, path.fixed_diamond_error
            for gate, i in zip(path.gates, indices):
                gamma = sum(abs(g) for g in gate.coefficients)
                if gate.coefficients[i] == 0:
                    p = 0
                    break
                p *= abs(gate.coefficients[i])/gamma
                weight *= gamma*(1 if gate.coefficients[i] > 0 else -1)
                c += gate.t_counts[i]
                e += gate.diamond_errors[i]
            if p == 0:
                continue
            # Actual quantum +/-1 outcome is included; squared outcome is 1.
            for y, py in ((-1, F(1, 3)), (1, F(2, 3))):
                out["second_moment"] += p*py*(weight*y)**2
                out["expected_T_count"] += p*py*c
                out["joint_weighted_T"] += p*py*(weight*y)**2*c
                out["synthesis_bias_upper"] += p*py*abs(weight)*e
                out["range"] = max(out["range"], abs(weight*y))
    return out


def matmul(a, b):
    return [[sum(a[i][k]*b[k][j] for k in range(len(b)))
             for j in range(len(b[0]))] for i in range(len(a))]


def adjoint(a):
    return [[a[j][i].conjugate() for j in range(len(a))] for i in range(len(a))]


def rotate(pauli, theta):
    return [[math.cos(theta/2)*int(i == j)-1j*math.sin(theta/2)*pauli[i][j]
             for j in range(len(pauli))] for i in range(len(pauli))]


def tensor(a, b):
    return [[x*y for x in ar for y in br] for ar in a for br in b]


def residual(a, b):
    return max(abs(x-y) for ar, br in zip(a, b) for x, y in zip(ar, br))


class AccountingTests(unittest.TestCase):
    def setUp(self):
        self.g1 = Gate((1, F(1, 4), -F(1, 4)), (0, 3, 2), (0, F(1, 100), F(1, 50)))
        self.g2 = Gate((F(5, 4), -F(1, 4), 0), (4, 0, 999), (F(1, 200), 0, 2))

    def test_controlled_pair_uses_product_not_sum(self):
        p = population_profile([Path(1, 1, (self.g1, self.g2))])
        self.assertEqual(p, enumerate_profile([Path(1, 1, (self.g1, self.g2))]))
        self.assertEqual(p["second_moment"], F(81, 16))
        self.assertNotEqual(p["second_moment"], F(9, 2))

    def test_outer_correlation_is_preserved(self):
        paths = [Path(F(1, 3), 2, (self.g1,), fixed_t=1),
                 Path(F(2, 3), -3, (self.g1, self.g2), fixed_t=7)]
        p = population_profile(paths)
        self.assertEqual(p, enumerate_profile(paths))
        self.assertNotEqual(p["joint_weighted_T"], p["second_moment"]*p["expected_T_count"])

    def test_equal_mean_length_does_not_fix_moment(self):
        fixed = population_profile([Path(1, 1, (self.g1,))])
        variable = population_profile([Path(F(1, 2), 1, ()),
                                       Path(F(1, 2), 1, (self.g1, self.g1))])
        self.assertEqual(fixed["expected_T_count"], variable["expected_T_count"])
        self.assertGreater(variable["second_moment"], fixed["second_moment"])

    def test_signed_outer_normalization_counted_once(self):
        p1 = population_profile([Path(1, 1, (self.g1,))])
        p2 = population_profile([Path(1, -3, (self.g1,))])
        self.assertEqual(p2["second_moment"], 9*p1["second_moment"])
        self.assertEqual(p2["range"], 3*p1["range"])
        self.assertEqual(p2["synthesis_bias_upper"], 3*p1["synthesis_bias_upper"])
        self.assertEqual(p2["expected_T_count"], p1["expected_T_count"])

    def test_exact_notch_and_zero_cost(self):
        g = Gate((1, 0, 0), (0, 99, 99), (0, 2, 2))
        p = population_profile([Path(1, 1, (g,))])
        self.assertEqual((p["second_moment"], p["range"], p["expected_T_count"],
                          p["synthesis_bias_upper"]), (1, 1, 0, 0))
        record = axis_budget(p, F(1, 10), F(1, 20), 10**9)
        self.assertEqual(record["expected_total_T"], 0)

    def test_fixed_cost_and_bias_are_weighted(self):
        p = population_profile([Path(1, 2, (), fixed_t=7, fixed_diamond_error=F(1, 100))])
        self.assertEqual(p["expected_T_count"], 7)
        self.assertEqual(p["synthesis_bias_upper"], F(1, 50))

    def test_bias_budget_exhaustion(self):
        p = population_profile([Path(1, 1, (), fixed_diamond_error=F(1, 10))])
        r = axis_budget(p, F(1, 10), F(1, 20), 100)
        self.assertEqual(r["status"], "BIAS_BUDGET_EXHAUSTED")
        self.assertIsNone(r["shots"])

    def test_shot_cap_is_not_clipped_or_accepted(self):
        p = population_profile([Path(1, 1, (), fixed_t=1)])
        r = axis_budget(p, F(1, 10), F(1, 20), 1)
        self.assertEqual(r["status"], "SHOT_CAP_EXCEEDED")
        self.assertGreater(r["shots"], 1)
        with self.assertRaises(ValueError):
            total_t((r, r))

    def test_ceiling_at_cap_and_initialization_scope(self):
        p = population_profile([Path(1, 1, (), fixed_t=3)])
        r = axis_budget(p, F(1, 10), F(1, 20), 10**9)
        at_cap = axis_budget(p, F(1, 10), F(1, 20), r["shots"])
        self.assertEqual(at_cap["status"], "ELIGIBLE")
        self.assertEqual(total_t((r, r), 5), 5+6*r["shots"])

    def test_finite_bias_increases_sufficient_shots(self):
        p = population_profile([Path(1, 1, (), fixed_t=1)])
        r0 = axis_budget(p, F(1, 10), F(1, 20), 10**9)
        r1 = axis_budget(p, F(1, 10), F(1, 20), 10**9,
                         model_bias=F(1, 100), numerical_bias=F(1, 1000))
        self.assertGreater(r1["shots"], r0["shots"])

    def test_rational_log_encloses_independent_math_reference(self):
        for x in (1, 2, 5, 40, 3840):
            bound = log_upper(x)
            self.assertGreaterEqual(float(bound)+1e-14, math.log(x))
            self.assertLess(float(bound)-math.log(x), 1e-10)
        self.assertGreater(log_upper(5, 2), log_upper(5, 3))

    def test_malformed_probabilities_and_costs_rejected(self):
        with self.assertRaises(ValueError):
            population_profile([Path(F(1, 2), 1, ())])
        with self.assertRaises(TypeError):
            Path(0.5, 1, ())
        with self.assertRaises(ValueError):
            Gate((1, -1), (1, 1), (0, 0))
        with self.assertRaises(ValueError):
            Gate((1,), (F(1, 2),), (0,))
        with self.assertRaises(ValueError):
            Path(1, 1, (), fixed_t=-1)


class SemanticDesignControls(unittest.TestCase):
    """Analytic design controls, not an implemented wrapper adapter validation."""
    def test_repeated_same_pauli_fuses_but_alternating_axes_do_not(self):
        z, x = [[1+0j, 0j], [0j, -1+0j]], [[0j, 1+0j], [1+0j, 0j]]
        rz, rx = rotate(z, 0.37), rotate(x, 0.37)
        same = matmul(rz, matmul(rz, rz))
        self.assertLess(residual(same, rotate(z, 3*0.37)), 1e-14)
        alternating = matmul(rz, matmul(rx, rz))
        self.assertGreater(residual(alternating, rotate(z, 3*0.37)), 0.1)

    def test_signed_controlled_joint_lowering(self):
        identity, z, x = [[1+0j, 0j], [0j, 1+0j]], [[1+0j, 0j], [0j, -1+0j]], [[0j, 1+0j], [1+0j, 0j]]
        for theta in (0.37, -0.37):
            target_u = rotate(x, theta)
            target = [[complex(int(i == j)) if i < 2 and j < 2 else
                       target_u[i-2][j-2] if i >= 2 and j >= 2 else 0j
                       for j in range(4)] for i in range(4)]
            lowered = matmul(rotate(tensor(identity, x), theta/2),
                             rotate(tensor(z, x), -theta/2))
            self.assertLess(residual(target, lowered), 1e-14)

    def test_system_global_phase_becomes_controlled_relative_phase(self):
        state = [1/math.sqrt(2), 1/math.sqrt(2)]
        self.assertAlmostEqual(2*(state[0].conjugate()*state[1]).real, 1)
        changed = [state[0], -state[1]]
        self.assertAlmostEqual(2*(changed[0].conjugate()*changed[1]).real, -1)

    def test_basis_inverse_order_matters(self):
        z, x = [[1+0j, 0j], [0j, -1+0j]], [[0j, 1+0j], [1+0j, 0j]]
        basis = rotate(x, 0.37)
        core = rotate(z, -0.29)
        correct = matmul(adjoint(basis), matmul(core, basis))
        wrong = matmul(basis, matmul(core, basis))
        self.assertGreater(residual(correct, wrong), 0.1)

    def test_shared_inverse_draw_does_not_implement_independent_product(self):
        # +/-phi draw in a channel and its inverse: sharing cancels every
        # sample, while independent draws attenuate coherence by cos(phi)^2.
        phi = 0.37
        independent = sum(cmath.exp(1j*(a-b)*phi)/4 for a in (-1, 1) for b in (-1, 1))
        shared = sum(cmath.exp(1j*(a-a)*phi)/2 for a in (-1, 1))
        self.assertAlmostEqual(independent.real, math.cos(phi)**2)
        self.assertAlmostEqual(shared.real, 1)
        self.assertGreater(abs(shared-independent), 0.1)


if __name__ == "__main__":
    unittest.main()
