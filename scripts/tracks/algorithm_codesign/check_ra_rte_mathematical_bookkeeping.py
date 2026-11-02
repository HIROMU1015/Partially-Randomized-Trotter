#!/usr/bin/env python3
"""Off-domain exact bookkeeping for the GPT RA-RTE design, not a science pilot.

No synthesis, sampling, solver, circuit, operator matrix, saved-resource scoring,
Hamiltonian, molecular file or GPU access. Finite lists are artificial inputs.
"""
import hashlib
import itertools
import json
from collections import defaultdict
from fractions import Fraction as F
from math import isqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/track_b_ra_rte_mathematical_audit/2026-10-06"
SCRIPT_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reduce_word(word):
    stack = []
    for letter in word:
        if stack and stack[-1] == letter:
            stack.pop()
        else:
            stack.append(letter)
    return tuple(stack)


def phase(sigma, degree):
    # Gaussian rational pair for (-i*sigma)**degree, without complex floats.
    return [(F(1), F(0)), (F(0), F(-sigma)),
            (F(-1), F(0)), (F(0), F(sigma))][degree % 4]


def gaussian_add(left, right):
    return left[0] + right[0], left[1] + right[1]


def scale(value, number):
    return value[0] * number, value[1] * number


def expected_free_word(degree, p):
    # Exhaustive symbolic monomials, no random draw or quantum trajectory.
    result = defaultdict(F)
    for letters in itertools.product(range(len(p)), repeat=degree):
        weight = F(1)
        for letter in letters:
            weight *= p[letter]
        result[reduce_word(letters)] += weight
    return dict(result)


def add_scaled(output, words, coefficient):
    for word, probability in words.items():
        output[word] = gaussian_add(output.get(word, (F(0), F(0))), scale(coefficient, probability))


def clean(values):
    return {k: v for k, v in values.items() if v != (0, 0)}


def direct_event_mean(k, c, s, sigma, p):
    result = {}
    for letters in itertools.product(range(len(p)), repeat=k + 1):
        q0, word = letters[0], tuple(reversed(letters[1:]))
        probability = F(1)
        for letter in letters:
            probability *= p[letter]
        add_scaled(result, {reduce_word(word): probability}, scale(phase(sigma, k), c))
        add_scaled(result, {reduce_word((q0,) + word): probability}, scale(phase(sigma, k + 1), s))
    return clean(result)


def dot(left, right):
    return sum((a * b for a, b in zip(left, right)), F(0))


def shot_condition(n, ell, h):
    return h > 0 and n * h * h >= ell * (2 + F(4, 3) * h)


def inverse_unitary_word(word):
    inverses = {"A": "A_dag", "A_dag": "A", "X": "X", "B_dag": "B"}
    stack = []
    for letter in word:
        if stack and inverses.get(letter) == stack[-1]:
            stack.pop()
        else:
            stack.append(letter)
    return stack


def main():
    protocol = json.loads((OUT / "bookkeeping_protocol_v1.json").read_text())
    require(protocol["analysis_script_sha256"] == SCRIPT_SHA, "script identity changed after protocol")
    require(not (OUT / "bookkeeping_checks_v1.json").exists(), "do not overwrite a completed check")
    meta = protocol["metadata"]
    checks = []

    def checked(name, condition, detail=None):
        require(condition, name)
        checks.append({"check": name, "status": "PASS", "detail": detail})

    p = (F(2, 7), F(5, 7)); c, s = F(3, 5), F(4, 5)
    for sigma in (-1, 1):
        for k in range(3):
            expected = {}
            add_scaled(expected, expected_free_word(k, p), scale(phase(sigma, k), c))
            add_scaled(expected, expected_free_word(k + 1, p), scale(phase(sigma, k + 1), s))
            checked(f"IID_free_word_mean_sigma_{sigma}_k_{k}", direct_event_mean(k, c, s, sigma, p) == clean(expected))
        terminal = direct_event_mean(3, F(1), F(0), sigma, p)
        target = {}; add_scaled(target, expected_free_word(3, p), phase(sigma, 3))
        checked(f"terminal_pure_word_sigma_{sigma}", terminal == clean(target))
        checked(f"terminal_mutant_detected_sigma_{sigma}", direct_event_mean(3, c, s, sigma, p) != terminal)
    checked("phase_mutant_detected", direct_event_mean(1, c, s, -1, p) != direct_event_mean(1, c, s, 1, p))
    checked("correlated_index_mutant_detected", expected_free_word(2, p) != {(): F(1)})

    D = ((F(1), c, F(0)), (F(0), s, F(1)))
    t = (F(1), F(2, 5)); d = (F(0), F(1, 100), F(0)); e, ell = F(1, 8), F(3)
    for u in (F(0), F(1, 4), F(9, 20), F(1, 2)):
        w = (1 - c * u, u, t[1] - s * u)
        B = sum(w); q = tuple(v / B for v in w); y = 1 / B
        slack = e - dot(d, w); h = e * y - dot(d, q)
        checked(f"bijection_and_bias_u_{u}", all(dot(row, w) == rhs for row, rhs in zip(D, t))
                and sum(q) == 1 and all(dot(row, q) == y * rhs for row, rhs in zip(D, t))
                and tuple(v / y for v in q) == w and h == slack / B)
        for n in (639, 640, 641):
            direct = slack > 0 and n * slack * slack >= ell * (2 * B * B + F(4, 3) * B * slack)
            checked(f"direct_normalized_shot_equivalence_u_{u}_n_{n}", direct == shot_condition(n, ell, h))
    kappa, n = F(1, 10), 640
    a = F(4, 3) * ell; discriminant = a * a + 8 * n * ell
    checked("positive_root_exact_witness", discriminant.denominator == 1
            and isqrt(discriminant.numerator) ** 2 == discriminant
            and (a + isqrt(discriminant.numerator)) / (2 * n) == kappa
            and shot_condition(n, ell, kappa) and not shot_condition(n - 1, ell, kappa))
    checked("rounded_down_kappa_mutant_detected", not shot_condition(n, ell, kappa - F(1, 10000)))
    u = F(9, 20); B = F(7, 5) - F(2, 5) * u
    true_h = (e - u / 100) / B; missing_factor2_h = (e - u / 200) / B
    checked("missing_factor2_bias_mutant_detected", not shot_condition(n, ell, true_h) and shot_condition(n, ell, missing_factor2_h))
    checked("sum_probability_mutant_detected", sum((F(1, 3), F(1, 3))) != 1)

    # Artificial LP primal/dual certificate. No solver or candidate search.
    q = (F(7, 12), F(5, 12), F(0)); y = F(5, 6); C = (F(5), F(1), F(3))
    dual_u = (F(53, 30), F(-197, 48)); zeta, lam = F(97, 30), F(1)
    primal = dot(C, q); dual = zeta + lam * kappa
    dual_feasible = dot(t, dual_u) >= lam * e and all(
        sum(D[row][j] * dual_u[row] for row in range(2)) + zeta <= C[j] + lam * d[j] for j in range(3))
    checked("fixed_n_primal_dual_exact_gap_zero", dual_feasible and e * y - dot(d, q) >= kappa and primal == dual == F(10, 3))
    checked("wrong_y_dual_sign_mutant_detected", not dot(t, (F(0), F(0))) >= lam * e)

    minus, plus, target = (F(1), F(0)), (F(3, 5), F(4, 5)), (F(4, 5), F(3, 5))
    lminus, lplus = F(7, 20), F(3, 4)
    checked("two_direction_exact_matching", tuple(lminus * a + lplus * b for a, b in zip(minus, plus)) == target)
    checked("two_direction_sec_squared_bound", (lminus + lplus) ** 2 <= 2 / (1 + dot(minus, plus)))
    checked("zero_gap_is_singular", dot(minus, (F(0), F(1))) == 0,
            "Identical bracket angles have sin(Delta)=0. Use one column directly, not the displayed quotient.")
    mixed = (F(39, 50), F(1, 10), F(1, 5), F(1, 5))
    mixed_D = ((F(1), F(3, 5), F(4, 5), F(0)), (F(0), F(4, 5), F(3, 5), F(1)))
    checked("mixed_angles_are_valid_extension", all(dot(row, mixed) == rhs for row, rhs in zip(mixed_D, t)) and mixed[1] > 0 and mixed[2] > 0)
    checked("one_angle_constraint_cannot_be_omitted", mixed[1] * mixed[2] != 0,
            "Two distinct positive angle columns at the same degree violate the old one-angle rule.")
    checked("mixed_normalization_lower_bound", sum(mixed) ** 2 >= t[0] ** 2 + t[1] ** 2)

    rho_residual, y_test = F(1, 1000), F(2, 3)
    checked("residual_bias_rescaling", rho_residual / y_test == F(3, 2000))
    checked("rounding_probabilities_needs_rechecking", sum((F(58, 100), F(42, 100))) == 1 and F(58, 100) != q[0])
    checked("workspace_is_not_expected_capacity", F(1, 100) * 100 == 1 and 100 > 2,
            "A rare workspace-100 column has expected workspace 1 but violates a peak-workspace cap 2.")
    checked("grid_core_confidence_monotonic", shot_condition(4, F(1), F(1)) and shot_condition(8, F(1), F(1)))
    checked("grid_unqualified_hard_cap_counterexample", 2 * 4 <= 8 < 2 * 8,
            {"n": 4, "rounded_grid_n": 8, "ratio_bound": 2, "per_shot_cost": 1,
             "two_axis_total_cost_before": 8, "after": 16, "unchanged_total_cap": 8,
             "interpretation": "Factor-r approximation holds, unchanged hard-cap feasibility does not."})
    costs, weights, sampling = (F(1), F(4)), (F(1), F(1)), (F(2, 3), F(1, 3))
    V = sum(w * w / r for w, r in zip(weights, sampling)); expected_cost = dot(costs, sampling)
    checked("known_IS_leading_optimum_identity", V * expected_cost == (weights[0] + 2 * weights[1]) ** 2 == 9)
    checked("adjoint_pair_control_zero_identity", inverse_unitary_word(("A_dag", "A")) == [])
    checked("independent_pair_zero_branch_mutant_detected", inverse_unitary_word(("B_dag", "A")) != [])
    checked("adjoint_pair_one_branch_word", inverse_unitary_word(("X", "A_dag", "X", "A")) == ["X", "A_dag", "X", "A"])
    checked("channel_mean_vs_coherent_mean_mutant_detected", F(1, 2) * 1 + F(1, 2) * (-1) == 0
            and F(1, 2) * 1 ** 2 + F(1, 2) * (-1) ** 2 == 1,
            "I/-I have zero first operator mean and the same identity channel; no matrix evaluation.")
    result = {**meta, "status": "ALL_FIXED_BOOKKEEPING_CHECKS_PASSED", "checks": checks,
              "check_count": len(checks), "science_runs": 0, "synthesis_calls": 0,
              "LP_or_SOCP_solver_calls": 0, "registered_R1_or_R1p5_resource_scoring": 0,
              "matrix_or_circuit_evaluations": 0, "proof_scope": "finite off-domain algebra witnesses; general proofs are in the audit document",
              "mandatory_STOP": True, "next_stage_authorized": False}
    with (OUT / "bookkeeping_checks_v1.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2); stream.write("\n")
    print(json.dumps({"checks": len(checks), "status": result["status"], "science_runs": 0}))


if __name__ == "__main__":
    main()
