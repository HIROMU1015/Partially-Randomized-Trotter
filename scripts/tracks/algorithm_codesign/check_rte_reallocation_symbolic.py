#!/usr/bin/env python3
"""R0 exact symbolic checks, not a science runner or RTE sampler.

Only Fraction arithmetic in the free product Q_i^2=I is used. No matrices,
random sampling, trotterlib, Hamiltonian input, solver or circuit operations.
The fixed fixtures and resource limits are part of this checker source.
"""
from fractions import Fraction as F
from itertools import product
from math import factorial, isqrt
import hashlib
import json
import resource
import sys
import time


DEGREES = (1, 3, 5, 7)
TIMES = (F(1, 8), F(1), F(8))
SIGNS = (-1, 1)
ETAS = (F(0), F(1, 2), F(1))
PROBABILITIES = (F(1, 2), F(1, 3), F(1, 6))
ZERO = (F(0), F(0))
ONE = (F(1), F(0))
SCALE = 1 << 96


def cmul(z, w):
    return z[0] * w[0] - z[1] * w[1], z[0] * w[1] + z[1] * w[0]


def phase(k, sigma):
    z = ONE
    for _ in range(k):
        z = cmul(z, (F(0), F(-sigma)))
    return z


def reduce_word(word):
    out = []
    for letter in word:
        if out and out[-1] == letter:
            out.pop()
        else:
            out.append(letter)
    return tuple(out)


def accumulate(poly, word, value):
    if value == ZERO:
        return
    word = reduce_word(word)
    old = poly.get(word, ZERO)
    new = old[0] + value[0], old[1] + value[1]
    if new != ZERO:
        poly[word] = new
    else:
        poly.pop(word, None)


def add(*polys):
    out = {}
    for poly in polys:
        for word, z in poly.items():
            accumulate(out, word, z)
    return out


def scale(poly, z):
    out = {}
    for word, value in poly.items():
        accumulate(out, word, cmul(z, value))
    return out


def mul(left, right):
    out = {}
    for lword, lz in left.items():
        for rword, rz in right.items():
            accumulate(out, lword + rword, cmul(lz, rz))
    return out


def powers(probabilities, maximum):
    r = {(j,): (p, F(0)) for j, p in enumerate(probabilities) if p}
    out = [{(): ONE}]
    for _ in range(maximum):
        out.append(mul(r, out[-1]))
    return out


def direct_taylor(probabilities, x, m, sigma):
    # Independent reference: literal ordered free words, not a matrix or event API.
    out = {}
    for n in range(m + 1):
        coefficient = x ** n / factorial(n)
        for word in product(range(len(probabilities)), repeat=n):
            weight = coefficient
            for letter in word:
                weight *= probabilities[letter]
            accumulate(out, word, cmul((weight, F(0)), phase(n, sigma)))
    return out


def paired_and_optimal(x, m):
    t = [x ** n / factorial(n) for n in range(m + 1)]
    paired = [(t[k], t[k + 1]) if k % 2 == 0 else (F(0), F(0))
              for k in range(m + 1)]
    if x == 0:
        return t, paired, paired, F(0)
    rho = sum(t[1::2]) / sum(t[::2])
    optimum = []
    e, o = F(0), F(0)
    for j in range((m + 1) // 2):
        e += t[2 * j]
        optimum.append((e - o / rho, rho * e - o))
        o += t[2 * j + 1]
        optimum.append((o - rho * e, o / rho - e))
    return t, paired, optimum, rho


def ensemble_mean(coefficients, rpowers, sigma, omit_odd_phase=False):
    out = {}
    for k, (a, b) in enumerate(coefficients):
        if a == b == 0:
            continue
        unnormalized = add(scale(rpowers[k], (a, F(0))),
                           scale(rpowers[k + 1], (F(0), -sigma * b)))
        prefactor = ONE if omit_odd_phase and k % 2 else phase(k, sigma)
        out = add(out, scale(unnormalized, prefactor))
    return out


def norm_interval(coefficients):
    low, high = F(0), F(0)
    for a, b in coefficients:
        squared = a * a + b * b
        n = isqrt(squared.numerator * SCALE * SCALE // squared.denominator)
        lower = F(n, SCALE)
        upper = lower if lower * lower == squared else F(n + 1, SCALE)
        low += lower
        high += upper
    return low, high


def digest(poly):
    rows = [[list(w), str(z[0]), str(z[1])] for w, z in sorted(poly.items())]
    return hashlib.sha256(json.dumps(rows, separators=(',', ':')).encode()).hexdigest()


def sqrt_series(squared, degree):
    # Formal Taylor square-root coefficients, no numerical fit.
    out = [F(1)]
    for n in range(1, degree + 1):
        cross = sum(out[j] * out[n - j] for j in range(1, n))
        out.append((squared.get(n, F(0)) - cross) / 2)
    return out


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (15, 15))
    resource.setrlimit(resource.RLIMIT_AS, (256 << 20, 256 << 20))
    start = time.monotonic()
    rows, witnesses = [], []
    rpowers = powers(PROBABILITIES, max(DEGREES) + 1)
    for m in DEGREES:
        for x in TIMES:
            t, paired, optimal, rho = paired_and_optimal(x, m)
            assert all(a >= 0 and b >= 0 for a, b in optimal)
            assert optimal[-1] == (0, 0)
            previous_b = F(0)
            recursive_a = F(1)
            for k, (a, b) in enumerate(optimal):
                assert a + previous_b == t[k]
                assert recursive_a == a
                assert b == (rho * a if k % 2 == 0 else a / rho)
                previous_b = b
                if k < m:
                    recursive_a = t[k + 1] - b
            e, o = sum(t[::2]), sum(t[1::2])
            parallel_scale = sum(a if k % 2 == 0 else b for k, (a, b) in enumerate(optimal))
            assert parallel_scale == e
            assert parallel_scale ** 2 * (1 + rho ** 2) == e ** 2 + o ** 2
            if m == 3:
                assert optimal[1] == (2 * x ** 3 / (3 * (x ** 2 + 2)), 2 * x ** 2 / (x ** 2 + 6))
                assert optimal[2][0] == x ** 2 * (x ** 2 + 2) / (2 * (x ** 2 + 6))
                assert (1 + x ** 2) * (1 + x ** 2 / 9) - (1 + x ** 2 / 3) ** 2 == 4 * x ** 2 / 9
            for eta in ETAS:
                coefficients = [((1 - eta) * a + eta * c, (1 - eta) * b + eta * d)
                                for (a, b), (c, d) in zip(paired, optimal)]
                ni = norm_interval(coefficients)
                if coefficients != paired:
                    assert ni[1] < norm_interval(paired)[0]
                for sigma in SIGNS:
                    actual = ensemble_mean(coefficients, rpowers, sigma)
                    expected = direct_taylor(PROBABILITIES, x, m, sigma)
                    assert actual == expected
                    rows.append({'m': m, 'x': str(x), 'sigma': sigma, 'eta': str(eta),
                                 'reduced_free_word_terms': len(actual), 'mean_sha256': digest(actual),
                                 'matching': True, 'scope': 'exact symbolic fixture, no resource outcome'})
    for m in DEGREES:
        t, paired, optimal, _ = paired_and_optimal(F(0), m)
        for sigma in SIGNS:
            assert ensemble_mean(optimal, rpowers, sigma) == {(): ONE}
            rows.append({'m': m, 'x': '0', 'sigma': sigma, 'eta': None, 'matching': True,
                         'scope': 'identity endpoint, no division by rho'})

    _, _, opt, _ = paired_and_optimal(F(1), 3)
    expected = direct_taylor(PROBABILITIES, F(1), 3, 1)
    assert ensemble_mean(opt, rpowers, 1, omit_odd_phase=True) != expected
    witnesses.append({'mutation': 'omit odd (-i sigma)^k phase', 'rejected': True})
    bad = opt.copy(); bad[-1] = (F(0), F(1, 7))
    assert ensemble_mean(bad, rpowers, 1) != expected
    witnesses.append({'mutation': 'nonzero terminal b_m creates degree m+1', 'rejected': True})
    a, b = F(2, 5), F(3, 7)
    ordered = {(2, 1): (a, F(0)), (0, 2, 1): (F(0), -b)}
    swapped = {(2, 1): (a, F(0)), (2, 1, 0): (F(0), -b)}
    assert ordered != swapped
    witnesses.append({'mutation': 'move rotation across noncommuting branch word', 'rejected': True,
                      'note': 'IID mean matching alone cannot detect this branch-order change'})
    for sigma in SIGNS:
        q = {(0,): ONE}; word = {(2, 1): ONE}
        left = scale(mul({(): (a, F(0)), (0,): (F(0), -sigma * b)}, word), phase(1, sigma))
        right = scale(mul(q, mul({(): (b, F(0)), (0,): (F(0), sigma * a)}, word)), phase(2, sigma))
        assert left == right
        # On joint diagonal blocks Ctrl(-U) differs from -Ctrl(U) at 00.
        ctrl_minus = {('00', ()): ONE, ('11', (0, 1)): (F(-1), F(0))}
        minus_ctrl = {('00', ()): (F(-1), F(0)), ('11', (0, 1)): (F(-1), F(0))}
        assert ctrl_minus != minus_ctrl
    witnesses.append({'mutation': 'drop relative controlled phase as system global phase', 'rejected': True})

    pair_s = sqrt_series({2: F(1)}, 6)
    second = sqrt_series({2: F(1, 9)}, 6)
    pair_s = [v + (second[n - 2] / 2 if n >= 2 else 0) for n, v in enumerate(pair_s)]
    opt_s = sqrt_series({2: F(2), 4: F(7, 12), 6: F(1, 36)}, 6)
    assert [pair_s[n] - opt_s[n] for n in range(5)] == [0, 0, 0, 0, F(1, 9)]
    assert pair_s[6] - opt_s[6] == F(-13, 81)

    b_rows = []
    for probabilities in (PROBABILITIES, (F(99, 100), F(1, 100), F(0)), (F(1), F(0), F(0))):
        ps = powers(probabilities, 3)
        s2 = sum(p * p for p in probabilities)
        d = add(ps[2], {(): (-s2, F(0))})
        for x in TIMES:
            aa, bb = 1 - s2 * x * x / 2, x - s2 * x ** 3 / 6
            for sigma in SIGNS:
                recovered = add({(): (aa, F(0))}, scale(ps[1], (F(0), -sigma * bb)),
                                scale(mul(add(ps[0], scale(ps[1], (F(0), -sigma * x / 3))), d), (-x * x / 2, F(0))))
                assert recovered == direct_taylor(probabilities, x, 3, sigma)
            if s2 == 1:
                pair_probabilities = []
                assert d == {}
            else:
                pair_probabilities = []
                for i, pi in enumerate(probabilities):
                    first = pi * (1 - pi) / (1 - s2)
                    if first == 0:
                        continue
                    conditional_sum = F(0)
                    for j, pj in enumerate(probabilities):
                        if i == j or pj == 0:
                            continue
                        conditional = pj / (1 - pi)
                        conditional_sum += conditional
                        assert first * conditional == pi * pj / (1 - s2)
                        pair_probabilities.append(first * conditional)
                    assert conditional_sum == 1
                assert sum(pair_probabilities) == 1
            b_rows.append({'p': [str(p) for p in probabilities], 'x': str(x), 's2': str(s2),
                           'mean_matching_both_signs': True, 'residual_support_pairs': len(pair_probabilities),
                           'sampling_calls': 0})
    # Counterexample only to *unrestricted* optimality, not to the proposed class.
    x = F(1)
    general_lcu_squared = (1 - x ** 2 / 2) ** 2 + (x - x ** 3 / 6) ** 2
    class_opt_squared = (1 + x ** 2 / 2) ** 2 + (x + x ** 3 / 6) ** 2
    assert general_lcu_squared == F(17, 18) and class_opt_squared == F(65, 18)
    assert general_lcu_squared < class_opt_squared
    witnesses.append({'overclaim': 'B_* is optimal over arbitrary LCU', 'counterexample': 'single involution, x=1',
                      'alternative_B_squared': '17/18', 'adjacent_nonnegative_B_squared': '65/18'})

    # Scalar P3 modulus squared; independent real/imag multiplication.
    real = {0: F(1), 2: F(-1, 2)}; imag = {1: F(-1), 3: F(1, 6)}
    sq = {}
    for poly in (real, imag):
        for i, v in poly.items():
            for j, w in poly.items():
                sq[i + j] = sq.get(i + j, F(0)) + v * w
    assert {n: v for n, v in sq.items() if v} == {0: F(1), 4: F(-1, 12), 6: F(1, 36)}
    runtime = time.monotonic() - start
    assert runtime < 30
    output = {'schema': 'track_b_rte_reallocation_r0_exact_symbolic_result_v1',
              'status': 'PASS_FIXED_EXACT_SYMBOLIC_FIXTURES_NOT_SCIENCE',
              'A_mean_fixtures': rows, 'A_mean_fixture_count': len(rows),
              'B_identity_probability_fixtures': b_rows, 'B_fixture_count': len(b_rows),
              'negative_witnesses': witnesses,
              'formal_small_x_delta': {'x4': '1/9', 'x6': '-13/81'},
              'P3_modulus_squared': '1-y^4/12+y^6/36',
              'source_sha256': hashlib.sha256(open(__file__, 'rb').read()).hexdigest(),
              'python': sys.version, 'CPU_cap_seconds': 15, 'address_space_cap_MiB': 256,
              'wall_cap_seconds': 30, 'technical_runtime_seconds': runtime,
              'technical_peak_RSS_KiB_linux': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'symbolic_checker_runs': 1, 'science_runs': 0, 'sampling_calls': 0,
              'matrix_solver_synthesis_compile_GPU_NPZ_Hamiltonian_calls': 0,
              'general_proof_is_separate_from_finite_fixtures': True,
              'new_method_priority_or_real_resource_improvement_established': False,
              'RUN_READY': False, 'science_execution_authorized': False, 'mandatory_STOP': True}
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
