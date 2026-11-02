#!/usr/bin/env python3
"""BM-0.5 formal word audit, truncated at degree three.

Standard-library Fraction arithmetic on abstract noncommuting letters only.
No matrices, Hamiltonians, states, science inputs, circuits, or physical provider.
Three independent constructions: formal exp/log product; Maxwell Eq. (10)/(19)
recursion; BM floor + internal decomposition. This is a finite algebra check,
not a proof for all group sizes or a finite-time error certificate.
"""

from fractions import Fraction as Q
from hashlib import sha256
import json
from math import factorial

DEGREE = 3


def add(*polys):
    out = {}
    for poly in polys:
        for word, value in poly.items():
            out[word] = out.get(word, Q(0)) + value
    return {word: value for word, value in out.items() if value}


def scale(poly, coefficient):
    return {word: value * coefficient for word, value in poly.items()
            if value * coefficient}


def mul(left, right):
    out = {}
    for word_l, value_l in left.items():
        for word_r, value_r in right.items():
            word = word_l + word_r
            if len(word) <= DEGREE:
                out[word] = out.get(word, Q(0)) + value_l * value_r
    return {word: value for word, value in out.items() if value}


def comm(left, right):
    return add(mul(left, right), scale(mul(right, left), -1))


def letter(name):
    return {(name,): Q(1)}


def exp_series(poly):
    assert () not in poly
    power = {(): Q(1)}
    out = power.copy()
    for k in range(1, DEGREE + 1):
        power = mul(power, poly)
        out = add(out, scale(power, Q(1, factorial(k))))
    return out


def log_series(product):
    assert product.get(()) == 1
    delta = add(product, {(): Q(-1)})
    power, out = {(): Q(1)}, {}
    for k in range(1, DEGREE + 1):
        power = mul(power, delta)
        out = add(out, scale(power, Q((-1) ** (k + 1), k)))
    return out


def product(*factors):
    out = {(): Q(1)}
    for factor in factors:
        out = mul(out, factor)
    return out


def power_product(poly, count):
    return product(*([poly] * count))


def sweep_product(names, u):
    # Literal forward/reverse half sweeps, with no BCH formula.
    halves = [exp_series(scale(letter(name), u / 2)) for name in names]
    return product(*(halves + halves[::-1]))


def literal_nested_log(a_names, b_names, m):
    a_half = power_product(sweep_product(a_names, Q(1, 2 * m)), m)
    b_halves = [exp_series(scale(letter(name), Q(1, 2))) for name in b_names]
    p = product(*(b_halves + [a_half, exp_series(letter('R')), a_half]
                  + b_halves[::-1]))
    return log_series(p)


def maxwell_wrap(inner, outer):
    # Maxwell convention: BCH(inner, outer) = log(e^(outer/2)e^inner e^(outer/2)).
    bracket = comm(inner, outer)
    return add(inner, outer,
               scale(comm(bracket, inner), Q(-1, 12)),
               scale(comm(bracket, outer), Q(-1, 24)))


def maxwell_sweep_log(names, u):
    current = scale(letter(names[-1]), u)
    for name in reversed(names[:-1]):
        current = maxwell_wrap(current, scale(letter(name), u))
    return current


def maxwell_nested_log(a_names, b_names, m):
    a_half_log = scale(maxwell_sweep_log(a_names, Q(1, 2 * m)), m)
    current = maxwell_wrap(letter('R'), scale(a_half_log, 2))
    for name in reversed(b_names):
        current = maxwell_wrap(current, letter(name))
    return current


def bm_c(outer, inner):
    return add(scale(comm(outer, comm(outer, inner)), Q(-1, 24)),
               scale(comm(inner, comm(inner, outer)), Q(1, 12)))


def bm_components(a_names, b_names):
    a = add(*(letter(name) for name in a_names))
    b = add(*(letter(name) for name in b_names))
    ka = add(*(bm_c(letter(name), add(*(letter(n) for n in a_names[i + 1:])))
               for i, name in enumerate(a_names[:-1])))
    floor = add(bm_c(a, letter('R')),
                *(bm_c(letter(name), add(a, letter('R'),
                                        *(letter(n) for n in b_names[i + 1:])))
                  for i, name in enumerate(b_names)))
    return add(a, b, letter('R')), floor, ka


def encoded(poly):
    return [[list(word), str(value)] for word, value in sorted(poly.items())]


def require_equal(left, right, label):
    residual = add(left, scale(right, -1))
    if residual:
        raise AssertionError((label, encoded(residual)))


def main():
    fixtures = []
    mutations = 0
    for a_count, b_count in ((1, 1), (2, 1), (2, 2)):
        aa = [f'A{i + 1}' for i in range(a_count)]
        bb = [f'B{i + 1}' for i in range(b_count)]
        linear, floor, ka = bm_components(aa, bb)
        for m in (1, 2, 4):
            literal = literal_nested_log(aa, bb, m)
            compact = maxwell_nested_log(aa, bb, m)
            bm = add(linear, floor, scale(ka, Q(1, 4 * m * m)))
            require_equal(literal, compact, 'literal_vs_compact')
            require_equal(compact, bm, 'compact_vs_bm')
            assert not any(len(word) == 2 for word in literal)
            # Sensitivity control: the full-interval coefficient is wrong here.
            mutation_detected = None
            if ka:
                wrong = add(linear, floor, scale(ka, Q(1, m * m)))
                mutation_detected = bool(add(literal, scale(wrong, -1)))
                assert mutation_detected
                mutations += 1
            fixtures.append({
                'a': a_count, 'b': b_count, 'm': m,
                'degree': DEGREE, 'all_exact_residuals_zero': True,
                'word_count': len(literal),
                'common_polynomial_sha256': sha256(
                    json.dumps(encoded(literal), separators=(',', ':')).encode()).hexdigest(),
                'wrong_internal_factor_detected': mutation_detected,
            })
    print(json.dumps({
        'schema': 'track_b.bm05.formal_word_audit.v1',
        'scope': 'ABSTRACT_FREE_WORDS_ONLY_NO_PHYSICAL_EVALUATION',
        'maxwell_source': 'https://arxiv.org/html/2606.30738v1',
        'maxwell_locators': ['III.2.1 Eq.(10)', 'III.2.3 Eq.(19)'],
        'base_bm0_commit': '3b2d624adde979f7d8f983bc7fdbbec88f90c500',
        'fixtures': fixtures, 'fixture_count': len(fixtures),
        'factor_mutation_controls_passed': mutations,
        'result': 'EXACT_EQUALITY_THROUGH_DEGREE_3_IN_ALL_REGISTERED_FIXTURES',
        'general_size_proof': 'See bm05_equivalence_and_method_delta_audit_v1.md induction.',
        'physical_matrix_evaluations': 0, 'science_runs': 0,
        'BM1_authorized': False, 'mandatory_stop': True,
    }, indent=2))


if __name__ == '__main__':
    main()
