#!/usr/bin/env python3
"""R0.5 fixed symbolic equivalence comparisons; no science/sampling/matrices.

The A construction is taken from the frozen R0 proof, not optimized again.
Free involution words and I1 one-qubit Pauli words are separate representations.
The padded CTS arm uses the explicit Hermitian-zero-sum rephasing documented
in the R0.5 audit; it is not asserted to be CTS with general I0 access.
"""
from fractions import Fraction as F
from itertools import product
from math import factorial, isqrt
from decimal import Decimal, localcontext
import hashlib
import json
import resource
import sys
import time

DEGREES = (3, 5, 7)
TIMES = (F(1, 8), F(1, 4), F(1))
SIGNS = (-1, 1)
P = (F(1, 2), F(1, 3), F(1, 6))
Z = (F(0), F(0))
ONE = (F(1), F(0))
SCALE = 1 << 96
PAULIS = ('X', 'Y', 'Z')
PAIR_TABLE = {
    ('X', 'Y'): ('Z', (0, 1)), ('Y', 'X'): ('Z', (0, -1)),
    ('Y', 'Z'): ('X', (0, 1)), ('Z', 'Y'): ('X', (0, -1)),
    ('Z', 'X'): ('Y', (0, 1)), ('X', 'Z'): ('Y', (0, -1)),
}


def cmul(a, b):
    return a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]


def phase(n, sigma):
    out = ONE
    for _ in range(n):
        out = cmul(out, (F(0), F(-sigma)))
    return out


def reduce_word(word):
    out = []
    for j in word:
        if out and out[-1] == j:
            out.pop()
        else:
            out.append(j)
    return tuple(out)


def pmul(left, right):
    if left == 'I':
        return right, ONE
    if right == 'I':
        return left, ONE
    if left == right:
        return 'I', ONE
    axis, z = PAIR_TABLE[(left, right)]
    return axis, (F(z[0]), F(z[1]))


def inc(poly, key, value):
    prev = poly.get(key, Z)
    new = prev[0] + value[0], prev[1] + value[1]
    if new == Z:
        poly.pop(key, None)
    else:
        poly[key] = new


def add(*polys):
    out = {}
    for poly in polys:
        for key, value in poly.items():
            inc(out, key, value)
    return out


def scale(poly, z):
    out = {}
    for key, value in poly.items():
        inc(out, key, cmul(value, z))
    return out


def mul(left, right, pauli=False):
    out = {}
    for a, za in left.items():
        for b, zb in right.items():
            key, extra = pmul(a, b) if pauli else (reduce_word(a + b), ONE)
            inc(out, key, cmul(cmul(za, zb), extra))
    return out


def powers(pauli=False):
    identity = 'I' if pauli else ()
    operator = {PAULIS[j] if pauli else (j,): (p, F(0)) for j, p in enumerate(P)}
    out = [{identity: ONE}]
    for _ in range(max(DEGREES)):
        out.append(mul(operator, out[-1], pauli))
    return out


def candidate(x, m):
    t = [x ** n / factorial(n) for n in range(m + 1)]
    e, o = sum(t[::2]), sum(t[1::2])
    rho = o / e
    ordinary = [(t[k], t[k + 1]) if k % 2 == 0 else (F(0), F(0))
                for k in range(m + 1)]
    optimum = []
    ep, op = F(0), F(0)
    for j in range((m + 1) // 2):
        ep += t[2 * j]
        optimum.append((ep - op / rho, rho * ep - op))
        op += t[2 * j + 1]
        optimum.append((op - rho * ep, op / rho - ep))
    return t, ordinary, optimum, e, o


def sqrt_interval(q):
    n = isqrt(q.numerator * SCALE * SCALE // q.denominator)
    lo = F(n, SCALE)
    hi = lo if lo * lo == q else F(n + 1, SCALE)
    return lo, hi


def interval_sum(*intervals):
    return sum(v[0] for v in intervals), sum(v[1] for v in intervals)


def record_interval(pair):
    with localcontext() as context:
        context.prec = 40
        mid = (pair[0] + pair[1]) / 2
        display = str(Decimal(mid.numerator) / Decimal(mid.denominator))
    return {'lower_exact': str(pair[0]), 'upper_exact': str(pair[1]),
            'midpoint_display_not_decision_value': display}


def relation(a, b):
    if a[1] < b[0]:
        return 'LESS'
    if a[0] > b[1]:
        return 'GREATER'
    return 'INTERVAL_OVERLAP'


def signature(poly):
    positive_scale = sum(abs(z[0]) + abs(z[1]) for z in poly.values())
    assert positive_scale > 0
    return tuple((axis, str(z[0] / positive_scale), str(z[1] / positive_scale))
                 for axis, z in sorted(poly.items()))


def digest(poly):
    rows = [(str(k), str(z[0]), str(z[1])) for k, z in sorted(poly.items())]
    return hashlib.sha256(json.dumps(rows, separators=(',', ':')).encode()).hexdigest()


def word_laws():
    # Small authorized technical fixture enumeration, not I0 algorithm preprocessing.
    out = [[('I', ONE, F(1))]]
    for n in range(1, max(DEGREES) + 1):
        rows = []
        for word in product(range(3), repeat=n):
            axis, z, probability = 'I', ONE, F(1)
            for j in word:
                axis, factor = pmul(axis, PAULIS[j])
                z = cmul(z, factor)
                probability *= P[j]
            rows.append((axis, z, probability))
        assert sum(row[2] for row in rows) == 1
        out.append(rows)
    return out


def A_pauli(coefficients, sigma, laws):
    mean, atoms, labelled = {}, set(), 0
    for k, (a, b) in enumerate(coefficients):
        if a == b == 0:
            continue
        for axis, z, probability in laws[k]:
            for j, pj in enumerate(P):
                atom = {axis: cmul(phase(k, sigma), cmul((a, F(0)), z))}
                result_axis, extra = pmul(PAULIS[j], axis)
                inc(atom, result_axis, cmul(phase(k, sigma), cmul((F(0), -sigma * b), cmul(z, extra))))
                mean = add(mean, scale(atom, (probability * pj, F(0))))
                atoms.add(signature(atom)); labelled += 1
    return mean, {'labelled_before_identical_atom_merge': labelled,
                  'distinct_phase_preserving_atoms': len(atoms)}


def ptsc_pauli(t, sigma, laws):
    mean, atoms, labelled = {}, set(), 0
    for j, pj in enumerate(P):
        atom = {'I': ONE, PAULIS[j]: (F(0), -sigma * t[1])}
        mean = add(mean, scale(atom, (pj, F(0))))
        atoms.add(signature(atom)); labelled += 1
    for n in range(2, len(t)):
        for axis, z, probability in laws[n]:
            atom = {axis: cmul(phase(n, sigma), z)}
            mean = add(mean, scale(atom, (t[n] * probability, F(0))))
            atoms.add(signature(atom)); labelled += 1
    return mean, {'labelled_before_identical_atom_merge': labelled,
                  'distinct_phase_preserving_atoms': len(atoms)}


def padded_cts(t, o, sigma, laws):
    mean, atoms, labelled = {}, set(), 0
    for n in range(1, len(t)):
        for axis, z, probability in laws[n]:
            # Rephase all anti-Hermitian words by the SAME -i. Their sum is zero.
            hermitian_sign = z[0] if z[1] == 0 else z[1]
            assert hermitian_sign in (-1, 1)
            if n % 2 == 0:
                atom = {axis: (F((-1) ** (n // 2)) * hermitian_sign, F(0))}
                mass = t[n] * probability
            else:
                signed_axis = -sigma * ((-1) ** ((n - 1) // 2)) * hermitian_sign
                atom = {'I': ONE}
                inc(atom, axis, (F(0), o * signed_axis))
                mass = t[n] * probability / o
            mean = add(mean, scale(atom, (mass, F(0))))
            atoms.add(signature(atom)); labelled += 1
    return mean, {'labelled_before_identical_atom_merge': labelled,
                  'distinct_phase_preserving_atoms': len(atoms),
                  'access': 'I1 per-word Pauli closure/rephasing; full coefficient collection not needed'}


def collected_cts(t, sigma, ppowers):
    even = add(*(scale(ppowers[n], cmul((t[n], F(0)), phase(n, sigma))) for n in range(2, len(t), 2)))
    odd = add(*(scale(ppowers[n], cmul((t[n], F(0)), phase(n, sigma))) for n in range(1, len(t), 2)))
    assert all(z[1] == 0 for z in even.values()) and all(z[0] == 0 for z in odd.values())
    lc, ls = sum(abs(z[0]) for z in even.values()), sum(abs(z[1]) for z in odd.values())
    mean = dict(even); atoms = set()
    for axis, z in even.items():
        atoms.add(signature({axis: (F(1 if z[0] > 0 else -1), F(0))}))
    if ls:
        for axis, z in odd.items():
            signed = F(1 if z[1] > 0 else -1)
            atom = {'I': ONE}; inc(atom, axis, (F(0), ls * signed))
            mean = add(mean, scale(atom, (abs(z[1]) / ls, F(0))))
            atoms.add(signature(atom))
    else:
        mean = add(mean, {'I': ONE}); atoms.add(signature({'I': ONE}))
    return mean, lc, ls, {'labelled_nonzero_collected_atoms': len(even) + (len(odd) if ls else 1),
                          'distinct_phase_preserving_atoms': len(atoms),
                          'real_coefficients_M_minus_I': {k: str(z[0]) for k, z in sorted(even.items())},
                          'imaginary_coefficients': {k: str(z[1]) for k, z in sorted(odd.items())}}


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    resource.setrlimit(resource.RLIMIT_AS, (256 << 20, 256 << 20))
    start = time.monotonic()
    fpowers, ppowers, laws = powers(), powers(True), word_laws()
    # Independent literal Pauli word expansion and Hermitian-zero-sum rephasing.
    for n, law in enumerate(laws):
        literal, rephased = {}, {}
        for axis, z, probability in law:
            inc(literal, axis, cmul((probability, F(0)), z))
            signed = z[0] if z[1] == 0 else z[1]
            inc(rephased, axis, (probability * signed, F(0)))
        assert literal == ppowers[n] == rephased
    rows = []
    for m in DEGREES:
        for x in TIMES:
            t, ordinary, optimum, e, o = candidate(x, m)
            BA = sqrt_interval(e * e + o * o)
            BP = interval_sum(*(sqrt_interval(a * a + b * b) for a, b in ordinary))
            P0 = interval_sum(sqrt_interval(1 + x * x), (sum(t[2:]), sum(t[2:])))
            CF = interval_sum((e - 1, e - 1), sqrt_interval(1 + o * o))
            assert relation(BA, BP) == relation(BA, P0) == relation(BA, CF) == 'LESS'
            per_sign = []
            lc_ls = None
            for sigma in SIGNS:
                target_free = add(*(scale(fpowers[n], cmul((t[n], F(0)), phase(n, sigma))) for n in range(m + 1)))
                for coefficients in (ordinary, optimum):
                    recovered = {}
                    for k, (a, b) in enumerate(coefficients):
                        if a == b == 0:
                            continue
                        term = add(scale(fpowers[k], (a, F(0))), scale(fpowers[k + 1], (F(0), -sigma * b)))
                        recovered = add(recovered, scale(term, phase(k, sigma)))
                    assert recovered == target_free
                ptsc_free = add(fpowers[0], scale(fpowers[1], (F(0), -sigma * x)),
                                *(scale(fpowers[n], cmul((t[n], F(0)), phase(n, sigma))) for n in range(2, m + 1)))
                assert ptsc_free == target_free
                target_pauli = add(*(scale(ppowers[n], cmul((t[n], F(0)), phase(n, sigma))) for n in range(m + 1)))
                am, ordinary_support = A_pauli(ordinary, sigma, laws)
                bm, optimum_support = A_pauli(optimum, sigma, laws)
                zm, zsupport = ptsc_pauli(t, sigma, laws)
                fm, fsupport = padded_cts(t, o, sigma, laws)
                cm, lc, ls, csupport = collected_cts(t, sigma, ppowers)
                assert am == bm == zm == fm == cm == target_pauli
                assert lc_ls is None or lc_ls == (lc, ls)
                lc_ls = lc, ls
                per_sign.append({'sigma': sigma, 'all_same_target_means_match': True,
                                 'free_target_sha256': digest(target_free), 'Pauli_target_sha256': digest(target_pauli),
                                 'ordinary_support': ordinary_support, 'A_optimum_support': optimum_support,
                                 'PTSC_K0_support': zsupport, 'CTS_padded_per_word_support': fsupport,
                                 'CTS_collected_support': csupport})
            lc, ls = lc_ls
            CC = interval_sum((lc, lc), sqrt_interval(1 + ls * ls))
            rows.append({'m': m, 'x': str(x), 'E': str(e), 'O': str(o),
                         'A_optimum_coefficients': [[str(a), str(b)] for a, b in optimum],
                         'normalization': {'A_ordinary': record_interval(BP), 'A_optimum': record_interval(BA),
                                           'PTSC_K0_finite': record_interval(P0), 'CTS_free_formula_padded_I1_only': record_interval(CF),
                                           'CTS_collected_I1': record_interval(CC)},
                         'CTS_literal_collected_Lc': str(lc), 'CTS_literal_collected_Ls': str(ls),
                         'relations': {'A_vs_ordinary': relation(BA, BP), 'A_vs_PTSC_K0': relation(BA, P0),
                                       'A_vs_CTS_free_formula': relation(BA, CF), 'CTS_collected_vs_A': relation(CC, BA)},
                         'I0_CTS_free_formula_executable_for_general_involutions': False,
                         'per_sign': per_sign})
    # W=Q0 Q1 Q2 is neither Hermitian nor anti-Hermitian in the free involution algebra.
    W, Wdag = {(0, 1, 2): ONE}, {(2, 1, 0): ONE}
    assert W != Wdag and W != scale(Wdag, (F(-1), F(0)))
    atom = add({(): ONE}, scale(W, (F(0), F(-1))))
    adjoint = add({(): ONE}, scale(Wdag, (F(0), F(1))))
    residual = add(mul(adjoint, atom), {(): (F(-2), F(0))})
    assert residual != {}
    # Wan's +time ensemble, ADJOINTED, has the ordinary A orientation exactly.
    for sigma in SIGNS:
        a, b = F(1, 2), F(1, 6)
        expected = scale(add({(2, 1): (a, F(0))}, {(0, 2, 1): (F(0), -sigma * b)}), phase(2, sigma))
        known_dagger = scale(mul(add({(): (a, F(0))}, {(0,): (F(0), -sigma * b)}), {(2, 1): ONE}), phase(2, sigma))
        assert expected == known_dagger
    # A odd event with Q0=X and W=Z has Y and Z terms, no identity.
    _, _, optimal, _, _ = candidate(F(1), 3)
    a, b = optimal[1]; assert a > 0 and b > 0
    axis, extra = pmul('X', 'Z')
    atom = scale(add({'Z': (a, F(0))}, {axis: cmul((F(0), -b), extra)}), phase(1, 1))
    assert set(atom) == {'Y', 'Z'} and all(v != Z for v in atom.values())
    elapsed = time.monotonic() - start
    assert elapsed < 60
    result = {'schema':'track_b_rte_reallocation_r05_symbolic_comparison_v1',
              'status':'PASS_FIXED_EXACT_SYMBOLIC_COMPARISONS_NOT_SCIENCE',
              'normalization_domain_count':len(rows), 'sign_mean_comparison_count':len(rows)*2,
              'free_model':{'generators':3,'probabilities':[str(p) for p in P], 'relations_only':'Q_i^2=I'},
              'Pauli_fixture':{'axes':list(PAULIS),'probabilities':[str(p) for p in P],
                               'access':'I1; explicit one-qubit Pauli multiplication, no matrices'},
              'rows':rows,
              'semantic_witnesses':{'general_word_common_angle_shortcut_is_not_unitary':True,
                                    'word_neither_Hermitian_nor_antiHermitian':[[0,1,2],[2,1,0]],
                                    'unitarity_residual_sha256':digest(residual),
                                    'Wan_adjoint_ordinary_orientation_matches_both_signs':True,
                                    'A_odd_atom_not_literal_CTS_atom':True,
                                    'A_odd_atom_numerator':{k:[str(z[0]),str(z[1])] for k,z in sorted(atom.items())}},
              'comparison_set_extended_after_results':False,
              'source_sha256':hashlib.sha256(open(__file__,'rb').read()).hexdigest(),
              'python':sys.version, 'technical_runtime_seconds':elapsed,
              'technical_peak_RSS_KiB_linux':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'caps':{'CPU_seconds':30,'address_space_MiB':256,'wall_seconds':60},
              'technical_checker_runs':1,'science_runs':0,'random_sampling_calls':0,
              'synthesis_compile_matrix_solver_molecule_DF_NPZ_GPU_calls':0,
              'new_algorithm_world_priority_established':False,
              'RUN_READY':False,'science_execution_authorized':False,'mandatory_STOP':True}
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
