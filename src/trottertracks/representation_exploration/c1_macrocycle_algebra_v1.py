"""Exact symbolic C1 preparation. No matrices, eigensolvers or science inputs."""
from fractions import Fraction as F
from itertools import permutations
from math import factorial

REPRESENTATIVES = (("A", "B", "C"), ("A", "C", "B"), ("B", "A", "C"))
WEIGHTS = (F(1, 8), F(1, 4), F(1, 3), F(3, 8))


def add(*polys):
    out = {}
    for poly in polys:
        for word, value in poly.items():
            out[word] = out.get(word, F(0)) + value
    return {word: value for word, value in out.items() if value}


def scale(poly, value):
    return {word: F(value) * x for word, x in poly.items() if value * x}


def multiply(left, right, degree=6):
    out = {}
    for x, a in left.items():
        for y, b in right.items():
            if len(x) + len(y) <= degree:
                out[x + y] = out.get(x + y, F(0)) + a * b
    return {word: value for word, value in out.items() if value}


def commutator(left, right):
    return add(multiply(left, right), scale(multiply(right, left), -1))


def symbol(name):
    return {(name,): F(1)}


def exponential(name, time, degree=6):
    return {(name,) * k: F(time) ** k / factorial(k) for k in range(degree + 1)}


def log_series(stages, degree=6):
    unitary = {(): F(1)}
    for name, time in stages:
        unitary = multiply(unitary, exponential(name, time, degree), degree)
    z = add(unitary, {(): F(-1)})
    result, power = {}, {(): F(1)}
    for k in range(1, degree + 1):
        power = multiply(power, z, degree)
        result = add(result, scale(power, F((-1) ** (k + 1), k)))
    return result


def kernel(order):
    if tuple(sorted(order)) != ("A", "B", "C"):
        raise ValueError("three distinct symbolic fragments required")
    a, b, c = order
    return ((a, F(1, 2)), (b, F(1, 2)), (c, F(1)),
            (b, F(1, 2)), (a, F(1, 2)))


def kernel_error(order):
    # log(prod exp(t A)) at cubic degree has opposite sign to physical D,
    # because (-i)^3=i and log S=-i h H-i h^3 D+... .
    return scale({w: v for w, v in log_series(kernel(order), 3).items()
                  if len(w) == 3}, -1)


def nested_error(order):
    # Independent recursive Strang formula, not an exp/log calculation.
    a, b, c = map(symbol, order)
    y = add(b, c)
    return add(scale(commutator(a, commutator(a, y)), F(1, 24)),
               scale(commutator(y, commutator(a, y)), F(1, 12)),
               scale(commutator(b, commutator(b, c)), F(1, 24)),
               scale(commutator(c, commutator(b, c)), F(1, 12)))


def canonical_stages(stages):
    """Only adjacent equal-generator merges; no noncommuting reordering."""
    out = []
    for name, value in stages:
        value = F(value)
        if not value:
            continue
        if out and out[-1][0] == name:
            value += out.pop()[1]
        if value:
            out.append((name, value))
    return tuple(out)


def macrocycle(j, k, a):
    if j == k or j not in range(3) or k not in range(3) or F(a) not in WEIGHTS:
        raise ValueError("fixed off-oracle macrocycle catalog required")
    a = F(a)
    raw = []
    for index, time in ((j, a), (k, 1 - 2 * a), (j, a)):
        raw.extend((name, time * value) for name, value in kernel(REPRESENTATIVES[index]))
    return tuple(raw)


def macro_error(j, k, a):
    a = F(a)
    return add(scale(kernel_error(REPRESENTATIVES[j]), 2 * a ** 3),
               scale(kernel_error(REPRESENTATIVES[k]), (1 - 2 * a) ** 3))


def catalog():
    return [{"id": f"M{j}{k}_{a.numerator}_{a.denominator}", "j": j, "k": k,
             "a": str(a), "b": str(1 - 2 * a)}
            for j in range(3) for k in range(3) if j != k for a in WEIGHTS]


def encode(poly):
    return [{"word": list(w), "coefficient": str(v)} for w, v in sorted(poly.items())]


def preparation_proofs():
    """Free-algebra identities only. No registered parameter substitution."""
    records = []
    for order in permutations(("A", "B", "C")):
        log = log_series(kernel(order), 4)
        assert {w: v for w, v in log.items() if len(w) == 1} == add(*map(symbol, order))
        assert not any(len(w) in (2, 4) for w in log)
        assert kernel_error(order) == nested_error(order)
        records.append({"order": list(order), "D": encode(kernel_error(order)),
                        "recursive_Strang_match": True,
                        "reverse_order_isospectral_partner": list(reversed(order))})
    for row in catalog():
        j, k, a = row["j"], row["k"], F(row["a"])
        stages = macrocycle(j, k, a)
        assert tuple(reversed(stages)) == stages
        assert {name: sum(v for n, v in stages if n == name)
                for name in ("A", "B", "C")} == {"A": 1, "B": 1, "C": 1}
        cubic = scale({w: v for w, v in log_series(stages, 3).items() if len(w) == 3}, -1)
        assert cubic == macro_error(j, k, a)
        assert log_series(canonical_stages(stages), 3) == log_series(stages, 3)
    return {"kind": "EXACT_FREE_ALGEBRA_PREPARATION", "kernels": records,
            "retained_representatives": [list(x) for x in REPRESENTATIVES],
            "raw_permutations": 6, "known_spectral_classes": 3,
            "macro_catalog_count": len(catalog()),
            "registered_Hamiltonian_substitutions": 0,
            "eigenvalue_or_PF_performance_evaluations": 0,
            "new_novelty_claim": False}
