"""Generator-only CAR actions, raw BCH rows and exact JW Pauli algebra."""

from fractions import Fraction as F
from math import isqrt

from ..a2_spec01_03.common import complex_rat, content_id, integer, rat

ZERO = (F(0), F(0))
ONE = (F(1), F(0))


def add(x, y):
    return x[0] + y[0], x[1] + y[1]


def mul(x, y):
    return x[0] * y[0] - x[1] * y[1], x[0] * y[1] + x[1] * y[0]


def scale(x, a):
    return x[0] * a, x[1] * a


def accumulate(table, key, value):
    value = add(table.get(key, ZERO), value)
    if value == ZERO:
        table.pop(key, None)
    else:
        table[key] = value


def action_column(matrix, bits):
    column = {}
    for p in range(len(matrix)):
        for q in range(len(matrix)):
            if not (bits >> q) & 1:
                continue
            intermediate = bits ^ (1 << q)
            if (intermediate >> p) & 1:
                continue
            sign = (-1) ** ((bits & ((1 << q) - 1)).bit_count()
                            + (intermediate & ((1 << p) - 1)).bit_count())
            accumulate(column, intermediate | (1 << p), scale(matrix[p][q], sign))
    return column


def column_record(column):
    return [{"occupation_bits": str(k), "amplitude": complex_rat(*column[k])} for k in sorted(column)]


def upper(c, q, squares, occupied, lower):
    bits = sum(1 << integer(v, natural=True) for v in occupied)
    qc = action_column(q, bits)
    contributions, columns = [], [{"operator": "one_body", "column": column_record(qc)}]
    value = c + qc.get(bits, ZERO)[0]
    for name, weight, matrix in squares:
        col = action_column(matrix, bits)
        norm = sum((a * a + b * b for a, b in col.values()), F(0))
        value += weight * norm
        columns.append({"operator": "df:" + name, "column": column_record(col)})
        contributions.append({"block_id": name, "lambda": rat(weight), "action_norm_squared": rat(norm),
                              "weighted_expectation": rat(weight * norm)})
    return {"method": "CAR_BITPREFIX_COLUMN_NORM_V1", "columns": columns,
            "constant": rat(c), "q_expectation": rat(qc.get(bits, ZERO)[0]),
            "square_contributions": contributions, "U": rat(value), "L": rat(lower),
            "W": rat(value - lower), "unit": "HA"}


def k_sos(caps, labels, bits):
    count, coeffs = len(caps), {}
    raw_count = 0
    for j in range(count - 1):
        for k in range(j + 1, count):
            key = (j, j, k)
            coeffs[key] = coeffs.get(key, F(0)) + F(1, 24)
            raw_count += 1
        for k in range(j + 1, count):
            for ell in range(j + 1, count):
                key = (k, ell, j)
                coeffs[key] = coeffs.get(key, F(0)) - F(1, 12)
                raw_count += 1
    roots = {}

    def root(i, j):
        pair = tuple(sorted((i, j)))
        if pair not in roots:
            z = caps[pair[0]] * caps[pair[1]]
            grid = 1 << bits
            floor = isqrt((z.numerator * grid * grid) // z.denominator)
            ceil = floor if F(floor * floor, grid * grid) == z else floor + 1
            roots[pair] = {"positions": [str(v + 1) for v in pair], "z": rat(z), "u": rat(F(ceil, grid))}
        from ..a2_spec01_03.common import read_rat
        return read_rat(roots[pair]["u"])

    rows = [F(0)] * count
    for (a, b, c), coefficient in coeffs.items():
        x = abs(coefficient) * caps[b] * root(a, c)
        y = abs(coefficient) * caps[c] * root(a, b)
        rows[a] += x + y
        rows[c] += x
        rows[b] += y
    dictionary = [{"positions": [str(i + 1) for i in key], "kappa": rat(value)}
                  for key, value in sorted(coeffs.items())]
    witnesses = [roots[key] for key in sorted(roots)]
    return {"method": "K0_RAW_BCH_STREAMING_DYADIC_UPPER_V1",
            "formula_ref": "58d8672702284d73321cc67a26c4640adaddde5f:K0_4_EQ6_8_9",
            "position_to_fragment": [{"position": str(i + 1), "fragment_id": label, "f": rat(caps[i])}
                                     for i, label in enumerate(labels)],
            "sqrt_bits": str(bits), "raw_count": str(raw_count), "dictionary": dictionary,
            "dictionary_digest": content_id("k0_raw_bch_v1", {"terms": dictionary}),
            "sqrt_witnesses": witnesses,
            "sqrt_witness_digest": content_id("sqrt_witnesses_v1", {"items": witnesses}),
            "row_sums_upper": [rat(r) for r in rows], "K_SOS_upper": rat(max(rows, default=F(0))),
            "meaning": "UPPER_BOUND_ON_K0_ROW_SUM_NOT_EXACT_IRRATIONAL_VALUE", "unit": "HA2"}


_LOCAL = {
    "II": ((1, 0), "I"), "IX": ((1, 0), "X"), "IY": ((1, 0), "Y"), "IZ": ((1, 0), "Z"),
    "XI": ((1, 0), "X"), "XX": ((1, 0), "I"), "XY": ((0, 1), "Z"), "XZ": ((0, -1), "Y"),
    "YI": ((1, 0), "Y"), "YX": ((0, -1), "Z"), "YY": ((1, 0), "I"), "YZ": ((0, 1), "X"),
    "ZI": ((1, 0), "Z"), "ZX": ((0, 1), "Y"), "ZY": ((0, -1), "X"), "ZZ": ((1, 0), "I"),
}


def product(left, right):
    result = {}
    for p, x in left.items():
        for q, y in right.items():
            phase, letters = ONE, []
            for a, b in zip(p, q, strict=True):
                step, letter = _LOCAL[a + b]
                phase = mul(phase, step)
                letters.append(letter)
            accumulate(result, "".join(letters), mul(mul(x, y), phase))
    return result


def linear(left, right, factor=F(1)):
    out = dict(left)
    for key, val in right.items():
        accumulate(out, key, scale(val, factor))
    return out


def commutator(left, right):
    return linear(product(left, right), product(right, left), F(-1))


def one_body_pauli(matrix):
    n, operators = len(matrix), []
    for p in range(n):
        prefix = "Z" * p
        suffix = "I" * (n - p - 1)
        operators.append(({prefix + "X" + suffix: (F(1, 2), F(0)),
                           prefix + "Y" + suffix: (F(0), F(-1, 2))},
                          {prefix + "X" + suffix: (F(1, 2), F(0)),
                           prefix + "Y" + suffix: (F(0), F(1, 2))}))
    result = {}
    for p in range(n):
        for q in range(n):
            for key, val in product(operators[p][0], operators[q][1]).items():
                accumulate(result, key, mul(matrix[p][q], val))
    return result


def pauli_record(poly):
    return [{"pauli_mode0_left": key, "coefficient": complex_rat(*poly[key])} for key in sorted(poly)]


def hermitian_l1(poly):
    from ..a2_spec01_03.common import require
    require(all(value[1] == 0 for value in poly.values()), "NON_HERMITIAN", "commutator", "Real Pauli coefficients required")
    return sum((abs(v[0]) for v in poly.values()), F(0))


def commutators(q, squares, fragments, caps):
    bases = {"one_body": one_body_pauli(q)}
    for name, weight, matrix in squares:
        a = one_body_pauli(matrix)
        bases["df:" + name] = {key: scale(value, weight) for key, value in product(a, a).items() if scale(value, weight) != ZERO}
    polys = [bases[f["fragment_id"]] for f in fragments]
    groups, total = [], F(0)
    for j in range(len(polys) - 1):
        tail = {}
        for p in polys[j + 1:]:
            tail = linear(tail, p)
        first = commutator(polys[j], commutator(polys[j], tail))
        second = commutator(tail, commutator(tail, polys[j]))
        a, b = caps[j] / 2, sum(caps[j + 1:], F(0)) / 2
        first_cap, second_cap = 4 * a * a * b, 4 * a * b * b
        first_l1, second_l1 = hermitian_l1(first), hermitian_l1(second)
        chosen1, chosen2 = min(first_cap, first_l1), min(second_cap, second_l1)
        total += chosen1 / 24 + chosen2 / 12
        groups.append({"position": str(j + 1), "fragment_id": fragments[j]["fragment_id"],
                       "tail_fragment_ids": [f["fragment_id"] for f in fragments[j + 1:]],
                       "centered_caps": {"a": rat(a), "b": rat(b)},
                       "first": {"pauli": pauli_record(first), "structure_l1": rat(first_l1), "center_cap": rat(first_cap), "selected": rat(chosen1)},
                       "second": {"pauli": pauli_record(second), "structure_l1": rat(second_l1), "center_cap": rat(second_cap), "selected": rat(chosen2)}})
    return {"method": "GROUPED_EXACT_JW_L1_AND_CENTERED_CAP_MIN_V1",
            "formula_ref": "db1819587df36ee8833b37227ca0083faa0408ec:EW2_5",
            "groups": groups, "Cbar2": rat(total), "unit": "HA3",
            "weights": {"first": rat(F(1, 24)), "second": rat(F(1, 12))},
            "scalar_scope": "EXTERNAL_PSD_SHIFTS_CANCEL_ONLY_INSIDE_COMMUTATORS",
            "structural_baseline": "COMPLETE_FOR_DECLARED_TWO_METHODS_NOT_BEST_KNOWN_ENVELOPE"}
