"""Checker-only occupation matrices, trace Pauli extraction and explicit K0 B.

No imports from producer_algebra/generator. Full Fock matrices are an independent
small-artificial-input reference, not a scalable molecular evaluation backend.
"""

from fractions import Fraction as F
from itertools import product as cartesian

from ..a2_spec01_03.common import (
    complex_rat, content_id, integer, keys_exact, rat, read_rat, require,
)


def plus(x, y):
    return x[0] + y[0], x[1] + y[1]


def times(x, y):
    a, b = x
    c, d = y
    return a * c - b * d, a * d + b * c


def matrix_add(a, b, weight=F(1)):
    out = dict(a)
    for key in b:
        x, y = b[key]
        value = plus(out.get(key, (F(0), F(0))), (weight * x, weight * y))
        if value == (0, 0):
            out.pop(key, None)
        else:
            out[key] = value
    return out


def matrix_product(a, b):
    out, by_row = {}, {}
    for (row, col), value in b.items():
        by_row.setdefault(row, []).append((col, value))
    for (row, mid), x in a.items():
        for col, y in by_row.get(mid, []):
            key = (row, col)
            value = plus(out.get(key, (F(0), F(0))), times(x, y))
            if value == (0, 0):
                out.pop(key, None)
            else:
                out[key] = value
    return out


def bracket(a, b):
    return matrix_add(matrix_product(a, b), matrix_product(b, a), F(-1))


def occupation_matrix(coefficients):
    n, result = len(coefficients), {}
    for ket in range(2 ** n):
        occupied = [p for p in range(n) if ket & (2 ** p)]
        for annihilate_index, q in enumerate(occupied):
            remaining = occupied[:annihilate_index] + occupied[annihilate_index + 1:]
            for p in range(n):
                if p in remaining:
                    continue
                # Insert into an ordered exterior-product list. This does not
                # use the generator's prefix-popcount hopping implementation.
                insertion_index = sum(j < p for j in remaining)
                sign = -1 if (annihilate_index + insertion_index) % 2 else 1
                bra = sum(2 ** j for j in remaining) + 2 ** p
                re, im = coefficients[p][q]
                key = (bra, ket)
                value = plus(result.get(key, (F(0), F(0))), (sign * re, sign * im))
                if value == (0, 0):
                    result.pop(key, None)
                else:
                    result[key] = value
    return result


def _column(matrix, column):
    return [{"occupation_bits": str(row), "amplitude": complex_rat(*value)}
            for (row, col), value in sorted(matrix.items()) if col == column]


def upper(c, q, squares, occupied, lower):
    ket = sum(2 ** integer(k, natural=True) for k in occupied)
    qm = occupation_matrix(q)
    q_value = qm.get((ket, ket), (F(0), F(0)))
    require(q_value[1] == 0, "NON_HERMITIAN", "q_expectation", "Real expectation required")
    columns, contributions = [{"operator": "one_body", "column": _column(qm, ket)}], []
    value = c + q_value[0]
    for name, weight, coefficients in squares:
        am = occupation_matrix(coefficients)
        diagonal = matrix_product(am, am).get((ket, ket), (F(0), F(0)))
        require(diagonal[1] == 0 and diagonal[0] >= 0, "EXPECTATION", "A_squared", "Hermitian square expectation")
        value += weight * diagonal[0]
        columns.append({"operator": "df:" + name, "column": _column(am, ket)})
        contributions.append({"block_id": name, "lambda": rat(weight), "action_norm_squared": rat(diagonal[0]),
                              "weighted_expectation": rat(weight * diagonal[0])})
    return {"method": "CAR_BITPREFIX_COLUMN_NORM_V1", "columns": columns,
            "constant": rat(c), "q_expectation": rat(q_value[0]),
            "square_contributions": contributions, "U": rat(value), "L": rat(lower),
            "W": rat(value - lower), "unit": "HA"}


def verify_k_sos(record, caps, labels, bits):
    m = len(caps)
    dictionary = {}
    # Reconstruct by all candidate tuples, separately from generator's two loops.
    raw_count = 0
    for a in range(m):
        for b in range(m):
            for c in range(m):
                value = F(0)
                if a == b and c > a:
                    value += F(1, 24)
                    raw_count += 1
                if a > c and b > c:
                    value -= F(1, 12)
                    raw_count += 1
                if value:
                    dictionary[(a, b, c)] = value
    terms = [{"positions": [str(i + 1) for i in key], "kappa": rat(value)}
             for key, value in sorted(dictionary.items())]
    wanted_pairs = {tuple(sorted((a, c))) for a, b, c in dictionary}
    wanted_pairs.update(tuple(sorted((a, b))) for a, b, c in dictionary)
    roots = {}
    require(type(record.get("sqrt_witnesses")) is list, "INCOMPLETE_CERTIFICATE", "sqrt", "Missing witnesses")
    require(len(record["sqrt_witnesses"]) == len(wanted_pairs), "SQRT_WITNESS", "sqrt", "Wrong witness count")
    grid = 2 ** bits
    for witness, pair in zip(record["sqrt_witnesses"], sorted(wanted_pairs), strict=True):
        keys_exact(witness, ("positions", "z", "u"), "sqrt_witness")
        require(witness["positions"] == [str(i + 1) for i in pair], "SQRT_WITNESS", "positions", "Canonical pair order required")
        z, u = read_rat(witness["z"]), read_rat(witness["u"])
        require(z == caps[pair[0]] * caps[pair[1]] and z >= 0 and u >= 0 and u * u >= z,
                "SQRT_WITNESS", "sqrt", "Not an upper witness for declared radicand")
        require((u * grid).denominator == 1 and (u == 0 if z == 0 else
                u > 0 and (u - F(1, grid)) ** 2 < z),
                "SQRT_GRID", "u", "Require minimal nonnegative upper point on explicit dyadic grid")
        roots[pair] = u
    matrix = [[F(0) for _ in range(m)] for _ in range(m)]
    for (a, b, c), kappa in dictionary.items():
        x = abs(kappa) * caps[b] * roots[tuple(sorted((a, c)))]
        y = abs(kappa) * caps[c] * roots[tuple(sorted((a, b)))]
        # Four B entries, including repeated coordinates. No deduplication.
        matrix[a][c] += x
        matrix[c][a] += x
        matrix[a][b] += y
        matrix[b][a] += y
    rows = [sum(row, F(0)) for row in matrix]
    expected = {"method": "K0_RAW_BCH_STREAMING_DYADIC_UPPER_V1",
                "formula_ref": "58d8672702284d73321cc67a26c4640adaddde5f:K0_4_EQ6_8_9",
                "position_to_fragment": [{"position": str(i + 1), "fragment_id": v, "f": rat(caps[i])} for i, v in enumerate(labels)],
                "sqrt_bits": str(bits), "raw_count": str(raw_count), "dictionary": terms,
                "dictionary_digest": content_id("k0_raw_bch_v1", {"terms": terms}),
                "sqrt_witnesses": record["sqrt_witnesses"],
                "sqrt_witness_digest": content_id("sqrt_witnesses_v1", {"items": record["sqrt_witnesses"]}),
                "row_sums_upper": [rat(v) for v in rows], "K_SOS_upper": rat(max(rows, default=F(0))),
                "meaning": "UPPER_BOUND_ON_K0_ROW_SUM_NOT_EXACT_IRRATIONAL_VALUE", "unit": "HA2"}
    require(record == expected, "K_SOS_RECOMPUTATION", "K_SOS", "Raw dictionary, row sum or identity differs")
    return max(rows, default=F(0))


def trace_pauli(matrix, n):
    # Same exact 2^-n Tr(PX) as frozen checker. Group by XOR displacement and
    # batch the (-1)^(row dot sign_bits) sums with an exact Walsh transform.
    # No JW multiplication table, floating-point transform or sector projection.
    dimension, by_flip = 2 ** n, {}
    for (row, col), value in matrix.items():
        flip = row ^ col
        table = by_flip.setdefault(flip, {})
        table[row] = plus(table.get(row, (F(0), F(0))), value)
    result = []
    for flip, diagonal in sorted(by_flip.items()):
        values = [diagonal.get(row, (F(0), F(0))) for row in range(dimension)]
        stride = 1
        while stride < dimension:
            for start in range(0, dimension, 2 * stride):
                for offset in range(stride):
                    i, j = start + offset, start + offset + stride
                    a, b = values[i], values[j]
                    values[i] = plus(a, b)
                    values[j] = (a[0] - b[0], a[1] - b[1])
            stride *= 2
        for sign_bits, value in enumerate(values):
            if value == (0, 0):
                continue
            letters = "".join("Y" if (flip >> k) & 1 and (sign_bits >> k) & 1 else
                              "X" if (flip >> k) & 1 else "Z" if (sign_bits >> k) & 1 else "I"
                              for k in range(n))
            phase = ((1, 0), (0, 1), (-1, 0), (0, -1))[(flip & sign_bits).bit_count() % 4]
            re, im = times(value, phase)
            result.append({"pauli_mode0_left": letters,
                           "coefficient": complex_rat(re / dimension, im / dimension)})
    return sorted(result, key=lambda t: t["pauli_mode0_left"])


def commutators(q, squares, fragments, caps):
    matrices = {"one_body": occupation_matrix(q)}
    for name, weight, g in squares:
        a = occupation_matrix(g)
        matrices["df:" + name] = {key: (v[0] * weight, v[1] * weight)
                                  for key, v in matrix_product(a, a).items() if weight}
    ordered = [matrices[f["fragment_id"]] for f in fragments]
    groups, total = [], F(0)
    for j, fj in enumerate(ordered[:-1]):
        tail = {}
        for other in ordered[j + 1:]:
            tail = matrix_add(tail, other)
        first = trace_pauli(bracket(fj, bracket(fj, tail)), len(q))
        second = trace_pauli(bracket(tail, bracket(tail, fj)), len(q))

        def norm(poly):
            from ..a2_spec01_03.common import read_complex
            coeffs = [read_complex(t["coefficient"]) for t in poly]
            require(all(i == 0 for _, i in coeffs), "NON_HERMITIAN", "double_commutator", "Nonreal Hermitian Pauli coefficient")
            return sum((abs(r) for r, _ in coeffs), F(0))

        a = caps[j] * F(1, 2)
        b = sum(caps[k] for k in range(j + 1, len(caps))) * F(1, 2)
        # Nested standard commutator norm bound, separately from generator.
        cap1, cap2 = 2 * a * (2 * a * b), 2 * b * (2 * b * a)
        l1, l2 = norm(first), norm(second)
        s1, s2 = min(cap1, l1), min(cap2, l2)
        total += F(1, 24) * s1 + F(1, 12) * s2
        groups.append({"position": str(j + 1), "fragment_id": fragments[j]["fragment_id"],
                       "tail_fragment_ids": [f["fragment_id"] for f in fragments[j + 1:]],
                       "centered_caps": {"a": rat(a), "b": rat(b)},
                       "first": {"pauli": first, "structure_l1": rat(l1), "center_cap": rat(cap1), "selected": rat(s1)},
                       "second": {"pauli": second, "structure_l1": rat(l2), "center_cap": rat(cap2), "selected": rat(s2)}})
    return {"method": "GROUPED_EXACT_JW_L1_AND_CENTERED_CAP_MIN_V1",
            "formula_ref": "db1819587df36ee8833b37227ca0083faa0408ec:EW2_5",
            "groups": groups, "Cbar2": rat(total), "unit": "HA3",
            "weights": {"first": rat(F(1, 24)), "second": rat(F(1, 12))},
            "scalar_scope": "EXTERNAL_PSD_SHIFTS_CANCEL_ONLY_INSIDE_COMMUTATORS",
            "structural_baseline": "COMPLETE_FOR_DECLARED_TWO_METHODS_NOT_BEST_KNOWN_ENVELOPE"}
