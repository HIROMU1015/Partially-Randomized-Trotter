"""SP-0.5 exact-angle catalogue, interval guards, and primitive accounting.

Only single-qubit gate strings are synthesized. Controlled gates are lowered
analytically first; their two joint-space Pauli rotations have native angles
+theta/2 and -theta/2. No trajectories, wrappers, or Hamiltonians are built.
"""
from __future__ import annotations

from decimal import Decimal, localcontext, ROUND_FLOOR, ROUND_CEILING
from fractions import Fraction
import hashlib
import importlib.metadata
from pathlib import Path
import time

import mpmath as mp


def configure():
    mp.mp.dps = 80
    mp.iv.dps = 90


def angle(unit, numerator, denominator=1):
    f = Fraction(numerator, denominator)
    if unit not in ("pi", "rad"):
        raise ValueError("unknown angle unit")
    return {"unit": unit, "numerator": f.numerator, "denominator": f.denominator}


def scaled(a, scale):
    f = Fraction(a["numerator"], a["denominator"]) * scale
    return angle(a["unit"], f.numerator, f.denominator)


def key(a):
    return f'{a["unit"]}:{a["numerator"]}/{a["denominator"]}'


def value(a, ctx=mp.mp):
    x = ctx.mpf(a["numerator"]) / a["denominator"]
    return x * ctx.pi if a["unit"] == "pi" else x


def _endpoint_decimal(raw, rounding):
    sign, mantissa, exponent, _ = raw
    f = Fraction((-1 if sign else 1) * mantissa)
    f *= Fraction(2) ** exponent
    with localcontext() as ctx:
        ctx.prec = 70
        ctx.rounding = rounding
        return str(Decimal(f.numerator) / Decimal(f.denominator))


def bounds(x):
    """Serialize with directed rounding, retaining the exact binary endpoints."""
    return {"lo": _endpoint_decimal(x._mpi_[0], ROUND_FLOOR),
            "hi": _endpoint_decimal(x._mpi_[1], ROUND_CEILING)}


def interval(b):
    return mp.iv.mpf([b["lo"], b["hi"]])


def eye(n):
    return [[mp.iv.mpc(int(i == j)) for j in range(n)] for i in range(n)]


def product(a, b):
    return [[sum((a[i][k] * b[k][j] for k in range(len(b))), mp.iv.mpc(0))
             for j in range(len(b[0]))] for i in range(len(a))]


def sequence_matrix(sequence):
    """Same written-product order as pygridsynth DOmegaUnitary.from_gates."""
    iv = mp.iv
    w = iv.exp(iv.j * iv.pi / 4)
    s = iv.sqrt(2)
    matrices = {
        "H": [[1/s, 1/s], [1/s, -1/s]],
        "T": [[1, 0], [0, w]], "t": [[1, 0], [0, 1/w]],
        "S": [[1, 0], [0, iv.j]], "X": [[0, 1], [1, 0]],
        "W": [[w, 0], [0, w]],
    }
    u = eye(2)
    for gate in sequence:
        if gate not in matrices:
            raise ValueError(f"unsupported gate {gate!r}")
        u = product(u, matrices[gate])
    return u


def error_guard(sequence, a):
    """Outward-rounded Frobenius upper bound >= projective operator error.

    Search only the 16 scalar phases exp(i*k*pi/8); each is a valid witness.
    This is never applied to a system gate before controlling it. Any scalar
    reported by the synthesizer multiplies the entire native joint operator
    and disappears from its channel. No arbitrary optimal-phase claim is made.
    """
    u = sequence_matrix(sequence)
    t = value(a, mp.iv)
    v = [[mp.iv.exp(-mp.iv.j*t/2), 0], [0, mp.iv.exp(mp.iv.j*t/2)]]
    witnesses = []
    for k in range(16):
        phase = mp.iv.exp(mp.iv.j * k * mp.iv.pi/8)
        e2 = sum((abs(phase*u[i][j]-v[i][j])**2
                  for i in range(2) for j in range(2)), mp.iv.mpf(0))
        witnesses.append((mp.iv.sqrt(e2), k))
    e, k = min(witnesses, key=lambda item: mp.mpf(item[0]._mpi_[1]))
    return {"projective_operator_upper": bounds(e)["hi"],
            "phase_witness_pi_over_8": k, "channel_diamond_upper": bounds(2*e)["hi"]}


def exact_sequence(a):
    if a["unit"] != "pi":
        return None
    k = 4 * Fraction(a["numerator"], a["denominator"])
    if k.denominator != 1:
        return None
    k = k.numerator % 8
    if k == 7:
        return "t"
    return "S" * (k // 2) + ("T" if k % 2 else "")


def pai(a):
    """Known three-notch channel interpolation on the one pi/4 catalogue."""
    iv = mp.iv
    delta = iv.pi / 4
    if a["unit"] == "pi":
        f = 4 * Fraction(a["numerator"], a["denominator"])
        k = f.numerator // f.denominator
        exact = f.denominator == 1
    else:
        k = int(mp.floor(value(a) / (mp.pi/4)))
        exact = False
    t = value(a, iv) - k*delta
    if exact:
        g = [iv.mpf(1), iv.mpf(0), iv.mpf(0)]
    else:
        if not (t.a > 0 and t.b < delta.a):
            raise ArithmeticError("notch-cell location numerically uncertain")
        g2 = iv.sin(t) / iv.sin(delta)
        g = [(1+iv.cos(t)-g2*(1+iv.cos(delta)))/2,
             g2, (1-iv.cos(t)-g2*(1-iv.cos(delta)))/2]
    gamma = sum((abs(x) for x in g), iv.mpf(0))
    return {"notch_indices": [k % 8, (k+1) % 8, (k+4) % 8],
            "g": [bounds(x) for x in g], "gamma": bounds(gamma),
            "p": [bounds(abs(x)/gamma) for x in g]}


def economics(interpolations, branch_costs, deterministic_cost):
    """Independent primitives: E[W² C] = prod(gamma²) sum(E[C_i])."""
    if deterministic_cost == 0:
        return {"J": None, "classification": "ZERO_COST_BASELINE_NO_STRICT_GAIN"}
    if deterministic_cost < 0:
        raise ValueError("negative cost")
    moment, mean_cost = mp.iv.mpf(1), mp.iv.mpf(0)
    for interpolation, costs in zip(interpolations, branch_costs, strict=True):
        if len(costs) != 3 or any(c < 0 for c in costs):
            raise ValueError("invalid branch costs")
        gamma = interval(interpolation["gamma"])
        moment *= gamma**2
        mean_cost += sum((interval(p)*c for p, c in zip(interpolation["p"], costs)),
                         mp.iv.mpf(0))
    j = moment*mean_cost/deterministic_cost
    classification = ("STRICT_TRADEOFF" if j.b < 1 else
                      "NO_STRICT_TRADEOFF" if j.a >= 1 else "NUMERIC_INCONCLUSIVE")
    return {"J": bounds(j), "classification": classification,
            "weight_second_moment": bounds(moment), "expected_T_count": bounds(mean_cost)}


def package_tree_sha(name):
    dist = importlib.metadata.distribution(name)
    files = sorted(str(p) for p in dist.files if str(p).endswith(".py"))
    h = hashlib.sha256()
    for relative in files:
        h.update(relative.encode() + b"\0")
        h.update(Path(dist.locate_file(relative)).read_bytes() + b"\0")
    return h.hexdigest()


def synthesize(a, config):
    """One fixed implementation; shared exact Clifford/T fast path for all arms."""
    configure()
    start = time.monotonic()
    sequence = exact_sequence(a)
    route = "exact_common_fast_path"
    if sequence is None:
        from pygridsynth.config import GridsynthConfig
        from pygridsynth.gridsynth import gridsynth_gates
        route = "pygridsynth_2.0.0"
        cfg = GridsynthConfig(**config["synthesizer_options"])
        # Reserve headroom for the independently checked Frobenius bound.
        sequence = gridsynth_gates(value(a), mp.mpf(config["operator_epsilon"])/4, cfg=cfg)
    guard = error_guard(sequence, a)
    passed = Decimal(guard["projective_operator_upper"]) <= Decimal(config["operator_epsilon"])
    return {"angle": a, "key": key(a), "route": route, "sequence": sequence,
            "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
            "T_count": sequence.count("T")+sequence.count("t"),
            "Clifford_count": sum(sequence.count(x) for x in "HSX"),
            "scalar_W_count": sequence.count("W"), "error_guard": guard,
            "error_pass": passed, "wall_seconds": time.monotonic()-start}
