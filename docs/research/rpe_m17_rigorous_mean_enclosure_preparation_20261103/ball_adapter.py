"""Acb centers + certified global spectral-norm radius. No real input loader.

The elementary exact-rational/endpoint/norm formulas reuse the design of
h4_ball_evidence.py and h4_arb_bridge.py, without their study entrypoints.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
import math

from flint import arb, acb, acb_mat, fmpq, ctx


def rational(value):
    if isinstance(value, Q):
        return value
    if type(value) is int:
        return Q(value)
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("Nonfinite binary64 input")
        return Q.from_float(value)
    if isinstance(value, str):
        return Q.from_float(float.fromhex(value)) if '0x' in value.lower() else Q(value)
    raise ValueError("Exact rational/binary64 input required")


def real(value):
    q = rational(value)
    return arb(fmpq(q.numerator, q.denominator))


def dyadic(value):
    if not value.is_finite() or not value.is_exact():
        raise ValueError("Finite exact endpoint required")
    m, e = value.man_exp()
    return Q(int(m))*Q(2)**int(e)


def upper(value):
    if not value.is_finite():
        raise ValueError("Nonfinite enclosure")
    return value.upper()


def endpoints(value):
    return {"lower": str(dyadic(value.lower())), "upper": str(dyadic(upper(value)))}


def zball(pair):
    if len(pair) != 2:
        raise ValueError("Complex pair required")
    return acb(real(pair[0]), real(pair[1]))


def dagger(matrix):
    return matrix.conjugate().transpose()


def frobenius(matrix):
    return upper(sum((upper(abs(matrix[i,j]))**2 for i in range(matrix.nrows())
                      for j in range(matrix.ncols())), arb(0)).sqrt())


def identity(n):
    if type(n) is not int or n < 1:
        raise ValueError("Positive dimension required")
    return acb_mat([[int(i == j) for j in range(n)] for i in range(n)])


@dataclass
class MatrixBall:
    """||target-center||_2 <= radius; ||target||_2 <= norm (all upper bounds)."""
    center: acb_mat
    radius: arb
    norm: arb


def recenter(matrix, error=0, norm=None):
    """Move rectangular radii into one global norm radius; never discard them."""
    error = real(error) if not isinstance(error, arb) else error
    if not error.is_finite() or not error >= 0:
        raise ValueError("Nonnegative certified radius required")
    center = acb_mat(matrix.nrows(), matrix.ncols())
    square = arb(0)
    for i in range(matrix.nrows()):
        for j in range(matrix.ncols()):
            value = matrix[i,j]
            if not value.is_finite():
                raise ValueError("Nonfinite matrix entry")
            re, im = value.real.mid(), value.imag.mid()
            center[i,j] = acb(re, im)
            square += upper(abs(value.real-re))**2 + upper(abs(value.imag-im))**2
    radius = upper(error + square.sqrt())
    if norm is None:
        norm = upper(frobenius(center)+radius)
    else:
        norm = upper(real(norm) if not isinstance(norm, arb) else norm)
    if not norm.is_finite() or not norm >= 0:
        raise ValueError("Invalid target norm bound")
    return MatrixBall(center, radius, norm)


def unit(matrix):
    # Only callers that established exact unitarity may select norm=1.
    return recenter(matrix, norm=1)


def product(a, b):
    if a.center.ncols() != b.center.nrows():
        raise ValueError("Matrix dimensions disagree")
    # AB-CD=(A-C)B+C(B-D); ||C|| <= ||A||+ea.
    error = upper(a.radius*b.norm+(a.norm+a.radius)*b.radius)
    return recenter(a.center*b.center, error, upper(a.norm*b.norm))


def adjoint(a):
    return MatrixBall(dagger(a.center), a.radius, a.norm)


def weighted_sum(terms, norm=None):
    if not terms:
        raise ValueError("Empty operator mixture")
    rows, cols = terms[0][1].center.nrows(), terms[0][1].center.ncols()
    result, error, target_norm = acb_mat(rows, cols), arb(0), arb(0)
    for weight, item in terms:
        if (item.center.nrows(), item.center.ncols()) != (rows, cols):
            raise ValueError("Mixture dimensions disagree")
        w = weight if isinstance(weight, (arb, acb)) else real(weight)
        if not w.is_finite():
            raise ValueError("Nonfinite mixture weight")
        result += w*item.center
        error += upper(abs(w))*item.radius
        target_norm += upper(abs(w))*item.norm
    return recenter(result, upper(error), upper(target_norm) if norm is None else norm)


def power(a, exponent):
    if type(exponent) is not int or exponent < 0 or a.center.nrows() != a.center.ncols():
        raise ValueError("Nonnegative integer exponent and square matrix required")
    if not exponent:
        return unit(identity(a.center.nrows()))
    factor, result = a, None
    while exponent:
        if exponent & 1:
            result = factor if result is None else product(result, factor)
        exponent >>= 1
        if exponent:
            factor = product(factor, factor)
    return result


def expectation(operator, state_pairs):
    """Normalized ray of the saved vector; raw state bytes remain unchanged."""
    if len(state_pairs) != operator.center.nrows() or operator.center.ncols() != len(state_pairs):
        raise ValueError("State dimension mismatch")
    psi = acb_mat([[zball(pair)] for pair in state_pairs])
    norm2 = (dagger(psi)*psi)[0,0].real
    if not norm2 > 0:
        raise ValueError("State norm is zero or not bounded away from zero")
    unnormalized = (dagger(psi)*operator.center*psi)[0,0]
    value = unnormalized/norm2
    # Both input vectors are normalized by this exact norm2, so transfer <= radius.
    value += acb(arb(0, operator.radius), arb(0, operator.radius))
    if not value.is_finite():
        raise ValueError("Nonfinite expectation enclosure")
    return value, {"saved_vector_norm_squared": endpoints(norm2),
                   "state_semantics": "normalized ray; exact division, no raw-vector rewrite",
                   "operator_radius_upper": str(dyadic(operator.radius))}


def classify_interval(interval, saved_mean, n, alpha, epsilon, bridge=None):
    """Outward strict sufficient condition; None bridge stays unknown."""
    if type(n) is not int or n < 1:
        raise ValueError("Positive fixed shot count required")
    alpha, epsilon = real(alpha), real(epsilon)
    if not (alpha > 0 and alpha < 1 and epsilon > 0):
        raise ValueError("Invalid fixed statistical inputs")
    lo, hi = interval.lower(), upper(interval)
    if lo > hi:
        raise ValueError("Invalid interval")
    mean = real(saved_mean)
    u = upper(abs(interval-mean))
    # Necessary lower bound on |true mean - saved mean| for an exclusion decision.
    minimum = arb(0)
    if lo > mean:
        minimum = (lo-mean).lower()
    elif hi < mean:
        minimum = (mean-hi).lower()
    shot = (2*(2/alpha).log()/n).sqrt()
    total = u+shot
    if upper(total) < epsilon.lower():
        status = "SUFFICIENT_CONDITION_CONFIRMED"
    elif (minimum+shot).lower() >= epsilon.upper():
        status = "SUFFICIENT_CONDITION_EXCLUDED"
    else:
        status = "UNDETERMINED"
    out = {"interval": endpoints(interval), "U_cat_upper": str(dyadic(u)),
           "shot_half_width": endpoints(shot), "categorical_decision": status,
           "exec_bridge_upper": None, "U_exec_upper": None, "exec_decision": "UNDETERMINED"}
    if bridge is not None:
        bridge = rational(bridge)
        if bridge < 0:
            raise ValueError("Negative bridge bound")
        full = upper(u+real(bridge))
        out.update(exec_bridge_upper=str(bridge), U_exec_upper=str(dyadic(full)),
                   exec_decision="SUFFICIENT_CONDITION_CONFIRMED" if upper(full+shot) < epsilon.lower() else "UNDETERMINED")
    return out


def encode_matrix_ball(value):
    return {"shape": [value.center.nrows(), value.center.ncols()],
            "center": [[str(dyadic(value.center[i,j].real)), str(dyadic(value.center[i,j].imag))]
                       for i in range(value.center.nrows()) for j in range(value.center.ncols())],
            "spectral_radius_upper": str(dyadic(value.radius)), "target_norm_upper": str(dyadic(value.norm))}
