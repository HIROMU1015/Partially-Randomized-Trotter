"""Precision adapters for exact binary64 inputs; no molecular input loader."""
from __future__ import annotations

import math
import time
import numpy as np

DIAGNOSTIC_ATOL = 1e-10
CONVERGENCE_ATOL = 1e-13  # Repeatability diagnostic, never an error bound.


def precision_ladder(dimension, q):
    if type(dimension) is not int or dimension < 1 or type(q) is not int or q < 1:
        raise ValueError("Positive integer dimension/q required")
    target_bits = math.ceil(-math.log2(DIAGNOSTIC_ATOL))
    estimate = target_bits + (q-1).bit_length() + (dimension-1).bit_length() + 32
    initial = ((estimate+31)//32)*32
    return initial, initial+64, initial+128


def validated_array(value, *, vector=False):
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype("complex128"):
        raise ValueError("Unmodified complex128 array required")
    if vector:
        valid = value.ndim == 1 and value.size > 0
    else:
        valid = value.ndim == 2 and value.shape[0] == value.shape[1] and value.shape[0] > 0
    if not valid or not np.isfinite(value).all():
        raise ValueError("Invalid shape or nonfinite entries")
    return value


def exact_real(ctx, value):
    """Lift a binary64 number as a dyadic, never through str(float)."""
    if ctx.prec < 53:
        raise ValueError("At least 53 bits needed to preserve binary64 mantissas")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Nonfinite binary64 input")
    numerator, denominator = value.as_integer_ratio()
    return ctx.mpf(numerator)/ctx.mpf(denominator)


def exact_matrix(ctx, array, *, vector=False):
    array = validated_array(array, vector=vector)
    raw = array.reshape((-1, 1)) if vector else array
    return ctx.matrix([[ctx.mpc(exact_real(ctx, z.real), exact_real(ctx, z.imag)) for z in row] for row in raw])


def binary_power(matrix, q):
    """Binary powering without mpmath's unused final square. No eigensolver."""
    if type(q) is not int or q < 1 or matrix.rows != matrix.cols:
        raise ValueError("Square matrix and positive integer q required")
    factor, result, exponent = matrix.copy(), None, q
    while exponent:
        if exponent & 1:
            result = factor.copy() if result is None else result*factor
        exponent >>= 1
        if exponent:
            factor = factor*factor
    return result


def float_baseline(factors, psi, q):
    """Same final operations as the original evaluate_context, saved tails only."""
    if type(q) is not int or q < 1:
        raise ValueError("Positive integer q required")
    psi = validated_array(psi, vector=True)
    out, steps = {}, {}
    for path in ("law", "reference"):
        reverse, tail, forward = (validated_array(factors[path][k]) for k in ("reverse", "tail", "forward"))
        if reverse.shape != tail.shape or tail.shape != forward.shape or len(psi) != len(tail):
            raise ValueError("Matrix/state dimension mismatch")
        step = reverse @ tail @ forward
        value = complex(np.vdot(psi, np.linalg.matrix_power(step, q) @ psi))
        if not np.isfinite(value):
            raise ValueError("Nonfinite float baseline")
        out[path] = {"real": value.real, "imag": value.imag}
        steps[path] = step
    return out, steps


def evaluate_job(job):
    """Picklable worker. Inputs are synthetic or separately authorized arrays."""
    import mpmath
    ctx = mpmath.mp.clone()
    ctx.prec = job["bits"]
    started = time.monotonic()
    psi = exact_matrix(ctx, job["psi"], vector=True)
    if job["mode"] == "fixed_step":
        step = exact_matrix(ctx, job["step"])
    elif job["mode"] == "fixed_factors":
        step = exact_matrix(ctx, job["reverse"])*exact_matrix(ctx, job["tail"])*exact_matrix(ctx, job["forward"])
    else:
        raise ValueError("Unsupported mode; no coefficient reconstruction in Task A")
    if step.rows != psi.rows:
        raise ValueError("Matrix/state dimension mismatch")
    value = binary_power(step, job["q"])*psi
    mean = ctx.fsum(ctx.conj(psi[i])*value[i] for i in range(psi.rows))
    if not ctx.isfinite(mean):
        raise ValueError("Nonfinite high-precision result")
    digits = ctx.dps+10
    return {"path": job["path"], "mode": job["mode"], "bits": job["bits"],
            "mean": {"real": ctx.nstr(mean.real, digits), "imag": ctx.nstr(mean.imag, digits)},
            "wall_seconds": time.monotonic()-started, "backend": "mpmath", "backend_version": mpmath.__version__,
            "interval_certified": False}


def scalar_context():
    import mpmath
    ctx = mpmath.mp.clone()
    ctx.prec = 288  # Above the 224-bit top matrix tier, for scalar comparisons.
    return ctx


def number(ctx, payload):
    def real(value):
        return exact_real(ctx, value) if isinstance(value, (float, int)) else ctx.mpf(value)
    return ctx.mpc(real(payload["real"]), real(payload["imag"]))


def complex_difference(left, right):
    ctx = scalar_context()
    value = number(ctx, left)-number(ctx, right)
    return {"real": ctx.nstr(value.real, 80), "imag": ctx.nstr(value.imag, 80), "abs": ctx.nstr(abs(value), 80)}


def convergence(previous, current):
    by_path = {path: complex_difference(current[path]["mean"], previous[path]["mean"])
               for path in ("law", "reference")}
    ctx = scalar_context()
    return {"path_changes": by_path, "criterion": CONVERGENCE_ATOL,
            "converged": all(ctx.mpf(v["abs"]) <= exact_real(ctx, CONVERGENCE_ATOL) for v in by_path.values()),
            "is_error_bound": False}


def gap_relation(pair):
    gap = complex_difference(pair["law"]["mean"], pair["reference"]["mean"])
    ctx = scalar_context()
    distance = ctx.mpf(gap["abs"])
    threshold = exact_real(ctx, DIAGNOSTIC_ATOL)
    band = exact_real(ctx, CONVERGENCE_ATOL)
    relation = "BELOW" if distance < threshold-band else "ABOVE" if distance > threshold+band else "NEAR_THRESHOLD"
    return {"gap": gap, "relation_to_original_atol": relation, "decision_band": CONVERGENCE_ATOL,
            "decision_band_is_rigorous": False}


def interpret(baseline_reproduced, baseline, step_stage, factor_stage=None):
    """Candidate labels only. Never rewrite the original scientific status."""
    result = {"candidate_labels": ["INSUFFICIENT_EVIDENCE"],
              "original_numeric_diagnostic_status": "INCONSISTENT", "u_num_bound": None,
              "RPE_tolerance_certified": False, "scientific_status": "UNDETERMINED"}
    if not baseline_reproduced or not step_stage["convergence"]["converged"]:
        return result
    baseline_gap = complex_difference(baseline["law"], baseline["reference"])
    result["baseline_gap"] = baseline_gap
    ctx = scalar_context()
    if ctx.mpf(baseline_gap["abs"]) <= exact_real(ctx, DIAGNOSTIC_ATOL)+exact_real(ctx, CONVERGENCE_ATOL):
        result["reason"] = "No stable threshold excess in the current float baseline"
        return result
    pair = step_stage["best_pair"]
    step_gap = gap_relation(pair)
    result["fixed_step"] = step_gap
    result["baseline_to_fixed_step"] = {path: complex_difference(baseline[path], pair[path]["mean"])
                                        for path in ("law", "reference")}
    if step_gap["relation_to_original_atol"] == "BELOW":
        result["candidate_labels"] = ["POWER_ROUNDOFF_PLAUSIBLE"]
        result["scope"] = "Post-step q-power/state-action/expectation arithmetic; no unique-cause proof"
    elif step_gap["relation_to_original_atol"] == "ABOVE" and factor_stage is not None and factor_stage["convergence"]["converged"]:
        factor_gap = gap_relation(factor_stage["best_pair"])
        result["fixed_factors"] = factor_gap
        result["fixed_step_to_fixed_factors"] = {path: complex_difference(pair[path]["mean"], factor_stage["best_pair"][path]["mean"])
                                                 for path in ("law", "reference")}
        if factor_gap["relation_to_original_atol"] == "BELOW":
            result["candidate_labels"] = ["CONSTRUCTION_ROUNDOFF_PLAUSIBLE"]
            result["scope"] = "Post-archive step assembly R*tail*F; pre-archive tail construction not tested"
        elif factor_gap["relation_to_original_atol"] == "ABOVE":
            result["candidate_labels"] = ["CONSTRUCTION_ROUNDOFF_PLAUSIBLE", "IMPLEMENTATION_OR_DEFINITION_DIFFERENCE_POSSIBLE"]
            result["scope"] = "Gap retained by saved factors; rounding versus definition upstream remains unresolved"
    return result
