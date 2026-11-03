"""B-1a only: fixed binary64 A, probabilities, tau/B and replayed theta64.

No molecular loader, component reconstruction, probability regeneration or B-2.
Numeric functions accept synthetic or separately authorized inputs.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "docs/research/rpe_m17_numerical_precision_preparation_20261103"))
import precision_adapter as precision
import run_precision_isolation as task_a


def replay_scalars(distribution, r):
    """New SOURCE replay values, NOT scalars recovered from the old NPZ."""
    orders = distribution["orders"]
    probabilities = distribution["order_probabilities"]
    k = distribution["finite_taylor_order"]
    tau = distribution["dimensionless_step_time"]
    b = distribution["exact_finite_distribution"]
    if (type(k) is not int or k < 0 or k % 2 or orders != list(range(0, k+1, 2))
            or type(r) is not int or r < 1 or len(probabilities) != len(orders)):
        raise ValueError("Invalid finite event schema")
    if (any(type(x) is not float or not math.isfinite(x) for x in [tau, b, *probabilities])
            or b <= 0 or any(p < 0 for p in probabilities)
            or not math.isclose(math.fsum(probabilities), 1., rel_tol=0., abs_tol=1e-12)):
        raise ValueError("Invalid fixed binary64 scalars/probabilities")
    # Preserve stored event probabilities without a second renormalization.
    theta = [math.atan(tau/(n+1)) for n in orders]
    log_b = math.log(b)
    normalizer = math.exp(r*log_b)
    if not math.isfinite(normalizer) or normalizer <= 0:
        raise ValueError("Invalid replay normalization")
    return {"orders": orders.copy(), "probabilities": probabilities.copy(), "tau": tau,
            "B": b, "K": k, "r": r, "theta64": theta,
            "cos64": [math.cos(t) for t in theta], "sin64": [math.sin(t) for t in theta],
            "logB64": log_b, "normalizer64": normalizer,
            "provenance": {"tau_B_probabilities": "old RESULT binary64 fields",
                           "theta64_cos64_sin64_logB64_normalizer64": "new SOURCE replay; not old saved values"}}


def scalar_trace(scalars):
    trace = dict(scalars)
    trace["binary64_hex"] = {key: ([x.hex() for x in value] if isinstance(value, list) else value.hex())
                             for key, value in scalars.items()
                             if type(value) is float or (isinstance(value, list) and value and type(value[0]) is float)}
    trace["B1a_boundary"] = "exact A/probabilities/tau/B/theta64; HP cos/sin, Taylor divisions, log/exp, matrix arithmetic"
    return trace


def float_replay(arrays, scalars):
    """Reuse both original independent float constructors; no RTEConfig rebuild."""
    sys.path.insert(0, str(ROOT / "src/trotterlib"))
    import rte
    if Path(rte.__file__).resolve() != ROOT/"src/trotterlib/rte.py":
        raise ValueError("Wrong finite RTE module")
    distribution = SimpleNamespace(orders=scalars["orders"], order_probabilities=scalars["probabilities"],
                                   dimensionless_step_time=scalars["tau"])
    event = task_a.original_runner.literal_law_mean(arrays["A_probability"], distribution)
    polynomial = rte.finite_taylor_operator(arrays["A_coefficient"], scalars["tau"], scalars["K"])
    return {"law_event_mean": event, "reference_polynomial": polynomial,
            "law_tail_mean": np.linalg.matrix_power(event, scalars["r"]),
            "reference_tail": np.linalg.matrix_power(polynomial, scalars["r"])/scalars["normalizer64"]}


def matrix_payload(ctx, matrix):
    digits = ctx.dps+10
    return {"shape": [matrix.rows, matrix.cols],
            "entries": [[ctx.nstr(z.real, digits), ctx.nstr(z.imag, digits)] for z in matrix],
            "bits": ctx.prec, "format": "row-major decimal round-trip; not an interval"}


def restore_matrix(ctx, payload):
    rows, cols = payload["shape"]
    if rows < 1 or cols < 1 or len(payload["entries"]) != rows*cols:
        raise ValueError("Invalid matrix payload")
    values = [ctx.mpc(ctx.mpf(re), ctx.mpf(im)) for re, im in payload["entries"]]
    if any(not ctx.isfinite(z) for z in values):
        raise ValueError("Nonfinite matrix payload")
    return ctx.matrix([values[i*cols:(i+1)*cols] for i in range(rows)])


def construct(ctx, path, a, scalars):
    """Law and Taylor remain distinct algorithms. Attenuation occurs once."""
    identity = ctx.eye(a.rows)
    tau = precision.exact_real(ctx, scalars["tau"])
    if path == "law":
        event = ctx.zeros(a.rows)
        for n, p, theta64 in zip(scalars["orders"], scalars["probabilities"], scalars["theta64"], strict=True):
            theta = precision.exact_real(ctx, theta64)
            product = identity if n == 0 else precision.binary_power(a, n)
            event += precision.exact_real(ctx, p)*(-1)**(n//2)*(ctx.cos(theta)*identity-ctx.j*ctx.sin(theta)*a)*product
        return event, precision.binary_power(event, scalars["r"])
    if path == "reference":
        polynomial, term = identity.copy(), identity.copy()
        for degree in range(1, scalars["K"]+2):
            term = term*((-ctx.j*tau/degree)*a)
            polynomial += term
        normalizer = ctx.exp(scalars["r"]*ctx.log(precision.exact_real(ctx, scalars["B"])))
        return polynomial, precision.binary_power(polynomial, scalars["r"])/normalizer
    raise ValueError("Unknown path")


def evaluate_job(job):
    import mpmath
    ctx = mpmath.mp.clone()
    ctx.prec = job["bits"]
    if ctx.prec < 53 or type(job["q"]) is not int or job["q"] < 1:
        raise ValueError("Invalid precision/q")
    started = time.monotonic()
    psi = precision.exact_matrix(ctx, job["psi"], vector=True)
    forward = precision.exact_matrix(ctx, job["forward"])
    reverse = precision.exact_matrix(ctx, job["reverse"])
    saved_tail = precision.exact_matrix(ctx, job["saved_tail"])
    if forward.rows != psi.rows or reverse.rows != psi.rows or saved_tail.rows != psi.rows:
        raise ValueError("Fixed dimension mismatch")
    if job["mode"] == "b1a":
        a = precision.exact_matrix(ctx, job["A"])
        if a.rows != psi.rows:
            raise ValueError("A/state dimension mismatch")
        one, tail = construct(ctx, job["path"], a, job["scalars"])
    elif job["mode"] == "replay_tail_baseline":
        one, tail = None, precision.exact_matrix(ctx, job["replay_tail"])
        if tail.rows != psi.rows:
            raise ValueError("Replay tail dimension mismatch")
    else:
        raise ValueError("B-1b/B-2/other modes are not implemented")
    # Match Task A left association. Never round the constructed tail to binary64.
    step = (reverse*tail)*forward
    state = precision.binary_power(step, job["q"])*psi
    mean = ctx.fsum(ctx.conj(psi[i])*state[i] for i in range(psi.rows))
    if not ctx.isfinite(mean):
        raise ValueError("Nonfinite B-1a mean")
    return {"mode": job["mode"], "path": job["path"], "bits": job["bits"],
            "mean": {"real": ctx.nstr(mean.real, ctx.dps+10), "imag": ctx.nstr(mean.imag, ctx.dps+10)},
            "tail_vs_saved_frobenius": ctx.nstr(ctx.norm(tail-saved_tail), ctx.dps+10),
            "tail_vs_replay_frobenius": ctx.nstr(ctx.norm(tail-precision.exact_matrix(ctx, job["replay_tail"])), ctx.dps+10),
            "operators": {"tail": matrix_payload(ctx, tail), **({"one_event_or_polynomial": matrix_payload(ctx, one)} if one is not None else {})},
            "wall_seconds": time.monotonic()-started, "backend": "mpmath", "backend_version": mpmath.__version__,
            "interval_certified": False}


def convergence(previous, current, q):
    ctx = precision.scalar_context()
    if set(previous) != set(current) or not current or not set(current) <= {"law", "reference"}:
        raise ValueError("Convergence path mismatch")
    check = {"path_changes": {path: precision.complex_difference(current[path]["mean"], previous[path]["mean"]) for path in current},
             "criterion": precision.CONVERGENCE_ATOL, "is_error_bound": False}
    check["converged"] = all(ctx.mpf(v["abs"]) <= precision.exact_real(ctx, precision.CONVERGENCE_ATOL) for v in check["path_changes"].values())
    changes = {}
    for path in current:
        before = restore_matrix(ctx, previous[path]["operators"]["tail"])
        after = restore_matrix(ctx, current[path]["operators"]["tail"])
        if before.rows != after.rows or before.cols != after.cols:
            raise ValueError("Convergence dimension mismatch")
        changes[path] = ctx.nstr(ctx.norm(after-before), 80)
    # Engineering guard for amplification by q; NOT a power perturbation bound.
    tolerance = precision.exact_real(ctx, precision.CONVERGENCE_ATOL)/q
    check.update(tail_changes_frobenius=changes, tail_criterion=ctx.nstr(tolerance, 80),
                 converged=check["converged"] and all(ctx.mpf(v) <= tolerance for v in changes.values()))
    return check
