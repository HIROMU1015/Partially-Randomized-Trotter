"""Fixed-time complex-signal Hoeffding accounting, independent of Hamiltonian.

Matches the headroom and safe-integer arithmetic of the frozen Track A
``ax2b_numerical_accounting.u_aware_shots``. This is conditional accounting;
an allowance supplied by the caller is not certified by this module.
"""
from __future__ import annotations

import math
from typing import Mapping

AXES = ("cosine", "sine")


def nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}: finite real required")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name}: finite nonnegative value required")
    return value


def fixed_signal_shots(*, epsilon_sig: float, bias: Mapping,
                       allowance: Mapping, log_B_upper: float | None,
                       evidence_kind="UNKNOWN", evidence_ref=None,
                       epsilon_axes=None, alpha_axes=None):
    """Return sufficient shots for corrected independent bounded ±B outcomes.

    ``h_a = epsilon_a - abs(bias_a) - allowance_a`` and
    ``N_a = ceil(2 B_upper**2 log(2/alpha_a) / h_a**2)``.
    Axis epsilon values have squared sum <= epsilon_sig**2; failure budgets
    sum to the recorded union budget. Missing evidence yields null N and G.
    """
    epsilon_sig = nonnegative(epsilon_sig, "epsilon_sig")
    if epsilon_sig == 0 or set(bias) != set(AXES) or set(allowance) != set(AXES):
        raise ValueError("PRECISION_OR_AXIS_SCHEMA")
    eps = dict({a: epsilon_sig / math.sqrt(2) for a in AXES} if epsilon_axes is None else epsilon_axes)
    alpha = dict({a: .025 for a in AXES} if alpha_axes is None else alpha_axes)
    if set(eps) != set(AXES) or set(alpha) != set(AXES):
        raise ValueError("AXIS_ALLOCATION_SCHEMA")
    eps = {a: nonnegative(eps[a], "epsilon_axis") for a in AXES}
    alpha = {a: nonnegative(alpha[a], "alpha_axis") for a in AXES}
    if any(eps[a] == 0 or not 0 < alpha[a] < 1 for a in AXES):
        raise ValueError("AXIS_ALLOCATION_RANGE")
    if math.hypot(*eps.values()) > epsilon_sig * (1 + 1e-14) or sum(alpha.values()) >= 1:
        raise ValueError("ALLOCATION_EXCEEDS_COMPLEX_PRECISION_OR_FAILURE_BUDGET")
    if evidence_kind not in {"UNKNOWN", "EMPIRICAL", "CERTIFIED"}:
        raise ValueError("EVIDENCE_KIND")
    if evidence_kind != "UNKNOWN" and not evidence_ref:
        raise ValueError("NUMERICAL_EVIDENCE_REFERENCE_REQUIRED")
    log_B = None if log_B_upper is None else nonnegative(log_B_upper, "log_B_upper")
    B = math.exp(log_B) if log_B is not None and log_B < 709 else None
    rows = {}
    for a in AXES:
        b = None if bias[a] is None else nonnegative(bias[a], "bias")
        u = None if allowance[a] is None else nonnegative(allowance[a], "allowance")
        if evidence_kind == "UNKNOWN":
            u = None
        h = None if b is None or u is None else eps[a] - b - u
        status = ("UNDETERMINED" if h is None else "ELIGIBLE" if h > 0
                  else "INELIGIBLE" if max(0., b - u) >= eps[a] else "UNDETERMINED")
        n = log_n = None
        if status == "ELIGIBLE" and log_B is not None:
            log_n = math.log(2) + 2 * log_B - 2 * math.log(h) + math.log(math.log(2 / alpha[a]))
            if log_n < math.log(2**52):
                try:
                    bound = 2 * B**2 / h**2 * math.log(2 / alpha[a])
                except (TypeError, OverflowError, ZeroDivisionError):
                    bound = math.inf
                n = max(1, math.ceil(bound if math.isfinite(bound) and bound else math.exp(log_n)))
        rows[a] = dict(status=status, epsilon=eps[a], alpha=alpha[a], bias=b,
                       allowance=u, headroom=h, shots=n, log_shot_bound=log_n)
    usable = all(rows[a]["status"] == "ELIGIBLE" and rows[a]["shots"] is not None for a in AXES)
    return dict(schema="fixed_complex_signal_hoeffding_v1", axes=rows, usable=usable,
                epsilon_sig=epsilon_sig, alpha_union=sum(alpha.values()), log_B_upper=log_B,
                N_total=sum(rows[a]["shots"] for a in AXES) if usable else None,
                evidence_kind=evidence_kind, evidence_ref=evidence_ref,
                certificate_verified=False, interpretation="CONDITIONAL_DIAGNOSTIC",
                estimator="B_times_fresh_independent_plus_minus_one_mean")
