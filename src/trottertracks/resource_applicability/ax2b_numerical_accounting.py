"""Declared-allowance diagnostics; no molecular I/O or certification engine."""
from __future__ import annotations

import math
import sys

from .ax2a_preparation import axis_headroom

AXES = ("real", "imag")


def _number(value, name, *, minimum=0.0):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(name + ": finite real required")
    value = float(value)
    if not math.isfinite(value) or value < minimum:
        raise ValueError(name + ": out of range")
    return value


def u_aware_shots(bias, allowance, *, epsilon, log_B_upper,
                  evidence_kind="UNKNOWN", evidence_ref=None, alpha_axis=0.025,
                  normalization_upper=None):
    """Three states under the symmetric-axis rule, with conditional labels.

    Allowance is the *combined* candidate/reference axis uncertainty. A caller
    must justify both it and the normalization upper value. No certificate is
    inferred from agreement, a residual, an empirical discrepancy, or this API.
    Integer shots use the legacy arithmetic order in its safe range; larger
    values remain log-domain diagnostics, rather than unreliable integers.
    """
    if set(bias) != set(AXES) or set(allowance) != set(AXES):
        raise ValueError("AXIS_SCHEMA")
    if evidence_kind not in ("UNKNOWN", "EMPIRICAL", "CERTIFIED"):
        raise ValueError("EVIDENCE_KIND")
    if evidence_kind != "UNKNOWN" and not evidence_ref:
        raise ValueError("NUMERICAL_EVIDENCE_REFERENCE_REQUIRED")
    epsilon = _number(epsilon, "epsilon")
    alpha_axis = _number(alpha_axis, "alpha_axis")
    if epsilon == 0 or not 0 < alpha_axis < 1:
        raise ValueError("PRECISION_OR_FAILURE_ALLOCATION")
    if log_B_upper is not None:
        log_B_upper = _number(log_B_upper, "log_B_upper")
    B = (math.exp(log_B_upper) if log_B_upper is not None
         and log_B_upper <= math.log(sys.float_info.max) else None)
    if normalization_upper is not None:
        B = _number(normalization_upper, "normalization_upper", minimum=1.0)
        if log_B_upper is None or math.log(B) != log_B_upper:
            raise ValueError("NORMALIZATION_LOG_BINDING")
        # Preserve the saved binary64 normalization for exact legacy integer
        # reproduction. exp(log(saved_B)) can change its last bit.
    rows = {}
    for axis in AXES:
        b, u = bias[axis], allowance[axis]
        if b is not None:
            b = _number(b, "bias")
        if u is not None:
            u = _number(u, "allowance")
        if evidence_kind == "UNKNOWN":
            u = None
        row = axis_headroom(epsilon / math.sqrt(2), b, u)
        row.update(shots=None, log_shot_bound=None, integer_status="NOT_APPLICABLE",
                   allowance=u, u_over_headroom=None)
        h = row["headroom"]
        if row["status"] == "ELIGIBLE" and log_B_upper is not None:
            log_n = (math.log(2) + 2 * log_B_upper - 2 * math.log(h)
                     + math.log(math.log(2 / alpha_axis)))
            row["log_shot_bound"] = log_n
            ratio = u / h
            row["u_over_headroom"] = ratio if math.isfinite(ratio) else None
            if log_n < math.log(2**52):
                # This operation order reproduces AX-1b when u=0, alpha=.025.
                try:
                    bound = 2 * B**2 / h**2 * math.log(2 / alpha_axis)
                except (TypeError, OverflowError, ZeroDivisionError):
                    bound = math.inf
                if not math.isfinite(bound) or bound == 0:
                    bound = math.exp(log_n)
                row.update(shots=max(1, math.ceil(bound)), integer_status="BINARY64_DIAGNOSTIC")
            else:
                row["integer_status"] = "LOG_DOMAIN_ONLY"
        rows[axis] = row
    statuses = {row["status"] for row in rows.values()}
    status = ("INELIGIBLE" if "INELIGIBLE" in statuses else
              "ELIGIBLE" if statuses == {"ELIGIBLE"} else "UNDETERMINED")
    total = (sum(row["shots"] for row in rows.values())
             if status == "ELIGIBLE" and all(row["shots"] is not None for row in rows.values()) else None)
    return {"schema": "track_a_u_aware_diagnostic_v1", "evidence_kind": evidence_kind,
            "evidence_ref": evidence_ref, "allowance_certification_claimed_by_caller": evidence_kind == "CERTIFIED",
            "certificate_verified_by_this_function": False,
            "eligibility_under_declared_allowance": status, "axes": rows, "N_total": total,
            "alpha_axis": alpha_axis, "familywise_winner_certified": False,
            "interpretation": "CONDITIONAL_DIAGNOSTIC"}


def propagate_allowance(local_errors, stage_norm_upper, *, initial_error=0.0):
    """e[j+1] <= L[j]*e[j]+eta[j], only as strong as caller's inputs."""
    if len(local_errors) != len(stage_norm_upper):
        raise ValueError("STAGE_LENGTH")
    error = _number(initial_error, "initial_error")
    history = []
    for eta, norm in zip(local_errors, stage_norm_upper, strict=True):
        error = _number(norm, "stage norm") * error + _number(eta, "local error")
        if not math.isfinite(error):
            raise ValueError("ALLOWANCE_PROPAGATION_OVERFLOW")
        history.append(error)
    return {"final": error, "stages": history, "certificate_verified": False}
