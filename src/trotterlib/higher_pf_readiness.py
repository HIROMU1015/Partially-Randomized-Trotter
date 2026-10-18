"""Pure evaluation kernels for separately authorized high-order PR studies.

No chemistry loader, compiler, optimizer, sampler or filesystem side effects.
The CLI loads this module directly to avoid trotterlib's scientific imports.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
from statistics import NormalDist

FORMULAS = {"2nd": "S2-saved-v1", "4th": "Yoshida4-saved-binary64-v1",
            "4th(new_2)": "new2-printed8-saved-binary64-v1"}
SERIES = "H46-new2-representative-native-explicit-XY-ordinary-v2"
Q_GRID = (1, 2, 4, 8)
PROFILE = "1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def number(value, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("FINITE_NUMBER_REQUIRED")
    if not math.isfinite(value) or (positive and value <= 0) or (nonnegative and value < 0):
        raise ValueError("NUMBER_RANGE")
    return value


def integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError("INTEGER_RANGE")
    return value


def moment_audit(values, representation):
    """Exact rational moments of decimal strings or actual binary64 values.

Zero residual of floating arithmetic is deliberately not an exact certificate.
This routine never labels a binary approximation as an algebraic formula.
"""
    if representation == "DECIMAL_LITERAL":
        if not all(isinstance(v, str) for v in values):
            raise ValueError("DECIMAL_STRINGS_REQUIRED")
        w = tuple(Fraction(v) for v in values)
    elif representation == "BINARY64_RATIONAL":
        w = tuple(Fraction.from_float(float(number(v))) for v in values)
    else:
        raise ValueError("REPRESENTATION")
    if not w:
        raise ValueError("EMPTY_COMPOSITION")
    first, cubic = sum(w), sum(x**3 for x in w)
    return {"representation": representation, "first_exact": str(first),
            "cubic_exact": str(cubic), "palindrome": w == w[::-1],
            "negative_time": any(x < 0 for x in w),
            "rational_order_conditions_met": first == 1 and cubic == 0 and w == w[::-1],
            "algebraic_formula_adopted": False}


def exact_family_polynomial(a):
    """Coefficients ascending in b for 2a^3+2b^3+(1-2a-2b)^3.

An exact real root b with an isolating interval defines a separate formula;
choosing a root/interval or applying it to a molecule is not done here.
"""
    a = Fraction(a)
    c = 1 - 2*a
    return (c**3 + 2*a**3, -6*c**2, 12*c, Fraction(-6))


def signed_schedule(weights, q, T, allocation):
    integer(q, 1); number(T, positive=True)
    if len(weights) != len(allocation) or not weights:
        raise ValueError("SCHEDULE_DIMENSION")
    for w in weights:
        number(w)
    for count in allocation:
        integer(count, 1)
    if sum(allocation) != 32:
        raise ValueError("RTE_STEPS_PER_OUTER_MUST_BE_32")
    if tuple(weights) != tuple(weights)[::-1] or abs(math.fsum(weights)-1) > 1e-12:
        raise ValueError("SCHEDULE_COMPOSITION")
    return [{"outer": outer, "piece": piece, "signed_time": T*w/q,
             "rte_steps": allocation[piece]}
            for outer in range(q) for piece, w in enumerate(weights)]


def sample_summary(values):
    """Exact rational mean/variance of integer compiled costs; SE is a diagnostic."""
    if not values:
        raise ValueError("EMPTY_SAMPLE")
    for x in values:
        integer(x)
    n = len(values)
    mean = sum(Fraction(v) for v in values) / n
    variance = sum((Fraction(v)-mean)**2 for v in values)/(n-1) if n > 1 else None
    return {"n": n, "mean_exact": str(mean), "mean": float(mean),
            "variance_exact": str(variance) if variance is not None else None,
            "variance": float(variance) if variance is not None else None,
            "SE": math.sqrt(float(variance)/n) if variance is not None else None,
            "population_mean_certified": False, "unobserved_event_cost_imputed": False}


def paired_costs(records, shots, identity):
    """Pair event-identical full wrappers; do not average axes before weighting."""
    expected = {"SOURCE", "input_sha256", "preparation_hash", "compiler_settings_hash",
                "series", "control", "formula_id", "system", "prefix", "q", "stage"}
    if set(identity) != expected or identity["series"] != SERIES or identity["control"] != "ordinary":
        raise ValueError("COST_IDENTITY_SCHEMA_OR_SERIES")
    if identity["compiler_settings_hash"] != PROFILE or identity["formula_id"] not in FORMULAS.values():
        raise ValueError("COMPILER_OR_FORMULA")
    if identity["q"] not in Q_GRID or identity["stage"] not in ("EXPLORATION", "CONFIRMATION"):
        raise ValueError("COST_SCOPE")
    for key in ("real", "imag"):
        integer(shots[key], 1)
    pairs = {}
    for record in records:
        if any(record.get(key) != value for key, value in identity.items()):
            raise ValueError("SOURCE_INPUT_PREPARATION_COMPILER_SCOPE_MISMATCH")
        if record.get("forced_event") is not False or record.get("cost_kind") != "MEASURED":
            raise ValueError("FORCED_OR_NONMEASURED_COST_SAMPLE")
        sample = record.get("replica_id")
        if not isinstance(sample, str) or not sample or not record.get("event_digest") or not record.get("seed_namespace"):
            raise ValueError("TRAJECTORY_IDENTITY")
        axis = record.get("axis")
        if axis not in ("cosine", "sine") or axis in pairs.setdefault(sample, {}):
            raise ValueError("DUPLICATE_OR_UNKNOWN_AXIS")
        for metric in ("rz_count", "cx_count"):
            integer(record[metric])
        pairs[sample][axis] = record
    values = {"RZ": [], "CX": [], "cosine_RZ": [], "sine_RZ": []}
    for sample in sorted(pairs):
        if set(pairs[sample]) != {"cosine", "sine"}:
            raise ValueError("MISSING_PAIRED_AXIS")
        cos, sin = pairs[sample]["cosine"], pairs[sample]["sine"]
        for key in ("event_digest", "seed_namespace"):
            if cos[key] != sin[key]:
                raise ValueError("UNPAIRED_EVENT_OR_SEED")
        # Repeated event outcomes are possible; replica identity must be unique.
        values["RZ"].append(shots["real"]*cos["rz_count"] + shots["imag"]*sin["rz_count"])
        values["CX"].append(shots["real"]*cos["cx_count"] + shots["imag"]*sin["cx_count"])
        values["cosine_RZ"].append(cos["rz_count"])
        values["sine_RZ"].append(sin["rz_count"])
    if not pairs:
        raise ValueError("EMPTY_SAMPLE")
    return {"summaries": {k: sample_summary(v) for k, v in values.items()},
            "paired_values": values, "replica_ids": sorted(pairs),
            "independence_between_formulas_certified": False}


def evaluate_row(row, shot_rule):
    """Saved finite signal only. No order-based bias prediction or interpolation."""
    bias = {axis: (abs(number(row["total_signed_bias"][axis]))
                   if row["total_signed_bias"][axis] is not None else None)
            for axis in ("real", "imag")}
    log_b = number(row["log_B"], nonnegative=True)
    shots = shot_rule(bias, {"real": 1e-9, "imag": 1e-9}, epsilon=.001,
                      log_B_upper=log_b+1e-12, evidence_kind="EMPIRICAL",
                      evidence_ref=row["signal_evidence"], alpha_axis=.025)
    status = shots["eligibility_under_declared_allowance"]
    result = {"candidate_id": row["candidate_id"], "system": row["identity"]["system"],
              "formula_id": row["identity"]["formula_id"], "q": row["identity"]["q"],
              "shots": shots, "eligibility": status, "cost_kind": row["cost_kind"],
              "cost": None, "mean_exact": None, "population_mean_certified": False}
    if row["cost_kind"] not in ("MEASURED", "MODELED", "MISSING"):
        raise ValueError("COST_KIND")
    if row["cost_kind"] == "MEASURED" and (not row.get("records") or row.get("model") is not None):
        raise ValueError("MEASURED_COST_SCHEMA")
    if row["cost_kind"] == "MISSING" and (row.get("records") or row.get("model") is not None):
        raise ValueError("MISSING_IS_NOT_ZERO")
    if row["cost_kind"] == "MODELED":
        if row.get("records") or not row.get("model", {}).get("evidence"):
            raise ValueError("MODELED_COST_SCHEMA")
        for a in ("real", "imag"):
            number(row["model"][a], nonnegative=True)
        result["model_only_cost"] = (sum(shots["axes"][a]["shots"]*row["model"][a] for a in ("real", "imag"))
                                     if shots["N_total"] is not None else None)
        result["model_SE"] = None
    if status == "ELIGIBLE" and shots["N_total"] is not None and row["cost_kind"] == "MEASURED":
        counts = {a: shots["axes"][a]["shots"] for a in ("real", "imag")}
        result["cost"] = paired_costs(row["records"], counts, row["identity"])
        result["mean_exact"] = result["cost"]["summaries"]["RZ"]["mean_exact"]
    return result


def select_q(rows):
    if sorted(r["q"] for r in rows) != list(Q_GRID):
        raise ValueError("COMPLETE_PREREGISTERED_Q_GRID_REQUIRED")
    if len({(r["system"], r["formula_id"]) for r in rows}) != 1:
        raise ValueError("MIXED_Q_STRATUM")
    available = [r for r in rows if r["mean_exact"] is not None]
    best = min(available, key=lambda r: (Fraction(r["mean_exact"]), r["q"], r["candidate_id"])) if available else None
    unresolved = [r["candidate_id"] for r in rows if r["eligibility"] == "UNDETERMINED" or
                  (r["eligibility"] == "ELIGIBLE" and r["mean_exact"] is None)]
    return {"full_grid_winner": best["candidate_id"] if best and not unresolved else None,
            "available_subset_point_best": best["candidate_id"] if best else None,
            "unresolved": unresolved, "tie_break": "exact_mean_then_smaller_q_then_candidate_id",
            "status": "INCOMPLETE" if unresolved else "COMPLETE_POINT_SELECTION",
            "global_or_population_winner_certified": False}


def comparison(left, right, *, independent_candidates):
    """Independent-candidate plug-in delta-method diagnostics, not guaranteed CIs.

No CRN support without a new explicit coupling contract. Matching replica index
alone does not establish covariance across formulas/q.
"""
    if independent_candidates is not True:
        raise ValueError("CRN_COUPLING_NOT_ESTABLISHED")
    x, y = left["mean"], right["mean"]
    for v in (x, y):
        number(v, nonnegative=True)
    sx, sy = left["SE"], right["SE"]
    if sx is None or sy is None or y <= 0:
        return {"status": "STATISTICAL_CONCLUSION_DEFERRED", "ratio": None, "difference_SE": None}
    for v in (sx, sy):
        number(v, nonnegative=True)
    return {"status": "EXPLORATORY_PLUGIN_ONLY", "difference": x-y,
            "difference_SE": math.hypot(sx, sy), "ratio": x/y,
            "ratio_delta_SE": math.hypot(sx/y, x*sy/y**2),
            "independent_candidate_assumption": True, "rigorous_interval": None,
            "rare_unobserved_layers_covered": False}


def seed_for(campaign, stage, candidate, replica, outer, piece, input_hash, occupied=()):
    if not campaign or stage not in ("EXPLORATION", "CONFIRMATION") or not candidate or not input_hash:
        raise ValueError("SEED_NAMESPACE")
    for value in (replica, outer, piece):
        integer(value)
    key = {"version": "HIGHER_PF_INDEPENDENT_V1", "campaign": campaign, "stage": stage,
           "candidate": candidate, "replica": replica, "outer": outer, "piece": piece,
           "input_sha256": input_hash}
    seed = int(digest(key)[:16], 16)
    if seed in occupied:
        raise ValueError("SEED_COLLISION_REQUIRES_NEW_APPROVED_NAMESPACE")
    return seed


def freeze_selection(candidate_ids, exploration_seed_ids, metadata_digest):
    if not candidate_ids or not exploration_seed_ids or len(set(exploration_seed_ids)) != len(exploration_seed_ids):
        raise ValueError("FREEZE_SELECTION_SCHEMA")
    payload = {"candidates": sorted(set(candidate_ids)),
               "exploration_seed_ids": sorted(exploration_seed_ids),
               "evidence_sha256": metadata_digest}
    return {**payload, "selection_sha256": digest(payload)}


def check_confirmation(selection, candidate_ids, confirmation_seed_ids, *, preregistered_selection_sha256):
    body = {k: selection[k] for k in ("candidates", "exploration_seed_ids", "evidence_sha256")}
    if digest(body) != selection["selection_sha256"] or selection["selection_sha256"] != preregistered_selection_sha256:
        raise ValueError("SELECTION_NOT_PREREGISTERED")
    if sorted(set(candidate_ids)) != selection["candidates"]:
        raise ValueError("CONFIRMATION_CANDIDATE_CHANGE")
    if not confirmation_seed_ids or len(set(confirmation_seed_ids)) != len(confirmation_seed_ids):
        raise ValueError("CONFIRMATION_SEED_SCHEMA")
    if set(confirmation_seed_ids) & set(selection["exploration_seed_ids"]):
        raise ValueError("EXPLORATION_CONFIRMATION_OVERLAP")
    return {"namespace_disjoint": True, "random_variable_independence_proved": False,
            "note": "also audit PRNG algorithm, event law, sample registry and all legacy seeds"}


def planning_n(variance, *, absolute_mcse=None, relative_mcse=None, reference_mean=None):
    """Plug-in planning only; targets must be supplied by a later protocol."""
    number(variance, nonnegative=True)
    if (absolute_mcse is None) == (relative_mcse is None):
        raise ValueError("ONE_MCSE_TARGET_REQUIRED")
    if relative_mcse is not None:
        number(relative_mcse, positive=True); number(reference_mean, positive=True)
        absolute_mcse = relative_mcse * reference_mean
    number(absolute_mcse, positive=True)
    return {"n_plugin": max(2, math.ceil(variance / absolute_mcse**2)),
            "adopted_n": None, "rare_tail_guarantee": False,
            "zero_variance_pilot_is_not_constant_cost_proof": variance == 0}


def planning_equal_n_difference(variance_left, variance_right, effect, *, alpha, power, comparisons):
    """Normal approximation + Bonferroni planning; not a power guarantee."""
    for v in (variance_left, variance_right):
        number(v, nonnegative=True)
    number(effect, positive=True); integer(comparisons, 1)
    if not 0 < alpha < 1 or not .5 < power < 1:
        raise ValueError("PLANNING_ALPHA_POWER")
    z = NormalDist().inv_cdf(1-alpha/(2*comparisons)) + NormalDist().inv_cdf(power)
    return {"equal_n_plugin": max(2, math.ceil((variance_left+variance_right)*z*z/effect**2)),
            "adopted_n": None, "coverage_or_power_guarantee": False}
