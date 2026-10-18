"""Copy saved scalars without numerical reanalysis; decode only after a grant.

V2 keeps normalization and signed-bias transformations as declarative recipes.
Preparing a projection never evaluates exp, abs/sums, headroom, shots or costs.
"""
from copy import deepcopy
import math

AXES = ("real", "imag")
SCHEMA = "PR_ALPHA_BOUND_NATIVE_SCALARS_V2"


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def scalar_axes(value):
    return {axis: deepcopy(value[axis]) for axis in AXES}


def make_record(*, system, cell_id, epsilon, mode, signal, baseline, cost,
                lineage, noncancellation=None):
    """Metadata projection only: saved floats are copied, never transformed."""
    require(mode in ("USUAL_SIGNED_SAVED", "EMPIRICAL_NONCANCELLATION"), "BIAS_MODE")
    r = dict(system=system, cell_id=cell_id, epsilon=epsilon, bias_kind=mode,
             allowance_factor=1., method=signal["cell"]["method"] if signal else
             ("B2" if cell_id.startswith("B2_") else cell_id.split("_")[0]),
             source_fields=deepcopy(lineage), scalar_encoding="SAVED_RECIPES_V1")
    missing = []
    if signal is None:
        missing.append("original_saved_signal")
    if baseline is None:
        missing.append("saved_baseline_accounting")
    if mode == "EMPIRICAL_NONCANCELLATION" and noncancellation is None and (
            signal is None or not signal["error_decomposition"].get("signed_parts")):
        missing.append("saved_signed_decomposition")
    if missing:
        r["missing_fields"] = missing
        return r
    require(signal["cell"]["id"] == cell_id, "SIGNAL_IDENTITY")
    allowance = signal["allowance"]
    require(allowance["kind"] == "EMPIRICAL" and allowance["certified"] is False,
            "SAVED_EMPIRICAL_ALLOWANCE")
    corrected = scalar_axes(signal["signals"]["corrected"])
    raw = signal["signals"].get("raw")
    if raw is None:
        require(r["method"] != "B2" and signal["log_B"] == 0., "MISSING_PR_RAW")
        raw = corrected
        raw_policy = "DETERMINISTIC_IDENTITY_B_EQUALS_ONE"
    else:
        raw_policy = "SAVED_RAW"
    if mode == "USUAL_SIGNED_SAVED":
        recipe = dict(kind="ABS_SAVED_AXIS_TOTAL", values=scalar_axes(signal["error_decomposition"]["total_signed"]))
    elif noncancellation is not None:
        recipe = dict(kind="COPY_SAVED_MAGNITUDE", values=scalar_axes(noncancellation))
    else:
        parts = signal["error_decomposition"]["signed_parts"]
        require(set(parts) in ({"discard", "PF"}, {"outer_PF", "finite_RTE"}, {"PF"}), "SIGNED_COMPONENT_SET")
        recipe = dict(kind="SUM_ABS_SAVED_PARTS", ordered_terms=[scalar_axes(v) for v in parts.values()], names=list(parts))
    r.update(contract="TRACK_A_NATIVE_SIGNAL_V1", bias_recipe=recipe,
        allowance=scalar_axes(allowance), log_B=deepcopy(signal["log_B"]),
        log_B_margin=deepcopy(allowance["log_B_margin"]),
        normalization_recipe="EXP_LOG_B_CORR_AND_EXP_LOG_B_PLUS_FACTOR_MARGIN_BUDGET",
        normalization_role="B_CORR_IS_ESTIMATOR_MULTIPLIER", budget_role="B_BUDGET_IS_BOUND_ONLY",
        evidence_kind=baseline["evidence_kind"], evidence_ref=baseline["evidence_ref"],
        certificate_verified=False, population_winner_certified=False,
        probability_evidence="FRESH_IID_HADAMARD_MODEL", fresh_iid=True,
        axis_sign_convention="BIT0_PLUS1_COS_REAL_SDG_SIN_IMAG", raw_policy=raw_policy,
        corrected=corrected, raw=scalar_axes(raw), reference=scalar_axes(signal["reference"]),
        saved_error_decomposition=deepcopy(signal["error_decomposition"]),
        baseline_accounting=deepcopy(baseline), cost=deepcopy(cost))
    return r


def decode_record(record):
    """Evaluate input recipes inside an independently approved execution only."""
    r = deepcopy(record)
    if r.get("missing_fields"):
        return r
    require(r.get("scalar_encoding") == "SAVED_RECIPES_V1", "SCALAR_ENCODING")
    require(r.get("normalization_recipe") == "EXP_LOG_B_CORR_AND_EXP_LOG_B_PLUS_FACTOR_MARGIN_BUDGET", "NORMALIZATION_RECIPE")
    for value in (r["log_B"], r["log_B_margin"]):
        require(type(value) in (int, float) and math.isfinite(value) and value >= 0, "INVALID_SAVED_LOG_NORMALIZATION")
    recipe = r["bias_recipe"]
    if recipe["kind"] == "ABS_SAVED_AXIS_TOTAL":
        r["bias"] = {a: abs(recipe["values"][a]) for a in AXES}
    elif recipe["kind"] == "COPY_SAVED_MAGNITUDE":
        r["bias"] = scalar_axes(recipe["values"])
    elif recipe["kind"] == "SUM_ABS_SAVED_PARTS":
        require(recipe["ordered_terms"] and len(recipe["ordered_terms"]) == len(recipe["names"]), "EMPTY_SIGNED_PARTS")
        r["bias"] = {a: sum(abs(term[a]) for term in recipe["ordered_terms"]) for a in AXES}
    else:
        raise ValueError("UNKNOWN_BIAS_RECIPE")
    # B2 log-domain accounting does not need exp(B), which could overflow.
    # Only B1 consumes explicit multipliers; defer their evaluation to B1.
    return r


def decode_binomial_normalization(record):
    r = deepcopy(record)
    if not r.get("missing_fields"):
        r["B_corr"] = math.exp(r["log_B"])
        r["B_budget"] = math.exp(r["log_B"] + r["allowance_factor"] * r["log_B_margin"])
    return r
