"""Pure Track A native-signal B-1/B-2 calculations; no real input I/O.

The public functions accept already bound scalar projections. Runners must
validate an independently approved one-shot grant before loading real inputs.
SciPy is imported lazily only inside the binomial backend. No sampling exists.
"""
from __future__ import annotations

from fractions import Fraction as F
import hashlib
import importlib.util
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
AXES = ("real", "imag")
CONTRACT = "TRACK_A_NATIVE_SIGNAL_V1"
RULE_SHA = "ed3f57937a231719583c2893677b7e82f934dd98bda35cd37fc0f17f83d4279f"
_RULE = None


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def rational(value):
    require(not isinstance(value, bool), "BOOLEAN_NOT_NUMBER")
    if isinstance(value, float):
        require(math.isfinite(value), "NONFINITE")
        return F.from_float(value)  # exact stored binary64, no epsilon snapping
    try:
        return F(value)
    except (ValueError, TypeError, OverflowError, ZeroDivisionError) as exc:
        raise ValueError("INVALID_RATIONAL") from exc


def frozen_rule():
    global _RULE
    if _RULE is None:
        path = ROOT / "scripts/d0_frozen_shot_rule_v1.py"
        require(hashlib.sha256(path.read_bytes()).hexdigest() == RULE_SHA, "FROZEN_RULE_CHANGED")
        spec = importlib.util.spec_from_file_location("_alpha_frozen_rule", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # inspected stdlib-only frozen function
        _RULE = module
    return _RULE


def native_record(record):
    require(record.get("contract") == CONTRACT, "WRONG_CONTRACT_NO_RPE_OR_H4_TRANSFER")
    require(record.get("system") in ("H6", "H8"), "SYSTEM_SCOPE")
    require(record.get("method") in ("B0", "B0I", "B1", "B2"), "METHOD_SCOPE")
    require(record.get("bias_kind") in ("USUAL_SIGNED_SAVED", "EMPIRICAL_NONCANCELLATION"), "BIAS_SCOPE")
    require(type(record.get("allowance_factor")) in (int, float) and record["allowance_factor"] in (1., 10.), "ALLOWANCE_FACTOR")
    require(record.get("evidence_kind") in ("UNKNOWN", "EMPIRICAL", "CERTIFIED"), "EVIDENCE_KIND")
    require(record.get("normalization_role") == "B_CORR_IS_ESTIMATOR_MULTIPLIER", "NORMALIZATION_ROLE")
    require(record.get("budget_role") == "B_BUDGET_IS_BOUND_ONLY", "BUDGET_ROLE")
    require(record.get("certificate_verified") is False, "CERTIFICATE_PROMOTION")
    require(record.get("population_winner_certified") is False, "WINNER_PROMOTION")
    require(record.get("probability_evidence") == "FRESH_IID_HADAMARD_MODEL", "COST_SE_IS_NOT_ALPHA_EVIDENCE")
    require(set(record["bias"]) == set(record["allowance"]) == set(AXES), "AXIS_SCHEMA")
    for axis in AXES:
        for name in ("bias", "allowance"):
            value = record[name][axis]
            if value is not None:
                require(rational(value) >= 0, "NEGATIVE_BIAS_OR_ALLOWANCE")


def binomial_boundaries(n, corrected, B_corr, headroom, event="ge"):
    require(type(n) is int and 0 < n < 2**52, "INTEGER_SHOT_DOMAIN")
    require(event in ("ge", "gt"), "FAILURE_EVENT")
    z, B, h = map(rational, (corrected, B_corr, headroom))
    require(B >= 1 and h > 0, "NORMALIZATION_OR_HEADROOM")
    require(abs(z) <= B, "PROBABILITY_DOMAIN_NO_CLIPPING")
    low = F(n, 2) * (1 + (z - h) / B)
    high = F(n, 2) * (1 + (z + h) / B)
    floor = lambda v: v.numerator // v.denominator
    ceil = lambda v: -((-v.numerator) // v.denominator)
    lower_last = floor(low) if event == "ge" else ceil(low) - 1
    upper_first = ceil(high) if event == "ge" else floor(high) + 1
    return dict(lower_last=min(n, max(-1, lower_last)),
                upper_first=min(n + 1, max(0, upper_first)),
                p_exact=(1 + z / B) / 2, event=event)


def scipy_tail(n, p, lower_last, upper_first):
    """Exact binomial distribution, floating numerical evaluation (not MC)."""
    require(type(n) is int and n < 2**52, "INTEGER_BACKEND_DOMAIN")
    q = float(p)
    require(0 <= q <= 1 and (p in (0, 1) or 0 < q < 1), "P_CONVERSION_LOST_INTERIOR")
    from scipy.stats import binom
    terms, logs = [], []
    if p in (0, 1):
        k = 0 if p == 0 else n
        value = float(k <= lower_last or k >= upper_first)
        return dict(probability=value, log_probability=math.log(value) if value else None,
                    numerical_status="DEGENERATE_EXACT", underflow=False)
    for side, index in (("left", lower_last), ("right", upper_first)):
        if (side == "left" and index < 0) or (side == "right" and index > n):
            terms.append(0.); logs.append(-math.inf); continue
        if side == "left":
            value, logv = float(binom.cdf(index, n, q)), float(binom.logcdf(index, n, q))
        else:
            value, logv = float(binom.sf(index - 1, n, q)), float(binom.logsf(index - 1, n, q))
        require(math.isfinite(value) and 0 <= value <= 1 and not math.isnan(logv), "BINOMIAL_BACKEND_NONFINITE")
        if value == 0. or not math.isfinite(logv):
            return dict(probability=None, log_probability=None,
                        numerical_status="UNRESOLVED_NONEMPTY_TAIL_UNDERFLOW", underflow=True)
        terms.append(value); logs.append(logv)
    value = math.fsum(terms)
    require(value <= 1 + 8 * math.ulp(1.), "BINOMIAL_SUM_DOMAIN")
    require(value <= 1, "BINOMIAL_SUM_ROUNDOFF_UNRESOLVED")
    return dict(probability=value, log_probability=math.log(value) if value else None,
                numerical_status="EXACT_DISTRIBUTION_FLOAT_EVALUATION", underflow=False)


def binomial_axis(*, n, corrected, raw, B_corr, B_budget, headroom, allowance,
                  alpha_axis=.025, event="ge", backend=scipy_tail):
    z, r, B, budget, h, u, alpha = map(rational,
        (corrected, raw, B_corr, B_budget, headroom, allowance, alpha_axis))
    require(B >= 1 and budget >= B and h > 0 and u >= 0, "NORMALIZATION_BOUND_OR_HEADROOM")
    require(0 < alpha < 1, "ALPHA_DOMAIN")
    require(abs(r) <= 1 and abs(B * r - z) <= u, "RAW_CORRECTED_MISMATCH_NO_REPLACEMENT")
    bounds = binomial_boundaries(n, z, B, h, event)
    tail = backend(n, bounds["p_exact"], bounds["lower_last"], bounds["upper_first"])
    log_hoeffding = math.log(2.) - float(F(n) * h * h / (2 * budget * budget))
    hoeffding = math.exp(log_hoeffding) if log_hoeffding > -745 else None
    probability = tail["probability"]
    if probability is None:
        status = "UNDETERMINED_NUMERICAL"
    elif abs(probability - float(alpha)) <= 32 * math.ulp(float(alpha)):
        status = "UNDETERMINED_AT_NUMERICAL_ALPHA_BOUNDARY"
    else:
        status = "NUMERICALLY_WITHIN_BUDGET" if probability < float(alpha) else "EXCEEDS_BUDGET"
    return dict(n=n, event=event, lower_last=bounds["lower_last"], upper_first=bounds["upper_first"],
        p_rational=str(bounds["p_exact"]), binomial=tail,
        hoeffding_bound=min(1., hoeffding) if hoeffding is not None else None,
        log_hoeffding_bound=min(0., log_hoeffding), alpha_axis=float(alpha), alpha_compliance=status,
        rigorous_signal_precision_certified=False, bias_allowance_conditional=True,
        floating_probability_is_interval_certificate=False)


def binomial_case(record, *, backend=scipy_tail):
    native_record(record)
    require(record.get("fresh_iid") is True, "FRESH_IID_REQUIRED")
    require(record.get("axis_sign_convention") == "BIT0_PLUS1_COS_REAL_SDG_SIN_IMAG", "HADAMARD_SIGN_CONVENTION")
    a = record["baseline_accounting"]
    require(a["alpha_axis"] == .025 and a["cost_acquisition_eligible"] is True, "SAVED_SHOT_CONTRACT")
    require(a["evidence_kind"] == record["evidence_kind"] and
            a["certificate_verified_by_this_function"] is False, "SAVED_EVIDENCE_CLASS")
    require(a["eligibility_under_declared_allowance"] == "ELIGIBLE", "SAVED_ELIGIBILITY")
    require(record["B_budget"] == math.exp(record["log_B"] + record["allowance_factor"] * record["log_B_margin"]),
            "SAVED_BUDGET_LOG_BINDING")
    axes = {}
    for axis in AXES:
        saved = a["axes"][axis]
        require(saved["integer_status"] == "BINARY64_DIAGNOSTIC", "NO_FINITE_SAVED_SHOT")
        u = record["allowance_factor"] * record["allowance"][axis]
        require(saved["allowance"] == u and saved["headroom"] == record["epsilon"] / math.sqrt(2) - record["bias"][axis] - u,
                "SAVED_HEADROOM_BINDING")
        axes[axis] = binomial_axis(n=saved["shots"], corrected=record["corrected"][axis],
            raw=record["raw"][axis], B_corr=record["B_corr"], B_budget=record["B_budget"],
            headroom=saved["headroom"], allowance=saved["allowance"], event="ge", backend=backend)
    values = [v["binomial"]["probability"] for v in axes.values()]
    return dict(cell_id=record["cell_id"], system=record["system"], epsilon=record["epsilon"], axes=axes,
        union_bound=min(1., math.fsum(values)) if all(v is not None for v in values) else None,
        union_bound_kind="SUM_OF_MARGINALS_NO_AXIS_INDEPENDENCE_ASSUMED",
        exact_joint_failure_probability=None, alpha_total=.05,
        rigorous_signal_precision_certified=False, population_cost_certified=False)


def reaccount(record, alpha_total):
    native_record(record)
    total = rational(alpha_total)
    require(0 < total < 1, "ALPHA_DOMAIN")
    alpha_axis = float(total / 2)
    require(alpha_axis > 0 and alpha_axis + alpha_axis == float(total), "ALPHA_SPLIT")
    f = record["allowance_factor"]
    allowance = {a: None if record["allowance"][a] is None else f * record["allowance"][a] for a in AXES}
    a = frozen_rule().u_aware_shots(record["bias"], allowance, epsilon=record["epsilon"],
        log_B_upper=record["log_B"] + f * record["log_B_margin"],
        evidence_kind=record["evidence_kind"], evidence_ref=record["evidence_ref"], alpha_axis=alpha_axis)
    if float(total) == .05:
        saved = record.get("baseline_accounting")
        require(saved is not None, "BASELINE_ACCOUNTING_MISSING")
        for key in ("alpha_axis", "N_total", "axes", "evidence_kind", "eligibility_under_declared_allowance"):
            require(a[key] == saved[key], "BASELINE_REPRODUCTION:" + key)
    status = a["eligibility_under_declared_allowance"]
    if any(x["integer_status"] == "LOG_DOMAIN_ONLY" for x in a["axes"].values()):
        status = "LOG_DOMAIN_ONLY"
    elif status == "ELIGIBLE" and (a["N_total"] is None or any(x["u_over_headroom"] is None or x["u_over_headroom"] > .01 for x in a["axes"].values())):
        status = "UNDETERMINED"
    result = dict(cell_id=record["cell_id"], system=record["system"], method=record["method"],
        epsilon=record["epsilon"], bias_kind=record["bias_kind"], allowance_factor=f,
        alpha_total=float(total), alpha_real=alpha_axis, alpha_imag=alpha_axis,
        status=status, accounting=a, G_exact=None, cost_SE=None,
        population_winner_certified=False, cost_uncertainty_is_alpha_evidence=False)
    if status != "ELIGIBLE":
        return result
    cost = record.get("cost")
    if not cost or cost["kind"] == "MISSING":
        result["status"] = "MISSING"; return result
    counts = {axis:a["axes"][axis]["shots"] for axis in AXES}
    if record["method"] == "B2":
        require(cost["kind"] == "PR_PAIRED_MEASURED" and cost["phase"] == "confirmation", "PR_CONFIRMATION_COST_REQUIRED")
        pairs = cost["pairs"]
        require(len(pairs) == cost["sample_count"] == 32, "CONFIRMATION_N32_REQUIRED")
        require(len({p["replica"] for p in pairs}) == 32 and len({p["seed"] for p in pairs}) == 32, "PAIRED_IDENTITY")
        require(all(p.get("event_digest") and p.get("compiler_hash") for p in pairs), "PAIRED_PROVENANCE")
        require(len({p["compiler_hash"] for p in pairs}) == 1, "PAIRED_COMPILER_CHANGED")
        require(all(type(p[axis]) is int and p[axis] >= 0 for p in pairs for axis in AXES), "INTEGER_COMPILED_COST")
        totals = [sum(counts[a] * p[a] for a in AXES) for p in pairs]
        point = F(sum(totals), 32)
        result.update(cost_SE=statistics.stdev(totals)/math.sqrt(32), paired_samples=32,
                      uncertainty_scope="PAIRED_SAMPLE_SE_NOT_CI")
        means = {axis:F(sum(p[axis] for p in pairs),32) for axis in AXES}
    else:
        require(cost["kind"] in ("DETERMINISTIC_MEASURED", "DETERMINISTIC_MODEL", "DETERMINISTIC_MIXED"), "DETERMINISTIC_COST_CLASS")
        if cost["kind"] == "DETERMINISTIC_MIXED":
            require(set(cost.get("axis_classes", {})) == set(AXES) and
                    set(cost["axis_classes"].values()) == {"MEASURED", "MODEL"}, "MIXED_AXIS_COST_CLASS")
            result["axis_cost_classes"] = dict(cost["axis_classes"])
        means = {axis:rational(cost[axis]) for axis in AXES}
        require(all(v > 0 for v in means.values()), "MISSING_IS_NOT_ZERO_COST")
        point = sum(counts[a] * means[a] for a in AXES)
    continuous = {}
    B = math.exp(record["log_B"] + f * record["log_B_margin"])
    for axis in AXES:
        h = a["axes"][axis]["headroom"]
        continuous[axis] = 2 * B**2 / h**2 * math.log(2 / alpha_axis)
    result.update(G_exact=str(point), cost_class=cost["kind"],
        continuous_shots_diagnostic=continuous,
        ceil_margin_diagnostic={axis:counts[axis]-continuous[axis] for axis in AXES},
        continuous_G_diagnostic=sum(float(means[axis])*continuous[axis] for axis in AXES))
    return result


def compare(rows):
    require(rows, "EMPTY_COMPARISON")
    keys = ("system", "epsilon", "bias_kind", "allowance_factor", "alpha_total")
    require(all(tuple(r[k] for k in keys) == tuple(rows[0][k] for k in keys) for r in rows), "UNEQUAL_PR_PF_CONDITIONS")
    require(len({r["cell_id"] for r in rows}) == len(rows), "DUPLICATE_CANDIDATE")
    best = {}
    for family in ("PR", "PF"):
        group = [r for r in rows if (r["method"] == "B2") == (family == "PR")]
        require(group, "MISSING_FAMILY")
        unresolved = [r["cell_id"] for r in group if r["status"] not in ("ELIGIBLE", "INELIGIBLE")]
        available = [r for r in group if r["G_exact"] is not None]
        if unresolved:
            best[family] = dict(status="UNRESOLVED_FULL_DOMAIN", best=None, unresolved=unresolved)
        elif not available:
            best[family] = dict(status="NO_ELIGIBLE_CANDIDATE", best=None)
        else:
            winner = min(available, key=lambda r:(F(r["G_exact"]),r["cell_id"]))
            best[family] = dict(status="CONDITIONAL_POINT_MINIMUM", best=winner)
    p, d = (best[k]["best"] for k in ("PR", "PF"))
    ratio = F(p["G_exact"]) / F(d["G_exact"]) if p and d else None
    return dict(**{k:rows[0][k] for k in keys}, families=best,
        PR_over_PF_exact=str(ratio) if ratio is not None else None,
        PR_reduction_percent_exact=str(100*(1-ratio)) if ratio is not None else None,
        rank="PR_LOWER" if ratio is not None and ratio < 1 else "PF_LOWER" if ratio is not None and ratio > 1 else "TIE" if ratio == 1 else "UNRESOLVED",
        population_winner_certified=False)
