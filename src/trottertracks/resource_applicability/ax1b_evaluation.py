"""Saved-cost evaluators. Pure arithmetic; no artifacts or scientific modules."""
from __future__ import annotations

import math

from .ax1b_contract import AXES, CASE_CONDITIONAL, canonical, digest, number, require


def cost_error(reference, prediction, unit="native_rz_count_per_shot"):
    require(unit == "native_rz_count_per_shot", "SCHEMA", "action index cannot be evaluated as RZ prediction")
    number(reference, "C_ref")
    if prediction is None:
        return dict(status="INPUT_MISSING", signed_relative=None, absolute_rz=None, absolute_relative=None,
                    log_ratio=None, underestimation_fraction=None)
    number(prediction, "C_pred")
    absolute = abs(prediction - reference)
    if reference == 0:
        return dict(status="ZERO_REFERENCE_COST", signed_relative=None, absolute_rz=absolute, absolute_relative=None,
                    log_ratio=None, underestimation_fraction=None)
    signed = (prediction - reference) / reference
    return dict(status="ZERO_PREDICTION" if prediction == 0 else "ERROR_DEFINED", signed_relative=signed,
                absolute_rz=absolute, absolute_relative=abs(signed), log_ratio=math.log(prediction / reference) if prediction else None,
                underestimation_fraction=max(0.0, -signed))


def error_summary(rows):
    """Every registered candidate/axis row retained, including missing/zero C."""
    valid = [r for r in rows if r["signed_relative"] is not None]
    scores = [r["absolute_relative"] for r in valid]
    sorted_scores = sorted(scores)
    n = len(scores)
    median = (sorted_scores[(n-1)//2] + sorted_scores[n//2]) / 2 if n else None
    return dict(registered_rows=len(rows), defined_predictions=sum(r["absolute_rz"] is not None for r in rows),
                coverage=sum(r["absolute_rz"] is not None for r in rows) / len(rows) if rows else None,
                missing_count=sum(r["status"] == "INPUT_MISSING" for r in rows),
                status_counts={s:sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})},
                positive_ref_denominator=n, mean_absolute_relative=math.fsum(scores)/n if n else None,
                median_absolute_relative=median, max_absolute_relative=max(scores) if n else None,
                underestimate_rate=sum(r["signed_relative"] < 0 for r in valid)/n if n else None,
                underestimate_more_than_10pct_rate=sum(r["signed_relative"] < -0.10 for r in valid)/n if n else None)


def rank_index(xs, ys):
    require(len(xs) == len(ys), "SCHEMA", "rank row alignment")
    if not xs:
        return dict(status="N_A_EMPTY_SUPPORT", spearman=None, unit="index_rank_only")
    def ranks(values):
        for v in values:
            number(v, "rank input")
        order = sorted(range(len(values)), key=lambda i: values[i])
        result = [0.0] * len(values)
        j = 0
        while j < len(order):
            end = j + 1
            while end < len(order) and values[order[end]] == values[order[j]]:
                end += 1
            for idx in order[j:end]:
                result[idx] = (j + 1 + end) / 2
            j = end
        return result
    x, y = ranks(xs), ranks(ys)
    mx, my = math.fsum(x)/len(x), math.fsum(y)/len(y)
    xx, yy = math.fsum((a-mx)**2 for a in x), math.fsum((b-my)**2 for b in y)
    rho = math.fsum((a-mx)*(b-my) for a,b in zip(x,y))/math.sqrt(xx*yy) if xx and yy else None
    return dict(status="INDEX_RANK_DIAGNOSTIC" if rho is not None else "N_A_CONSTANT_RANK", spearman=rho, unit="index_rank_only", denominator=len(xs))


def reference_shots(bias, B, epsilon):
    """I4 reference accounting, never an operational bias/shot predictor."""
    number(epsilon, "epsilon", positive=True)
    require(set(bias) == {"real", "imag"}, "SCHEMA", "axis bias keys")
    for b in bias.values():
        number(b, "reference bias")
    number(B, "B", positive=True)
    require(B >= 1, "SCHEMA", "normalization below1")
    boundary = math.sqrt(2) * max(bias.values())
    margins = {a:epsilon/math.sqrt(2)-b for a,b in bias.items()}
    shots = {}
    for a, margin in margins.items():
        if margin <= 0 or epsilon <= math.sqrt(2)*bias[a]:
            shots[a] = None
        else:
            try:
                bound = 2 * B**2 / margin**2 * math.log(2/0.025)
            except (OverflowError, ZeroDivisionError):
                bound = math.inf
            if not math.isfinite(bound):
                return dict(eligible_ref=None, axis_shots=None, status="NUMERICAL_UNDEFINED", information_class="I4_CONDITIONAL_ORACLE")
            shots[a] = math.ceil(bound)
    eligible = epsilon > boundary and all(n is not None for n in shots.values())
    return dict(eligible_ref=eligible, axis_shots=shots, N_total=sum(shots.values()) if eligible else None,
                epsilon_min=boundary, margins=margins, boundary_sensitive=abs(epsilon-boundary)<=1e-12,
                status="REFERENCE_ACCOUNTING_ONLY", information_class="I4_CONDITIONAL_ORACLE")


def conditional_work(shots, costs):
    if shots is None or costs is None or any(shots.get(k) is None for k in ("real", "imag")) or any(costs.get(k) is None for k in AXES):
        return None
    return math.fsum(number(shots[s], "N_ref") * number(costs[c], "one-shot cost") for s,c in [("real","cosine"),("imag","sine")])


def selection(refs, predicted_G, diagnostic_kind, chosen=None, predicted_eligible=None):
    """Fixed direct set. Explicit chosen is an audit path for failure statuses."""
    base = dict(evaluation_case_id=CASE_CONDITIONAL, diagnostic_kind=diagnostic_kind,
                information_class="I4_CONDITIONAL_ORACLE", regret=None, common_support_regret=None,
                selected=None, false_acceptance="NOT_APPLICABLE_ORACLE_ELIGIBILITY", independent_test=False)
    if chosen is not None:
        if chosen not in refs:
            return {**base,"status":"SELECTED_OUTSIDE_DIRECT_SET"}
        if refs[chosen]["eligible_ref"] is False:
            return {**base,"status":"SELECTED_REFERENCE_INELIGIBLE","selected":chosen,"false_acceptance":True}
        if refs[chosen]["eligible_ref"] is None:
            return {**base,"status":"SELECTED_ELIGIBILITY_UNDETERMINED","selected":chosen}
    eligible = {k:r for k,r in refs.items() if r["eligible_ref"] is True}
    if not eligible:
        return {**base,"status":"REF_ELIGIBLE_EMPTY"}
    candidates = [k for k in eligible if predicted_eligible is None or predicted_eligible.get(k) is True]
    if not candidates:
        return {**base,"status":"MODEL_ALL_REJECTED"}
    present = {k:predicted_G.get(k) for k in candidates if predicted_G.get(k) is not None}
    if not present:
        return {**base,"status":"NO_PREDICTIONS","missing_candidates":sorted(candidates)}
    for v in present.values():
        number(v, "G_conditional_pred")
    if chosen is None:
        minimum = min(present.values())
        ties = [k for k,v in present.items() if abs(v-minimum)<=1e-9 and abs(v-minimum)<=1e-10*max(abs(v),abs(minimum))]
        chosen = min(ties, key=lambda k:(canonical(refs[k]["candidate"]),k))
    if chosen not in present:
        return {**base,"status":"MISSING_COST_PREDICTIONS","selected":chosen}
    if eligible[chosen].get("G_ref") is None or any(r.get("G_ref") is None for r in eligible.values()):
        return {**base,"status":"REFERENCE_COST_UNDEFINED","selected":chosen}
    for r in eligible.values():
        number(r["G_ref"],"G_ref")
    denominator = min(r["G_ref"] for r in eligible.values())
    if denominator == 0:
        return {**base,"status":"ZERO_REGRET_DENOMINATOR","selected":chosen}
    missing = sorted(set(eligible)-set(present))
    support_min = min(eligible[k]["G_ref"] for k in present)
    support_regret = eligible[chosen]["G_ref"]/support_min-1 if support_min else None
    return {**base,"status":"INCOMPLETE_PREDICTION_COVERAGE" if missing else "VALID_CONDITIONAL_ORACLE",
            "selected":chosen,"regret":None if missing else eligible[chosen]["G_ref"]/denominator-1,
            "selected_G_conditional_oracle":predicted_G.get(chosen),"selected_G_ref":eligible[chosen]["G_ref"],"full_set_min_G_ref":denominator,
            "common_support_regret":support_regret, "common_support_sha256":digest(sorted(present)),
            "missing_candidates":missing,"coverage_denominator_reference_eligible":len(eligible),
            "covered_reference_eligible":len(present),"registered_direct_count":len(refs)}


def paired_statistics(left, right, random=True, expected_count=None):
    expected = expected_count if expected_count is not None else (32 if random else 1)
    require(len(left)==len(right)==expected, "PAIR_IDENTITY", "missing/truncated paired costs")
    def keyed(rows):
        output = {}
        for r in rows:
            key = (r["trajectory_index"], r["trajectory_seed"])
            require(key not in output,"PAIR_IDENTITY","duplicate trajectory")
            output[key]=r
        return output
    L,R=keyed(left),keyed(right)
    require(set(L)==set(R),"PAIR_IDENTITY","cosine/sine index or seed mismatch")
    for k in L:
        for field in ["step_seeds","evolution_circuit_semantics_fingerprint"]:
            require(field in L[k] and field in R[k] and L[k][field]==R[k][field],"PAIR_IDENTITY",f"pair {field} mismatch/missing")
    xs=[number(L[k]["cost"],"cosine cost") for k in sorted(L)]
    ys=[number(R[k]["cost"],"sine cost") for k in sorted(L)]
    n=len(xs)
    require(n>1 if random else n==1,"PAIR_IDENTITY","invalid stochastic sample count")
    mx,my=math.fsum(xs)/n,math.fsum(ys)/n
    cc=math.fsum((x-mx)**2 for x in xs)/(n-1) if random else 0.0
    ss=math.fsum((y-my)**2 for y in ys)/(n-1) if random else 0.0
    cs=math.fsum((x-mx)*(y-my) for x,y in zip(xs,ys))/(n-1) if random else 0.0
    return dict(n=n,mean_cosine=mx,mean_sine=my,var_cosine=cc,var_sine=ss,cov_cos_sin=cs,
                interval_kind="ENGINEERING_INTERVAL_ONLY",formal_ci=False,cross_candidate_covariance=None,
                rare_event_cost_tail="UNRESOLVED",quantum_shot_uncertainty=False)


def common_support_selection(refs,models,cost_support,diagnostic_kind):
    """Same registered direct subset for every available calibrated model."""
    support=set(refs)
    for model in models:
        support.intersection_update(cost_support[model])
    return [dict(model_id=model,metric_name="regret_conditional_oracle_common_support",
                 direct_support_sha256=digest(sorted(support)),excluded_direct_candidates=sorted(set(refs)-support),
                 full_set_diagnostic=selection(refs,prediction,diagnostic_kind),
                 common_set_diagnostic=selection({fp:refs[fp] for fp in support},{fp:prediction.get(fp) for fp in support},diagnostic_kind+"_COMMON_SUPPORT"))
            for model,prediction in models.items()]


def paired_total(stats, N_real, N_imag):
    number(N_real,"N_ref_real");number(N_imag,"N_ref_imag")
    value=N_real*stats["mean_cosine"]+N_imag*stats["mean_sine"]
    variance=(N_real**2*stats["var_cosine"]+N_imag**2*stats["var_sine"]+2*N_real*N_imag*stats["cov_cos_sin"])/stats["n"]
    require(math.isfinite(variance) and variance>=0,"PAIR_IDENTITY","invalid covariance quadratic form")
    se=math.sqrt(variance)
    return dict(point=value,SE=se,engineering_interval=[value-2*se,value+2*se],formal_ci=False,
                interval_kind="ENGINEERING_INTERVAL_ONLY",rare_event_cost_tail="UNRESOLVED")
