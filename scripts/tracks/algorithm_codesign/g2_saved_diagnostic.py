"""Bounded post-hoc arithmetic, not a registered RA-D0 optimizer or sampler.

Stdlib only. Reads published JSON; no execution-code imports, solver, circuit,
matrix, synthesis or stochastic sampling. Main creates an exclusive new marker.
"""
from decimal import Decimal, localcontext
from fractions import Fraction as F
from hashlib import sha256
from itertools import product
from math import isqrt
from pathlib import Path
import json
import resource
import signal
import sys
import time

BASE = "4f2a08c80a293513761508421a76be2005d4c9ec"
OUT = "artifacts/track_b_g2_saved_diagnostic/2026-10-09"
TABLE = "artifacts/track_b_ra_d0_preparation/2026-10-06/candidate_table_v1.json"
RAW = "artifacts/track_b_rte_reallocation_r1_result/2026-10-06/v1/result.json"
R1_CONTRACT = "artifacts/track_b_rte_reallocation_r1_source/2026-10-06/contract_v2.json"
EPS = ("1e-3", "1e-4", "1e-6")
ORDER = ("O0", "O2", "P2", "P3", "A0", "A1", "A2")
RESOURCES = ("T", "CX", "1Q")
SCALE = 10**60


def digest_bytes(raw):
    return sha256(raw).hexdigest()


def digest(value):
    return digest_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


class I:
    """Outward rounded rational intervals; endpoints always multiples of 10^-60."""
    def __init__(self, lo, hi=None):
        lo, hi = F(lo), F(lo if hi is None else hi)
        if lo > hi:
            raise ValueError("reversed interval")
        self.lo = F((lo*SCALE).__floor__(), SCALE)
        self.hi = F((hi*SCALE).__ceil__(), SCALE)

    def __add__(self, other):
        o = as_i(other)
        return I(self.lo+o.lo, self.hi+o.hi)
    __radd__ = __add__

    def __neg__(self):
        return I(-self.hi, -self.lo)

    def __sub__(self, other):
        return self+-as_i(other)

    def __mul__(self, other):
        o = as_i(other)
        values = [a*b for a in (self.lo, self.hi) for b in (o.lo, o.hi)]
        return I(min(values), max(values))
    __rmul__ = __mul__

    def __truediv__(self, other):
        o = as_i(other)
        if o.lo <= 0 <= o.hi:
            raise ZeroDivisionError("interval denominator crosses zero")
        return self*I(1/o.hi, 1/o.lo)

    def __rtruediv__(self, other):
        return as_i(other)/self

    def square(self):
        if self.lo < 0:
            raise ValueError("square expects nonnegative interval")
        return self*self

    def json(self):
        return {"lo": str(self.lo), "hi": str(self.hi),
                "mid_display": float((self.lo+self.hi)/2)}


def as_i(value):
    return value if isinstance(value, I) else I(value)


def sqrt_i(value):
    v = as_i(value)
    if v.lo < 0:
        raise ValueError("negative root")
    def lower(z):
        return F(isqrt(z.numerator*SCALE*SCALE//z.denominator), SCALE)
    lo, hi = lower(v.lo), lower(v.hi)
    return I(lo, hi if hi*hi == v.hi else hi+F(1, SCALE))


def log_integer(value):
    if type(value) is not int or value <= 0:
        raise ValueError("exact positive integer required")
    with localcontext() as ctx:
        ctx.prec = 80
        v = Decimal(value).ln()
        return I(F(ctx.next_minus(v)), F(ctx.next_plus(v)))


def vectors(x):
    x = F(x)
    rho = (x+x**3/6)/(1+x*x/2)
    return {"O0": (0, F(1), x), "O2": (2, x*x/2, x**3/6),
            "P2": (2, x*x/2, F(0)), "P3": (3, x**3/6, F(0)),
            "A0": (0, F(1), rho),
            "A1": (1, 2*x**3/(3*(x*x+2)), 2*x*x/(x*x+6)),
            "A2": (2, x*x*(x*x+2)/(2*(x*x+6)), x**3/6)}


def gamma(x, s, r, b):
    mu = (x*x+2)/(x*x+6)
    values = (s, b, mu+(1-mu)*s-mu*r-b, 1-r-b, 1-s, 1-s, r)
    return dict(zip(ORDER, values, strict=True))


def vertices(x):
    x = F(x)
    mu = (x*x+2)/(x*x+6)
    return {name: gamma(x, *map(F, point)) for name, point in {
        "ordinary": (1, 0, 1), "PTSC_K0": (1, 0, 0), "A": (0, 1, 0),
        "J1": (0, 0, 0), "J2": (0, 0, mu), "J3": (1, 1, 0)}.items()}


def mean_check(x, weights):
    vec = vectors(x)
    result = [F(0)]*4
    for name, g in weights.items():
        if g < 0:
            raise ValueError("negative coefficient")
        degree, a, b = vec[name]
        result[degree] += g*a
        if b:
            result[degree+1] += g*b
    if result != [F(1), x, x*x/2, x**3/6]:
        raise ValueError("ideal degree mean mismatch")


def verify_inputs(table, raw):
    """Replay saved-value identity checks, not R1 build/compile or table extraction."""
    if table["source_result_sha256"] != digest_bytes(raw):
        raise ValueError("raw result hash mismatch")
    r = json.loads(raw)
    if (r["runs"], r["retries"], r["mandatory_STOP"]) != (1, 0, True):
        raise ValueError("saved run provenance")
    synth = r["synthesis_rows"]
    for s in synth:
        sequence = s["sequence"]
        if (digest_bytes(sequence.encode()) != s["sequence_sha256"] or
            sequence.count("T")+sequence.count("t") != s["T_count"] or
            sequence.count("t") != s["Tdagger_count"] or not s["error_pass"] or
            F(s["strict_operator_error_upper"]) < 0):
            raise ValueError("saved synthesis identity/error")
    comparisons, events = 0, 0
    arm_degree = {"ordinary": {0: "O0", 2: "O2"},
                  "PTSC_K0": {0: "O0", 2: "P2", 3: "P3"},
                  "A": {0: "A0", 1: "A1", 2: "A2"}}
    for xs, t in table["tables"].items():
        if t["distinct_columns"] != 21 or t["workspace_exclusions"]:
            raise ValueError("unexpected fixed table size/workspace")
        cols = {c["id"]: c for c in t["columns"]}
        if set(cols) != {g+":"+p for g in ORDER for p in EPS}:
            raise ValueError("unexpected column inventory")
        for c in cols.values():
            wanted = digest({k: c[k] for k in (
                "D_intervals", "costs", "d_upper", "workspace_peak", "events")})
            if wanted != c["implementation_identity_sha256"]:
                raise ValueError("column identity mismatch")
            degree, a, b = vectors(F(xs))[c["prototype"]]
            if (list(map(F, c["saved_ideal_ab_exact"])) != [a, b]):
                raise ValueError("ideal prototype identity")
            for k, (lo, hi) in enumerate(c["D_intervals"]):
                numer = a if k == degree else b if k == degree+1 else F(0)
                if not (0 <= F(lo) <= F(hi) and F(lo)**2*(a*a+b*b) <= numer**2 <= F(hi)**2*(a*a+b*b)):
                    raise ValueError("direction does not enclose exact normalized coefficient")
            if c["workspace_peak"] != 1:
                raise ValueError("workspace mismatch")
            ev = c["events"]
            if sum(F(e["label_probability"]) for e in ev) != 1:
                raise ValueError("conditional law")
            for e in ev:
                labels = e["source_label"].split(":")[1].strip("() ,")
                indices = [int(j.strip()) for j in labels.split(",") if j.strip()]
                p = F(1)
                for j in indices:
                    p *= (F(3, 4), F(1, 4))[j]
                if (p != F(e["label_probability"]) or len(indices) != degree+int(b != 0) or
                    e["word"] != (indices[1:] if b else indices) or e["rotation"] != (indices[0] if b else None) or
                    e["rotation_sign"] != 1 or e["phase_i_power"] != (-degree)%4 or
                    e["complement"] != (c["prototype"] == "A1")):
                    raise ValueError("event probability/word/phase/complement")
                nc = e["native_cost"]
                if len(nc["IR_sha256"]) != 64 or any(F(nc[k]) < 0 for k in (*RESOURCES, "strict_event_error_upper")):
                    raise ValueError("native identity/cost/error")
            for k in RESOURCES:
                if sum(F(e["label_probability"])*F(e["native_cost"][k]) for e in ev) != F(c["costs"][k]):
                    raise ValueError("conditional expected cost")
            if 2*sum(F(e["label_probability"])*F(e["native_cost"]["strict_event_error_upper"]) for e in ev) != F(c["d_upper"]):
                raise ValueError("conditional bias")
            events += len(ev)
        for row in r["resource_rows"]:
            if row["context"] != "distinct_basis" or not row["controlled"] or row["x"] != xs or row["arm"] not in arm_degree:
                continue
            for e in row["profile"]["events"]:
                degree = int(e["label"].split(":")[0])
                c = cols[arm_degree[row["arm"]][degree]+":"+row["epsilon"]]
                found = [v for v in c["events"] if v["source_label"] == e["label"]]
                if len(found) != 1:
                    raise ValueError("event missing from raw")
                saved = found[0]
                fields = (*RESOURCES, "strict_event_error_upper")
                if any(F(saved["native_cost"][k]) != F(e["native_cost"][k]) for k in fields):
                    raise ValueError("raw sign-pair cost/error mismatch")
                if F(e["label_probability"]) != F(saved["label_probability"]):
                    raise ValueError("raw probability mismatch")
                if row["sigma"] == 1:
                    for k in ("word", "rotation", "rotation_sign", "phase_i_power", "complement", "native_cost"):
                        if saved[k] != e[k]:
                            raise ValueError("raw positive event signature mismatch")
                comparisons += 1
    return {"saved_synthesis_rows_verified": len(synth), "column_events_verified": events,
            "raw_event_sign_cost_error_comparisons": comparisons,
            "sign_minus_role": "same-context equality check, not replication",
            "IR_rebuilt": False, "operator_error_recomputed": False,
            "strict_error_bounds_trusted_from_saved_source": True}


def column_data(c):
    a, b = map(F, c["saved_ideal_ab_exact"])
    return {"norm": sqrt_i(a*a+b*b), "bias": I(c["d_upper"]),
            "ec": {k: I(c["costs"][k]) for k in RESOURCES},
            "h": {k: sum((F(e["label_probability"])*sqrt_i(e["native_cost"][k])
                         for e in c["events"]), I(0)) for k in RESOURCES},
            "events": c["events"]}


def finite_confidence(m2, bound, remaining, ec, ell, cap):
    if remaining.lo <= 0:
        return {"status": "BIAS_EXHAUSTED_OR_UNRESOLVED"}
    n = ell*(2*m2/remaining.square()+F(4, 3)*bound/remaining)
    shot_lo, shot_hi = n.lo.__ceil__(), n.hi.__ceil__()
    return {"status": "CONDITIONAL_ANALYTIC_PASS" if shot_hi <= cap else "SHOT_CAP",
            "shots_per_axis_enclosure": [shot_lo, shot_hi],
            "range": bound.json(), "second_moment": m2.json(),
            "expected_resource_vector": {k: v.json() for k, v in ec.items()},
            "total_two_axes_resource": {k: (I(shot_hi)*(2*v+(5 if k == "1Q" else 0))).json() for k, v in ec.items()},
            "sampler_built": False, "numeric_mean_certificate": False,
            "remaining_stat": remaining.json()}


def evaluate_profile(xs, vertex, weights, precision, cols, plan):
    x = F(xs)
    mean_check(x, weights)
    selected = [(g, weights[g], cols[g+":"+precision[g]]) for g in ORDER if weights[g]]
    B = sum((g*c["norm"] for _, g, c in selected), I(0))
    bias = sum((g*c["norm"]*c["bias"] for _, g, c in selected), I(0))
    s = I(plan["epsilon_axis"])-bias
    ec = {k: sum((g*c["norm"]*c["ec"][k] for _, g, c in selected), I(0))/B for k in RESOURCES}
    log_argument = 2/F(plan["alpha_axis"])
    if log_argument.denominator != 1:
        raise ValueError("confidence allocation requires exact integer log input")
    ell = log_integer(log_argument.numerator)
    row = {"id": xs+"/"+vertex+"/"+",".join(g+"="+precision[g] for g, _, _ in selected),
           "x": xs, "vertex": vertex, "precision": precision,
           "uniform_precision": len(set(precision.values())) == 1,
           "gamma_exact": {g: str(weights[g]) for g in ORDER},
           "B": B.json(), "bias_upper_enclosure": bias.json(), "remaining_stat": s.json(),
           "ideal_degree_match_exact": True, "numeric_mean_certificate": False,
           "workspace_peak": 1, "canonical": {}, "IS": {}}
    row["canonical"]["finite_confidence"] = finite_confidence(B.square(), B, s, ec, ell, plan["shot_cap_per_axis"])
    for k in RESOURCES:
        K = sum((g*c["norm"]*c["h"][k] for _, g, c in selected), I(0))
        zeros = [name+":"+e["source_label"] for name, _, c in selected for e in c["events"] if F(e["native_cost"][k]) == 0]
        item = {"K": K.json(), "zero_cost_events": zeros,
                "attainment": "UNATTAINED_NET_COST_INFIMUM" if zeros and K.hi > 0 else "ATTAINED_IDEAL_REAL_PROPOSAL",
                "linear_fractional_squared": (K/s).square().json() if s.lo > 0 else None,
                "finite_confidence": {"status": "MISSING_FINITE_PROPOSAL_ZERO_COST", "sampler_built": False}}
        row["canonical"][k] = {"net_cost": (B.square()*ec[k]).json(),
                              "bias_adjusted_net_cost": (B.square()*ec[k]/s.square()).json() if s.lo > 0 else None}
        if not zeros:
            S = sum((g*c["norm"]*sum((F(e["label_probability"])/sqrt_i(e["native_cost"][k])
                       for e in c["events"]), I(0)) for _, g, c in selected), I(0))
            ec_is = {j: sum((g*c["norm"]*sum((F(e["label_probability"])*F(e["native_cost"][j])/sqrt_i(e["native_cost"][k])
                         for e in c["events"]), I(0)) for _, g, c in selected), I(0))/S for j in RESOURCES}
            max_c = max(F(e["native_cost"][k]) for _, _, c in selected for e in c["events"])
            item["S"] = S.json()
            item["finite_confidence"] = finite_confidence(S*K, S*sqrt_i(max_c), s, ec_is, ell, plan["shot_cap_per_axis"])
        row["IS"][k] = item
    return row


def min_record(rows, getter):
    valid = [(r, getter(r)) for r in rows if getter(r) is not None]
    if not valid:
        return {"status": "NO_ELIGIBLE_VALUE"}
    best = min(valid, key=lambda z: F(z[1]["hi"]))
    lower = min(F(v["lo"]) for _, v in valid)
    other = [F(v["lo"]) for r, v in valid if r["id"] != best[0]["id"]]
    return {"id": best[0]["id"], "vertex": best[0]["vertex"],
            "precision": best[0]["precision"], "value": I(lower, F(best[1]["hi"])).json(),
            "strict_unique_minimum_enclosure": bool(other) and F(best[1]["hi"]) < min(other),
            "eligible_profiles": len(valid)}


def summarize(rows):
    out = {}
    for xs in ("1/8", "1/4"):
        selected = [r for r in rows if r["x"] == xs]
        per_x = {"profiles": len(selected), "axes": {}}
        for k in RESOURCES:
            get_is = lambda r: r["IS"][k]["linear_fractional_squared"]
            old = [r for r in selected if r["vertex"] in ("ordinary", "PTSC_K0", "A")]
            b2, b3 = min_record(old, get_is), min_record(selected, get_is)
            ratio = I(b3["value"]["lo"], b3["value"]["hi"])/I(b2["value"]["lo"], b2["value"]["hi"])
            canonical = min_record(old, lambda r: r["canonical"][k]["bias_adjusted_net_cost"])
            per_x["axes"][k] = {"same_IS_original_vertices": b2, "same_IS_all_vertices": b3,
                "B3_over_B2_ideal_diagnostic": ratio.json(),
                "strict_improvement_of_ideal_infimum": F(b3["value"]["hi"]) < F(b2["value"]["lo"]),
                "canonical_original_profile_subset": canonical,
                "canonical_subset_is_not_global_mixed_B2_minimum": True,
                "same_IS_original_uniform_precision": min_record([r for r in old if r["uniform_precision"]], get_is),
                "same_IS_all_uniform_precision": min_record([r for r in selected if r["uniform_precision"]], get_is),
                "per_vertex": {v: min_record([r for r in selected if r["vertex"] == v], get_is)
                               for v in vertices(F(xs))},
                "IS_unattained_profiles": sum(r["IS"][k]["attainment"] == "UNATTAINED_NET_COST_INFIMUM" for r in selected),
                "IS_confidence_status_counts": {s: sum(r["IS"][k]["finite_confidence"]["status"] == s for r in selected)
                    for s in sorted({r["IS"][k]["finite_confidence"]["status"] for r in selected})}}
        out[xs] = per_x
    return out


def price_rows(table):
    rows = []
    for xs, t in table["tables"].items():
        x, mu = F(xs), (F(xs)**2+2)/(F(xs)**2+6)
        cols = {c["id"]: column_data(c) for c in t["columns"]}
        for p, k, mode in product(EPS, RESOURCES, ("IS_E_sqrt_C", "affine_expected_C_price")):
            ell = {g: cols[g+":"+p]["norm"]*(cols[g+":"+p]["h"][k] if mode == "IS_E_sqrt_C" else cols[g+":"+p]["ec"][k]) for g in ORDER}
            alpha = ell["O0"]-ell["A0"]-ell["A1"]+(1-mu)*ell["P2"]
            beta = ell["A2"]-mu*ell["P2"]-ell["P3"]
            zeta = ell["O2"]-ell["P2"]-ell["P3"]
            increments = {"ordinary": alpha+zeta, "PTSC_K0": alpha, "A": beta,
                          "J1": I(0), "J2": mu*zeta, "J3": alpha+beta}
            sign = lambda v: "POSITIVE" if v.lo > 0 else "NEGATIVE" if v.hi < 0 else "ZERO" if v.lo == v.hi == 0 else "UNRESOLVED"
            rows.append({"x": xs, "precision": p, "resource": k, "price_mode": mode,
                         "alpha": alpha.json(), "beta": beta.json(), "zeta": zeta.json(),
                         "signs": {a: sign(v) for a, v in (("alpha", alpha), ("beta", beta), ("zeta", zeta))},
                         "increments": {g: v.json() for g, v in increments.items()},
                         "scope": "affine fixed-price, bias/shot/caps not optimized"})
    return rows


def return_rows(table):
    result = []
    chi = F(5, 8)
    for xs, t in table["tables"].items():
        x = F(xs)
        a0, b0 = 1-chi*x*x/2, x-chi*x**3/6
        A, b, c = sqrt_i((1+x*x/2)**2+(x+x**3/6)**2), sqrt_i(x**4/4+x**6/36), x*x/2+x**4/6
        v = b*(A-b)
        threshold = (v-I(c))/(v+I(c))
        ret = sqrt_i(a0*a0+b0*b0)+(1-chi)*b
        reused = []
        for p in EPS:
            col = next(c for c in t["columns"] if c["id"] == "O2:"+p)
            events = [e for e in col["events"] if e["word"][0] != e["word"][1]]
            if len(events) != 4 or sum(F(e["label_probability"]) for e in events) != 1-chi:
                raise ValueError("return off-diagonal support")
            reused.append({"precision": p, "column_id": col["id"],
                           "source_labels": [e["source_label"] for e in events],
                           "conditional_law_exact": [str(F(e["label_probability"])/(1-chi)) for e in events],
                           "conditional_costs": {k: str(sum(F(e["label_probability"])*F(e["native_cost"][k]) for e in events)/(1-chi)) for k in RESOURCES},
                           "conditional_d_upper": str(2*sum(F(e["label_probability"])*F(e["native_cost"]["strict_event_error_upper"]) for e in events)/(1-chi)),
                           "IR_ids": [e["native_cost"]["IR_sha256"] for e in events]})
        result.append({"x": xs, "chi": str(chi), "threshold": threshold.json(),
                       "B_A": A.json(), "B_return": ret.json(), "B2_return_over_A": (ret/A).square().json(),
                       "returned_zero_degree_ab": [str(a0), str(b0)],
                       "returned_ratio": str(b0/a0), "rotation_angle_evaluated": False,
                       "off_diagonal_existing_O2_reuse": reused,
                       "returned_zero_degree_two_label_cost_error_IR": "MISSING",
                       "return_finite_confidence_native_comparison": "MISSING_NOT_IMPUTED",
                       "CTS_distinct_basis_same_target_full_collection_acquisition_cost_error": "MISSING",
                       "no_new_angle_dictionary_or_synthesis": True})
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")


def main():
    root = Path(__file__).resolve().parents[3]
    folder = root/OUT
    plan_raw = (folder/"diagnostic_scope_v2.json").read_bytes()
    plan = json.loads(plan_raw)
    for path, expected in plan["input_and_source_hashes"].items():
        if digest_bytes((root/path).read_bytes()) != expected:
            raise PermissionError("fixed input/source mismatch: "+path)
    marker = folder/"diagnostic_profiles_consumed.json"
    with marker.open("x") as f:
        json.dump({"scope_sha256": digest_bytes(plan_raw), "runs": 1, "retries": 0,
                   "kind": "POSTHOC_SAVED_VALUE_DIAGNOSTIC_NOT_OLD_SCIENCE_RUN"}, f, sort_keys=True)
        f.write("\n")
    start, cpu = time.monotonic(), time.process_time()
    old_as = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (512*1024**2, old_as[1]))
    resource.setrlimit(resource.RLIMIT_CPU, (120, 121))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("wall cap")))
    signal.alarm(180)
    try:
        table = json.loads((root/TABLE).read_bytes())
        audit = verify_inputs(table, (root/RAW).read_bytes())
        rows = []
        for xs, t in table["tables"].items():
            cols = {c["id"]: column_data(c) for c in t["columns"]}
            local = []
            for vertex, w in vertices(F(xs)).items():
                active = [g for g in ORDER if w[g]]
                for precs in product(EPS, repeat=len(active)):
                    local.append(evaluate_profile(xs, vertex, w, dict(zip(active, precs)), cols, plan))
                    if len(local) > 252:
                        raise PermissionError("profile cap")
            if len(local) != 252:
                raise ValueError("incomplete profile inventory")
            rows.extend(local)
        write_json(folder/"profile_rows_v1.json", rows)
        write_json(folder/"linear_price_rows_v1.json", price_rows(table))
        write_json(folder/"known_return_rows_v1.json", return_rows(table))
        write_json(folder/"saved_identity_audit_v1.json", audit)
        result = {"status": "G2_BOUNDED_DIAGNOSTIC_COMPLETED_WITH_DECLARED_LIMITS",
                  "profiles_evaluated": len(rows), "summary": summarize(rows),
                  "kind": "POSTHOC_DEVELOPMENT", "old_RA_D0_witnesses": 0,
                  "registered_LP_calls": 0, "solver_calls": 0, "synthesis_calls": 0,
                  "science_circuit_matrix_trajectory_DF_NPZ_GPU_calls": 0,
                  "registered_science_retries": 0, "diagnostic_preprofile_source_corrections": 1,
                  "diagnostic_process_invocations": 2, "completed_profile_passes": 1,
                  "mandatory_STOP": True, "next_stage_authorized": False,
                  "source_base": BASE, "source_and_inputs": plan["input_and_source_hashes"],
                  "caps": {"wall_seconds": 180, "CPU_seconds": 120, "AS_bytes": 512*1024**2,
                           "output_bytes": 16*1024**2, "profiles": 504},
                  "runtime": {"python": sys.version, "executable": sys.executable},
                  "wall_seconds": time.monotonic()-start, "CPU_seconds": time.process_time()-cpu,
                  "peak_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  "numeric_method": "outward rational 10^-60 intervals; integer sqrt; Decimal correctly-rounded ln with neighbours",
                  "sample_law_status": "analytic ideal real proposals only; no dyadic sampler/mean certificate",
                  "global_finite_confidence_optimality": False}
        write_json(folder/"result_v1.json", result)
        count = sum(p.stat().st_size for p in folder.glob("*.json"))
        if count > 16*1024**2:
            raise MemoryError("output cap")
        print(json.dumps({"status": result["status"], "profiles": len(rows), "output_bytes": count,
                          "wall_seconds": result["wall_seconds"]}))
    except Exception as e:
        write_json(folder/"failure_v2.json", {"status": "G2_TECHNICAL_INCONCLUSIVE", "error_type": type(e).__name__,
                   "error": str(e), "retries": 0, "mandatory_STOP": True,
                   "partial_prefix_not_final_evidence": True})
        raise
    finally:
        signal.alarm(0)
        write_json(folder/"STOP_v2.json", {"mandatory_STOP": True, "next_stage_authorized": False,
                                       "retries": 0, "decider": "GPT_G2"})


if __name__ == "__main__":
    main()
