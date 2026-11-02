"""Extract only fixed saved R1 records, without importing any R1 execution code."""
from fractions import Fraction as F
from hashlib import sha256
import json
from .exact import normalize_direction

RESULT_SHA256 = "f726ad70cb2643533f0d037b518cde1b702724adb4e6571fea25e26e4bfdd61e"
EPSILONS = ("1e-3", "1e-4", "1e-6")
ARMS = ("ordinary", "PTSC_K0", "A")
GROUPS = {"ordinary": {0: "O0", 2: "O2"},
          "PTSC_K0": {0: "O0", 2: "P2", 3: "P3"},
          "A": {0: "A0", 1: "A1", 2: "A2"}}


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def event_signature(event):
    return {k: event[k] for k in ("label_probability", "word", "rotation",
                                 "rotation_sign", "phase_i_power", "complement")} | {
        "native_cost": event["native_cost"]}


def extract_saved_table(raw):
    if sha256(raw).hexdigest() != RESULT_SHA256:
        raise PermissionError("R1 result identity mismatch")
    result = json.loads(raw)
    if result["runs"] != 1 or result["retries"] != 0 or not result["mandatory_STOP"]:
        raise PermissionError("R1 provenance mismatch")
    for saved in result["synthesis_rows"]:
        sequence = saved["sequence"]
        if (sha256(sequence.encode()).hexdigest() != saved["sequence_sha256"]
                or sequence.count("T")+sequence.count("t") != saved["T_count"]
                or sequence.count("t") != saved["Tdagger_count"]
                or not saved["error_pass"] or F(saved["strict_operator_error_upper"]) < 0):
            raise PermissionError("saved synthesis sequence identity/count/error flag mismatch")
    rows = [r for r in result["resource_rows"] if r["context"] == "distinct_basis"
            and r["controlled"] and r["arm"] in ARMS]
    tables = {}
    for xs in ("1/8", "1/4"):
        x = F(xs)
        rho = (x+x**3/6)/(1+x*x/2)
        formulas = {"O0": (0, F(1), x), "O2": (2, x*x/2, x**3/6),
                    "P2": (2, x*x/2, F(0)), "P3": (3, x**3/6, F(0)),
                    "A0": (0, F(1), rho),
                    "A1": (1, 2*x**3/(3*(x*x+2)), 2*x*x/(x*x+6)),
                    "A2": (2, x*x*(x*x+2)/(2*(x*x+6)), x**3/6)}
        candidates, profiles, by_id = [], [], {}
        sign_pairs, exclusions = [], []
        for arm in ARMS:
            for epsilon in EPSILONS:
                pair = {}
                for sigma in (1, -1):
                    found = [r for r in rows if (r["x"], r["arm"], r["epsilon"], r["sigma"])
                             == (xs, arm, epsilon, sigma)]
                    if len(found) != 1:
                        raise ValueError("candidate table identity failure")
                    pair[sigma] = found[0]
                plus, minus = pair[1]["profile"], pair[-1]["profile"]
                em = {e["label"]: e for e in minus["events"]}
                if len(em) != len(minus["events"]) or set(em) != {e["label"] for e in plus["events"]}:
                    raise ValueError("INCONCLUSIVE_SIGN_TABLE_MISMATCH")
                for e in plus["events"]:
                    m = em[e["label"]]
                    if any(F(e["native_cost"][k]) != F(m["native_cost"][k])
                           for k in ("T", "CX", "1Q", "strict_event_error_upper")):
                        raise ValueError("INCONCLUSIVE_SIGN_TABLE_MISMATCH")
                if plus["workspace_qubits_beyond_2_system"] != minus["workspace_qubits_beyond_2_system"]:
                    raise ValueError("INCONCLUSIVE_SIGN_TABLE_MISMATCH")
                sign_pairs.append({"arm": arm, "epsilon": epsilon, "cost_error_equal": True,
                                   "IR_equal_required_across_signs": False})
                memberships = []
                for degree, prototype in GROUPS[arm].items():
                    events = [e for e in plus["events"] if e["label"].split(":")[0] == str(degree)]
                    k, a, b = formulas[prototype]
                    if not events or any((F(e["a"]), F(e["b"])) != (a, b) for e in events):
                        raise ValueError("saved prototype mismatch")
                    law = [F(e["label_probability"]) for e in events]
                    if sum(law) != 1 or any(p <= 0 for p in law):
                        raise ValueError("invalid conditional law")
                    for e in events:
                        labels = e["label"].split(":")[1].strip("() ,")
                        indices = [] if not labels else [int(j.strip()) for j in labels.split(",") if j.strip()]
                        iid = F(1)
                        for j in indices:
                            iid *= (F(3, 4), F(1, 4))[j]
                        if iid != F(e["label_probability"]):
                            raise ValueError("saved conditional IID law mismatch")
                        if e["rotation_sign"] != 1 or e["phase_i_power"] != (-degree) % 4:
                            raise ValueError("phase or signed-time semantic mismatch")
                        if e["complement"] != (arm == "A" and degree % 2 == 1):
                            raise ValueError("odd complement semantic mismatch")
                        nc = e["native_cost"]
                        if len(nc["IR_sha256"]) != 64 or any(F(nc[c]) < 0 for c in ("T", "CX", "1Q", "strict_event_error_upper")):
                            raise ValueError("invalid saved cost/IR/error identity")
                    column, norm = normalize_direction(a, b, k)
                    ident = f"{prototype}:{epsilon}"
                    costs = {c: str(sum(p*F(e["native_cost"][c]) for p, e in zip(law, events)))
                             for c in ("T", "CX", "1Q")}
                    d = 2*sum(p*F(e["native_cost"]["strict_event_error_upper"]) for p, e in zip(law, events))
                    item = {"id": ident, "prototype": prototype, "epsilon": epsilon, "degree": degree,
                            "saved_ideal_ab_exact": [str(a), str(b)],
                            "direction_ratio_exact": str(b/a),
                            "D_intervals": [[str(lo), str(hi)] for lo, hi in column],
                            "costs": costs, "d_upper": str(d),
                            "workspace_peak": plus["workspace_qubits_beyond_2_system"],
                            "workspace_source": "saved uniform controlled context, not expected capacity",
                            "events": [{"source_label": e["label"], **event_signature(e)} for e in events]}
                    identity = digest({k: item[k] for k in ("D_intervals", "costs", "d_upper", "workspace_peak", "events")})
                    item["implementation_identity_sha256"] = identity
                    if ident in by_id:
                        if by_id[ident]["implementation_identity_sha256"] != identity:
                            raise ValueError("same prototype native identity mismatch")
                    elif item["workspace_peak"] > 1:
                        exclusions.append(ident)
                    else:
                        by_id[ident] = item
                        candidates.append(item)
                    memberships.append({"column_id": ident, "ideal_weight_interval": list(map(str, norm)),
                                        "saved_midpoint_weight": str(sum(F(e["implemented_coefficient"]) for e in events))})
                profiles.append({"arm": arm, "epsilon": epsilon, "memberships": memberships,
                                 "original_profile": {k: v for k, v in plus.items() if k != "events"},
                                 "original_finite_confidence": pair[1]["finite_confidence"],
                                 "original_event_law_sha256": digest(plus["events"])})
        tables[xs] = {"columns": sorted(candidates, key=lambda c: c["id"]), "B0_saved_profiles": profiles,
                      "target": list(map(str, (F(1), x, x*x/2, x**3/6))), "sign_checks": sign_pairs,
                      "workspace_exclusions": exclusions,
                      "distinct_columns": len(candidates), "O0_cross_arm_duplicate_verified": True}
    return {"schema": "ra_d0_saved_candidate_table_v1", "source_result_sha256": RESULT_SHA256,
            "no_angles_generated": True, "no_circuit_build_or_synthesis": True, "tables": tables}
