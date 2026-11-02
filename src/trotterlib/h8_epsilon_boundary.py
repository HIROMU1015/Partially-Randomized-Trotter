"""New epsilon endpoints over frozen H8 candidates; scalar arithmetic only.

Uses the unchanged D0 shot rule, old component validation and n32 paired metric.
No sampler, signal provider, circuit, model prediction, or compiler is imported.
"""
from copy import deepcopy
from fractions import Fraction as F
import h8_noncancellation as old

EPS = (0.04, 0.025, 0.015, 0.000875, 0.00075, 0.000625)
ANCHORS = (0.05, 0.01, 0.001, 0.0005)
MODES = ("USUAL_SIGNED_SAVED", "EMPIRICAL_NONCANCELLATION")
require = old.require


def accounting(record, magnitude, epsilon):
    """Reuse exact D0 expression; only new epsilon and declared bias change."""
    if magnitude is None:
        return None
    require(set(magnitude) == set(old.AXES), "BIAS_AXES")
    for v in magnitude.values():
        require(old.finite(v) >= 0, "NEGATIVE_BIAS")
    view = dict(record, error_decomposition=dict(total_signed=magnitude))
    return old.d0.accounting(view, epsilon, 1.0)


def cost_row(record, magnitude, epsilon, mode, unit=None, pairs=None):
    a = accounting(record, magnitude, epsilon)
    ok, shots = (None, None) if a is None else old.eligibility(a)
    row = dict(cell_id=record["cell"]["id"], method=record["cell"]["method"],
        epsilon=epsilon, factor=1.0, bias_kind=mode, eligible=ok, accounting=a,
        shots=shots, G_exact=None, G_standard_error=None, axis_costs=unit,
        cost_status="NOT_ELIGIBLE_OR_UNRESOLVED", P=0,
        rigorous_bias_guarantee=None, population_mean_guaranteed=False)
    if ok is not True:
        return row
    if row["method"] == "B2":
        require(pairs is not None and len(pairs) == 32, "PAIRED_N32_REQUIRED")
        metric = old.d0.weighted_metric(pairs,
            {axis: shots[ch] for axis, ch in old.CHANNEL.items()}, "RZ",
            point_rule="sum_axis_products")
        row.update(G_exact=str(F(int(metric["G_exact"]["numerator"]), int(metric["G_exact"]["denominator"]))),
            G_standard_error=metric["G_standard_error"], paired_samples=metric["paired_samples"],
            cost_status="PR_FINITE_SAMPLE_MEAN_CONFIRMATION_N32")
    else:
        require(unit is not None and set(unit) == set(old.CHANNEL), "SAVED_UNIT_PRICE_REQUIRED")
        require(all(F(v["RZ"]) > 0 for v in unit.values()), "POSITIVE_SAVED_UNIT_PRICE")
        row.update(G_exact=str(sum(F(unit[a]["RZ"])*shots[ch] for a,ch in old.CHANNEL.items())),
            cost_status="SAVED_MEASURED_POINT" if all(x["measured"] for x in unit.values())
                        else "MODEL_CONDITIONAL_POINT")
    return row


def compare(rows, det_ids, PR_ids, epsilon, mode):
    require({r["cell_id"] for r in rows} == set(det_ids) | set(PR_ids), "FULL_DOMAIN_REQUIRED")
    d = old.minimum([r for r in rows if r["cell_id"] in det_ids])
    p = old.minimum([r for r in rows if r["cell_id"] in PR_ids])
    reduction = None
    if d["best"] is not None and p["best"] is not None:
        reduction = str(100*(1-F(p["best"]["G_exact"])/F(d["best"]["G_exact"])))
    return dict(epsilon=epsilon, factor=1, bias_kind=mode, scope="strengthened184",
        deterministic=d, PR=p, reduction_percent_exact=reduction,
        interpretation="CONDITIONAL_POINT_COMPARISON_NOT_POPULATION_OR_GLOBAL_OPTIMUM")


def evaluate(data, registry, six_rows, saved_noncancellation, guard=lambda: None):
    """Reproduce four old anchors before generating six new precision endpoints."""
    cells, union, records = old.bind_domain(data, registry)
    require(len(cells) == 184 and len(union) == 6, "FROZEN_184_PLUS_6")
    signed, unresolved = old.signed_index(data["signed"], cells)
    usual = old.usual_index(data["saved_usual"], list(cells)+union)
    prior_nc = {(r["cell_id"], r["epsilon"]):r for r in saved_noncancellation if r["factor"] == 1}
    prior = {}
    units = {}
    for entry in six_rows:
        r = entry["updated"]
        key = entry["bias_kind"], r["cell_id"], r["epsilon"]
        if r["factor"] != 1:
            continue
        require(key not in prior, "DUPLICATE_SAVED_ENDPOINT")
        prior[key] = r
        if r.get("axis_costs"):
            cid = r["cell_id"]
            signature = lambda v: {a:(F(x["RZ"]),x["measured"]) for a,x in v.items()}
            require(cid in cells and (cid not in units or signature(units[cid]) == signature(r["axis_costs"])),
                    "SAVED_UNIT_PRICE_INCONSISTENT")
            units[cid] = deepcopy(r["axis_costs"])
    require(set(prior) == {(m,c,e) for m in MODES for c in [*cells,*union] for e in old.EPSILONS},
            "EXACT_1900_FACTOR1_SAVED_ENDPOINTS")
    pairs = old.PR_pairs(data["wrappers"], union, data["confirmation"]["registrations"],
                         data["wrapper_binding"]["allowed_sources"])
    require(data["wrapper_binding"]["compiler"] == old.COMPILER, "COMPILER_BINDING")
    magnitudes = {}
    for cid in [*cells,*union]:
        guard()
        rec = records[cid]
        if cid in union:
            old.PR_normalization(rec)
            checked = old.validate_PR(rec, data["full_reference"])
            require(checked["first_failure"] is None, "PR_COMPONENT_CLOSURE")
        values = []
        for axis in old.AXES:
            if (cid,axis) in unresolved:
                values.append(None)
                continue
            parts = checked["signed_parts"][axis] if cid in union else signed[cid,axis]["signed_parts"]
            values.append(old.envelope(rec["cell"]["method"], parts,
                           rec["error_decomposition"]["total_signed"][axis]))
        for mode, field in zip(MODES, ("usual_signed", "empirical_noncancellation")):
            magnitudes[mode,cid] = None if None in values else {
                a:v[field] for a,v in zip(old.AXES,values)}
    reproduced = 0
    def make(cid, e, mode):
        return cost_row(records[cid], magnitudes[mode,cid], e, mode,
                        unit=units.get(cid), pairs=pairs.get(cid))
    for mode in MODES:
        for e in ANCHORS:
            for cid in [*cells,*union]:
                guard()
                r = make(cid,e,mode)
                a = usual[cid,e,1.] if mode == MODES[0] else prior_nc[cid,e]["accounting"]
                require(r["accounting"] == a, "ANCHOR_SHOT_REPRODUCTION:"+mode+":"+cid)
                previous = prior[mode,cid,e]
                require(r["eligible"] is previous["eligible"] and r["G_exact"] == previous["G_exact"],
                        "ANCHOR_PRICE_REPRODUCTION:"+mode+":"+cid)
                if cid in union and r["eligible"]:
                    se = previous.get("saved_paired_SE",previous.get("G_standard_error"))
                    require(r["G_standard_error"] == se, "ANCHOR_PAIRED_SE_REPRODUCTION")
                reproduced += 1
    rows, comparisons = [], []
    for mode in MODES:
        for e in EPS:
            current=[]
            for cid in [*cells,*union]:
                guard()
                current.append(make(cid,e,mode))
            rows.extend(current)
            comparisons.append(compare(current,cells,union,e,mode))
    return dict(status="COMPLETE_MANDATORY_STOP", accounting_rows=rows, comparisons=comparisons,
        anchor_rows_reproduced=reproduced, new_endpoint_rows=len(rows),
        native_compile=0, new_sampling=0, new_signal=0, new_model_prediction=0,
        next_stage_authorized=False, rigorous_bias_guarantee=None,
        population_mean_guaranteed=False, unit_prices="SAVED_SIX_AXIS_REFLECTED_ONLY")
