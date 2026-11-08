"""Prepared saved-RZ diagnostic. Real inputs require separate execution permission.

Importing this file does not read research data or import quantum/science code.
The only public CLI reads the three fixed Git blobs below; it never builds a
circuit, changes a shot schedule, or evaluates a new price/law/phase policy.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import json
import lzma
from pathlib import Path
import subprocess

BASE = "654b8501c1348cd0694daeefd19391df6b928354"
INPUTS = {
    "bprime": "artifacts/pr_reference_bprime/2026-11-07/TOTALS_18ROUND.json",
    "reference": "artifacts/pr_reference_transfer_qualification/2026-11-06/STAGE4_RPE.json",
    "roles": "artifacts/pr_reference_rotation_v3/2026-11-07/ANGLE_KEY_MULTIPLICITIES.json.xz",
}
ROLES = (
    "D_unfused", "D_boundary_fusion_difference", "RTE_basis_frames",
    "RTE_central_rotation", "RTE_products", "RTE_event_sign_phase",
    "scalar_or_vacuum_correction", "CSWAP", "XY_wrapper",
)


def interval(values):
    if len(values) != 2:
        raise ValueError("Two endpoints required")
    lo, hi = map(F, values)
    if lo > hi:
        raise ValueError("Reversed interval")
    return lo, hi


def split_cost(nb, nr, pb, pr):
    """Exact four-corner accounting; keeps attribution's shared endpoints.

    Signed prices are allowed for the saved D-fusion difference role.
    Component interval sums need not equal the tighter direct-change interval.
    No component extrema are treated as independent realizable prices.
    """
    if type(nb) is not int or type(nr) is not int or min(nb, nr) < 0:
        raise ValueError("Nonnegative integer shots required")
    pb, pr = interval(pb), interval(pr)
    corners = []
    for b in pb:
        for r in pr:
            change = nr * r - nb * b
            b_anchor = (nb * (r - b), (nr - nb) * r)
            r_anchor = (nr * (r - b), (nr - nb) * b)
            if sum(b_anchor) != change or sum(r_anchor) != change:
                raise ArithmeticError("Coupled accounting identity")
            corners.append((change, b_anchor, r_anchor))
    def envelope(xs):
        xs = tuple(xs)
        return min(xs), max(xs)
    return {
        "Bprime_cost": envelope(nb * b for b in pb),
        "reference_cost": envelope(nr * r for r in pr),
        "change": envelope(c[0] for c in corners),
        "Bprime_shot_anchor": {
            "price": envelope(c[1][0] for c in corners),
            "shots": envelope(c[1][1] for c in corners),
        },
        "reference_shot_anchor": {
            "price": envelope(c[2][0] for c in corners),
            "shots": envelope(c[2][1] for c in corners),
        },
    }


def role_prices(record, reference):
    if set(record["parts"]) != set(ROLES):
        raise ValueError("Missing/extra role; no zero completion")
    result = {}
    for role in ROLES:
        price = sum((F(n) for n in record["parts"][role].values()), F())
        # V3 source's lower bound discards every reference product-Z RZ.
        result[role] = (F() if reference and role == "RTE_products" else price, price)
    return result


def indexed(rows, key):
    result = {}
    for row in rows:
        k = key(row)
        if k in result:
            raise ValueError("Duplicate saved record")
        result[k] = row
    return result


def add_intervals(values):
    values = list(values)
    return tuple(sum((v[k] for v in values), F()) for k in (0, 1))


def aggregate_splits(rows):
    rows = list(rows)
    out = {k: add_intervals(row[k] for row in rows)
           for k in ("Bprime_cost", "reference_cost", "change")}
    for anchor in ("Bprime_shot_anchor", "reference_shot_anchor"):
        out[anchor] = {k: add_intervals(row[anchor][k] for row in rows)
                       for k in ("price", "shots")}
    return out


def diagnose(bprime, reference, roles):
    bs = indexed(bprime["rows"], lambda r: r["m"])
    rs = indexed(reference["rows"], lambda r: r["m"])
    vs = indexed(roles["rows"], lambda r: (r["m"], r["arm"], r["axis"]))
    if set(bs) != set(range(18)) or set(rs) != set(bs):
        raise ValueError("Expected all 18 saved rounds")
    expected = {(m, a, x) for m in bs for a in ("registered", "Bprime", "reference") for x in "XY"}
    if set(vs) != expected:
        raise ValueError("V3 coverage mismatch")
    result = []
    for m in range(18):
        b, r = bs[m], rs[m]
        identity = (m, b["q"], b["r"], b["bits"])
        if (r["m"], r["q"], r["r"], r["bits"]) != identity:
            raise ValueError("Schedule identity mismatch")
        nb, nr = b["shots_per_axis"], r["arms"]["reference"]["shots_per_axis"]
        for axis in "XY":
            prices = b["price_per_axis"][axis]
            pb = interval((prices["Bprime_RZ_lower"], prices["Bprime_RZ_upper"]))
            pr = interval((prices["reference_RZ_lower"], prices["reference_RZ_upper"]))
            if min(pb[0], pr[0]) < 0:
                raise ValueError("Negative full price")
            bp, rp = vs[m, "Bprime", axis], vs[m, "reference", axis]
            for v, n in ((bp, nb), (rp, nr)):
                if (v["m"], v["q"], v["r"], v["bits"]) != identity or v["shots"] != n:
                    raise ValueError("V3 schedule/shot mismatch")
            rb, rr = role_prices(bp, False), role_prices(rp, True)
            if add_intervals(rb.values()) != pb or add_intervals(rr.values()) != pr:
                raise ArithmeticError("V3 roles do not recover saved price bounds")
            result.append({
                "m": m, "q": b["q"], "r": b["r"], "bits": b["bits"], "axis": axis,
                "Bprime_shots": nb, "reference_shots": nr,
                **split_cost(nb, nr, pb, pr),
                "roles": {k: split_cost(nb, nr, rb[k], rr[k]) for k in ROLES},
            })
    totals = aggregate_splits(result)
    exact_b = bprime["exact_Bprime_total_fractions"]
    if totals["Bprime_cost"] != (F(exact_b["RZ_lower"]), F(exact_b["RZ_upper"])):
        raise ArithmeticError("Exact Bprime total mismatch")
    if totals["reference_cost"][0] != F(bprime["exact_reference_RZ_lower_fraction"]):
        raise ArithmeticError("Exact reference lower mismatch")
    for arm, key in (("Bprime", "Bprime_cost"), ("reference", "reference_cost")):
        saved = bprime["totals"][arm]
        if not (F.from_float(saved["total_RZ_lower"]) <= totals[key][0]
                and totals[key][1] <= F.from_float(saved["total_RZ_upper"])):
            raise ArithmeticError("Saved outward total does not enclose exact sum")
    rounds = [{"m": m, **aggregate_splits(x for x in result if x["m"] == m)}
              for m in range(18)]
    return {
        "input_commit": BASE, "input_paths": INPUTS,
        "scope": "saved RZ bounds and unchanged sufficient shots; no new prices or RPE",
        "interval_kind": "exact rational enclosures, not feasible endpoint policies or confidence intervals",
        "component_extrema_are_correlated": True,
        "rows": result, "round_costs": rounds, "totals": totals,
        "role_totals": {k: aggregate_splits(x["roles"][k] for x in result) for k in ROLES},
        "new_CX_depth_size_T_or_success_qualification": False,
        "next_action": "MANDATORY_STOP",
    }


def json_ready(value):
    if isinstance(value, F):
        return str(value)
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Fresh JSON path, after execution permission")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Existing output preserved")
    root = Path(__file__).resolve().parents[1]
    saved = {}
    for name, path in INPUTS.items():
        raw = subprocess.check_output(["git", "show", BASE + ":" + path], cwd=root)
        if path.endswith(".xz"):
            raw = lzma.decompress(raw)
        saved[name] = json.loads(raw)
    result = diagnose(**saved)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(json_ready(result), stream, ensure_ascii=False, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
