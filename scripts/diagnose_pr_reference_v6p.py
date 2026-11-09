"""V6-P preparation: fixed saved-bound screen, only after execution permission.

Import and --inspect-inputs never evaluate saved scientific numbers. The run
subcommand is the single scientific campaign; it is NOT authorized by preparing
this adapter. No phase provider, circuit, sampler or price provider is included.
"""
from __future__ import annotations

import argparse
import ast
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import json
import lzma
import math
from pathlib import Path
import subprocess
import sys
import types

BASE = "f2638a7845685a119619382b850441f5d8d33713"
PLAN = "e5d6e388ee2543b47ebcc1dfd9e19862d08ac327"
ROOT = Path(__file__).resolve().parents[1]
SELF = "scripts/diagnose_pr_reference_v6p.py"
INPUTS = {
    "stage3": "artifacts/pr_reference_transfer_qualification/2026-11-06/STAGE3_ALL_SUPPORT.json",
    "rounds": "artifacts/track_c_guarantee_cn_price/2026-11-06/A_guarantee/verified/ROUNDS.json",
    "stage4": "artifacts/pr_reference_transfer_qualification/2026-11-06/STAGE4_RPE.json",
    "bprime": "artifacts/pr_reference_bprime/2026-11-07/ERROR_RPE.json",
    "prices": "artifacts/pr_reference_bprime/2026-11-07/TOTALS_18ROUND.json",
    "d0": "artifacts/pr_reference_cost_structure/2026-11-08/D0_run01/RESULTS.json.xz",
}
REUSE = {
    "iv": "src/trottertracks/track_c_compiler_a_18round.py",
    "cost": "scripts/diagnose_pr_reference_cost_structure.py",
    "check": "scripts/check_pr_reference_cost_structure_d0.py",
}
TARGET = (17, 131072, 32, 88)


def git_bytes(commit, path):
    return subprocess.check_output(["git", "show", commit + ":" + path], cwd=ROOT)


def inventory():
    """Byte provenance only; no JSON decoding or numerical evaluation."""
    result = {}
    for path in (*INPUTS.values(), *REUSE.values()):
        raw = git_bytes(BASE, path)
        blob = subprocess.check_output(
            ["git", "rev-parse", BASE + ":" + path], cwd=ROOT, text=True).strip()
        result[path] = {"git_blob": blob, "sha256": hashlib.sha256(raw).hexdigest(),
                        "bytes": len(raw)}
    return {"input_commit": BASE, "plan_commit": PLAN, "files": result,
            "scientific_values_evaluated": False}


@lru_cache(maxsize=None)
def reused(name):
    """Use pinned helpers without importing quantum builders or running D0."""
    path = REUSE[name]
    raw = git_bytes(BASE, path)
    module = types.ModuleType("_v6p_reuse_" + name)
    module.__file__ = str(ROOT / path)
    if name == "iv":
        # Copy no equations: compile the original three helper definitions.
        # Importing the original full module would import Qiskit/builders.
        import mpmath as mp
        tree = ast.parse(raw, filename=path)
        defs = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name in ("I", "upper", "lower")]
        if {n.name for n in defs} != {"I", "upper", "lower"}:
            raise ValueError("Pinned interval helpers missing")
        module.mp, module.math = mp, math
        exec(compile(ast.Module(body=defs, type_ignores=[]), path, "exec"), vars(module))
        mp.iv.prec = 192
    else:
        # Both pinned scripts are passive at import; their main() is not called.
        exec(compile(raw, path, "exec"), vars(module))
    return module


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    require(type(value) in (int, float, str), "Numeric endpoint required")
    require(math.isfinite(float(value)), "Finite endpoint required")
    return value


def exact_iv_endpoint(value):
    sign, mantissa, exponent, bitcount = value
    require(bitcount >= 0, "Nonfinite interval endpoint")
    return F(-mantissa if sign else mantissa) * F(2) ** exponent


def shot_ceiling(radius, beta):
    """Original outward-endpoint/192-bit ceil rule, pure scalar arithmetic."""
    iv = reused("iv")
    mp, I = iv.mp, iv.I
    mp.iv.prec = 192
    radius, beta = number(radius), number(beta)
    require(float(radius) > 0 and 0 < float(beta) < math.pi / 2,
            "Nonpositive radius/statistical margin")
    epsilon = I(radius) * mp.iv.sin(I(beta)) / mp.iv.sqrt(I(2))
    required = 2 * mp.iv.log(I(1440)) / epsilon ** 2
    with mp.workprec(192):
        n = int(mp.ceil(mp.mpf(required._mpi_[1])))
    lo, hi = map(exact_iv_endpoint, required._mpi_)
    require(F(n) >= hi and F(n - 1) < hi, "Upper endpoint ceil failed")
    return {"shots_per_axis": n, "required_shots_interval": (lo, hi)}


def bound_and_shots(nominal, pf, shared, native_raw, phase):
    """No total-minus-phase subtraction; input components are raw-disk bounds."""
    iv = reused("iv")
    mp, I = iv.mp, iv.I
    mp.iv.prec = 192
    for x in (*nominal, pf, shared, native_raw, phase):
        number(x)
    require(all(float(x) >= 0 for x in (pf, shared, native_raw, phase)),
            "Negative error bound")
    rho0 = I(nominal[0]) + I(nominal[1])
    other = I(shared) + I(native_raw)
    disk = other + I(phase)
    require(iv.lower(rho0) > 0 and iv.upper(disk) < iv.lower(rho0),
            "Raw disk does not fit nominal radius")
    ratio = disk / rho0
    rho = rho0 - disk
    beta = I('.4') - I(pf) - mp.iv.atan2(ratio, mp.iv.sqrt(1 - ratio ** 2))
    rl, bl = iv.lower(rho), iv.lower(beta)
    return {"phase_upper": phase, "other_raw_disk_upper": iv.upper(other),
            "total_raw_disk_upper": iv.upper(disk), "raw_radius_lower": rl,
            "statistical_phase_lower": bl, **shot_ceiling(rl, bl)}


def candidate_bound(prefix, normalization):
    iv = reused("iv")
    M, norm = iv.I(number(prefix)), iv.I(number(normalization))
    require(float(prefix) >= 0 and float(normalization) >= 0, "Negative phase bound")
    single_round = 2 * iv.I(2) ** -53 * M
    return {"normalization_upper": normalization,
            "single_round_raw_phase_upper": iv.upper(single_round),
            "phase_upper": iv.upper(norm + single_round),
            "formula": "saved_B_norm + 2 * 2**(-53) * saved_M"}


def scaled(n, price):
    p = reused("cost").interval(price)
    return min(n * x for x in p), max(n * x for x in p)


def recost(d0, prices, new_n):
    """Reweight saved prices; preserve correlation when shot change is negative."""
    cost, check = reused("cost"), reused("check")
    require(type(new_n) is int and new_n > 0, "Positive integer candidate shots")
    rows = cost.indexed(d0["rows"], lambda x: (x["m"], x["axis"]))
    ps = cost.indexed(prices["rows"], lambda x: x["m"])
    require(set(rows) == {(m, a) for m in range(18) for a in "XY"}
            and set(ps) == set(range(18)), "Saved price coverage")
    out = []
    for (m, axis), row in sorted(rows.items()):
        require(set(row["roles"]) == set(cost.ROLES), "Missing/extra saved role")
        nb, old_nr = row["Bprime_shots"], row["reference_shots"]
        require(type(nb) is int and type(old_nr) is int and min(nb, old_nr) > 0,
                "Invalid saved shot count")
        require(tuple(row[k] for k in ("m", "q", "r", "bits"))
                == tuple(ps[m][k] for k in ("m", "q", "r", "bits")), "Price identity")
        require(nb == ps[m]["shots_per_axis"], "Bprime schedule changed")
        p = ps[m]["price_per_axis"][axis]
        pb = cost.interval((p["Bprime_RZ_lower"], p["Bprime_RZ_upper"]))
        pr = cost.interval((p["reference_RZ_lower"], p["reference_RZ_upper"]))
        require(pb[0] >= 0 and pr[0] >= 0, "Negative whole-circuit price")
        require(scaled(nb, pb) == check.pair(row["Bprime_cost"])
                and scaled(old_nr, pr) == check.pair(row["reference_cost"]), "D0 whole price mismatch")
        for field in ("Bprime_cost", "reference_cost"):
            require(check.add(check.pair(x[field]) for x in row["roles"].values())
                    == check.pair(row[field]), "Saved roles fail to recover whole cost")
        if m != 17:
            out.append({"m": m, "axis": axis, "shots_per_axis": old_nr,
                        "reference_cost": check.pair(row["reference_cost"]),
                        "roles": {k: check.pair(row["roles"][k]["reference_cost"])
                                  for k in cost.ROLES}})
            continue
        require((m, row["q"], row["r"], row["bits"]) == TARGET
                and (nb, old_nr) == (626, 630), "Fixed m17 schedule mismatch")
        role_new = {}
        for role in cost.ROLES:
            role = str(role)
            rb = tuple(x / nb for x in check.pair(row["roles"][role]["Bprime_cost"]))
            rr = tuple(x / old_nr for x in check.pair(row["roles"][role]["reference_cost"]))
            split = cost.split_cost(nb, new_n, rb, rr)
            check.check_split(split, nb, new_n, rb, rr)
            role_new[role] = split["reference_cost"]
        require(check.add(role_new.values()) == scaled(new_n, pr), "New role/whole sum mismatch")
        # Couple the same price endpoint in old and new cost: independent
        # interval subtraction/addition would widen this when new_n < old_nr.
        for price in pr:
            require(old_nr * price + (new_n - old_nr) * price == new_n * price,
                    "Same-endpoint cost identity")
        out.append({"m": m, "axis": axis, "shots_per_axis": new_n,
                    "reference_cost": scaled(new_n, pr), "roles": role_new,
                    "delta_to_saved_reference": scaled(new_n - old_nr, pr)})
    old_ref = check.add(check.pair(x["reference_cost"]) for x in rows.values())
    old_b = check.add(check.pair(x["Bprime_cost"]) for x in rows.values())
    require(old_ref == check.pair(d0["totals"]["reference_cost"])
            and old_b == check.pair(d0["totals"]["Bprime_cost"]), "Saved total mismatch")
    require(old_b == check.pair((prices["exact_Bprime_total_fractions"]["RZ_lower"],
                                prices["exact_Bprime_total_fractions"]["RZ_upper"]))
            and old_ref[0] == F(prices["exact_reference_RZ_lower_fraction"]), "Original total mismatch")
    new_ref = check.add(x["reference_cost"] for x in out)
    difference = new_ref[0] - old_b[1], new_ref[1] - old_b[0]
    role_totals = {role: check.add(x["roles"][role] for x in out) for role in cost.ROLES}
    require(check.add(role_totals.values()) == new_ref, "Global roles fail total")
    return {"Bprime_total_unchanged": old_b, "saved_reference_total": old_ref,
            "conditional_reference_total": new_ref, "reference_minus_Bprime": difference,
            "delta_to_saved_reference": check.add(x["delta_to_saved_reference"] for x in out if x["m"] == 17),
            "conditional_role_totals": role_totals, "rows": out,
            "cost_intervals_are_not_feasible_endpoint_policies": True,
            "price_unchanged_is_conditional": True}


def campaign(saved):
    """Only this function combines real saved bounds, N and cost. Not a preflight."""
    cost = reused("cost")
    def by_m(name):
        rows = saved[name]["rows"]
        result = cost.indexed(rows, lambda x: x["m"])
        require(set(result) == set(range(18)), name + " round coverage")
        return result
    s3, s4, bs, ps = (by_m(k) for k in ("stage3", "stage4", "bprime", "prices"))
    cases = [x for x in saved["rounds"] if x["name"] == "CN_S2_h0.02_ld3"]
    require(len(cases) == 1, "Unique original case required")
    rs = cost.indexed(cases[0]["rows"], lambda x: x["m"])
    require(set(rs) == set(range(18)), "Original case coverage")
    for m in range(18):
        for source in (s3, s4, bs, rs):
            require(tuple(source[m][k] for k in ("m", "q", "r"))
                    == tuple(ps[m][k] for k in ("m", "q", "r")), "Round identity")
        require(s4[m]["bits"] == bs[m]["bits"] == ps[m]["bits"]
                == rs[m]["finite_bit_proposal"]["bits_per_uniform_draw"], "Finite-bit identity")
        require(bs[m]["shots_per_axis"] == ps[m]["shots_per_axis"], "Bprime N identity")
        require(s4[m]["arms"]["reference"]["alpha_axis"] == "1/720", "Alpha changed")
    require(tuple(ps[17][k] for k in ("m", "q", "r", "bits")) == TARGET, "Target changed")
    require(saved["bprime"]["interval_bits"] == 192
            and saved["bprime"]["alpha_per_axis"] == "1/720"
            and saved["bprime"]["energy_accuracy_Ha"] == 1.5936e-4
            and saved["bprime"]["phase_cap_rad"] == .4
            and saved["bprime"]["failure_total"] == "36/720=.05"
            and saved["stage4"]["full_failure_probability"] == "36*(1/720)=0.05"
            and saved["stage4"]["energy_accuracy_Ha"] == 1.5936e-4
            and saved["stage4"]["phase_budget"] == ".4 rad", "Success convention changed")
    row, original = s3[17], rs[17]
    ref = row["arms"]["reference"]
    require(ref["all_support_sufficient"] and s4[17]["arms"]["reference"]["success_conditions_closed"],
            "Missing saved all-support qualification")
    require(bs[17]["shots_per_axis"] == 626
            and s4[17]["arms"]["reference"]["shots_per_axis"] == 630, "Old target N changed")
    nominal = original["raw_radius_lower"], original["raw_extra_disk_upper"]
    args = (nominal, original["PF_phase_upper"], row["old_source_shared_raw_disk_upper"],
            ref["native_raw_disk_upper"])
    old = bound_and_shots(*args, ref["phase_generation_extra_raw_disk_upper"])
    b_check = shot_ceiling(bs[17]["raw_radius_lower"], bs[17]["statistical_phase_lower"])
    require(old["shots_per_axis"] == 630 and b_check["shots_per_axis"] == 626, "Old N reproduction failed")
    ph = row["phase_arithmetic"]
    candidate = candidate_bound(ph["exact_prefix_absolute_upper"], ph["normalization_vs_controlled_source_upper"])
    new = bound_and_shots(*args, candidate["phase_upper"])
    # Validate D0 N/schedule against saved success rows, without recomputing m0..16 N.
    for x in saved["d0"]["rows"]:
        m = x["m"]
        require(x["reference_shots"] == s4[m]["arms"]["reference"]["shots_per_axis"]
                and x["Bprime_shots"] == bs[m]["shots_per_axis"], "D0/success N mismatch")
    accounts = recost(saved["d0"], saved["prices"], new["shots_per_axis"])
    diff = accounts["reference_minus_Bprime"]
    if candidate["phase_upper"] >= ph["phase_error_upper"]:
        status = "NO_PHASE_BOUND_IMPROVEMENT"
    elif new["shots_per_axis"] >= 630:
        status = "NO_SUFFICIENT_SHOT_REDUCTION"
    elif diff[1] < 0:
        status = "CONDITIONAL_B1_TOTAL_RZ_ADVANTAGE"
    elif diff[0] > 0:
        status = "SHOT_REDUCTION_WITHOUT_TOTAL_RZ_ADVANTAGE"
    else:
        status = "INCONCLUSIVE"
    return {"status": status, "fixed_target": TARGET, "old_reference_reconstructed": old,
            "old_reference_saved": ref, "Bprime_saved": bs[17], "Bprime_m17_ceil_check": b_check,
            "old_phase_breakdown": ph, "candidate_phase_bound": candidate,
            "conditional_reference": new, "accounting_RZ_only": accounts,
            "assumptions": ["same binary literals, exact sum, one final binary64 RN",
                            "old arithmetic/normalization model and all-support M",
                            "unchanged native context; final P still one RZ"],
            "old_internal_intervals_bitwise_reproduced": False,
            "new_phase_provider_implemented_or_qualified": False,
            "new_RPE_N_evaluated_only_for_reference_m17": True,
            "next_action": "MANDATORY_STOP"}


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(reused("cost").json_ready(value), stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("inspect-inputs", help="Provenance only; no scientific numerical execution")
    run = sub.add_parser("run", help="Scientific campaign ONLY after explicit V6-P execution permission")
    run.add_argument("--source-commit", required=True, help="Published commit fixing this adapter")
    run.add_argument("--output", type=Path, required=True, help="Fresh campaign directory")
    args = parser.parse_args()
    if args.mode == "inspect-inputs":
        print(json.dumps(inventory(), indent=2))
        return
    require(sys.flags.optimize == 0, "Keep original checker assertions enabled")
    require(git_bytes(args.source_commit, SELF) == Path(__file__).read_bytes(), "Execution SOURCE changed")
    # Fresh output preserves a failed/finished one-shot attempt; no resume loop.
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        meta = inventory()
        write_json(args.output / "SOURCE.json", {"execution_commit": args.source_commit,
                   "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "inputs": meta, "python": sys.version, "scientific_campaign_limit": 1})
        saved = {}
        for name, path in INPUTS.items():
            raw = git_bytes(BASE, path)
            saved[name] = json.loads(lzma.decompress(raw) if path.endswith(".xz") else raw)
        result = campaign(saved)
        write_json(args.output / "RESULTS.json", result)
        write_json(args.output / "STOP.json", {"status": result["status"], "next_action": "MANDATORY_STOP",
                   "new_phase_provider_or_circuit": False, "V6Q_started": False})
    except Exception as exc:
        write_json(args.output / "STOP.json", {"status": "FAIL", "error": str(exc),
                   "error_type": type(exc).__name__, "next_action": "MANDATORY_STOP"})
        raise


if __name__ == "__main__":
    main()
