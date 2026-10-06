#!/usr/bin/env python3
"""R1.5 post-hoc attribution using saved R1 fractions/counts only.

No scientific modules, synthesis, angle evaluation, circuits, matrices, shots,
trajectories, Hamiltonians, molecular files, GPU or new candidate generation.
"""
import collections
import csv
import hashlib
import json
import subprocess
from fractions import Fraction as F
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = "24bfeb84a4ce87b56985d174dd98d1d5e1702a2b"
SOURCE = "d43d64a821a0249a0dfab12a2472bd3a72fdee74"
AUTH = "411f08f768244fe87b600d82308c3851847fe9e4"
R1 = ROOT / "artifacts/track_b_rte_reallocation_r1_result/2026-10-06/v1"
OUT = ROOT / "artifacts/track_b_r1p5_saved_attribution/2026-10-06"
RESULT_SHA = "f726ad70cb2643533f0d037b518cde1b702724adb4e6571fea25e26e4bfdd61e"
MARKER_SHA = "f25000ee5e3b94eb499b4a89bb28814e3de1aea641b249bcbb9d953d427c2ced"
PRECISIONS = ("1e-3", "1e-4", "1e-6")
ARMS = ("ordinary", "PTSC_K0", "A")
SCRIPT_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
META = {
    "input_R1_commit": BASE, "original_result_sha256": RESULT_SHA,
    "analysis_script_sha256": SCRIPT_SHA, "analysis_classification": "POSTHOC_ATTRIBUTION_DESIGN_INPUT",
    "no_science_rerun": True, "no_synthesis_rerun": True, "no_new_candidates": True,
    "R1_scientific_classification_unchanged": True, "mandatory_STOP": True,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def blob(path, commit=BASE):
    require(not str(path).lower().endswith(".npz"), "NPZ access excluded")
    return subprocess.check_output(["git", "-C", str(ROOT), "show", commit + ":" + str(path)])


def load(path):
    return json.loads(path.read_bytes())


def display(value):
    return format(float(F(value)), ".12g")


def direction(value):
    return "LOWER" if value < 1 else "HIGHER" if value > 1 else "EQUAL"


def interval(lo, hi=None):
    lo, hi = F(lo), F(lo if hi is None else hi)
    require(0 <= lo <= hi, "invalid saved interval")
    return {"lo": str(lo), "hi": str(hi)}


def bounds(value):
    return F(value["lo"]), F(value["hi"])


def dominates(left, right, coordinates):
    """Conservative robust dominance: all upper<=lower, at least one strict."""
    comparisons = [(bounds(left[k])[1], bounds(right[k])[0]) for k in coordinates]
    return all(a <= b for a, b in comparisons) and any(a < b for a, b in comparisons)


def row_identity(row):
    return {k: row[k] for k in ("context", "x", "sigma", "controlled", "epsilon", "arm")}


def shot_inputs(row):
    p, f = row["profile"], row["finite_confidence"]
    coefficient_bias = F(p["coefficient_L1_bias_upper"])
    bias = F(p["coefficient_and_strict_synthesis_bias_upper"])
    return {
        "ideal_B_squared_interval": p["ideal_B_squared"],
        "implemented_B": p["implemented_B"], "implemented_B_squared": p["implemented_weight_second_moment"],
        "coherent_measurement_bias_upper": str(bias), "coefficient_bias_upper": str(coefficient_bias),
        "coherent_synthesis_bias_upper": str(bias - coefficient_bias),
        "remaining_axis_budget": f["epsilon_stat_axis_lower"],
        "sufficient_shots_per_axis": f["sufficient_shots_per_axis"],
        "sufficient_shots_total": f["sufficient_shots_total"],
    }


def point(row):
    p, f = row["profile"], row["finite_confidence"]
    out = {**row_identity(row), "point_id": row["arm"] + ":" + row["epsilon"],
           "implemented_B_squared": interval(p["implemented_weight_second_moment"]),
           "ideal_B_squared": p["ideal_B_squared"],
           "sufficient_shots_per_axis": f["sufficient_shots_per_axis"],
           "coherent_measurement_bias_upper": p["coefficient_and_strict_synthesis_bias_upper"],
           "workspace": p["workspace_qubits_beyond_2_system"]}
    for name in ("T", "CX", "1Q"):
        out["E_" + name] = interval(p["E_native_cost"][name])
    for name, saved in (("G_T", "G_T"), ("G_CX", "G_CX"), ("G_1Q", "G_1Q_with_Hadamard_preparation_readout")):
        out[name] = interval(f[saved])
    return out


def pareto(primary):
    coordinate_sets = {
        "task_resource": ("G_T", "G_CX", "G_1Q"),
        "mechanism_implemented": ("implemented_B_squared", "E_T", "E_CX", "E_1Q"),
        "mechanism_ideal_interval_conservative": ("ideal_B_squared", "E_T", "E_CX", "E_1Q"),
    }
    groups, csv_rows = [], []
    for x in ("1/8", "1/4"):
        for sigma in (-1, 1):
            points = [point(r) for r in primary if r["x"] == x and r["sigma"] == sigma]
            require(len(points) == 9, "precision envelope does not have nine registered points")
            fronts = {}
            for name, coordinates in coordinate_sets.items():
                fronts[name] = {}
                for target in points:
                    fronts[name][target["point_id"]] = [p["point_id"] for p in points if dominates(p, target, coordinates)]
            baseline_checks = []
            for target in points:
                record = {**target}
                for name in coordinate_sets:
                    record[name + "_dominators"] = fronts[name][target["point_id"]]
                    record[name + "_nondominated"] = not record[name + "_dominators"]
                csv_rows.append(record)
                if target["arm"] == "A":
                    for candidate in points:
                        if candidate["arm"] == "A":
                            continue
                        baseline_checks.append({"A_point": target["point_id"], "baseline_point": candidate["point_id"],
                            "dominates_A": dominates(candidate, target, coordinate_sets["task_resource"]),
                            "coordinate_relations": {k: direction(F(candidate[k]["lo"]) / F(target[k]["lo"]))
                                                     for k in coordinate_sets["task_resource"]}})
            groups.append({"x": x, "sigma": sigma, "points": points,
                           "front_point_ids": {name: [key for key, value in values.items() if not value]
                                               for name, values in fronts.items()},
                           "dominators": fronts, "all_registered_baseline_checks_against_A": baseline_checks})
    sigma_checks = []
    fields = ["implemented_B_squared", "ideal_B_squared", "E_T", "E_CX", "E_1Q", "G_T", "G_CX", "G_1Q",
              "coherent_measurement_bias_upper", "sufficient_shots_per_axis", "workspace"]
    for x in ("1/8", "1/4"):
        minus, plus = [g for g in groups if g["x"] == x]
        negative = {p["point_id"]: p for p in minus["points"]}
        equal = all(all(negative[p["point_id"]][k] == p[k] for k in fields) for p in plus["points"])
        sigma_checks.append({"x": x, "saved_coordinates_exactly_equal": equal,
                             "independent_replication_count": 0,
                             "sign_control_policy": "fronts calculated separately; equality only permits display deduplication"})
    return {"coordinate_sets": coordinate_sets, "groups": groups, "sigma_control_checks": sigma_checks,
            "dominance_rule": "all coordinate upper(left)<=lower(right), at least one strict; exact fractions only",
            "interval_note": "G and implemented moment are saved exact model values; ideal B-squared retains its saved enclosure. Equal nondegenerate ideal intervals do not prove dominance under this conservative rule."}, csv_rows


def factorization(rows):
    indexed = {(r["context"], r["x"], r["sigma"], r["epsilon"], r["arm"]): r for r in rows if r["controlled"]}
    output = []
    for a in rows:
        if not a["controlled"] or a["arm"] != "A":
            continue
        for baseline in ("ordinary", "PTSC_K0"):
            b = indexed[a["context"], a["x"], a["sigma"], a["epsilon"], baseline]
            N_a, N_b = a["finite_confidence"]["sufficient_shots_total"], b["finite_confidence"]["sufficient_shots_total"]
            nr = F(N_a, N_b)
            for q in ("T", "CX", "1Q"):
                ea, eb = F(a["profile"]["E_native_cost"][q]), F(b["profile"]["E_native_cost"][q])
                overhead = F(5, 2) if q == "1Q" else F(0)
                task_ea, task_eb = ea + overhead, eb + overhead
                name = "G_1Q_with_Hadamard_preparation_readout" if q == "1Q" else "G_" + q
                ga, gb = F(a["finite_confidence"][name]), F(b["finite_confidence"][name])
                require(ga == N_a * task_ea and gb == N_b * task_eb, "task per-shot accounting mismatch")
                cr, gr = task_ea / task_eb, ga / gb
                require(gr == nr * cr, "exact G factorization mismatch")
                output.append({**row_identity(a), "baseline": baseline, "Q": q,
                    "A_shot_inputs": shot_inputs(a), "baseline_shot_inputs": shot_inputs(b),
                    "A_E_native": str(ea), "baseline_E_native": str(eb), "common_task_overhead_per_shot": str(overhead),
                    "A_E_task_per_shot": str(task_ea), "baseline_E_task_per_shot": str(task_eb),
                    "A_G": str(ga), "baseline_G": str(gb), "N_total_ratio": str(nr),
                    "E_task_per_shot_ratio": str(cr), "G_ratio": str(gr), "exact_product_verified": True,
                    "E_native_ratio": str(ea / eb) if eb else None,
                    "directions": {"shots": direction(nr), "task_per_shot_cost": direction(cr), "G": direction(gr)},
                    "causal_percentage_attribution": None})
    return output


def saved_angle_attribution(rows, synthesis):
    cache = {r["key"]: r for r in synthesis}
    contributions, row_totals, angle_users = [], [], collections.defaultdict(list)
    checked_events, basis_pairs = 0, 0
    rho_by_x = collections.defaultdict(set)
    for row in rows:
        eps = row["epsilon"]
        basis_keys = ["pi:1/8:scale:" + str(s) + ":epsilon:" + eps for s in (1, -1)]
        basis_T = sum(cache[k]["T_count"] for k in basis_keys)
        weights = collections.defaultdict(F)
        role_weights = collections.defaultdict(lambda: collections.defaultdict(F))
        clifford_one = F(0)
        for event in row["profile"]["events"]:
            checked_events += 1
            probability = F(event["canonical_probability_exact"])
            if event["complement"]:
                q = F(event["a"]) / F(event["b"])
                sign = -event["rotation_sign"]
            else:
                q = F(event["b"]) / F(event["a"]) if F(event["b"]) else F(0)
                sign = event["rotation_sign"]
            occurrences = collections.Counter()
            if q:
                scales = (sign, -sign) if row["controlled"] else (2 * sign,)
                for scale in scales:
                    key = "atan:" + str(q) + ":scale:" + str(scale) + ":epsilon:" + eps
                    require(key in cache, "angle identity is not an existing synthesis key")
                    occurrences[key] += 1
                if row["arm"] == "A":
                    rho_by_x[row["x"]].add(str(q))
                    role = "A_rho_odd_complement" if event["complement"] else "A_rho_even_order_" + event["label"].split(":")[0]
                elif row["arm"] == "CTS_collected":
                    role = "CTS_collected_common_angle"
                else:
                    role = row["arm"] + ("_x" if q == F(row["x"]) else "_x_over_3")
                for key, count in occurrences.items():
                    role_weights[key][role] += probability * count
            rotation_T = sum(count * cache[key]["T_count"] for key, count in occurrences.items())
            residual_T = event["native_cost"]["T"] - rotation_T
            if row["context"] == "distinct_basis":
                pairs = F(residual_T, basis_T)
                require(pairs >= 0 and pairs.denominator == 1, "basis cost residual has no integer pair decomposition")
                pairs = int(pairs)
            else:
                require(residual_T == 0, "unexplained T cost in a Pauli control")
                pairs = 0
            basis_pairs += pairs
            for key in basis_keys:
                if pairs:
                    occurrences[key] += pairs
                    role_weights[key]["distinct_basis_conjugator_saved_cost_residual"] += probability * pairs
            # Each frozen conjugator contributes both signs. This uses saved
            # cost residuals only, and never constructs an IR or reduces a word.
            reconstructed_T = sum(count * cache[key]["T_count"] for key, count in occurrences.items())
            reconstructed_error = sum(count * F(cache[key]["strict_operator_error_upper"]) for key, count in occurrences.items())
            require(reconstructed_T == event["native_cost"]["T"], "saved angle T sum mismatch")
            require(reconstructed_error == F(event["native_cost"]["strict_event_error_upper"]), "saved angle error sum mismatch")
            exact_one = event["native_cost"]["1Q"] - sum(count * cache[key]["one_qubit_count"] for key, count in occurrences.items())
            require(type(exact_one) is int and exact_one >= 0, "negative saved Clifford residual")
            clifford_one += probability * exact_one
            for key, count in occurrences.items():
                weights[key] += probability * count
        rotation_contribution, basis_contribution, weighted_one = F(0), F(0), F(0)
        for key in sorted(weights):
            saved = cache[key]
            w = weights[key]
            T = w * saved["T_count"]
            one = w * saved["one_qubit_count"]
            record = {**row_identity(row), "synthesis_key": key, "angle_key": saved["angle_key"],
                      "expected_occurrences_per_event": str(w), "weighted_T_contribution": str(T),
                      "weighted_1Q_contribution": str(one),
                      "weighted_strict_error_upper": str(w * F(saved["strict_operator_error_upper"])),
                      "role_expected_occurrences": {k: str(v) for k, v in role_weights[key].items()},
                      "T_count": saved["T_count"], "sequence_sha256": saved["sequence_sha256"]}
            contributions.append(record)
            angle_users[key].append({**row_identity(row), "roles": list(role_weights[key]),
                                     "expected_occurrences_per_event": str(w)})
            if saved["angle_key"].startswith("pi:"):
                basis_contribution += T
            else:
                rotation_contribution += T
            weighted_one += one
        require(rotation_contribution + basis_contribution == F(row["profile"]["E_native_cost"]["T"]), "row T attribution mismatch")
        require(weighted_one + clifford_one == F(row["profile"]["E_native_cost"]["1Q"]), "row 1Q attribution mismatch")
        row_totals.append({**row_identity(row), "E_T_rotation_keys": str(rotation_contribution),
                           "E_T_basis_keys": str(basis_contribution), "E_T_total": row["profile"]["E_native_cost"]["T"],
                           "E_1Q_synthesized_keys": str(weighted_one), "E_1Q_exact_Clifford_residual": str(clifford_one)})
    require(set(angle_users) == set(cache), "not all existing synthesis keys have a saved usage")
    angle_rows = []
    for saved in synthesis:
        kind, raw, _, scale = saved["angle_key"].split(":")
        users = angle_users[saved["key"]]
        angle_rows.append({k: saved[k] for k in ("key", "angle_key", "epsilon", "T_count", "Tdagger_count",
                          "one_qubit_count", "strict_operator_error_upper", "sequence_sha256")}
                         | {"T_gate_only_count": saved["T_count"] - saved["Tdagger_count"],
                            "sequence_length": len(saved["sequence"]),
                            "symbolic_RZ_angle_identity": scale + "*" + ("atan(" + raw + ")" if kind == "atan" else "pi*" + raw),
                            "angle_kind": kind, "saved_rational_argument": raw, "saved_scale": scale,
                            "used_by_arms": sorted({u["arm"] for u in users}),
                            "used_by_contexts": sorted({u["context"] for u in users}),
                            "native_modes": sorted({"controlled_half_angle_pair" if u["controlled"] and kind == "atan" else
                                                   "ordinary_rotation" if kind == "atan" else "basis_conjugator" for u in users}),
                            "role_labels": sorted({role for u in users for role in u["roles"]}),
                            "usage_rows": users})
    jumps = []
    by_angle = collections.defaultdict(dict)
    for row in angle_rows:
        by_angle[row["angle_key"]][row["epsilon"]] = row
    for key, precisions in by_angle.items():
        for before, after in zip(PRECISIONS, PRECISIONS[1:]):
            left, right = precisions[before], precisions[after]
            jumps.append({"angle_key": key, "from_precision": before, "to_precision": after,
                          "T_before": left["T_count"], "T_after": right["T_count"],
                          "T_delta": right["T_count"] - left["T_count"],
                          "T_relative_delta": str(F(right["T_count"] - left["T_count"], left["T_count"])),
                          "one_qubit_delta": right["one_qubit_count"] - left["one_qubit_count"],
                          "strict_error_before": left["strict_operator_error_upper"], "strict_error_after": right["strict_operator_error_upper"],
                          "role_labels": left["role_labels"], "resonance_threshold": None})
    contrasts = []
    for x, values in rho_by_x.items():
        require(len(values) == 1, "saved A has more than one rho argument")
        rho = next(iter(values))
        for eps in PRECISIONS:
            for scale in (-2, -1, 1, 2):
                akey = f"atan:{rho}:scale:{scale}:epsilon:{eps}"
                for label, argument in (("ordinary_PTSC_order0_x", F(x)), ("ordinary_order2_x_over_3", F(x) / 3)):
                    bkey = f"atan:{argument}:scale:{scale}:epsilon:{eps}"
                    a, b = cache[akey], cache[bkey]
                    contrasts.append({"x": x, "epsilon": eps, "scale": scale, "A_rho_key": akey,
                                      "baseline_angle_role": label, "baseline_key": bkey,
                                      "A_T": a["T_count"], "baseline_T": b["T_count"],
                                      "T_delta": a["T_count"] - b["T_count"],
                                      "T_ratio": str(F(a["T_count"], b["T_count"])),
                                      "counterfactual_or_new_synthesis": False})
    return {"angle_rows": angle_rows, "weighted_contributions": contributions, "row_totals": row_totals,
            "precision_jumps": jumps, "rho_contrasts": contrasts,
            "verification": {"saved_event_records_checked": checked_events, "saved_basis_pairs_inferred": basis_pairs,
                             "existing_keys_used": len(cache), "row_T_and_1Q_conservation": True,
                             "event_T_and_strict_error_conservation": True, "circuits_reconstructed": 0},
            "attribution_method": "rotation keys from stored a/b/complement/sign; signed basis pairs from integer saved-T residual, verified against saved strict-error sums; no IR construction"}


def precision_curves(rows):
    grouped = collections.defaultdict(dict)
    for row in rows:
        if row["controlled"]:
            grouped[row["context"], row["x"], row["sigma"], row["arm"]][row["epsilon"]] = row
    curves, flat = [], []
    for (context, x, sigma, arm), registered in grouped.items():
        records = []
        for eps in PRECISIONS:
            row = registered[eps]
            record = {**row_identity(row), **shot_inputs(row),
                      "E_T": row["profile"]["E_native_cost"]["T"], "G_T": row["finite_confidence"]["G_T"]}
            records.append(record)
            flat.append(record)
        strictest = F(records[-1]["G_T"])
        curves.append({"context": context, "x": x, "sigma": sigma, "arm": arm, "registered_points": records,
                       "strictest_G_T_compared_with_coarser_registered": [
                           {"coarser_precision": p["epsilon"], "ratio": str(strictest / F(p["G_T"])),
                            "direction": direction(strictest / F(p["G_T"]))} for p in records[:-1]],
                       "precision_selection_performed": False})
    return curves, flat


def table_json(name, rows, **extra):
    return {**META, "schema": "track_b_R1p5_" + name + "_v1", "rows": rows, **extra}


def write_json(name, data):
    with (OUT / name).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def write_csv(name, rows):
    flattened = []
    for record in rows:
        values = dict(META)
        for key, value in record.items():
            if isinstance(value, dict) and set(value) == {"lo", "hi"}:
                values[key + "_lo"] = value["lo"]
                values[key + "_hi"] = value["hi"]
                values[key + "_display"] = display(value["lo"])
            elif isinstance(value, (dict, list, tuple)):
                values[key] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            else:
                values[key] = value
        flattened.append(values)
    fields = list(dict.fromkeys(k for row in flattened for k in row))
    with (OUT / name).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(flattened)


def fixed_inputs():
    paths = [
        "docs/tracks/algorithm_codesign/r1_one_shot_result_validation_20261006.md",
        "docs/tracks/algorithm_codesign/r1_post_run_gpt_review_request_20261006.md",
    ]
    paths += ["artifacts/track_b_rte_reallocation_r1_result/2026-10-06/v1/" + name for name in
              ("result.json", "one_shot_consumed.json", "descriptive_summary_v1.json", "resource_rows_display_v1.csv",
               "saved_field_audit_v1.json", "evidence_manifest_v1.json")]
    paths += ["artifacts/track_b_rte_reallocation_r1_source/2026-10-06/" + name for name in
              ("authorization.json", "contract_v2.json", "source_manifest_v1.json")]
    paths += ["docs/tracks/algorithm_codesign/" + name for name in
              ("rte_reallocation_r1_native_semantics_v1.md", "rte_reallocation_r1_preregistration_v2.md",
               "rte_reallocation_r0_independent_proof_v1.md", "rte_reallocation_r05_equivalence_novelty_audit_v1.md")]
    identities = {}
    for path in paths:
        actual = (ROOT / path).read_bytes()
        require(actual == blob(path), "fixed input changed: " + path)
        identities[path] = {"commit": BASE, "sha256": sha(actual), "bytes": len(actual)}
    raw = (R1 / "result.json").read_bytes()
    require(sha(raw) == RESULT_SHA and sha((R1 / "one_shot_consumed.json").read_bytes()) == MARKER_SHA,
            "original result/marker identity mismatch")
    result = json.loads(raw)
    require(result["status"] == "R1_RESOURCE_MAP_COMPLETE_AWAITING_GPT_REVIEW"
            and result["source_commit"] == SOURCE and result["authorization_commit"] == AUTH,
            "R1 source/status mismatch")
    require(result["runs"] == 1 and result["retries"] == 0 and result["mandatory_STOP"] is True,
            "R1 run/STOP record mismatch")
    require(len(result["synthesis_rows"]) == 126 and len(result["resource_rows"]) == 264,
            "R1 inventory incomplete")
    require(all(row["error_pass"] is True for row in result["synthesis_rows"]), "saved guard failed")
    for row in result["synthesis_rows"]:
        require(sha(row["sequence"].encode()) == row["sequence_sha256"], "saved sequence identity mismatch")
        require(row["T_count"] == row["sequence"].count("T") + row["sequence"].count("t"), "saved T count mismatch")
    instruction = ROOT / "docs/tracks/algorithm_codesign/inputs/r1p5_saved_attribution_user_instruction_20261006.txt"
    require(sha(instruction.read_bytes()) == "5c4690ed459a99298bcb3bf8550cede9c924451cb07f873dc01736fd18d01ed1", "instruction identity changed")
    identities[str(instruction.relative_to(ROOT))] = {"role": "byte-exact user instruction snapshot",
        "sha256": sha(instruction.read_bytes()), "bytes": instruction.stat().st_size}
    native_path = "src/trottertracks/algorithm_codesign/rte_reallocation/native.py"
    native_reference = {"commit": SOURCE, "path": native_path, "sha256": sha(blob(native_path, SOURCE)),
                        "role": "frozen symbolic lowering convention reference only; never imported or executed"}
    return result, identities, native_reference


def method_checks():
    # Synthetic bookkeeping checks have no quantum/task input.
    a = {"u": interval(1), "v": interval(3)}
    b = {"u": interval(2), "v": interval(2)}
    require(not dominates(a, b, ("u", "v")) and not dominates(b, a, ("u", "v")), "trade-off collapsed")
    require(not dominates(a, a, ("u", "v")), "equality treated as strict dominance")
    uncertain = {"u": interval(1, 3), "v": interval(1)}
    require(not dominates(uncertain, b, ("u", "v")), "interval overlap treated as robust dominance")
    strict = {"u": interval(1), "v": interval(1)}
    require(dominates(strict, b, ("u", "v")), "strict coordinate dominance missing")
    na, nb, ea, eb = 6, 8, F(10), F(9)
    ratio = F(na, nb) * (ea + F(5, 2)) / (eb + F(5, 2))
    require(ratio == F(na) * (ea + F(5, 2)) / (nb * (eb + F(5, 2)))
            and ratio != F(na, nb) * ea / eb, "1Q task overhead omitted")
    return {"synthetic_bookkeeping_checks": 5, "tradeoff_equality_interval_and_1Q_overhead_checks": "PASS",
            "quantum_or_registered_science_test_calls": 0}


def main():
    require(not OUT.exists(), "analysis output already exists; do not overwrite attribution artifacts")
    checks = method_checks()
    result, identities, native_reference = fixed_inputs()
    rows = result["resource_rows"]
    primary = [r for r in rows if r["context"] == "distinct_basis" and r["controlled"]]
    require(len(primary) == 36, "primary row inventory mismatch")
    envelope, pareto_rows = pareto(primary)
    factors = factorization(rows)
    require(len(factors) == 216, "factorization inventory mismatch")
    angles = saved_angle_attribution(rows, result["synthesis_rows"])
    curves, curve_rows = precision_curves(rows)
    total_index = {(r["context"], r["x"], r["sigma"], r["controlled"], r["epsilon"], r["arm"]): r
                   for r in angles["row_totals"]}
    focal = []
    for x, eps in (("1/4", "1e-3"), ("1/8", "1e-4")):
        for baseline in ("ordinary", "PTSC_K0"):
            aa = total_index["distinct_basis", x, 1, True, eps, "A"]
            bb = total_index["distinct_basis", x, 1, True, eps, baseline]
            focal.append({"x": x, "epsilon": eps, "sigma_display": 1, "baseline": baseline,
                "factorizations": [f for f in factors if f["context"] == "distinct_basis" and f["x"] == x
                                   and f["epsilon"] == eps and f["sigma"] == 1 and f["baseline"] == baseline],
                "E_T_additive_saved_component_difference_A_minus_baseline": {
                    "rotation_keys": str(F(aa["E_T_rotation_keys"]) - F(bb["E_T_rotation_keys"])),
                    "basis_keys": str(F(aa["E_T_basis_keys"]) - F(bb["E_T_basis_keys"])),
                    "total": str(F(aa["E_T_total"]) - F(bb["E_T_total"]))},
                "causal_percentage_attribution": None})
    OUT.mkdir(parents=True)
    policy = {**META, "scope": "registered saved R1 data only; no scalar score, threshold, materiality, new precision or eta",
              "primary": "distinct_basis controlled ordinary/PTSC_K0/A; per-x per-sign nine-point registered envelope",
              "G_coordinates": ["G_T", "G_CX", "G_1Q_with_Hadamard_preparation_readout"],
              "dominance": envelope["dominance_rule"], "1Q_task_overhead": "5/2 native-independent 1Q gates per shot averaged over Re/Im",
              "shot_attribution": "N ratio times per-shot cost ratio; nonlinear shot inputs listed, never causally percent-decomposed",
              "angle_attribution": angles["attribution_method"], "native_reference": native_reference,
              "resonance_threshold": None, "counterfactuals_or_interpolation": False,
              "R1_classification": result["status"], "algorithm_adoption": False}
    write_json("analysis_policy_v1.json", policy)
    write_json("input_identity_v1.json", {**META, "inputs": identities, "native_reference_only": native_reference,
                                         "source_commit": SOURCE, "authorization_commit": AUTH, "original_marker_sha256": MARKER_SHA})
    write_json("precision_envelope_pareto_v1.json", {**META, **envelope})
    write_csv("precision_envelope_pareto_v1.csv", pareto_rows)
    tables = {
        "factorization_table": factors, "synthesis_angle_table": angles["angle_rows"],
        "angle_usage_contributions": angles["weighted_contributions"], "angle_row_component_totals": angles["row_totals"],
        "angle_precision_jumps": angles["precision_jumps"], "rho_angle_contrasts": angles["rho_contrasts"],
        "precision_bias_shot": curve_rows,
    }
    for name, records in tables.items():
        write_json(name + "_v1.json", table_json(name, records))
        write_csv(name + "_v1.csv", records)
    fronts = [{"x": g["x"], "sigma": g["sigma"], "all_front_points": g["front_point_ids"]["task_resource"],
               "A_front_points": [p for p in g["front_point_ids"]["task_resource"] if p.startswith("A:")]}
              for g in envelope["groups"]]
    primary_curves = [c for c in curves if c["context"] == "distinct_basis"]
    summary = {**META, "schema": "track_b_R1p5_attribution_summary_v1", "source_commit": SOURCE,
               "authorization_commit": AUTH, "original_marker_sha256": MARKER_SHA,
               "R1_terminal_status_unchanged": result["status"], "primary_envelope_fronts": fronts,
               "sigma_controls": envelope["sigma_control_checks"], "focal_registered_conditions": focal,
               "precision_curves_all_controlled": curves,
               "strictest_precision_has_higher_G_T_than_both_coarser_in_primary_curves": all(
                   all(v["direction"] == "HIGHER" for v in c["strictest_G_T_compared_with_coarser_registered"]) for c in primary_curves),
               "T_jump_raw_range": [min(j["T_delta"] for j in angles["precision_jumps"]), max(j["T_delta"] for j in angles["precision_jumps"])],
               "resonance_threshold_or_causal_percentage": None, "algorithm_adoption": False,
               "research_decision_owner": "GPT", "next_stage_authorized": False}
    write_json("attribution_summary_v1.json", summary)
    write_json("verification_v1.json", {**META, **checks, **angles["verification"],
        "Pareto_primary_rows": 36, "Pareto_per_x_per_sign_registered_points": 9,
        "exact_factorizations_verified": len(factors), "angle_rows": len(angles["angle_rows"]),
        "angle_precision_jumps": len(angles["precision_jumps"]), "rho_angle_contrasts": len(angles["rho_contrasts"]),
        "all_controlled_precision_rows": len(curve_rows), "original_result_marker_unchanged": True,
        "science_runner_synthesis_compile_matrix_trajectory_GPU_NPZ_calls": 0,
        "new_candidates_or_precision_points": 0, "saved_value_analysis_script_invocations": 1})
    print(json.dumps({"status": "POSTHOC_ATTRIBUTION_COMPLETE", "fronts": fronts,
                      "exact_factorizations": len(factors), "angle_rows": len(angles["angle_rows"]),
                      "science_calls": 0, "mandatory_STOP": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
