#!/usr/bin/env python3
"""One post-hoc fixed shift on one saved T0.1 law; stdlib/Fraction only.

No T0/T0.1/v3 imports, solver, quantization, synthesis, operator evaluation,
new authorization, or marker. Existing input files are strictly read-only.
"""
from decimal import Decimal, localcontext
from fractions import Fraction as F
from pathlib import Path
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
BASE = "5cf56e4a5949d64c24eac127bac0c223d431df87"
BRANCH = "track-b-ra-d0-t02-exact-certificate-20261008"
T01 = "artifacts/track_b_ra_d0_t01_active_support/2026-10-08"
OUTPUT = "artifacts/track_b_ra_d0_t02_exact_certificate/2026-10-08"
TABLE = "artifacts/track_b_ra_d0_preparation/2026-10-06/candidate_table_v1.json"
POLICY = "artifacts/track_b_ra_d0_source_review_v2/2026-10-07/numerical_baseline_amendment_v2.json"
SOURCE_CONTRACT = "artifacts/track_b_ra_d0_source_review_v3/2026-10-07/execution_contract_v3.json"
SCRIPT = "scripts/tracks/algorithm_codesign/audit_ra_d0_t02_exact_certificate.py"
CONTRACT_SHA256 = "e74862fd14c23147fd4431117cea5b3c15fe8cdc2b973316feb3afe476b4cdd8"
RESOURCES = ("T", "CX", "1Q")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def read_json(path):
    return json.loads((ROOT / path).read_bytes())


def git(*args):
    require(args[0] in ("show", "rev-parse", "branch", "remote", "status"), "non-read-only Git command")
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def quantity(value):
    value = F(value)
    with localcontext() as ctx:
        ctx.prec = 32
        display = format(Decimal(value.numerator) / Decimal(value.denominator), ".25E") if value else "0"
    return {"exact": str(value), "decimal": display}


def check_protected(protected):
    for path, digest in protected.items():
        require(not path.lower().endswith(".npz"), "molecular input path prohibited")
        raw = (ROOT / path).read_bytes()
        require(sha(raw) == digest and raw == git("show", BASE + ":" + path), "protected input mismatch: " + path)


def certificate(data, labels, groups, q, y, z, denominator, n, e, delta_num, kappa):
    """Independently evaluate fixed-law interval and scalar certificate fields."""
    lookup = {c["id"]: c for c in data["columns"]}
    counts = [v * denominator for v in q]
    law = {"q_nonnegative": all(v >= 0 for v in q), "sum_q_one": sum(q) == 1,
           "common_denominator": str(denominator), "counts_integral": all(v.denominator == 1 for v in counts),
           "counts_sum_denominator": sum(counts) == denominator, "y_positive": y > 0,
           "y_on_fixed_grid": (y * denominator).denominator == 1,
           "latent_nonnegative": all(v >= 0 for v in z.values()), "latent_sum_y": sum(z.values()) == y}
    members = []
    for group in groups:
        indices = group["indices"]
        mass = sum(q[j] for j in indices)
        lo, hi = group["interval"]
        latent = z[group["arm"]]
        residual = max(abs(mass - latent * lo), abs(mass - latent * hi))
        tau = (3 + hi / 2) / denominator
        members.append({"arm": group["arm"], "prototype": group["prototype"], "mass": str(mass),
                        "latent_z": str(latent), "coefficient_interval": list(map(str, (lo, hi))),
                        "residual_upper": quantity(residual), "tau": quantity(tau),
                        "margin": quantity(tau - residual), "pass": residual <= tau})
    endpoints = []
    for degree, target in enumerate(map(F, data["target"])):
        low = sum(q[j] * F(lookup[col]["D_intervals"][degree][0]) for j, (_, col) in enumerate(labels)) - y * target
        high = sum(q[j] * F(lookup[col]["D_intervals"][degree][1]) for j, (_, col) in enumerate(labels)) - y * target
        endpoints.append({"degree": degree, "lower": str(low), "upper": str(high),
                          "absolute_upper": str(max(abs(low), abs(high)))})
    residuals = [F(row["absolute_upper"]) for row in endpoints]
    xi = sum(residuals)
    dq = sum(q[j] * F(lookup[col]["d_upper"]) for j, (_, col) in enumerate(labels))
    h = e * y - dq - xi
    workspace = max([0] + [lookup[col]["workspace_peak"] for j, (_, col) in enumerate(labels) if q[j] > 0])
    costs = {r: sum(q[j] * F(lookup[col]["costs"][r]) for j, (_, col) in enumerate(labels)) for r in RESOURCES}
    overhead = {"T": F(0), "CX": F(0), "1Q": F(5, 2)}
    resource = {r: 2 * n * (costs[r] + overhead[r]) for r in RESOURCES}
    flags = {"sampler": all(v for k, v in law.items() if k != "common_denominator"),
             "membership": all(g["pass"] for g in members), "mean": xi <= y * delta_num,
             "confidence": h >= kappa, "workspace": workspace <= 1}
    return {"flags": flags, "all_five_pass": all(flags.values()), "sampler": law,
            "q_integer_counts": [int(v) for v in counts] if law["counts_integral"] else None,
            "membership_groups": members, "mean_endpoints": endpoints, "residual_by_degree": list(map(str, residuals)),
            "xi_upper": quantity(xi), "mean_bound": str(y * delta_num), "mean_margin": quantity(y * delta_num - xi),
            "d_dot_q": str(dq), "h_lower": str(h), "kappa_upper": str(kappa), "confidence_margin": quantity(h - kappa),
            "workspace_peak": workspace, "expected_native_cost": {r: str(v) for r, v in costs.items()},
            "measurement_overhead": {r: str(v) for r, v in overhead.items()},
            "resources": {r: str(v) for r, v in resource.items()}, "diagnostic_cost_only": True,
            "B": str(1 / y), "B_squared": str(1 / y ** 2)}


def shift_once(q, source_index, destination_index, delta):
    require(source_index != destination_index and q[source_index] >= delta, "negative shifted weight rejected; no alternate delta")
    shifted = list(q)
    shifted[source_index] -= delta
    shifted[destination_index] += delta
    return shifted


def audit(contract, counters):
    require(git("rev-parse", "HEAD").decode().strip() == BASE, "diagnostic must start at fixed T01 HEAD")
    require(git("branch", "--show-current").decode().strip() == BRANCH, "independent branch mismatch")
    remote = git("remote", "get-url", "origin").decode().strip()
    require(remote == "git@github.com:HIROMU1015/Partially-Randomized-Trotter.git", "repository ownership mismatch")
    old_identity = read_json(T01 + "/input_identity_v1.json")
    old_manifest = read_json(T01 + "/evidence_manifest_v1.json")
    protected = dict(old_identity["protected_sha256"])
    protected.update({p: record["sha256"] for p, record in old_manifest["files"].items()})
    protected[T01 + "/evidence_manifest_v1.json"] = sha(git("show", BASE + ":" + T01 + "/evidence_manifest_v1.json"))
    check_protected(protected)
    for key in ("source_commit", "authorization_commit", "result_commit", "T0_commit"):
        require(old_identity[key] == contract[key], "fixed commit identity mismatch: " + key)
    require(contract["T01_commit"] == BASE, "T01 identity mismatch")
    identity = {"schema": "ra_d0_t02_input_identity_v1", "T01_commit": BASE,
                **{k: contract[k] for k in ("source_commit", "authorization_commit", "result_commit", "T0_commit")},
                "only_task": contract["only_task"], "protected_sha256": protected,
                "marker_sha256": old_identity["marker_sha256"], "candidate_table_sha256": protected[TABLE],
                "source_contract_sha256": protected[SOURCE_CONTRACT], "fixed_shift_contract_sha256": CONTRACT_SHA256,
                "audit_script_sha256": sha((ROOT / SCRIPT).read_bytes()), "runtime": {"executable": sys.executable, "version": sys.version, "libraries": "stdlib only"}}
    prior = read_json(T01 + "/projection_result_v1.json")
    prior_cert = read_json(T01 + "/certificate_comparison_v1.json")
    prior_verification = read_json(T01 + "/verification_v1.json")
    require(prior["task"] == prior_cert["task"] == contract["only_task"], "other saved task prohibited")
    require(prior["classification"] == prior_verification["classification"] == "T01_MEMBERSHIP_REPAIRED_OTHER_CONSTRAINT_FAILED", "T01 classification mismatch")
    require(prior["original_v3_classification_unchanged"] == "D0_TECHNICAL_INCONCLUSIVE", "original classification mismatch")
    data = read_json(TABLE)["tables"][contract["only_x"]]
    policy = read_json(POLICY)
    source_contract = read_json(SOURCE_CONTRACT)
    require(source_contract["numerical_policy"] == POLICY, "numerical contract path mismatch")
    denominator, n, delta = int(contract["denominator"]), contract["only_n"], F(contract["delta"])
    require(denominator == 2 ** 60 == int(policy["denominator"]) and n == 767135 and delta == F(1, 2 ** 40), "fixed scale mismatch")
    require(delta * denominator == contract["shifted_integer_counts"] == 2 ** 20, "shift count mismatch")
    e, delta_num = F(policy["e"]), F(policy["delta_num"])
    require(e == F(1, 200) and delta_num == F(1, 10 ** 12), "fixed tolerance mismatch")
    constants = read_json("artifacts/track_b_ra_d0_t0_read_only_failure_audit/2026-10-07/nominal_residual_audit_v1.json")["confidence_constants"]
    require(prior_cert["ell_upper"] == constants["ell_upper"] and prior_cert["kappa_upper"] == constants["kappa_upper"], "saved ell/kappa mismatch")
    kappa = F(prior_cert["kappa_upper"])
    labels, groups = [], []
    for profile in data["B0_saved_profiles"]:
        if profile["epsilon"] != "1e-3":
            continue
        for member in profile["memberships"]:
            prototype = member["column_id"].split(":")[0]
            indices = list(range(len(labels), len(labels) + 3))
            labels.extend([profile["arm"], prototype + ":" + ep] for ep in contract["precision_order"])
            groups.append({"arm": profile["arm"], "prototype": prototype, "indices": indices,
                           "interval": tuple(map(F, member["ideal_weight_interval"]))})
    require(labels == prior["variable_labels"], "independent variable order reconstruction mismatch")
    fixed = prior["fixed_point"]
    receipt = fixed["rounding"]
    require(int(receipt["common_denominator"]) == denominator and receipt["q_counts_sum"] == denominator, "T01 sampler receipt mismatch")
    q = [F(v, denominator) for v in receipt["q_integer_counts"]]
    y = F(receipt["y_integer_count"], denominator)
    z = {arm: F(value) * y for arm, value in prior["relative_representation_shares"].items()}
    require(len(q) == len(labels) and list(map(str, q)) == fixed["q"] and str(y) == fixed["y"]
            and {r: str(v) for r, v in z.items()} == fixed["z"], "independent T01 fixed point reconstruction mismatch")
    before = certificate(data, labels, groups, q, y, z, denominator, n, e, delta_num, kappa)
    saved = prior_cert["points"]["PROJECTED_FIXED_DYADIC"]
    for key in ("residual_by_degree", "mean_endpoints", "mean_bound", "mean_margin", "xi_upper", "d_dot_q", "h_lower", "kappa_upper", "confidence_margin", "resources", "workspace_peak", "B", "B_squared"):
        require(before[key] == saved[key], "independent T01 certificate reconstruction mismatch: " + key)
    require({k: v for k, v in before["flags"].items() if k != "sampler"} == saved["flags"] and before["flags"]["sampler"], "T01 certificate flags mismatch")
    for new, old in zip(before["membership_groups"], saved["membership_groups"]):
        for key, value in new.items():
            require(value == old[key], "T01 membership reconstruction mismatch: " + key)
    # ID-based resolution in both the unique candidate table and B2 variable list.
    source_id, destination_id = contract["source_column_id"], contract["destination_column_id"]
    table_ids = [col["id"] for col in data["columns"]]
    require(len(set(table_ids)) == len(table_ids), "duplicate candidate ID")
    table_indices = [table_ids.index(cid) for cid in (source_id, destination_id)]
    variable_indices = [labels.index(["ordinary", cid]) for cid in (source_id, destination_id)]
    require(table_indices == contract["expected_candidate_table_indices_zero_based"] == [13, 14], "candidate index mismatch")
    require(variable_indices == contract["expected_variable_indices_zero_based"] == [4, 5], "variable index mismatch")
    src, dst = [data["columns"][j] for j in table_indices]
    logical_keys = ("prototype", "degree", "saved_ideal_ab_exact", "direction_ratio_exact")
    require(src["prototype"] == dst["prototype"] == "O2" and all(src[k] == dst[k] for k in logical_keys), "logical O2 identity mismatch")
    metadata_src = [{k: v for k, v in event.items() if k != "native_cost"} for event in src["events"]]
    metadata_dst = [{k: v for k, v in event.items() if k != "native_cost"} for event in dst["events"]]
    require(metadata_src == metadata_dst, "logical event labels/words/phases differ")
    require(canonical(src["D_intervals"]) == canonical(dst["D_intervals"]), "D intervals byte-canonical mismatch")
    require([[F(v) for v in pair] for pair in src["D_intervals"]] == [[F(v) for v in pair] for pair in dst["D_intervals"]], "D intervals exact mismatch")
    require(src["workspace_peak"] == dst["workspace_peak"] == 1 and src["workspace_source"] == dst["workspace_source"], "workspace identity mismatch")
    require(F(dst["d_upper"]) < F(src["d_upper"]), "destination error bound not strictly smaller")
    require(all(F(col["costs"][r]) >= 0 for col in (src, dst) for r in RESOURCES), "saved resource costs invalid")
    i, j = variable_indices
    require(q[i] >= delta, "insufficient source weight; no alternate delta")
    pair_identity = {"source_id": source_id, "destination_id": destination_id, "candidate_table_indices_zero_based": table_indices,
                     "variable_indices_zero_based": variable_indices, "same_logical_O2": True,
                     "logical_event_metadata_equal": True, "D_intervals_byte_canonical_equal": True,
                     "D_intervals_exact_equal": True, "D_intervals_sha256": sha(canonical(src["D_intervals"])),
                     "logical_event_metadata_sha256": sha(canonical(metadata_src)), "workspace_equal": True,
                     "workspace_peak": 1, "source_d_upper": quantity(F(src["d_upper"])),
                     "destination_d_upper": quantity(F(dst["d_upper"])), "destination_error_bound_strictly_smaller": True,
                     "source_costs": src["costs"], "destination_costs": dst["costs"], "source_weight_at_least_delta": True}
    counters["weight_shift_applications"] += 1
    shifted = shift_once(q, i, j, delta)
    after = certificate(data, labels, groups, shifted, y, z, denominator, n, e, delta_num, kappa)
    require(all(shifted[k] == q[k] for k in range(len(q)) if k not in (i, j)), "other q changed")
    require(after["membership_groups"] == before["membership_groups"], "group mass/membership changed")
    require(after["mean_endpoints"] == before["mean_endpoints"] and after["xi_upper"] == before["xi_upper"], "mean residual changed")
    require(after["workspace_peak"] == before["workspace_peak"], "workspace changed")
    require(all(shifted[k] == q[k] == 0 for k, (arm, _) in enumerate(labels) if arm in ("PTSC_K0", "A")), "inactive support changed")
    observed_h_delta = F(after["h_lower"]) - F(before["h_lower"])
    formula_h_delta = delta * (F(src["d_upper"]) - F(dst["d_upper"]))
    require(observed_h_delta == formula_h_delta == F(after["confidence_margin"]["exact"]) - F(before["confidence_margin"]["exact"]), "confidence delta exact identity failed")
    actual_resource_delta = {r: F(after["resources"][r]) - F(before["resources"][r]) for r in RESOURCES}
    formula_resource_delta = {r: 2 * n * delta * (F(dst["costs"][r]) - F(src["costs"][r])) for r in RESOURCES}
    require(actual_resource_delta == formula_resource_delta, "resource delta exact identity failed")
    require(before["q_integer_counts"][i] - after["q_integer_counts"][i] == after["q_integer_counts"][j] - before["q_integer_counts"][j] == 2 ** 20, "sampler count delta failed")
    if any(not passed for flag, passed in after["flags"].items() if flag != "confidence"):
        classification = "T02_OTHER_CERTIFICATE_FAIL"
    elif not after["flags"]["confidence"]:
        classification = "T02_CONFIDENCE_STILL_FAIL"
    else:
        classification = "T02_FULL_CERT_PASS"
    check_protected(protected)
    failed = [k for k, v in after["flags"].items() if not v]
    result = {"schema": "ra_d0_t02_weight_shift_result_v1", "classification": classification, "only_task": contract["only_task"],
              "delta": str(delta), "shifted_counts": 2 ** 20, "denominator": str(denominator), "pair_identity": pair_identity,
              "variable_labels": labels, "before_point": {"q": list(map(str, q)), "y": str(y), "z": {r: str(v) for r, v in z.items()}},
              "after_point": {"q": list(map(str, shifted)), "y": str(y), "z": {r: str(v) for r, v in z.items()}},
              "confidence_margin_before": before["confidence_margin"], "confidence_margin_after": after["confidence_margin"],
              "confidence_delta": quantity(observed_h_delta), "confidence_delta_formula": quantity(formula_h_delta),
              "weight_shift_applications": counters["weight_shift_applications"], "quantization_applications": 0,
              "failed_certificates": failed, "post_hoc_delta_choice": True, "diagnostic_single_point_only": True,
              "original_classifications_unchanged": contract["original_classifications_unchanged"],
              "B2_optimum_or_new_minimum_claim": False, "B2_B3_superiority_or_novelty_claim": False,
              "general_numerical_repair_claim": False, "mandatory_STOP": True, "next_stage_authorized": False}
    comparison = {"schema": "ra_d0_t02_certificate_comparison_v1", "only_task": contract["only_task"],
                  "ell_upper": prior_cert["ell_upper"], "kappa_upper": str(kappa), "e": str(e), "delta_num": str(delta_num),
                  "points": {"T01_PROJECTED_FIXED_DYADIC": before, "T02_FIXED_WEIGHT_SHIFT": after}, "classification": classification}
    resources = {"schema": "ra_d0_t02_resource_delta_v1", "only_task": contract["only_task"], "resource_interpretation": "diagnostic point costs; not minima or improvement evidence",
                 "formula": "2*n*delta*(cost_destination-cost_source)", "n": n, "weight_delta": str(delta),
                 "before": before["resources"], "after": after["resources"],
                 "delta_exact": {r: str(v) for r, v in actual_resource_delta.items()},
                 "delta": {r: quantity(v) for r, v in actual_resource_delta.items()},
                 "formula_delta_exact": {r: str(v) for r, v in formula_resource_delta.items()}, "exact_identity": True}
    verification = {"schema": "ra_d0_t02_verification_v1", "classification": classification, "status": "PASS_EXACT_SINGLE_POINT_VERIFICATION",
                    "independent_T01_count_receipt_reconstruction": True, "independent_T01_all_certificate_fields_match": True,
                    "pair_identity": pair_identity, "moved_counts_exactly_2_power_20": True,
                    "all_other_q_unchanged": True, "y_and_latent_z_unchanged": True, "all_group_masses_and_membership_unchanged": True,
                    "inactive_support_unchanged": True, "Dq_interval_endpoints_and_xi_exactly_unchanged": True,
                    "confidence_delta_exact_identity": True, "confidence_margin_sign_by_exact_fraction": True,
                    "resource_delta_exact_identity": True, "all_five_certificates_evaluated": True, "after_flags": after["flags"],
                    "failed_certificates": failed, "protected_files_count": len(protected), "protected_bytes_and_hashes_unchanged": True,
                    "T01_continuous_point_unchanged": True, "technical_blocking_issue": None,
                    "diagnostic_counts": dict(counters), "original_classifications_unchanged": contract["original_classifications_unchanged"],
                    "mandatory_STOP": True, "next_stage_authorized": False}
    return identity, result, comparison, resources, verification


def main():
    counters = {"weight_shift_applications": 0, "quantization_applications": 0}
    names = ("input_identity_v1.json", "weight_shift_result_v1.json", "certificate_comparison_v1.json", "resource_delta_v1.json", "verification_v1.json")
    out = ROOT / OUTPUT
    require(not any((out / name).exists() for name in names), "T0.2 outputs exist; refuse repeat/overwrite")
    require(sha((out / "fixed_shift_contract_v1.json").read_bytes()) == CONTRACT_SHA256, "pre-diagnostic contract changed")
    contract = read_json(OUTPUT + "/fixed_shift_contract_v1.json")
    require(contract["only_task"] == "P1_ANCHORS:1/8:767135:minimum:T" and contract["weight_shift_applications_max"] == 1, "fixed scope mismatch")
    try:
        bodies = audit(contract, counters)
    except Exception as error:
        details = {"classification": "T02_TECHNICAL_INCONCLUSIVE", "technical_reason": type(error).__name__ + ": " + str(error),
                   "diagnostic_counts": dict(counters), "mandatory_STOP": True, "next_stage_authorized": False}
        bodies = ({"schema": "ra_d0_t02_input_identity_v1", "T01_commit": BASE, "input_verification_incomplete": True,
                   "audit_script_sha256": sha((ROOT / SCRIPT).read_bytes()), "fixed_shift_contract_sha256": CONTRACT_SHA256},
                  {"schema": "ra_d0_t02_weight_shift_result_v1", **details},
                  {"schema": "ra_d0_t02_certificate_comparison_v1", "points": {}, **details},
                  {"schema": "ra_d0_t02_resource_delta_v1", "incomplete": True, **details},
                  {"schema": "ra_d0_t02_verification_v1", "status": "TECHNICAL_SINGLE_POINT_FAILURE", **details})
    forbidden_modules = [name for name in sys.modules if name.split(".")[0] in ("numpy", "scipy", "highspy", "trotterlib", "trottertracks")]
    require(not forbidden_modules, "forbidden modules imported")
    bodies[-1]["forbidden_modules_imported"] = forbidden_modules
    bodies[-1]["execution_counts_this_T02"] = {key: 0 for key in contract["forbidden_operations"]}
    bodies[-1]["execution_counts_this_T02"].update({"solver_calls": 0, "runner_calls": 0})
    for name, body in zip(names, bodies):
        with (out / name).open("x", encoding="utf-8") as stream:
            json.dump(body, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
    print(json.dumps({"classification": bodies[1]["classification"], "technical_reason": bodies[1].get("technical_reason"),
                      "points": {name: {"flags": cert["flags"], "confidence_margin": cert["confidence_margin"]}
                                 for name, cert in bodies[2].get("points", {}).items()},
                      "resource_delta": bodies[3].get("delta_exact"), "diagnostic_counts": counters,
                      "solver_calls": 0, "RA_D0_runner": 0, "mandatory_STOP": True}, indent=2))


if __name__ == "__main__":
    main()
