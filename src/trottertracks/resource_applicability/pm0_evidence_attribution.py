"""PM-0: fixed saved-JSON/source attribution. Stdlib only, stdout-only output.

No molecular data, cache, runtime, science module import, compile or sampling.
Input names are an explicit allowlist; paths embedded in JSON are never opened.
Floating checks below are bookkeeping checks, not modified science thresholds.
"""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess

EVIDENCE_COMMIT = "b6e65c6123475add5e620ec1064f361378bead95"
METRICS = ("rz_count", "rz_depth", "cx_count", "cx_depth", "total_depth", "circuit_size")
INPUTS = {
    "m1a": "artifacts/pr2_matched_accuracy_m1_execution/2026-09-30/pr2_matched_accuracy_m1_a_result_v1.json",
    "m1b": "artifacts/pr2_matched_accuracy_m1_b1_execution/2026-09-30/pr2_matched_accuracy_m1_b1_compile_map_result_v2.json",
    "validation": "artifacts/pr2_matched_accuracy_m1_b1_result_validation/2026-10-03/pr2_matched_accuracy_m1_b1_result_validation_v1.json",
    "m2": "artifacts/pr2_matched_accuracy_m2_transfer_execution/2026-10-04/pr2_matched_accuracy_m2_transfer_result_v2.json",
    "s2": "artifacts/pr2_v4_s2_development/2026-09-29/pr2_s2_development_resource_result_parallel_v1.json",
    "signal_source": "src/trotterlib/pr2_matched_accuracy_m1_execution.py",
    "contract_source": "src/trotterlib/pr2_matched_accuracy_m1_contract.py",
    "block_source": "src/trotterlib/df_partial_s2.py",
    "repeated_source": "src/trotterlib/df_partial_s2_repeated.py",
}
COMMON_FIVE = (
    "B2-rank3-q1-r4-K2", "B2-rank3-q1-r8-K2", "B0-rank6-q1-r0-K0",
    "B1-rank12-q1-r0-K0", "B3-rank0-q8-r32-K4",
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def near(a, b):
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9)


def verified_bytes(current, committed):
    require(current == committed, "input differs from evidence commit")
    return hashlib.sha256(current).hexdigest()


def load_inputs(root):
    """Read exactly five JSON and four source files; Git blobs must match."""
    loaded, audit = {}, {}
    for name, relative in INPUTS.items():
        data = (root / relative).read_bytes()
        blob = subprocess.check_output(
            ["git", "show", f"{EVIDENCE_COMMIT}:{relative}"], cwd=root
        )
        audit[name] = {"path": relative, "bytes": len(data),
                       "sha256": verified_bytes(data, blob), "commit_blob_identical": True}
        loaded[name] = json.loads(data) if relative.endswith(".json") else data.decode()
    return loaded, audit


def frontier(rows, metrics=METRICS):
    return [r for r in rows if not any(
        all(s[m] <= r[m] for m in metrics) and any(s[m] < r[m] for m in metrics)
        for s in rows
    )]


def minimum(rows, metric="rz_count"):
    return min(rows, key=lambda r: (r[metric], r["candidate_id"]))


def group_summary(rows):
    eligible = [r for r in rows if r["accuracy_eligible"]]
    by_method = {}
    for method in sorted({r["method"] for r in eligible}):
        best = minimum([r for r in eligible if r["method"] == method])
        by_method[method] = {k: best[k] for k in ("candidate_id", "rz_count", "total_shots", "effective_rz_cost")}
    return {"registered_count": len(rows), "eligible_count": len(eligible),
            "best_by_method": by_method,
            "point_pareto": sorted(r["candidate_id"] for r in frontier(eligible))}


def lower_envelope(rows):
    """Exact affine inequalities; no numerical P grid. Shared-boundary ties kept.

    A line may be competitive only at a crossing (zero-width interval).
    Identical lines each retain their interval, rather than silently dropping ties.
    """
    answer = []
    for row in rows:
        lo, hi = 0.0, math.inf
        for other in rows:
            a = row["total_shots"] - other["total_shots"]
            b = other["rz_count"] - row["rz_count"]
            if a == 0:
                if b < 0:
                    hi = -1.0
                    break
            elif a > 0:
                hi = min(hi, b / a)
            else:
                lo = max(lo, b / a)
        if hi >= lo:
            answer.append({"candidate_id": row["candidate_id"], "P_min": lo,
                           "P_max": None if math.isinf(hi) else hi,
                           "shot_slope": row["total_shots"], "rz_intercept": row["rz_count"]})
    return sorted(answer, key=lambda r: (r["P_min"], r["candidate_id"]))


def complex_bias(signal):
    as_complex = lambda x: complex(x["real"], x["imag"])
    total = as_complex(signal["corrected_mean"]) - as_complex(signal["exact_target"])
    result = {"total_bias_real_signed": total.real, "total_bias_imag_signed": total.imag,
              "total_bias_abs": abs(total)}
    if signal["candidate"]["method"] == "B0":
        # Saved pf_exact_tail_signal is the approximate discarded-H evolution,
        # not the exact truncated-H reference z_D required to split this bias.
        result.update(discard_bias_abs=None, pure_discard_pf_bias_abs=None,
                      bias_decomposition_status="MISSING_EXACT_TRUNCATED_H_SIGNAL")
    else:
        outer = as_complex(signal["pf_exact_tail_signal"]) - as_complex(signal["exact_target"])
        finite = as_complex(signal["corrected_mean"]) - as_complex(signal["pf_exact_tail_signal"])
        require(near(abs(outer + finite - total), 0), "complex bias closure")
        result.update(outer_bias_real_signed=outer.real, outer_bias_imag_signed=outer.imag,
                      finite_bias_real_signed=finite.real, finite_bias_imag_signed=finite.imag,
                      bias_decomposition_status="SAVED_COMPLEX_DIFFERENCES_NOT_SUM_OF_MAGNITUDES")
    return result


def row_from_saved(candidate, signal, means, work, stage):
    shots = signal["axis_shots"]
    eligible = signal["accuracy_eligible"]
    n = shots["real"] + shots["imag"] if eligible else None
    row = {k: candidate[k] for k in ("candidate_id", "method", "rank", "q", "r", "K")}
    row.update(stage=stage, T=candidate["T"], delta=candidate["delta"], R=candidate["q"]*candidate["r"],
               candidate_fingerprint=signal["candidate_fingerprint"], accuracy_eligible=eligible,
               total_shots=n, normalization=signal["normalization_multiplier"],
               normalization_squared=signal["normalization_multiplier"]**2,
               n_det=signal["n_det"], n_rand_ceil=signal["n_rand"], n_fixed=signal["n_fixed"],
               expected_random_applications=signal.get("expected_random_applications_exact", 0.0),
               tau=signal.get("finite_distribution", {}).get("dimensionless_step_time"),
               legacy_outer_pf_bias_abs=signal["outer_pf_bias_abs"],
               finite_truncation_bias_abs=signal["finite_truncation_bias_abs"])
    for axis, wrapper in (("real", "cosine"), ("imag", "sine")):
        row.update({f"{axis}_{key}": signal[f"axis_{key}"][axis] for key in ("shots", "bias", "allowance")})
        row[f"{axis}_one_shot_rz"] = means[wrapper]["rz_count"]
    for metric in METRICS:
        value = None if not eligible else sum(shots[a]*means[w][metric] for a,w in (("real","cosine"),("imag","sine")))
        require(not eligible or (work[metric] is not None and near(value, work[metric])), "N x cost closure")
        # Retain saved point values after closure checking; do not let last-bit
        # reordering of floating arithmetic create a new winner or remove a tie.
        row[metric] = work[metric] if eligible else None
    row["effective_rz_cost"] = row["rz_count"]/n if eligible else None
    row.update(complex_bias(signal))
    return row


def same_R_groups(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["method"] in ("B2", "B3"):
            groups[(row["method"],row["rank"],row["K"],row["T"],row["R"])].append(row)
    answer = []
    for key, group in sorted(groups.items()):
        if len(group) < 2:
            continue
        spreads = {name: max(r[name] for r in group)-min(r[name] for r in group)
                   for name in ("tau", "normalization", "expected_random_applications")}
        require(all(near(group[0][name], r[name]) for r in group for name in spreads), "same R normalization/action mismatch")
        answer.append({"method": key[0], "rank": key[1], "K": key[2], "T": key[3], "R": key[4],
                       "spreads": spreads, "candidate_ids": [r["candidate_id"] for r in sorted(group,key=lambda x:x["q"])],
                       "interpretation": "SAME_TAU_NORMALIZATION_RANDOM_EXPECTATION_NOT_SAME_SIGNAL_OR_COMPILED_COST"})
    return answer


def function_excerpt(source, name):
    tree = ast.parse(source)
    matches = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    require(len(matches) == 1, "missing/ambiguous static source function")
    node = matches[0]
    return {"function": name, "line": node.lineno, "end_line": node.end_lineno,
            "source_excerpt": "\n".join(source.splitlines()[node.lineno-1:node.end_lineno])}


def csv_text(rows):
    fields = sorted(set().union(*(r.keys() for r in rows)))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: "MISSING" if v is None else v for k,v in row.items()})
    return stream.getvalue()


def endpoint_decomposition(rows, domain):
    eligible = [r for r in rows if r["accuracy_eligible"]]
    partial = minimum([r for r in eligible if r["method"] == "B2"])
    result = []
    for method in ("B0", "B1", "B3"):
        endpoint = minimum([r for r in eligible if r["method"] == method])
        n_ratio = partial["total_shots"] / endpoint["total_shots"]
        c_ratio = partial["effective_rz_cost"] / endpoint["effective_rz_cost"]
        g_ratio = partial["rz_count"] / endpoint["rz_count"]
        require(near(n_ratio*c_ratio, g_ratio), "ratio decomposition closure")
        result.append({"domain":domain,"partial_id":partial["candidate_id"],
                       "endpoint_id":endpoint["candidate_id"],"N_ratio":n_ratio,
                       "effective_one_shot_RZ_ratio":c_ratio,"G_RZ_ratio":g_ratio})
    return result


def analyze(data, input_audit):
    a, b, v, m2, s2 = (data[k] for k in ("m1a", "m1b", "validation", "m2", "s2"))
    require(len(a["signal_records"]) == len(a["candidate_ledger"]) == len(b["compile_map"]) == 210, "M1 cardinality")
    signals = {r["candidate_fingerprint"]: r for r in a["signal_records"]}
    require(len(signals) == 210 and set(signals) == {r["candidate_fingerprint"] for r in a["candidate_ledger"]}, "M1 ledger identity")
    rows = []
    for cell in b["compile_map"]:
        signal = signals[cell["candidate"]["candidate_fingerprint"]]
        require(cell["candidate"] == signal["candidate"], "candidate signal/cost mismatch")
        require(cell["signal_record_fingerprint"] == digest(signal), "signal record fingerprint mismatch")
        require(cell["accuracy_eligible"] == signal["accuracy_eligible"] and cell["axis_shots"] == signal["axis_shots"], "eligibility/shots changed")
        rows.append(row_from_saved(cell["candidate"],signal,
                    {axis: cell["compiled_axes"][axis]["cost"] for axis in ("cosine","sine")},
                    cell["matched_accuracy_compiled_work_no_state_preparation"],"M1"))
    require(len({r["candidate_fingerprint"] for r in rows}) == 210, "duplicate M1 compile cell")
    eligible = [r for r in rows if r["accuracy_eligible"]]
    require(len(eligible) == 206, "M1 eligible cardinality")
    by_id = {r["candidate_id"]:r for r in rows}
    transfer = []
    require({r["candidate_id"] for r in m2["candidate_results"]} == set(COMMON_FIVE) and len(m2["candidate_results"]) == 5, "M2 frozen domain")
    for cell in m2["candidate_results"]:
        require(cell["development_candidate_fingerprint"] == by_id[cell["candidate_id"]]["candidate_fingerprint"], "M2 development identity")
        candidate = cell["signal"]["candidate"]
        require(all(candidate[k] == by_id[cell["candidate_id"]][k] for k in ("method","rank","q","r","K","T","delta")), "M2 transfer reoptimized")
        transfer.append(row_from_saved(candidate,cell["signal"],cell["axis_one_shot_compiled_means"],cell["work_by_metric"],"M2"))
    selected_fp = {r["candidate_fingerprint"] for r in a["compile_selection"]["selected"]}
    # Old selector selected 16 random cells; deterministic/discard baseline cells
    # were separate. The historical validation comparator is these 16, not 32.
    selected = [r for r in eligible if r["candidate_fingerprint"] in selected_fp]
    require(len(selected) == len(selected_fp) == 16, "old selector domain")
    proxy_rows = [{"candidate_id":s["candidate_id"], "total_shots":s["total_shots"], "n_det":s["n_det"], "n_rand":s["n_rand"]}
                  for s in signals.values() if s["candidate"]["method"] in ("B2","B3") and s["accuracy_eligible"]]
    require(len(proxy_rows) == 194, "random cardinality")
    proxy_ids = {r["candidate_id"] for r in frontier(proxy_rows,("total_shots","n_det","n_rand"))}
    actual_ids = {r["candidate_id"] for r in frontier(eligible)}
    regrets = []
    for metric in METRICS:
        best, chosen = minimum(eligible,metric), minimum(selected,metric)
        regret = chosen[metric]/best[metric]-1
        require(near(regret,v["analysis"]["old_selector_actual_comparison"]["by_metric"][metric]["old_selector_regret_fraction"]), "saved regret crosscheck")
        regrets.append({"metric":metric,"all_min_id":best["candidate_id"],"all_min_work":best[metric],
                        "selected_min_id":chosen["candidate_id"],"selected_min_work":chosen[metric],"regret_fraction":regret})
    common = [by_id[name] for name in COMMON_FIVE]
    domains = {"M1_all":group_summary(rows), "M1_q8":group_summary([r for r in rows if r["q"] == 8]),
               "M1_old_selector16":group_summary(selected), "M1_common5":group_summary(common), "M2_common5":group_summary(transfer)}
    endpoint_ratios = [item for name, domain in (("M1_all",rows),("M1_q8",[r for r in rows if r["q"]==8]),
                                                ("M1_common5",common),("M2_common5",transfer))
                       for item in endpoint_decomposition(domain,name)]
    q8_by_rank = [minimum([r for r in eligible if r["q"] == 8 and r["method"] == method and r["rank"] == rank])
                  for method,rank in sorted({(r["method"],r["rank"]) for r in eligible if r["q"] == 8})]
    # Presence matrix at identical q/r/K; an unregistered cell is not ineligible.
    rank_matrix = []
    for method, ranks in (("B0",(3,4,5,6,9)),("B2",(3,6,9))):
        settings = sorted({(r["q"],r["r"],r["K"]) for r in rows if r["method"] == method})
        for q,r,K in settings:
            for rank in ranks:
                name = f"{method}-rank{rank}-q{q}-r{r}-K{K}"
                cell = by_id.get(name)
                rank_matrix.append({"method":method,"q":q,"r":r,"K":K,"rank":rank,"candidate_id":name,
                                    "status":"NOT_REGISTERED" if cell is None else "ELIGIBLE" if cell["accuracy_eligible"] else "INELIGIBLE",
                                    "rz_count":None if cell is None else cell["rz_count"]})
    boundary = [r["candidate_id"] for r in rows if r["r"] == 64]
    require(sorted(boundary) == sorted(["B3-rank0-q8-r64-K2","B2-rank3-q1-r64-K2"]), "r64 boundary domain")
    static = {
        "uncontrolled_basis_change":function_excerpt(data["block_source"],"_append_basis"),
        "controlled_diagonal_primitives":function_excerpt(data["block_source"],"_append_block"),
        "equal_block_step_boundary_fusion":function_excerpt(data["repeated_source"],"_build_boundary_optimized"),
        "discard_bias_label":function_excerpt(data["signal_source"],"_deterministic_signal_record"),
        "general_relative_gaussian_or_control_aware_optimality":"UNVERIFIED_NO_COST_OR_OPERATOR_TEST_ADDED",
        "literature_coverage":"NOT_REAUDITED_IN_PM0_NO_NEW_NOVELTY_CLAIM",
    }
    summary = {
        "schema_version":"pr2_track_a_pm0_posthoc_v1", "status":"POSTHOC_PM0_COMPLETE_NO_NEW_SCIENCE_STOP",
        "evidence_commit":EVIDENCE_COMMIT, "input_identity":input_audit,
        "scope":{"model":"H4 linear STO-3G DF rank12; 8 system qubits", "geometries_angstrom":[1.0,1.3],
                 "T":0.8,"complex_signal_error":0.05,"axis_error":0.05/math.sqrt(2),"axis_alpha":0.025,
                 "M1_splits":[0,3,6,9,12],"q":[1,2,4,8],"delta":[0.8,0.4,0.2,0.1],
                 "compiler":"Qiskit1.3.0 rz/sx/x/cx opt1 seed17 no coupling/backend",
                 "evidence_kind":"LOCAL_POSTHOC_SAVED_VALUES_NOT_NEW_CI_OR_VALIDATION",
                 "formal_confidence_interval_claimed":False,"epsilon_sweep_performed":False},
        "counts":{"m1_registered":210,"m1_eligible":206,"random":194,"m2_frozen":5,
                  "old_selector":16,"proxy_frontier":len(proxy_ids),"actual_frontier":len(actual_ids),
                  "actual_frontier_in_proxy":len(actual_ids & proxy_ids),
                  "actual_frontier_in_selector":len(actual_ids & {r['candidate_id'] for r in selected})},
        "common_domains":domains,
        "endpoint_N_cost_ratio_decomposition":endpoint_ratios,
        "q8_best_by_method_rank":[{k:r[k] for k in ("candidate_id","method","rank","rz_count","total_shots","effective_rz_cost")} for r in q8_by_rank],
        "selector_regret":regrets,
        "state_preparation_point_lower_envelopes":{
            "M1_all":lower_envelope(eligible),"M1_common5":lower_envelope(common),"M2_common5":lower_envelope(transfer)},
        "same_R_groups":same_R_groups(rows),
        "candidate_domain_audit":{"r64_only":boundary,"B0_rank4_rank5":"NOT_REGISTERED_IN_M1_OR_M2",
                                  "B3_q1_r256":"NOT_REGISTERED_NO_INFERENCE_OR_NEW_RUN",
                                  "rank_matrix_missing_cells":sum(r["status"]=="NOT_REGISTERED" for r in rank_matrix)},
        "old_S2_scope":{"q":s2["task"]["q"],"delta":s2["task"]["delta"],
                        "primary_partial_ranks":sorted({r["rank"] for r in s2["B2_candidates"]}),
                        "control_partial_ranks":sorted({r["rank"] for r in s2["rank3_rank9_controls"]}),
                        "decision_preserved":s2["decision"]["status"],
                        "interpretation":"DOMAIN_CHANGE_CONFOUNDS_ATTRIBUTION_TO_Q_ALONE"},
        "static_baseline_audit":static,
        "claim_audit":[
            {"claim":"registered-grid intermediate partial remains competitive at fixed q8 and free q", "status":"SUPPORTED_SCOPED_POINT_VALUES"},
            {"claim":"q optimization first reveals rank3 or reverses the best method", "status":"NOT_SUPPORTED_SAME_DOMAIN_Q8_ALREADY_B2_RANK3"},
            {"claim":"old selector loses the primary RZ optimum", "status":"NOT_SUPPORTED_RZ_REGRET_ZERO"},
            {"claim":"old selector loses complete six-metric Pareto membership", "status":"SUPPORTED_ONE_OF_TWO_LOST_PROXY_FRONTIER_RETAINS_BOTH"},
            {"claim":"M1 vs M2 P envelope change establishes geometry-induced method change", "status":"NOT_SUPPORTED_DOMAIN_CONFOUNDED_COMMON5_BOTH_REACH_B1"},
            {"claim":"B0 outer_pf_bias_abs is pure PF bias", "status":"NOT_SUPPORTED_DISCARD_PLUS_PF_TOTAL_BIAS"},
            {"claim":"five frozen configurations transfer on H4 1.30 Angstrom", "status":"EXISTING_M2_SUPPORTED_NOT_METHOD_OPTIMALITY"},
            {"claim":"chemical accuracy energy/RPE end-to-end or strong-baseline superiority", "status":"NOT_ESTABLISHED"}],
        "missing_evidence":["Exact truncated-H signal z_D: cannot separate B0 discard from PF bias.",
                            "B0 rank4/5 under matched task: absent from this M1/M2 comparison domain.",
                            "Operator/relative-phase and actual cost tests for any stronger synthesis policy.",
                            "General/high-order deterministic method comparison and energy/RPE accuracy connection.",
                            "Compiled RZ allocation by deterministic/random/basis/fixed primitive category: only full-wrapper aggregates and action proxies used; no allocation inferred."],
        "minimal_PM1_proposal":{"selected":"DEVELOPMENT_B0_RANK4_RANK5_NEARBY_DISCARD_FALSIFICATION",
             "reason":"Concrete registered-domain gap; pure-PF/discard label cannot determine the two missing ranks. No proven unused cost-reducing optimizer was established by static inspection.",
             "proposed_q":[1,2,4,8],"T":0.8,"max_new_signal_records":8,"max_full_wrappers":16,
             "random_trajectories":0,"held_out_access":False,"authorized":False,
             "alternative_if_no_broader_claim":"Close a registered-rank, fixed-S2-implementation note without this test.",
             "strong_synthesis_baseline":"Unresolved; audit and separately authorize one applicable policy before claiming superiority to strong deterministic implementations."},
        "access_audit":{"unique_saved_json_input_files":5,"unique_source_input_files":4,
                        "session_total_saved_file_reads_measured":False,
                        "embedded_artifact_paths_followed":0,
                        "npz_resolve_stat_hash_load":0,"runtime_cache_reads":0,"science_imports":0,
                        "signal_evaluations":0,"trajectories_sampled":0,"circuit_build_compile":0,"gpu_operations":0},
        "original_statuses_preserved":{"M1A":a["status"],"M1B1":b["status"],"M2":m2["status"]},
        "mandatory_stop":True,"pm1_authorized":False,"next_stage_authorized":False,
    }
    require(len(proxy_ids) == 64 and len(actual_ids) == 2, "frontier count crosscheck")
    summary["summary_fingerprint"] = digest(summary)
    return summary, rows+transfer, rank_matrix


def build_bundle(root):
    data, audit = load_inputs(root)
    summary, rows, rank_matrix = analyze(data,audit)
    files = {"summary.json":json.dumps(summary,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n",
             "candidate_decomposition.csv":csv_text(rows),"fixed_q_r_K_rank_presence.csv":csv_text(rank_matrix),
             "selector_metric_regret.csv":csv_text(summary["selector_regret"]),
             "endpoint_N_cost_ratios.csv":csv_text(summary["endpoint_N_cost_ratio_decomposition"]),
             "same_R_candidate_comparison.csv":csv_text([r for r in rows if r["stage"]=="M1" and
                 r["candidate_id"] in {name for g in summary["same_R_groups"] for name in g["candidate_ids"]}])}
    # Second byte check also catches input changes during a long analysis.
    _, after = load_inputs(root)
    require(after == audit,"input changed during PM0")
    return files
