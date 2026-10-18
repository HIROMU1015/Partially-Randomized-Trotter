"""Check and publish saved B-2 output; never dispatch or recalculate shots.

Only the fixed run01 schema/scope is supported. Private scalar inputs are read
for exact baseline and missing-field comparisons, never decoded or published.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from fractions import Fraction as F
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
KEYS = ("system", "cell_id", "epsilon", "bias_kind", "allowance_factor")
GROUP = ("system", "epsilon", "bias_kind", "allowance_factor")
BASELINE_FIELDS = ("alpha_axis", "N_total", "axes", "evidence_kind",
                   "eligibility_under_declared_allowance")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    def unique(items):
        result = {}
        for k, v in items:
            if k in result:
                raise ValueError("DUPLICATE_JSON_KEY")
            result[k] = v
        return result
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def check(condition, name):
    if not condition:
        raise ValueError(name)


def key(row, fields=KEYS):
    return tuple(row[k] for k in fields)


def collect(run, plan_path, output):
    plan = read(plan_path)
    grant = read(run / "grant_B2_PRIVATE.json")
    approval = read(run / "approval_record_PRIVATE.json")
    gate = read(run / "prelaunch_gate.json")
    marker = read(run / "result01.oneshot")
    terminal = read(run / "result01/terminal.json")
    native = read(run / "result01/native_wait.json")
    launcher = read(run / "launcher_wait_receipt.json")
    raw = (run / "result01/result.json").read_bytes()
    result = read(run / "result01/result.json")
    input_path = Path(plan["input"]["path"])
    check(sha(input_path.read_bytes()) == plan["input"]["sha256"], "INPUT_HASH")
    inputs = read(input_path)["records"]
    check(grant["stage"] == approval["stage"] == marker["stage"] == "B2", "STAGE")
    check(grant["approved_by_user"] and grant["runtime_authorization"], "APPROVAL")
    check(grant["plan_sha256"] == marker["plan_sha256"] == sha(canonical(plan)), "PLAN")
    check(grant["approval_record_canonical_sha256"] == sha(canonical(approval)), "APPROVAL_BINDING")
    check(grant["input_sha256"] == terminal["input_sha256"] == plan["input"]["sha256"], "INPUT_BINDING")
    check(plan["stage"] == "B2" and plan["scope"]["alpha_total_grid"] == ["0.01", "0.05", "0.10"], "SCOPE")
    check(approval["alpha_total_grid"] == grant["approved_alpha_total_grid"] == plan["scope"]["alpha_total_grid"], "GRID_APPROVAL")
    check(marker["consumed"] and grant["one_shot"] and not grant["retry"]
          and not grant["auto_next_stage"], "ONE_SHOT")
    check(native["native_exit_code"] == launcher["native_exit_code"] == 0
          and native["wait_completed"] and launcher["wait_completed"]
          and launcher["dispatch_count"] == 1, "NATIVE_WAIT")
    check(terminal["status"] == "COMPLETE_WITH_INCONCLUSIVE_MANDATORY_STOP"
          and terminal["mandatory_stop"] and not terminal["retry"]
          and not terminal["auto_next_stage"], "TERMINAL")
    for item in plan["source_files"]:
        p = ROOT / item["path"]
        check(sha(p.read_bytes()) == item["sha256"], "SOURCE_HASH")
        original = subprocess.check_output(["git", "-C", str(ROOT), "show",
                                             plan["source_commit"] + ":" + item["path"]])
        check(p.read_bytes() == original, "SOURCE_COMMIT")
    preserved = []
    for i, item in enumerate(gate["preserved_files"]):
        p = Path(item["path"])
        check(p.stat().st_size == item["bytes"] and sha(p.read_bytes()) == item["sha256"], "PRESERVATION")
        public = item["category"] == "public_previous_alpha_evidence"
        preserved.append(dict(label=str(p.relative_to(ROOT)) if public else f"private_{i}_{p.name}",
                              sha256=item["sha256"], bytes=item["bytes"], category=item["category"]))
    rows = result["rows"]
    check(len(inputs) == 3720 and len(rows) == result["accounting_rows"] == 11160, "ROW_COUNTS")
    check({key(r) for r in inputs} == {tuple(k) for k in plan["scope"]["case_keys"]}, "INPUT_SCOPE")
    by_case = defaultdict(dict)
    by_group = defaultdict(list)
    for row in rows:
        alpha = row["alpha_total"]
        check(alpha in (.01, .05, .10) and row["alpha_real"] == row["alpha_imag"] == alpha / 2, "ALPHA_SPLIT")
        check(alpha not in by_case[key(row)], "DUPLICATE_CASE_ALPHA")
        by_case[key(row)][alpha] = row
        by_group[key(row, GROUP) + (alpha,)].append(row)
        check(row["population_winner_certified"] is False, "POINT_ONLY")
        if row["accounting"] is not None:
            a = row["accounting"]
            check(a["evidence_kind"] == "EMPIRICAL" and not a["certificate_verified_by_this_function"]
                  and not a["familywise_winner_certified"], "CONDITIONAL_EVIDENCE")
        if row["status"] == "ELIGIBLE":
            check(F(row["G_exact"]) > 0 and row["cost_uncertainty_is_alpha_evidence"] is False, "POSITIVE_COST")
            if row["method"] == "B2":
                check(row["cost_class"] == "PR_PAIRED_MEASURED" and row["paired_samples"] == 32
                      and row["uncertainty_scope"] == "PAIRED_SAMPLE_SE_NOT_CI", "PAIRED_N32")
        else:
            check(row["G_exact"] is None, "NO_ZERO_FILL")
    baseline_checks = missing = 0
    for record in inputs:
        triple = by_case[key(record)]
        check(set(triple) == {.01, .05, .10}, "COMPLETE_GRID")
        check(len({r["status"] for r in triple.values()}) == 1, "STATUS_GRID_INVARIANT")
        if record.get("missing_fields"):
            missing += 1
            for row in triple.values():
                check(row["status"] == "MISSING" and row["accounting"] is None
                      and row["missing_fields"] == record["missing_fields"], "RETAIN_MISSING")
        else:
            a, baseline = triple[.05]["accounting"], record["baseline_accounting"]
            check(all(canonical(a[k]) == canonical(baseline[k]) for k in BASELINE_FIELDS), "SAVED_BASELINE")
            baseline_checks += 1
        if triple[.05]["status"] == "ELIGIBLE":
            for axis in ("real", "imag"):
                check(triple[.01]["accounting"]["axes"][axis]["shots"] >= triple[.05]["accounting"]["axes"][axis]["shots"]
                      >= triple[.1]["accounting"]["axes"][axis]["shots"], "SHOT_MONOTONICITY")
            check(F(triple[.01]["G_exact"]) >= F(triple[.05]["G_exact"]) >= F(triple[.1]["G_exact"]), "COST_MONOTONICITY")
    check(baseline_checks == 3660 and missing == 60 and len(by_case) == 3720, "BASELINE_MISSING_COUNTS")
    check(result["accounting_evaluations"] == 10980 and result["binomial_axis_evaluations"] == 0, "CALL_BUDGET")
    comparisons = result["comparisons"]
    check(len(comparisons) == len(by_group) == len(result["switches"]) == 66, "COMPARISONS")
    for c in comparisons:
        group = by_group[key(c, GROUP) + (c["alpha_total"],)]
        for family in ("PR", "PF"):
            candidates = [r for r in group if (r["method"] == "B2") == (family == "PR")]
            unresolved = sorted(r["cell_id"] for r in candidates if r["status"] not in ("ELIGIBLE", "INELIGIBLE"))
            saved = c["families"][family]
            if unresolved:
                check(saved["status"] == "UNRESOLVED_FULL_DOMAIN" and saved["best"] is None
                      and sorted(saved["unresolved"]) == unresolved, "FULL_DOMAIN_MISSING")
            else:
                available = [r for r in candidates if r["G_exact"] is not None]
                check(bool(available), "EXPECTED_ELIGIBLE_FAMILY")
                best = min(available, key=lambda r: (F(r["G_exact"]), r["cell_id"]))
                check(saved["status"] == "CONDITIONAL_POINT_MINIMUM" and saved["best"] == best, "SAVED_POINT_MINIMUM")
        p, d = (c["families"][f]["best"] for f in ("PR", "PF"))
        if p and d:
            ratio = F(p["G_exact"]) / F(d["G_exact"])
            check(F(c["PR_over_PF_exact"]) == ratio and F(c["PR_reduction_percent_exact"]) == 100 * (1-ratio), "SAVED_RATIO")
            check(c["rank"] == ("PR_LOWER" if ratio < 1 else "PF_LOWER" if ratio > 1 else "TIE"), "SAVED_RANK")
        else:
            check(c["rank"] == "UNRESOLVED" and c["PR_over_PF_exact"] is None, "UNRESOLVED_RANK")
    comparison_map = {key(c, GROUP) + (c["alpha_total"],): c for c in comparisons}
    for s in result["switches"]:
        g = (s["system"], s["epsilon"], s["bias_kind"], 1.0)
        c, b = comparison_map[g + (s["alpha_total"],)], comparison_map[g + (.05,)]
        check(s["rank_changed"] == (c["rank"] != b["rank"]), "RANK_SWITCH")
        for f in ("PR", "PF"):
            check(s["winner_changed"][f] == ((c["families"][f]["best"] or {}).get("cell_id")
                  != (b["families"][f]["best"] or {}).get("cell_id")), "WINNER_SWITCH")
    nonbaseline = [c for c in comparisons if c["alpha_total"] != .05]
    resolved = [c for c in nonbaseline if c["rank"] != "UNRESOLVED"]
    max_deviation = max((abs(F(c["PR_reduction_percent_exact"]) - F(comparison_map[key(c, GROUP) + (.05,)]["PR_reduction_percent_exact"])), c)
                        for c in resolved)
    # Explicitly allow only public derived accounting fields in the full output.
    check(not any(t in raw for t in (b'"raw":', b'"corrected":', b'"reference":', b'"p_rational":', b'/home/')), "PRIVATE_RAW_IN_OUTPUT")
    summary = dict(status=terminal["status"], source_commit=plan["source_commit"],
        alpha_total_grid=[.01,.05,.1], allocation="EQUAL_TWO_AXES", input_rows=3720,
        complete_input_rows=3660, missing_input_rows=60, output_rows=11160,
        accounting_evaluations=10980, exact_baseline_reproductions=3660,
        status_counts=dict(Counter(r["status"] for r in rows)),
        eligible_cost_classes=dict(Counter(r["cost_class"] for r in rows if r["status"] == "ELIGIBLE")),
        comparisons=66, rank_counts=dict(Counter(c["rank"] for c in comparisons)),
        nonbaseline_resolved_rank_checks=len(resolved), nonbaseline_unresolved_rank_checks=len(nonbaseline)-len(resolved),
        rank_changes=sum(s["rank_changed"] for s in result["switches"]),
        family_winner_changes={f:sum(s["winner_changed"][f] for s in result["switches"]) for f in ("PR","PF")},
        nonbaseline_resolved_family_checks={f:sum(c["families"][f]["best"] is not None for c in nonbaseline) for f in ("PR","PF")},
        maximum_reduction_change_pp=float(max_deviation[0]), maximum_reduction_change_pp_exact=str(max_deviation[0]),
        maximum_change_condition={k:max_deviation[1][k] for k in GROUP + ("alpha_total",)},
        evidence_kind="EMPIRICAL", population_winner_certified=False,
        cost_SE_is_measurement_alpha_evidence=False, multi_signal_postselection_RPE_guarantee=False,
        scientific_dispatches=1, automatic_retry=False, mandatory_stop=True,
        new_sampling=0,new_signal=0,new_compile=0,new_fit=0,binomial_axis_evaluations=0,
        native_driver_exit=0,native_launcher_exit=0,driver_wall_seconds=terminal["wall_seconds"],
        launcher_wall_seconds=launcher["wall_seconds"], independent_review=False)
    check(not output.exists(), "PUBLIC_OUTPUT_EXISTS")
    output.mkdir(parents=True)
    def write(name, value):
        (output/name).write_bytes(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False).encode()+b"\n")
    (output/"result.json.gz").write_bytes(gzip.compress(raw,mtime=0))
    check(gzip.decompress((output/"result.json.gz").read_bytes()) == raw, "LOSSLESS_EXPORT")
    write("summary.json",summary)
    write("comparisons.json",comparisons)
    write("switches.json",result["switches"])
    table = []
    for c in sorted(comparisons,key=lambda c:key(c,GROUP)+(c["alpha_total"],)):
        row = {k:c[k] for k in GROUP+("alpha_total","rank","PR_over_PF_exact","PR_reduction_percent_exact")}
        row["PR_reduction_percent_display"] = float(F(c["PR_reduction_percent_exact"])) if c["PR_reduction_percent_exact"] else None
        for f in ("PR","PF"):
            family,best=c["families"][f],c["families"][f]["best"]
            row[f+"_status"]=family["status"]
            for field in ("cell_id","G_exact","cost_class","cost_SE"):
                row[f+"_"+field]=best.get(field) if best else None
            row[f+"_N_total"]=best["accounting"]["N_total"] if best else None
        table.append(row)
    with (output/"comparisons.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]),lineterminator="\n");writer.writeheader();writer.writerows(table)
    for name, obj in (("terminal.json",terminal),("native_wait.json",native),("launcher_wait_receipt.json",launcher)):
        write(name,obj)
    private_files = [plan_path,input_path,run/"approval_record_PRIVATE.json",run/"grant_B2_PRIVATE.json",
                     run/"prelaunch_gate.json",run/"result01.oneshot",run/"result01/result.json"]
    write("execution_binding.json",dict(stage="B2",source_commit=plan["source_commit"],
        source_files=plan["source_files"],runtime=plan["runtime"],
        plan_canonical_sha256=grant["plan_sha256"],input_sha256=grant["input_sha256"],
        preceding_published_commit=gate["published_prior_commit"],
        private_receipts=[dict(role=p.name,bytes=p.stat().st_size,sha256=sha(p.read_bytes())) for p in private_files],
        approval_scope="User approved immediately preceding B2 .01/.05/.10 one-shot proposal; original unapproved plan preserved",
        private_raw_or_grant_originals_published=False,one_shot_marker_consumed=True,mandatory_stop=True))
    write("preservation_audit.json",dict(status="PASS",files=preserved,total_files=len(preserved),
        used_source_files_unchanged=len(plan["source_files"]),old_B1_pack_plan_grant_marker_result_unchanged=True,
        old_STOP_preserved=True,other_worktrees_modified=False,unrelated_worktrees_frozen=False))
    write("verification.json",dict(status="PASS_SAVED_RESULT_CHECKS",baseline_reproductions=baseline_checks,
        retained_missing_input_rows=missing,scope_and_grid="PASS",positive_cost_and_monotonicity="PASS",
        saved_point_minima_ratios_switches="PASS",native_waits_and_marker="PASS",lossless_gzip="PASS",
        conditional_evidence_and_cost_classes="PASS",preserved_files=len(preserved),
        synthetic_62_and_CLI_8="REUSED_UNCHANGED_FROZEN_SOURCE_NOT_RERUN",
        shot_cost_or_binomial_function_calls_during_collection=0,independent_review=False))
    print(json.dumps(summary,ensure_ascii=False))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-run",type=Path,required=True)
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    collect(args.private_run,args.plan,args.out)
