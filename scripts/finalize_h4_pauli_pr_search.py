#!/usr/bin/env python3
"""Five-epsilon comparison from frozen independent-confirmation results."""
import argparse
from collections import Counter
import csv
from fractions import Fraction as Q
import importlib.util
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("pr_final_saved_summary",ROOT/"scripts/summarize_h4_pauli_pr_search.py")
T=importlib.util.module_from_spec(spec);spec.loader.exec_module(T)
R=T.R;OUT=R.OUT


def table(path,rows):
    with path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader()
        for r in rows:writer.writerow({k:"null" if v is None else v for k,v in r.items()})


def run(snapshot):
    if (OUT/"result_summary.json").exists():raise FileExistsError("FINAL_RESULT_ALREADY_SAVED")
    source=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    for code in ("scripts/finalize_h4_pauli_pr_search.py","scripts/summarize_h4_pauli_pr_search.py",*R.CODE):
        if subprocess.check_output(["git","show",source+":"+code],cwd=ROOT)!=(ROOT/code).read_bytes():raise ValueError("COMMIT_FINAL_ANALYSIS_SOURCE_FIRST")
    inputs=[OUT/snapshot/"summary.json",OUT/snapshot/"precision_rows.json",OUT/snapshot/"candidate_catalog.json",OUT/"confirmation_nominations.json",ROOT/"artifacts/non_df_h4_central_conditional_cng/2026-10-24/recovery02/run01/conditional_results.json"]
    R.save(OUT/"final_analysis_inputs.json",dict(SOURCE=source,input_sha256={str(p.relative_to(ROOT)):R.sha(p) for p in inputs}))
    summary,rows,catalog,nominations,legacy=map(R.read,inputs)
    outputs=[];csv_rows=[]
    for eps in R.F.EPSILONS:
        nominees=nominations["by_epsilon"][eps]
        selected=next(r for r in rows if r["candidate_id"]==nominees[0] and r["cohort"]=="confirmation" and r["epsilon_sig"]==eps)
        confirmed=sorted([r for r in rows if r["cohort"]=="confirmation" and r["epsilon_sig"]==eps and r["G_exact"] is not None],key=lambda r:(Q(r["G_exact"]),r["candidate_id"]))
        historic=next(r for r in legacy if r["case"]=="pauli_jw_PR" and r["epsilon_sig"]==eps)
        common=next(r for r in rows if r["candidate_id"]=="p027_q8_r4_K4" and r["cohort"]=="historical_baseline" and r["epsilon_sig"]==eps)
        deterministic=next(x["saved_deterministic"] for x in summary["minima"] if x["epsilon_sig"]==eps)
        det=deterministic["best_deterministic"];full=deterministic["best_full_PF"];b0=deterministic["best_truncated_B0_B0I"]
        G=Q(selected["G_exact"]);D=Q(det["G_exact"]);H=Q(historic["G_exact"]);C=Q(common["G_exact"])
        ratio=G/D;savings=100*(1-ratio)
        comparison=R.F.compare_ranges(det["G_guard_range_exact"],selected["G_guard_range_exact"])
        output=dict(epsilon_sig=eps,primary_exploration_nominee_confirmed=selected,
            confirmation_minimum_post_selection=confirmed[0],confirmation_top2_post_selection=confirmed[:2],
            deterministic_union_best=det,best_truncated_B0_B0I=b0,best_full_PF=full,
            fixed_PR_legacy_quoted={k:historic[k] for k in ("N_cos","N_sin","N_total","G_exact","G_guard_range_exact")},fixed_PR_common_policy=common,
            PR_over_best_deterministic_exact=str(ratio),PR_savings_percent_exact=str(savings),
            PR_over_fixed_legacy_exact=str(G/H),PR_over_fixed_common_policy_exact=str(G/C),
            rho_guard_comparison_holding_cohort_mean_prices_fixed=comparison,
            conditional_point_order="PR_LOWER" if G<D else "DETERMINISTIC_LOWER" if G>D else "TIE",
            actual_rho=None,unconditional_G=None,population_winner_proved=False)
        outputs.append(output)
        p=selected["parameters"];g=selected["G_statistics"]
        csv_rows.append(dict(epsilon_sig=eps,selected_PR=selected["candidate_id"],**p,confirmation_n=selected["cohort_n"],
            mean_C_cos_exact=selected["mean_RZ_exact"]["cosine"],mean_C_sin_exact=selected["mean_RZ_exact"]["sine"],
            B=selected["B_saved_binary64"],N_cos=selected["N_cos"],N_sin=selected["N_sin"],N_total=selected["N_total"],
            PR_G_exact=selected["G_exact"],PR_G_SE=g["standard_error"],PR_G_nominal_t95_halfwidth=g["nominal95_halfwidth"],
            B0_prefix=b0["prefix"],B0_q=b0["q"],B0_G_exact=b0["G_exact"],full_PF_q=full["q"],full_PF_G_exact=full["G_exact"],
            fixed_PR_legacy_G_exact=historic["G_exact"],fixed_PR_common_G_exact=common["G_exact"],
            PR_over_best_deterministic_exact=str(ratio),PR_savings_percent=float(savings),
            confirmation_post_selection_min=confirmed[0]["candidate_id"],guard_order=comparison["independent_guard_order"]))
    counts=Counter()
    for stage in summary["stage_records"]:counts.update(stage["counts"])
    scope=R.read(OUT/"active_scope.json")["call_caps"]
    assert counts["candidates"]<=scope["total_registered_candidates"]
    assert counts["mean_state_actions"]<=scope["Arb_mean_state_actions_max"]
    assert counts["trajectory_samples"]<=scope["total_new_paired_trajectories_max"]
    assert counts["top_level_compiles"]<=scope["new_top_level_compiles_max"]
    event_counts={cohort:Counter() for cohort in ("exploration","confirmation")};sample_counts=Counter()
    for path in OUT.glob("*/run01/prices/*/*/*/trajectory.json"):
        row=R.read(path);event_counts[row["cohort"]].update(row["order_counts"]);sample_counts[row["cohort"]]+=1
    native=[]
    for path in OUT.glob("*/run01/STOP.json"):
        stop=R.read(path);parent=R.read(path.parent.parent/"native_parent_exit.json")
        if parent["native_exit"]!=0 or not parent["reaped"] or not stop["all_children_reaped"] or any(stop["native_child_exits"]):raise ValueError("NATIVE_COMPLETION")
        native.append(dict(stage=path.parent.parent.name,child_count=stop["children"],parent_native_exit=0,children_native_exit=0,all_reaped=True,elapsed_seconds=stop["elapsed_seconds"]))
    result=dict(status="CONDITIONAL_EMPIRICAL_PR_PARAMETER_COMPARISON_COMPLETE",final_analysis_SOURCE=source,
        parameters_registered=len(catalog),science_counts=dict(counts),new_trajectory_counts=dict(sample_counts),
        observed_event_orders={k:dict(v) for k,v in event_counts.items()},precision_bits_counts=dict(Counter(str(c["precision_bits"]) for c in catalog)),
        initial_candidates=100,adaptive_candidates=len(catalog)-100,adaptive_rounds=2,confirmation_nominations=nominations,
        epsilon_results=outputs,native_completion=native,
        eligibility_counts={eps:dict(Counter(next(r["eligibility"] for r in rows if r["candidate_id"]==c["candidate_id"] and r["epsilon_sig"]==eps) for c in catalog)) for eps in R.F.EPSILONS},
        actual_rho=None,formal_certificate_accepted=False,PR_population_cost_tail_guaranteed=False,
        global_parameter_optimum_proved=False,important_claim_formally_adopted=False,
        precision_grid_newly_evaluated=False,
        study_scope="Fixed H4 1.00A/STO-3G Pauli184; S2/T.8; q1,2,4,8/r2,4,8/K2,4; central control; Qiskit1.3.0 opt1 seed17; stateprep excluded",
        inference="Primary uses frozen exploration nominee on independent confirmation. Confirmation winner is separately labelled post-selection. All cost/CI are finite-sample conditional estimates; no law-tail, actual-rho or global-optimum guarantee.")
    R.save(OUT/"result_summary.json",result);table(OUT/"best_by_precision.csv",csv_rows)
    table(OUT/"candidate_precision_table.csv",[dict(candidate_id=r["candidate_id"],**r["parameters"],epsilon_sig=r["epsilon_sig"],cohort=r["cohort"],n=r["cohort_n"],eligibility=r["eligibility"],N_cos=r["N_cos"],N_sin=r["N_sin"],N_total=r["N_total"],G_exact=r["G_exact"],G_SE=None if r["G_statistics"] is None else r["G_statistics"]["standard_error"],C_cos=r["mean_RZ_exact"]["cosine"],C_sin=r["mean_RZ_exact"]["sine"],B=r["B_saved_binary64"]) for r in rows])
    R.save(OUT/"candidate_catalog.json",catalog)
    print(__import__("json").dumps(dict(candidates=len(catalog),counts=dict(counts),results=csv_rows),ensure_ascii=False))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--snapshot",required=True);args=p.parse_args();run(args.snapshot)
