#!/usr/bin/env python3
"""Frozen saved-value comparison: registered minima and an optional epsilon grid."""
import argparse
from collections import Counter
import csv
from fractions import Fraction as Q
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import types

ROOT=Path(__file__).resolve().parents[1]
pkg=types.ModuleType("b0_saved_analysis");pkg.__path__=[str(ROOT/"src/trotterlib")];sys.modules[pkg.__name__]=pkg
F=importlib.import_module(pkg.__name__+".h4_pauli_b0_comparison")
A=importlib.import_module(pkg.__name__+".h4_conditional_cng")
OUT=ROOT/"artifacts/non_df_h4_pauli_b0_comparison/2026-10-25"


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
def table(p,rows):
    with Path(p).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader()
        for row in rows:w.writerow({k:"null" if v is None else v for k,v in row.items()})


def compact_minimum(best):
    if best is None:return None
    s=best["selected"]
    return dict(candidate_ids=best["candidate_ids"],prefix=s["prefix"],q=s["q"],
                G_exact=best["G_exact"],G_guard_range_exact=best["independent_guard_range_exact"],
                selected_candidate_guard_range=s["G_guard_range_exact"],
                N_cos=s["N_cos"],N_sin=s["N_sin"],N_total=s["N_total"],
                selection_scope="FINITE_REGISTERED_SET; EACH_CANDIDATE_HAS_INDEPENDENT_RHO_GUARD")


def summary_row(epsilon,rows,PR):
    best=F.finite_minimum(rows)
    truncated=F.finite_minimum([r for r in rows if r["family"]=="TRUNCATED_B0_B0I"])
    full=F.finite_minimum([r for r in rows if r["family"]=="FULL_PF"])
    selected=compact_minimum(best)
    return dict(epsilon_sig=epsilon,best_deterministic=selected,
                best_truncated_B0_B0I=compact_minimum(truncated),best_full_PF=compact_minimum(full),
                PR_saved_or_reaccounted={k:PR[k] for k in ("N_cos","N_sin","N_total","G_exact","G_guard_range_exact")},
                comparison=F.compare_ranges(None if best is None else best["independent_guard_range_exact"],PR["G_guard_range_exact"]))


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--plan",type=Path,required=True);args=parser.parse_args()
    plan=read(args.plan)
    if not plan["execution_enabled"]:raise ValueError("ANALYSIS_DISABLED")
    if (OUT/"result_summary.json").exists():raise FileExistsError("COMPLETED_ANALYSIS_PRESERVED")
    for p,h in plan["input_sha256"].items():
        if sha(ROOT/p)!=h:raise ValueError("FROZEN_ANALYSIS_INPUT_CHANGED:"+p)
    source=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    code=("scripts/summarize_h4_pauli_b0_comparison.py","src/trotterlib/h4_pauli_b0_comparison.py",
          "src/trotterlib/h4_conditional_cng.py","scripts/resource_applicability/integrate_h4_b0i_manuscript.py")
    for p in code:
        if subprocess.check_output(["git","show",source+":"+p],cwd=ROOT)!=(ROOT/p).read_bytes():raise ValueError("ANALYSIS_SOURCE_NOT_COMMITTED")
    registry=[];all_rows=[];targets={};seen=set();batch_counts=[]
    for batch in plan["batches"]:
        r=read(OUT/batch/"run01/batch_results.json");batch_counts.append(dict(batch=batch,SOURCE=r["SOURCE"],**r["counts"]))
        for c in r["candidates"]:
            spec=c["candidate"];cid=spec["candidate_id"]
            if cid in seen:raise ValueError("DUPLICATE_REGISTERED_CANDIDATE")
            seen.add(cid);targets[cid]=read(ROOT/c["target_path"])
            registry.append(dict(candidate_id=cid,prefix=spec["prefix"],q=spec["q"],family=spec["family"],
                                 batch=batch,SOURCE=r["SOURCE"],plan_sha256=r["plan_sha256"],
                                 prices=c["prices"],metrics=c["metrics"],target_path=c["target_path"],
                                 bias_axes=targets[cid]["axes"],precision_bits=c["precision_bits"],
                                 target_reuse=targets[cid].get("reuse"),price_reuse=spec.get("price_reuse")))
            all_rows.extend(c["rows"])
    if len(registry)!=plan["registered_candidates"]:raise ValueError("ANALYSIS_REGISTRY_COUNT")
    PR_source=read(ROOT/plan["saved_PR_results"])
    PR_fixed={e:next(r for r in PR_source if r["case"]=="pauli_jw_PR" and r["epsilon_sig"]==e) for e in F.EPSILONS}
    main_rows=[summary_row(e,[r for r in all_rows if r["epsilon_sig"]==e],PR_fixed[e]) for e in F.EPSILONS]
    candidate_table=[]
    lookup={r["candidate_id"]:r for r in registry}
    for r in all_rows:
        c=lookup[r["candidate_id"]]
        candidate_table.append(dict(candidate_id=r["candidate_id"],batch=c["batch"],prefix=r["prefix"],q=r["q"],family=r["family"],epsilon_sig=r["epsilon_sig"],eligibility=r["eligibility"],RZ_cos=c["prices"]["cosine"],RZ_sin=c["prices"]["sine"],N_cos=r["N_cos"],N_sin=r["N_sin"],N_total=r["N_total"],G_exact=r["G_exact"],G_guard_lower=None if r["G_guard_range_exact"] is None else r["G_guard_range_exact"][0],G_guard_upper=None if r["G_guard_range_exact"] is None else r["G_guard_range_exact"][1],target_precision_bits=c["precision_bits"]))
    table(OUT/"candidate_precision_table.csv",candidate_table)
    save(OUT/"candidate_registry.json",dict(registry=registry,all_prefixes=sorted({r["prefix"] for r in registry}),q_values=[1,2,4,8],batch_counts=batch_counts))
    scale=A.scale_bounds(1,0)
    PR_axes=read(ROOT/plan["saved_PR_axes"])
    PR_target=dict(axes={a:{k:next(r for r in PR_axes if r["case"]=="pauli_jw_PR" and r["axis"]==a)[k] for k in ("signed_bias_exact","u_enc_exact")} for a in F.AXES})
    PR_scales=[r["scale"] for r in PR_axes if r["case"]=="pauli_jw_PR"]
    if any(s!=PR_scales[0] for s in PR_scales):raise ValueError("PR_SCALE_NOT_PRECISION_INDEPENDENT")
    statistics=read(ROOT/plan["axis_statistics"])
    PR_prices={a:statistics["PR"]["axes"][a]["mean_exact"] for a in F.AXES}
    grid_rows=[];grid_minima=[];reaccounted=0
    for eps in plan["epsilon_grid"]:
        if eps in F.EPSILONS:
            rows=[r for r in all_rows if r["epsilon_sig"]==eps];PR=PR_fixed[eps]
        else:
            rows=[]
            for c in registry:
                row=F.account(targets[c["candidate_id"]],c["prices"],eps,scale=scale)
                if row["N_total"] is not None and any(v is None for v in c["prices"].values()):raise ValueError("GRID_ELIGIBLE_CANDIDATE_MISSING_PRICE")
                row.update(candidate_id=c["candidate_id"],prefix=c["prefix"],q=c["q"],family=c["family"]);rows.append(row)
            PR=F.account(PR_target,PR_prices,eps,scale=PR_scales[0]);reaccounted+=len(registry)+1
        s=summary_row(eps,rows,PR);grid_minima.append(s)
        b=s["best_deterministic"];f=s["best_full_PF"];tr=s["best_truncated_B0_B0I"]
        grid_rows.append(dict(epsilon=eps,best_deterministic_ids=";".join(b["candidate_ids"]),best_prefix=b["prefix"],best_q=b["q"],best_G_exact=b["G_exact"],best_G_guard_upper=b["G_guard_range_exact"][1],best_truncated_G_exact=None if tr is None else tr["G_exact"],best_full_PF_G_exact=None if f is None else f["G_exact"],PR_G_exact=PR["G_exact"],PR_G_guard_upper=PR["G_guard_range_exact"][1],rho_zero_order=s["comparison"]["rho_zero_order"],independent_guard_order=s["comparison"]["independent_guard_order"],deterministic_over_PR_exact=s["comparison"]["ratio_exact"]))
    table(OUT/"precision_grid.csv",grid_rows)
    # Reuse the DF study's pure display helper; it reports grid brackets only.
    segments=A.extract("scripts/resource_applicability/integrate_h4_b0i_manuscript.py","grid_segments",{})
    regions=segments(grid_rows,lambda r:r["rho_zero_order"])
    guard_regions=segments(grid_rows,lambda r:r["independent_guard_order"])
    summary=dict(status="CONDITIONAL_FINITE_REGISTERED_COMPARISON_COMPLETE",analysis_SOURCE=source,
                 analysis_plan_sha256=sha(args.plan),candidate_count=len(registry),batch_counts=batch_counts,
                 fixed_precision_results=main_rows,eligibility_counts={e:dict(Counter(r["eligibility"] for r in all_rows if r["epsilon_sig"]==e)) for e in F.EPSILONS},
                 grid=dict(points=len(grid_rows),range=[grid_rows[0]["epsilon"],grid_rows[-1]["epsilon"]],winner_counts=dict(Counter(r["rho_zero_order"] for r in grid_rows)),winner_regions=regions,independent_guard_regions=guard_regions,new_saved_value_point_reaccountings=reaccounted,continuous_boundary_certified=False,independent_experiments=False),
                 historical_PF_q8_accounting=[{k:r[k] for k in ("epsilon_sig","N_total","G_exact","G_guard_range_exact")} for r in PR_source if r["case"]=="pauli_jw_PF"],
                 normalization_policy="New common unitary deterministic B=1/logB=0 outward scale; old PF log margin preserved as historical separate baseline; saved PR scale unchanged.",
                 actual_compiled_rho=None,unconditional_G=None,PR_population_cost_tail_guaranteed=False,
                 formal_certificate_accepted=False,global_prefix_optimum_proved=False,important_claim_formally_adopted=False)
    save(OUT/"result_summary.json",summary)
    table(OUT/"best_by_precision.csv",[dict(epsilon_sig=r["epsilon_sig"],best_deterministic_ids=";".join(r["best_deterministic"]["candidate_ids"]),best_prefix=r["best_deterministic"]["prefix"],best_q=r["best_deterministic"]["q"],best_G_exact=r["best_deterministic"]["G_exact"],best_truncated_ids=";".join(r["best_truncated_B0_B0I"]["candidate_ids"]),best_full_PF_ids=";".join(r["best_full_PF"]["candidate_ids"]),best_full_PF_G_exact=r["best_full_PF"]["G_exact"],PR_G_exact=r["PR_saved_or_reaccounted"]["G_exact"],rho_zero_order=r["comparison"]["rho_zero_order"],independent_guard_order=r["comparison"]["independent_guard_order"],ratio_exact=r["comparison"]["ratio_exact"]) for r in main_rows])
    save(OUT/"analysis_provenance.json",dict(SOURCE=source,code_sha256={p:sha(ROOT/p) for p in code},input_sha256=plan["input_sha256"],state_actions=0,circuit_builds=0,compiles=0,PR_sampling=0,fixed_five_PR_N_G_reused=True,grid_PR_saved_value_reaccounting=True,DF_function_reused="grid_segments"))
    print(json.dumps(dict(candidate_count=len(registry),main=[dict(epsilon=r["epsilon_sig"],best=r["best_deterministic"]["candidate_ids"],G=r["best_deterministic"]["G_exact"],comparison=r["comparison"]) for r in main_rows],grid_winners=summary["grid"]["winner_counts"]),ensure_ascii=False))


if __name__=="__main__":main()
