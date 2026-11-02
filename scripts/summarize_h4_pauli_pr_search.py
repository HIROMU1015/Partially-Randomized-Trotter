#!/usr/bin/env python3
"""Saved-value PR comparison and finite selection, with separate cohorts."""
import argparse
from collections import Counter
from fractions import Fraction as Q
import importlib.util
import math
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("pr_summary_reused_runner",ROOT/"scripts/run_h4_pauli_pr_search.py")
R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)
S,A,F,E=R.S,R.A,R.F,R.E
OUT=R.OUT


def statistics():
    import numpy as np
    return A.extract("src/trotterlib/fixed_signal_evaluation.py","_statistics",dict(np=np,math=math,Fraction=Q))


def registry():
    candidates={};trajectories={};stage_records=[]
    for p in sorted(OUT.glob("*/run01/stage_results.json")):
        record=R.read(p);stage_records.append(dict(path=str(p.relative_to(ROOT)),sha256=R.sha(p),SOURCE=record["SOURCE"],counts=record["counts"]))
        for c in record.get("candidates",[]):
            cid=R.spec_id(c["candidate"])
            if cid in candidates:raise ValueError("DUPLICATE_CANDIDATE_REGISTRATION")
            candidates[cid]=c
        for path in record.get("trajectory_paths",[]):
            row=R.read(ROOT/path);cid=row["candidate_id"];cohort=row["cohort"]
            key=(cid,cohort,row["index"])
            if key in trajectories:raise ValueError("DUPLICATE_TRAJECTORY_ADDRESS")
            trajectories[key]=dict(row,path=path)
    return candidates,list(trajectories.values()),stage_records


def ranked(rows,epsilon,cohort="exploration"):
    return sorted([r for r in rows if r["epsilon_sig"]==epsilon and r["cohort"] in (cohort,"historical_baseline") and r["G_exact"] is not None],key=lambda r:(Q(r["G_exact"]),r["candidate_id"]))


def proposals(rows,registered,limit=32):
    """Pre-agreed one-coordinate neighborhoods and epsilon round-robin."""
    queues={};seen=set(registered);out=[]
    for eps in F.EPSILONS:
        queue=[]
        for parent in ranked(rows,eps)[:2]:
            c=parent["parameters"]
            changes=[]
            for delta in (-1,1,-4,4,-8,8):
                p=c["L_D"]+delta
                if 0<=p<=183:changes.append(dict(c,L_D=p))
            for field,domain in (("q",[1,2,4,8]),("r",[2,4,8]),("K",[2,4])):
                index=domain.index(c[field])
                for j in (index-1,index+1):
                    if 0<=j<len(domain):changes.append(dict(c,**{field:domain[j]}))
            for new in changes:
                cid=R.spec_id(new)
                queue.append((Q(parent["G_exact"]),abs(new["L_D"]-c["L_D"]),cid,new,parent["candidate_id"]))
        queues[eps]=sorted(queue,key=lambda x:x[:3])
    while len(out)<limit:
        progress=False
        for eps in F.EPSILONS:
            while queues[eps] and queues[eps][0][2] in seen:queues[eps].pop(0)
            if not queues[eps]:continue
            _,_,cid,c,parent=queues[eps].pop(0);seen.add(cid)
            out.append(dict(candidate_id=cid,**c,adaptive_parent=parent,target_epsilon=eps,initial_block="ADAPTIVE_POST_SELECTION"));progress=True
            if len(out)==limit:break
        if not progress:break
    return out


def summarize(name):
    destination=OUT/name
    if destination.exists():raise FileExistsError("SAVED_ANALYSIS_PRESERVED")
    source=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    for code in ("scripts/summarize_h4_pauli_pr_search.py",*R.CODE):
        if subprocess.check_output(["git","show",source+":"+code],cwd=ROOT)!=(ROOT/code).read_bytes():raise ValueError("COMMIT_ANALYSIS_SOURCE_FIRST")
    candidates,trajectories,stages=registry();destination.mkdir()
    # Freeze the actual scientific saved inputs before evaluating new statistics.
    paths=[ROOT/R.MODEL,ROOT/R.REFERENCE,ROOT/R.OLD_PR,
           ROOT/"artifacts/non_df_h4_pauli_b0_comparison/2026-10-25/result_summary.json"]
    for c in candidates.values():paths.extend([ROOT/c["law_path"],ROOT/c["target_path"]])
    paths.extend(ROOT/t["path"] for t in trajectories)
    R.save(destination/"analysis_inputs.json",dict(SOURCE=source,input_sha256={str(p.relative_to(ROOT)):R.sha(p) for p in paths},stage_records=stages))
    stats=statistics();rows=[];catalog=[]
    old=[t for t in R.read(ROOT/R.OLD_PR)["rows"] if t["case"]=="PR"]
    fixed="p027_q8_r4_K4"
    for cid,c in sorted(candidates.items()):
        d=R.read(ROOT/c["law_path"]);target=R.read(ROOT/c["target_path"])
        params={k:c["candidate"][k] for k in ("L_D","q","r","K")}
        cohorts={cohort:[t for t in trajectories if t["candidate_id"]==cid and t["cohort"]==cohort] for cohort in ("exploration","confirmation")}
        if cid==fixed:
            cohorts["historical_baseline"]=[dict(RZ={a:t[a]["RZ"] for a in F.AXES},order_counts=t["order_counts"]) for t in old]
        cost_records={}
        for cohort,samples in cohorts.items():
            if not samples:continue
            prices=[t["RZ"] for t in samples]
            costs=A.cost_summary(prices,stats)
            order_counts=Counter()
            for t in samples:order_counts.update(t["order_counts"])
            costs["observed_event_orders"]=dict(order_counts);cost_records[cohort]=costs
            mu={a:costs["axes"][a]["mean_exact"] for a in F.AXES}
            for eps in F.EPSILONS:
                row=S.account(target,mu,eps,d)
                gstats=None if row["N_total"] is None else A.paired(prices,dict(zip(F.AXES,(row["N_cos"],row["N_sin"]))),stats)
                if gstats is not None and Q(gstats["mean_exact"])!=Q(row["G_exact"]):raise ValueError("PAIRED_G_MEAN")
                row.update(candidate_id=cid,parameters=params,cohort=cohort,cohort_n=len(samples),G_statistics=gstats,
                    mean_RZ_exact=mu,B_saved_binary64=d["B_saved_binary64"],eligibility=F.joint_status(target,eps))
                rows.append(row)
        if not cost_records:
            for eps in F.EPSILONS:
                row=S.account(target,{a:None for a in F.AXES},eps,d)
                row.update(candidate_id=cid,parameters=params,cohort="UNMEASURED",cohort_n=0,G_statistics=None,mean_RZ_exact={a:None for a in F.AXES},B_saved_binary64=d["B_saved_binary64"],eligibility=F.joint_status(target,eps));rows.append(row)
        catalog.append(dict(candidate_id=cid,parameters=params,registration=c["candidate"],law_path=c["law_path"],target_path=c["target_path"],B=d["B_saved_binary64"],lambda_r=d["lambda_r"],bias_axes=target["axes"],precision_bits=target["bits"],state_actions=c["state_actions"],order_probabilities=d["order_probabilities"],effective_order_probabilities=d["order_mapping"]["exact_masses"],costs=cost_records))
    deterministic=R.read(paths[3])["fixed_precision_results"]
    minima=[]
    for eps in F.EPSILONS:
        top=ranked(rows,eps);confirmation=ranked([r for r in rows if r["cohort"]=="confirmation"],eps,"confirmation")
        det=next(x for x in deterministic if x["epsilon_sig"]==eps)
        minima.append(dict(epsilon_sig=eps,exploration_top2=top[:2],confirmation_top2=confirmation[:2],saved_deterministic=det))
    summary=dict(status="CONDITIONAL_EMPIRICAL_SAVED_VALUE_SNAPSHOT",SOURCE=source,
        candidates=len(candidates),new_trajectories=len(trajectories),stage_records=stages,
        minima=minima,proposed_next32=proposals(rows,candidates),
        actual_rho=None,formal_certificate_accepted=False,PR_population_cost_tail_guaranteed=False,
        selection_not_global_optimum=True,cohorts_not_pooled=True)
    R.save(destination/"candidate_catalog.json",catalog);R.save(destination/"precision_rows.json",rows)
    R.save(destination/"summary.json",summary)
    print(__import__("json").dumps(dict(snapshot=name,candidates=len(candidates),trajectories=len(trajectories),best=[dict(epsilon=x["epsilon_sig"],PR=x["exploration_top2"][0]["candidate_id"],G=x["exploration_top2"][0]["G_exact"],G_SE=x["exploration_top2"][0]["G_statistics"]["standard_error"]) for x in minima]),ensure_ascii=False))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--snapshot",required=True);args=p.parse_args();summarize(args.snapshot)
