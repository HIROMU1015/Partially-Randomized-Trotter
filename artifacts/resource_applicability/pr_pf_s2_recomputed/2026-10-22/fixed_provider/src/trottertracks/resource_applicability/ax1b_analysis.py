"""Future authorized saved-value orchestration. Preparation never calls this."""
from __future__ import annotations

import csv
import io
import math

from .ax1b_contract import AXES, METRICS, CASE_CONDITIONAL, canonical, digest, operational_na, require
from .ax1b_data import folds
from .ax1b_models import fit_cost,structural_na,finite_normalization,complexity_gate
from .ax1b_evaluation import cost_error,error_summary,rank_index,reference_shots,conditional_work,selection,paired_statistics,paired_total,common_support_selection,selection_status_summary

CALIBRATED=("PRED_BASE_SINGLE_COEFF","PRED_BASE_FEW_PARAM")


def predict_row(model,row):
    if row.dataset=="DIAG_PM1_8" and row.features.n_fixed is None:
        return dict(C_pred=None,status="INPUT_MISSING",missing_reason="N_A_PM1_PREDICTION_ANCHOR_INVARIANT_UNPROVEN",extrapolation_flags=[])
    return model.predict(row.features)


def analyze(rows,plan,values,allowlist):
    """No file IO or science generation; all inputs already authorized/projected."""
    training=[r for r in rows if r.dataset=="TRAIN_M1_210"]
    pm1=[r for r in rows if r.dataset=="DIAG_PM1_8"]
    m2=[r for r in rows if r.dataset=="DIAG_M2_5"]
    require((len(training),len(pm1),len(m2))==(210,8,5),"INPUT_IDENTITY","registered dataset sizes")
    predictions=[];fits=[];errors=[];normalization=[];statistics=[];selections=[];rank_diagnostics=[]
    stats_by_fp={}
    for r in rows:
        stats_by_fp[r.fingerprint]={}
        for metric in METRICS:
            left,right=r.pairs_by_metric[metric]
            stat=paired_statistics(left,right,r.candidate["method"] in {"B2","B3"})
            for axis in AXES:
                reference=r.reference_costs[axis][metric];mean=stat["mean_"+axis]
                require(abs(mean-reference)<=1e-9+1e-10*abs(reference),"PAIR_IDENTITY","saved cost mean differs from paired samples")
            stats_by_fp[r.fingerprint][metric]=stat
            statistics.append(dict(candidate_fingerprint=r.fingerprint,metric=metric,statistics=stat))
        if r.finite_input:
            s=r.finite_input;c=r.candidate
            record=finite_normalization(s["exact_rte_lambda_r"],c["T"],c["q"],c["r"],c["K"])
            require(abs(record["B"]-r.B)<=1e-12+1e-10*abs(r.B),"SCHEMA","finite B reproduction mismatch")
            require(abs(record["log_B"]-s["normalization_log"])<=1e-10,"SCHEMA","finite log B mismatch")
            require(record["log_bound_slack"]>=-1e-10,"SCHEMA","paper bound violation; inspect source/input rounding")
            normalization.append(dict(candidate_fingerprint=r.fingerprint,**record))
    partitions=[dict(fold_id="full210",diagnostic_family="full210",train=training,test=training+pm1+m2,diagnostic_kind="IN_SAMPLE_PLUS_OBSERVED_DIAGNOSTICS")]+folds(training)
    for partition in partitions:
        train,test=partition["train"],partition["test"]
        for model_id in CALIBRATED:
            model={axis:fit_cost(model_id,[r.features for r in train],[r.reference_costs[axis]["rz_count"] for r in train],[r.fingerprint for r in train]) for axis in AXES}
            fits.append(dict(fold_id=partition["fold_id"],model_id=model_id,axes={a:m.record() for a,m in model.items()}))
            for r in test:
                kind=partition["diagnostic_kind"]
                if partition["fold_id"]=="full210":
                    kind={"TRAIN_M1_210":"IN_SAMPLE_DEVELOPMENT","DIAG_PM1_8":"OBSERVED_PREFIX_DIAGNOSTIC","DIAG_M2_5":"OBSERVED_GEOMETRY_DIAGNOSTIC"}[r.dataset]
                predicted={a:predict_row(model[a],r) for a in AXES}
                pred=dict(schema_version="track_a_ax1b_prediction_v1",dataset_id=r.dataset,candidate_fingerprint=r.fingerprint,
                          model_id=model_id,model_version=1,fold_id=partition["fold_id"],diagnostic_family=partition["diagnostic_family"],diagnostic_kind=kind,
                          evaluation_case_id=CASE_CONDITIONAL,prediction_information_class="I1_FEATURES_H4_TRAINING_CALIBRATION",oracle_flags={"cost_features":False,"conditional_shots":True},
                          unit="native_rz_count_per_shot",scope_identity=r.candidate["compiler_identity"],
                          training_membership_sha256=digest(sorted(r.fingerprint for r in train)),feature_provenance=r.feature_provenance,
                          C_pred_by_axis={a:predicted[a]["C_pred"] for a in AXES},axis_prediction_details=predicted,B_pred=r.B,
                          G_conditional_oracle=None,G_conditional_oracle_status="EPSILON_DEPENDENT_VALUES_IN_SELECTION_TABLE",
                          cost_prediction_availability="DEFINED" if all(predicted[a]["C_pred"] is not None for a in AXES) else "INPUT_OR_FIT_UNAVAILABLE",**operational_na())
                predictions.append(pred)
                for a in AXES:
                    error=cost_error(r.reference_costs[a]["rz_count"],pred["C_pred_by_axis"][a])
                    errors.append(dict(model_id=model_id,fold_id=partition["fold_id"],diagnostic_kind=kind,axis=a,method=r.candidate["method"],prefix=r.candidate["rank"],
                         q=r.candidate["q"],R=r.candidate["q"]*r.candidate["r"],random_K=r.candidate["K"] if r.candidate["method"] in {"B2","B3"} else None,
                         candidate_fingerprint=r.fingerprint,extrapolation_flags=predicted[a]["extrapolation_flags"],prediction_missing_reason=predicted[a]["missing_reason"],**error))
    summaries=[]
    for model in CALIBRATED:
        for fold_id in sorted({e["fold_id"] for e in errors}):
            cells=[e for e in errors if e["model_id"]==model and e["fold_id"]==fold_id]
            for group in ["axis","method","prefix","q","R","random_K","diagnostic_kind"]:
                for key in sorted({canonical(e[group]) for e in cells}):
                    subset=[e for e in cells if canonical(e[group])==key]
                    summaries.append(dict(model_id=model,fold_id=fold_id,group=group,group_value=key,**error_summary(subset)))
    primary={}
    for model in CALIBRATED:
        primary[model]={}
        for q in [1,2,4,8]:
            cells=[e for e in errors if e["model_id"]==model and e["fold_id"]==f"leave_one_q_out:{q}"]
            summary=error_summary(cells)
            primary[model][q]=dict(score=summary["mean_absolute_relative"],full_coverage=summary["coverage"]==1 and bool(cells),support_hash=digest(sorted((e["candidate_fingerprint"],e["axis"]) for e in cells if e["absolute_rz"] is not None)))
        pooled=[e for e in errors if e["model_id"]==model and e["fold_id"].startswith("leave_one_q_out:")]
        summaries.append(dict(model_id=model,fold_id="POOLED_OUT_OF_FOLD_COST",diagnostic_kind="CROSS_FITTED_INTERNAL_GROUP",single_frozen_model=False,**error_summary(pooled)))
    adopted=complexity_gate(primary[CALIBRATED[0]],primary[CALIBRATED[1]])
    by_fp={r.fingerprint:r for r in rows}
    for epsilon in plan["metrics"]["epsilon_anchors"]:
        refs={r.fingerprint:dict(candidate=r.candidate,**reference_shots(r.reference_bias,r.B,epsilon)) for r in rows}
        for fp,ref in refs.items():
            r=by_fp[fp]
            ref["G_ref"]=conditional_work(ref.get("axis_shots"),{a:r.reference_costs[a]["rz_count"] for a in AXES}) if ref["eligible_ref"] else None
        for p in predictions:
            fp=p["candidate_fingerprint"];ref=refs[fp]
            selections.append(dict(row_kind="candidate_conditional_oracle_work",epsilon=epsilon,model_id=p["model_id"],fold_id=p["fold_id"],
                 diagnostic_kind=p["diagnostic_kind"],candidate_fingerprint=fp,evaluation_case_id=CASE_CONDITIONAL,
                 G_conditional_oracle=conditional_work(ref.get("axis_shots"),p["C_pred_by_axis"]),G_ref=ref["G_ref"],
                 reference_axis_shots=ref.get("axis_shots"),reference_eligible=ref["eligible_ref"],information_class="I4_CONDITIONAL_ORACLE"))
        for model_id in CALIBRATED:
            for partition in partitions:
                for group in ["DEVELOPMENT","M2_GEOMETRY"] if partition["fold_id"]=="full210" else ["FOLD_TEST"]:
                    test=partition["test"]
                    test=[r for r in test if (r.dataset=="DIAG_M2_5")== (group=="M2_GEOMETRY")] if partition["fold_id"]=="full210" else test
                    fps={r.fingerprint for r in test}
                    support={fp:refs[fp] for fp in fps}
                    pp=[p for p in predictions if p["model_id"]==model_id and p["fold_id"]==partition["fold_id"] and p["candidate_fingerprint"] in fps]
                    g={p["candidate_fingerprint"]:conditional_work(refs[p["candidate_fingerprint"]].get("axis_shots"),p["C_pred_by_axis"]) for p in pp}
                    selections.append(dict(model_id=model_id,fold_id=partition["fold_id"],epsilon=epsilon,dataset_scope=group,**selection(support,g,"FOLD_SPECIFIC_INTERNAL_GROUP" if group=="FOLD_TEST" else "IN_SAMPLE_PLUS_OBSERVED_PREFIX" if group=="DEVELOPMENT" else "OBSERVED_GEOMETRY")))
            pp=[p for p in predictions if p["model_id"]==model_id and p["fold_id"].startswith("leave_one_q_out:")]
            g={p["candidate_fingerprint"]:conditional_work(refs[p["candidate_fingerprint"]].get("axis_shots"),p["C_pred_by_axis"]) for p in pp}
            selections.append(dict(model_id=model_id,fold_id="POOLED_Q_CROSS_FITTED",epsilon=epsilon,
                  metric_name="regret_conditional_oracle_internal_group_210_cross_fitted",single_frozen_model=False,
                  **selection({r.fingerprint:refs[r.fingerprint] for r in training},g,"CROSS_FITTED_INTERNAL_GROUP")))
        for r in rows:
            ref=refs[r.fingerprint]
            if ref["eligible_ref"]:
                statistics.append(dict(candidate_fingerprint=r.fingerprint,epsilon=epsilon,reference_RZ_total=paired_total(stats_by_fp[r.fingerprint]["rz_count"],ref["axis_shots"]["real"],ref["axis_shots"]["imag"])))
        for part in partitions+[dict(fold_id="POOLED_Q_CROSS_FITTED",test=training)]:
            for group in ["DEVELOPMENT","M2_GEOMETRY"] if part["fold_id"]=="full210" else ["FOLD_TEST"]:
                test=part["test"]
                if part["fold_id"]=="full210":
                    test=[r for r in test if (r.dataset=="DIAG_M2_5")==(group=="M2_GEOMETRY")]
                support_refs={r.fingerprint:refs[r.fingerprint] for r in test}
                model_G={};cost_support={}
                for model in CALIBRATED:
                    pp=[p for p in predictions if p["model_id"]==model and p["candidate_fingerprint"] in support_refs and
                        (p["fold_id"].startswith("leave_one_q_out:") if part["fold_id"]=="POOLED_Q_CROSS_FITTED" else p["fold_id"]==part["fold_id"])]
                    cost_support[model]={p["candidate_fingerprint"] for p in pp if all(p["C_pred_by_axis"][a] is not None for a in AXES)}
                    model_G[model]={p["candidate_fingerprint"]:conditional_work(refs[p["candidate_fingerprint"]].get("axis_shots"),p["C_pred_by_axis"]) for p in pp}
                selections.extend(dict(epsilon=epsilon,fold_id=part["fold_id"],dataset_scope=group,
                    single_frozen_model=part["fold_id"]=="full210",**record) for record in
                    common_support_selection(support_refs,model_G,cost_support,"CROSS_FITTED_INTERNAL_GROUP" if part["fold_id"]=="POOLED_Q_CROSS_FITTED" else group))
    for dataset in [training,pm1,m2]:
        supported=[r for r in dataset if r.features.actions()["A_ceil"] is not None]
        for axis in AXES:
            rank_diagnostics.append(dict(dataset=supported[0].dataset if supported else None,axis=axis,
                 **rank_index([r.features.actions()["A_ceil"] for r in supported],[r.reference_costs[axis]["rz_count"] for r in supported])))
    # Saved PM2 grid is reference reproduction, not repeated model training.
    reproduction_count=0
    for e in allowlist["entries"]:
        if e["path"].endswith("/precision_ledger.csv"):
            for old in csv.DictReader(io.StringIO(values[e["path"]].decode())):
                r=by_fp.get(old["candidate_fingerprint"])
                require(r is not None,"INPUT_IDENTITY","PM2 ledger candidate outside direct set")
                shot=reference_shots(r.reference_bias,r.B,float(old["epsilon"]))
                require(shot["eligible_ref"]==(old["accuracy_eligible"].lower()=="true"),"SCHEMA","PM2 strict eligibility mismatch")
                if shot["eligible_ref"]:
                    for a,k in [("real","N_real"),("imag","N_imag")]:
                        require(shot["axis_shots"][a]==int(old[k]),"SCHEMA","PM2 shot rounding mismatch")
                    G=conditional_work(shot["axis_shots"],{a:r.reference_costs[a]["rz_count"] for a in AXES})
                    saved=float(old["primary_RZ_P0"])
                    require(abs(G-saved)<=1e-9+1e-10*abs(saved),"SCHEMA","PM2 reference RZ work mismatch")
                reproduction_count+=1
    # Prediction rows store one-shot C; epsilon-dependent conditional G is in selection records.
    return {"feature_provenance.json":[dict(candidate_fingerprint=r.fingerprint,**r.feature_provenance) for r in rows],
            "model_fits.json":dict(fits=fits,complexity_gate=adopted,index_diagnostics=rank_diagnostics),"predictions.jsonl":predictions,
            "cost_metrics.csv":[dict(row_kind="candidate_axis_error",**e) for e in errors]+[dict(row_kind="group_summary",**s) for s in summaries],
            "normalization_audit.csv":normalization,"shot_availability.json":dict(**operational_na(),reference_reproduced_rows=reproduction_count,structural_model=structural_na(),selection_status_counts=selection_status_summary(selections)),
            "conditional_oracle_selection.csv":selections,"paired_cost_statistics.csv":statistics,
            "report.md":"# AX-1b saved H4 model development diagnostics\n\nRQ-P1 and CONDITIONAL_ORACLE only. H4 internal/observed/cross-fitted results are development diagnostics. Operational shots/work/regret and full structural model remain N/A. Undetermined reference eligibility blocks full-set regret; known-eligible subset diagnostics have separate fields. Common-support membership retains undetermined candidates and its regret is scoped to that registered support. Engineering intervals are not formal CI; rare-event cost tail and cross-candidate covariance remain unresolved.\n\nMandatory STOP; next stage not authorized."}
