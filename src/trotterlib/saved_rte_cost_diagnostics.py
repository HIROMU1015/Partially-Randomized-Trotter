"""Read-only reconciliation of stored H2 order counts and paired MC costs.

No RNG replay, circuit generation, transpilation, molecular input reconstruction
or population-cost estimate is performed. Missing event-to-trajectory joins stay
missing: cost differences and seeds alone do not identify Taylor events.
"""
from __future__ import annotations

from fractions import Fraction
import math
from pathlib import Path

from .isolated_qiskit_execution import read, save, sha


def diagnose(root):
    root=Path(root)
    old=root/"artifacts/non_df_h2_process_recovery/2026-10-23"
    comparison=read(old/"comparison.json")
    stop=read(old/"STOP.json")
    if stop["result_inventory_sha256"]!=sha(old/"result_inventory.json"):
        raise ValueError("OLD_INVENTORY_BINDING")
    inventory=read(old/"result_inventory.json")["files"]
    if inventory["comparison.json"]!=sha(old/"comparison.json"):
        raise ValueError("COMPARISON_HASH")
    reports=[]
    for name in ["pauli_jw_PR","df_native_PR"]:
        saved=next(c for c in comparison["cases"] if c["case"]==name)
        folder=root/saved["run"]
        if sha(folder/"result.json")!=saved["result_sha256"]:
            raise ValueError("RESULT_HASH:"+name)
        result=read(folder/"result.json")
        validation=read(folder/"validation.json")
        if validation!=saved["validation"] or validation["status"]!="PASS":
            raise ValueError("VALIDATION_BINDING")
        samples=result["circuit_cost"]["samples"]
        count=len(samples)
        if count!=64 or result["config"]["method"]!="PR":
            raise ValueError("EXPECTED_STORED_H2_PR_COHORT")
        events=validation["sampled_orders"]
        positions=count*result["config"]["q"]*result["config"]["r"]
        if sum(e["observed"] for e in events)!=positions or any(e["event_positions"]!=positions for e in events):
            raise ValueError("STORED_EVENT_COUNT_RECONCILIATION")
        if not math.isclose(sum(e["probability"] for e in events),1.,abs_tol=1e-14):
            raise ValueError("STORED_ORDER_PROBABILITY_SUM")
        for axis in ["cosine","sine"]:
            exact=Fraction(sum(s["RZ"][axis] for s in samples),count)
            if exact!=Fraction(result["circuit_cost"]["axes"][axis]["mean_exact"]):
                raise ValueError("STORED_MC_MEAN_IDENTITY")
        has_events=all("events" in s or "event_orders" in s for s in samples)
        all_zero=all(e["observed"]==0 for e in events if e["order"]!=0)
        # Current saved schema has only seed and paired RZ, not event logs. Do
        # not replay PCG64 or infer a label from an unusual circuit price.
        joined=[dict(index=i,seed=s["seed"],RZ=s["RZ"],
            saved_keys=sorted(s),Taylor_orders=s.get("event_orders",[0]*(positions//count) if all_zero else None),
            event_to_cost_join=("AVAILABLE" if "event_orders" in s else
                "ALL_ORDER0_INFERRED_FROM_COMPLETE_AGGREGATE; COMPONENT_LOG_MISSING" if all_zero else "MISSING_STORED_EVENT_LOG"))
            for i,s in enumerate(samples)]
        reports.append(dict(case=name,SOURCE=saved["SOURCE"],run=saved["run"],
            result_sha256=saved["result_sha256"],validation_sha256=sha(folder/"validation.json"),
            Hamiltonian_hash=result["model"]["hamiltonian_hash"],state_hash=result["state"]["hash"],
            config=result["config"],normalization=result["normalization"],
            order_rows=events,event_positions=positions,trajectory_count=count,
            law_probabilities_and_counts_copied_without_sampling=True,
            paired_axes=result["circuit_cost"]["paired_axes"],saved_cost_statistics=result["circuit_cost"]["axes"],
            saved_precision_rows=saved["precision_rows"],samples=joined,
            per_trajectory_event_log_available=has_events,
            trajectory_order_cost_link_status=("AVAILABLE" if has_events else
                "ORDERS_ALL_ZERO_FROM_AGGREGATE_COMPONENT_LOG_MISSING" if all_zero else "NOT_IDENTIFIABLE_FROM_SAVED_SCHEMA"),
            forced_orders=validation["forced_event_orders"],forced_orders_are_cost_MC=False,
            nominal_interval_scope="Student-t diagnostic using realized trajectory sample variance; assumes IID trajectory draws",
            limitations=["Unobserved orders do not have zero compiled cost",
                "Sample variance cannot account for unobserved order-specific cost magnitudes",
                "Seed alone is not a stored event record; no RNG replay was performed",
                ("Complete aggregate counts imply all eight orders are zero for each saved DF trajectory, but component IDs/phases were not saved" if all_zero else
                 "Aggregate order counts cannot locate the two order2 events among the 64 saved Pauli trajectory costs"),
                "Whole-wrapper compiler cancellations prevent additive event-cost inference",
                "Pauli and DF component laws and circuits differ; no cross-representation cost imputation",
                "No certified population mean, rare-event contribution or scientific re-ranking"] ))
    return dict(status="SAVED_EVIDENCE_DIAGNOSIS_COMPLETE",scope="H2 1.00A STO-3G; DF tol-only1e-8 rank3; T=.8 q4 delta=.2 S2; PR r2 K4; Pauli LD7/14 and DF LD1/3 plus one-body",
        comparison_sha256=sha(old/"comparison.json"),old_STOP_sha256=sha(old/"STOP.json"),
        reports=reports,new_sampling=0,RNG_replay=0,new_circuit_builds=0,new_compile=0,
        new_population_cost_estimates=0,new_scientific_ranking=0,scientific_guarantee_upgraded=False)


def write_diagnosis(root,destination):
    destination=Path(destination)
    report=diagnose(root)
    save(destination/"rare_event_diagnosis.json",report)
    lines=["# H2保存済み希少Taylor eventの技術診断", "", report["scope"], "",
        "保存された法則・集計・価格を照合した。新sampling・RNG replay・回路生成・compileは0。", "",
        "|case|order|法則確率/event|観測数 / 512位置|保存された未観測確率|母平均費用寄与|",
        "|---|---:|---:|---:|---:|---|"]
    for case in report["reports"]:
        for row in case["order_rows"]:
            lines.append(f"|{case['case']}|{row['order']}|{row['probability']}|{row['observed']}|{row['probability_of_zero']}|未同定|")
    lines += ["", "各caseの64標本はseedとcos/sinのRZ価格を保存しているが、trajectory別event列・order位置・component IDを保存していない。",
        "したがってPauliのorder2×2件がどの価格に対応するかは、今回の保存値だけでは結合できない。",
        "DFは全512位置がorder0なので、64標本すべてのorder列は0×8と集計から確定する。ただしcomponent ID/位相の個別記録はない。",
        "DFのorder2/4未観測をPauliの価格で補完しない。JSONに128標本の価格・seedと欠測joinを明示した。", "",
        "名目SE/t95%は実現した64 trajectoryの標本分散の診断。未観測orderの大きな費用・母平均・厳密coverageは保証しない。",
        "強制order0/2/4の代数的回路検査は保存されているが、cost MCへの寄与測定ではない。",
        "既存C/N/G・順位・図は変更しない。追加情報取得案は別の未承認契約とし、実行しない。"]
    (destination/"rare_event_diagnosis.md").write_text("\n".join(lines)+"\n")
    return report
