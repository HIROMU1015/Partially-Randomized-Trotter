"""Read-only integration and arithmetic checks of four saved H2 case records."""
from fractions import Fraction
import math
from pathlib import Path

import numpy as np

from .fixed_signal_accounting import AXES
from .fixed_signal_evaluation import validate_fixed_signal_result
from .h2_non_df_pilot import precision_accounting, seed_for
from .isolated_qiskit_execution import read, save, sha
from . import h2_non_df_process_recovery as recovery


def validate_saved_case(name, result, rows, validation, plan):
    validate_fixed_signal_result(result)
    rep, method = name.rsplit("_",1)
    config = result["config"]
    expected_ld = (7 if rep=="pauli_jw" else 1) if method=="PR" else None
    for k,v in dict(hamiltonian_type=rep, method=method, T=.8, q=4, r=2, K=4,
                    L_D=expected_ld, cost_samples=64, seed=seed_for(rep,method),
                    product_formula="S2", randomization_law="finite_rte_even_paired").items():
        if config[k] != v:
            raise ValueError("SAVED_COMPARISON_CONFIG:"+name+"/"+k)
    if validation["status"] != "PASS" or result["circuit_cost"]["compiled_signal_max_error"] > 1e-9:
        raise ValueError("SAVED_VALIDATION:"+name)
    if [r["epsilon_sig"] for r in rows] != plan["evaluation"]["epsilon_sig"]:
        raise ValueError("PRECISION_SET")
    prices = np.asarray([[r["RZ"][a] for a in AXES] for r in result["circuit_cost"]["samples"]], dtype=float)
    count = len(prices)
    if count != (64 if method=="PR" else 1) or not result["circuit_cost"]["paired_axes"]:
        raise ValueError("MC_SAMPLE_DESIGN")
    cov = np.cov(prices, rowvar=False, ddof=1) if count>1 else np.zeros((2,2))
    arithmetic = []
    for row in rows:
        rebuilt = precision_accounting(result, row["epsilon_sig"], plan)
        if rebuilt != row:
            raise ValueError("SAVED_PRECISION_ACCOUNTING:"+name)
        if row["shots"]["usable"]:
            ns = [row["shots"]["axes"][a]["shots"] for a in AXES]
            expected = sum(n*Fraction(result["circuit_cost"]["axes"][a]["mean_exact"]) for n,a in zip(ns, AXES))
            if expected != Fraction(row["total_RZ"]["mean_exact"]):
                raise ValueError("SAVED_G_EXACT_IDENTITY")
            se = math.sqrt(max(0., float(np.asarray(ns)@cov@np.asarray(ns))/count))
            if not math.isclose(se,row["total_RZ"]["standard_error"], rel_tol=1e-12, abs_tol=1e-8):
                raise ValueError("PAIRED_COVARIANCE_SE")
            arithmetic.append(dict(epsilon_sig=row["epsilon_sig"], exact_G_identity=True,
                                   paired_covariance_SE=se))
    return dict(status="PASS", arithmetic=arithmetic, cost_sample_count=count,
        covariance_cosine_sine=cov.tolist(), covariance_included_in_G_SE=True,
        nominal95_is_population_guarantee=False, sampled_orders=validation["sampled_orders"])


def integrate():
    plan, source, grant = recovery.checked_binding()
    run = recovery.ROOT/plan["execution"]["run_namespace"]
    if read(run/"STOP.json")["status"] != "COMPLETE_MANDATORY_STOP":
        raise ValueError("SCIENTIFIC_RUN_NOT_COMPLETE_NO_INTEGRATION")
    old_verified = recovery.verify_old()
    receipt = read(recovery.OLD/"run01/inputs/receipt.json")
    cases, table, checks = [], [], {}
    compiler = None
    for name in ["pauli_jw_PF", "pauli_jw_PR", "df_native_PF", "df_native_PR"]:
        reused = name in recovery.REUSED
        folder = recovery.OLD/"run01"/name if reused else run/name
        result, rows, validation = [read(folder/n) for n in ["result.json","precision_rows.json","validation.json"]]
        if read(folder/"status.json")["status"] != "COMPLETE_FIXED_CASE":
            raise ValueError("UNACCEPTED_CASE")
        expected_source = recovery.OLD_SOURCE if reused else source["commit"]
        p = result["provenance"]["source"]
        if p["SOURCE"] != expected_source or p["input_snapshot_sha256"] != plan["input_snapshot_sha256"]:
            raise ValueError("RESULT_SOURCE_INPUT_BINDING")
        if result["state"]["hash"] != receipt["state_hash"]:
            raise ValueError("RESULT_STATE_BINDING")
        expected_ham = receipt["Pauli_hamiltonian_hash"] if name.startswith("pauli") else read(recovery.OLD/"run01/df_native_PR/result.json")["model"]["hamiltonian_hash"]
        if result["model"]["hamiltonian_hash"] != expected_ham:
            raise ValueError("RESULT_HAMILTONIAN_BINDING")
        if compiler is not None and result["compiler"] != compiler:
            raise ValueError("COMPILER_COMPARISON_CHANGED")
        compiler = result["compiler"]
        checks[name] = validate_saved_case(name,result,rows,validation,plan["science"])
        case = dict(case=name, reused_read_only=reused, SOURCE=expected_source,
            run=str(folder.relative_to(recovery.ROOT)), result_sha256=sha(folder/"result.json"),
            precision_rows_sha256=sha(folder/"precision_rows.json"), model=result["model"], state=result["state"],
            reference=result["reference"], signal=result["signal"], normalization=result["normalization"],
            circuit_cost={k:v for k,v in result["circuit_cost"].items() if k!="samples"},
            precision_rows=rows, validation=validation, compiler=compiler,
            comparison_checks=checks[name])
        cases.append(case)
        for row in rows:
            table.append(dict(case=name, epsilon_sig=row["epsilon_sig"], SOURCE=expected_source,
                C={a:result["circuit_cost"]["axes"][a] for a in AXES},
                N={a:row["shots"]["axes"][a]["shots"] for a in AXES},
                N_total=row["shots"]["N_total"], G=row["total_RZ"],
                eligible=row["shots"]["usable"], interpretation="CONDITIONAL_EMPIRICAL_ALLOWANCE"))
    output = dict(status="FOUR_FIXED_H2_CASES_COMPLETE_LOCAL_CONDITIONAL_EVIDENCE", SOURCE=source["commit"],
        old_source=recovery.OLD_SOURCE, contract_sha256=sha(recovery.BASE/"execution_contract.json"),
        input_sha256=plan["input_snapshot_sha256"], input_verification=old_verified,
        system=plan["science"]["system"], conditions=plan["science"]["evaluation"], cases=cases,
        table=table, same_L_D_is_same_partition=False,
        limitations=["fixed H2 geometry/basis/T/q/splits only; no general superiority or scaling conclusion",
            "empirical numerical allowance; Hoeffding sufficient N conditional on declared error bounds",
            "PR C/G are 64 paired MC sample means; nominal t interval is not a population guarantee",
            "unobserved Taylor orders retain unknown cost contribution, never zero-filled",
            "no new SCF/FCI/DF/state solve, candidate search, independent confirmation or detailed science review"],
        original_STOP_unchanged=True, automatic_next_stage=False)
    save(recovery.BASE/"comparison.json", output)
    lines=["# H2固定4方式の保存値比較", "", "H2 / 1.00 Å / STO-3G / T=.8, q4, δ=.2 / S2 / P=0。",
        "DFはtol-only1e-8 rank3。Pauli PR L_D7/14項、DF PR L_D1/3 fragment（one-bodyを別途保持）、r2/K4。",
        "PFは各表現の全block保持。分割単位が異なる。各軸ε/√2、α=.025、経験的allowance下のHoeffding十分N。",
        "PRは64 paired MC平均。Cはfull ordinary-control Hadamardのcompiled RZ/shot。", "",
        "|方式|ε_sig|C_cos|C_sin|N_cos|N_sin|G [RZ]|G SE|適格性|", "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in table:
        g=row["G"]
        lines.append(f"|{row['case']}|{row['epsilon_sig']}|{row['C']['cosine']['mean_exact']}|{row['C']['sine']['mean_exact']}|{row['N']['cosine']}|{row['N']['sine']}|{g['mean_exact'] if g else 'null'}|{g['standard_error'] if g else 'null'}|{'条件付き適格' if row['eligible'] else '未解決'}|")
    lines += ["", "Gは正確な有理数。誤差保証・母平均保証・全RPE費用・一般的な方式優位性を示さない。",
              "signal・bias・B・hash・SOURCE・検証gate・未観測orderは [comparison.json](comparison.json) を参照。"]
    (recovery.BASE/"comparison.md").write_text("\n".join(lines)+"\n")
    return output
