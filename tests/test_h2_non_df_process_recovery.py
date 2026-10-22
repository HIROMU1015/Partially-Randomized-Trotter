"""Authorization, one-shot, process cleanup and comparison gates; no molecules."""
import copy
import json
from pathlib import Path
import sys

import pytest

from trotterlib import h2_non_df_process_recovery as recovery
from trotterlib.h2_non_df_pilot import contract as old_contract
from trotterlib.isolated_qiskit_execution import compare_artificial, read, run_processes, save
from trotterlib.h2_non_df_process_integration import validate_saved_case


def test_science_contract_is_exactly_preserved_with_two_cases_only():
    plan = recovery.contract()
    original = old_contract()
    assert plan["science"] == {k:original[k] for k in plan["science"]}
    assert plan["acquisition_cases"] == ["pauli_jw_PR", "df_native_PF"]
    assert plan["reuse_without_recomputation"] == ["pauli_jw_PF", "df_native_PR"]
    for key in ["new_SCF", "new_FCI", "new_DF_decompositions", "new_state_solves"]:
        assert plan["execution"][key] == 0
    assert plan["execution"]["one_shot"] and not plan["execution"]["automatic_retry"]


@pytest.mark.parametrize("name", ["pauli_jw_PF", "df_native_PR", "pauli_jw_B0", "H4"])
def test_scientific_worker_rejects_unapproved_cases(name):
    with pytest.raises(ValueError, match="NOT_AUTHORIZED"):
        recovery.worker(name, Path("unused"))


def test_contract_mutation_rejected_before_science():
    plan = recovery.contract()
    plan["science"]["evaluation"]["PR_cost_samples"] = 65
    with pytest.raises(ValueError, match="CONTRACT_CHANGED"):
        recovery.verify_binding(plan, {})


def test_consumed_run_rejects_second_launch(tmp_path, monkeypatch):
    (tmp_path/"run").mkdir()
    monkeypatch.setattr(recovery, "ROOT", tmp_path)
    monkeypatch.setattr(recovery, "checked_binding", lambda: (
        {"execution":{"run_namespace":"run"}}, {"commit":"synthetic"}, {"grant_id":"test"}))
    with pytest.raises(FileExistsError):
        recovery.execute()


def test_consumed_worker_rejects_second_invocation(tmp_path, monkeypatch):
    out = tmp_path/"run/pauli_jw_PR"
    save(out/"marker.json", dict(grant_id="test", case="pauli_jw_PR"))
    save(out/"worker_consumed.json", {"pid":0})
    monkeypatch.setattr(recovery, "ROOT", tmp_path)
    monkeypatch.setattr(recovery, "checked_binding", lambda: (
        {"execution":{"run_namespace":"run"}}, {"commit":"synthetic"}, {"grant_id":"test"}))
    with pytest.raises(FileExistsError):
        recovery.worker("pauli_jw_PR", tmp_path/"unused")


def test_process_failure_retains_exit_and_logs_and_reaps_child(tmp_path):
    out = tmp_path/"case"
    out.mkdir()
    gate = tmp_path/"release.json"
    code = ("import pathlib,time,sys,os; out=pathlib.Path(sys.argv[1]); "
            "(out/'ready.json').write_text('{}'); "
            "exec('while not pathlib.Path(sys.argv[2]).exists(): time.sleep(.01)'); "
            "print('synthetic failure',file=sys.stderr); sys.exit(7)")
    reports = run_processes([dict(case="mock", output=out,
        argv=[sys.executable,"-c",code,str(out),str(gate)])], gate, cwd=tmp_path)
    assert reports[0]["exit_code"] == 7
    assert read(out/"process.json")["exit_code"] == 7
    assert "synthetic failure" in (out/"stderr.log").read_text()
    assert not Path(f"/proc/{reports[0]['pid']}").exists()


def test_process_output_collision_rejected_before_launch(tmp_path):
    jobs = [dict(case=str(i), output=tmp_path, argv=["does-not-exist"]) for i in range(2)]
    with pytest.raises(ValueError, match="COLLISION"):
        run_processes(jobs, tmp_path/"gate", cwd=tmp_path)


def test_pre_release_crash_is_reaped_and_recorded(tmp_path):
    jobs=[]
    for i in range(2):
        out=tmp_path/str(i)
        out.mkdir()
        jobs.append(dict(case=str(i), output=out, argv=[sys.executable,"-c","raise RuntimeError('startup')"]))
    with pytest.raises(RuntimeError, match="BEFORE_RELEASE"):
        run_processes(jobs, tmp_path/"gate", cwd=tmp_path)
    assert all(read(j["output"]/"process.json")["exit_code"] != 0 for j in jobs)


@pytest.mark.parametrize("mutation", ["counts", "operator", "overlap", "pids"])
def test_preflight_rejects_semantic_or_concurrency_mismatch(mutation):
    semantic=dict(config={}, samples=[{"seed":1,"RZ":{"cosine":1,"sine":2}}], probe_RZ=4,
        operator_real=[[1]], operator_imag=[[0]], normalization={"log_B":0.},
        signal={k:{"real":0.,"imag":0.} for k in ["raw","corrected","signed_bias"]},
        reference={"signal":{"real":0.,"imag":0.}})
    sequential={str(i):dict(semantic=copy.deepcopy(semantic),pid=i,started=1.,ended=3.) for i in range(4)}
    parallel=copy.deepcopy(sequential)
    if mutation=="counts": parallel["0"]["semantic"]["samples"][0]["RZ"]["cosine"]=3
    if mutation=="operator": parallel["0"]["semantic"]["operator_real"]=[[.9]]
    if mutation=="overlap": parallel["0"]["started"]=4.
    if mutation=="pids": parallel["0"]["pid"]=1
    with pytest.raises(ValueError):
        compare_artificial(sequential,parallel)


@pytest.mark.parametrize("name", recovery.REUSED)
def test_accepted_saved_accounting_and_covariance_without_molecular_execution(name):
    folder=recovery.OLD/"run01"/name
    result,rows,validation=[read(folder/p) for p in ["result.json","precision_rows.json","validation.json"]]
    report=validate_saved_case(name,result,rows,validation,old_contract())
    assert report["status"]=="PASS" and len(report["arithmetic"])==5


def test_saved_paired_covariance_gate_detects_wrong_G_standard_error():
    folder=recovery.OLD/"run01/df_native_PR"
    result,rows,validation=[read(folder/p) for p in ["result.json","precision_rows.json","validation.json"]]
    rows[0]["total_RZ"]["standard_error"]+=1
    with pytest.raises(ValueError,match="SAVED_PRECISION_ACCOUNTING"):
        validate_saved_case("df_native_PR",result,rows,validation,old_contract())
