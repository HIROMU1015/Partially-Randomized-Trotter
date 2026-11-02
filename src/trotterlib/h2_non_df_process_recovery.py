"""One-shot acquisition of only the two authorized missing H2 cases.

All scientific kernels and tolerances are imported unchanged from the old
pilot. Inputs and accepted results are hash-checked read-only snapshots.
"""
from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

from .isolated_qiskit_execution import digest, read, save, sha, run_processes, await_release

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT/"artifacts/non_df_h2_pilot/2026-10-22"
BASE = ROOT/"artifacts/non_df_h2_process_recovery/2026-10-23"
CASES = ("pauli_jw_PR", "df_native_PF")
REUSED = ("pauli_jw_PF", "df_native_PR")
BASE_COMMIT = "bd2f3ce9b2035d9caefac75785e6fe70a3dcbca2"
OLD_SOURCE = "b5ed2c6e2399b55a0c24eb8437f842892a6703ef"


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def resources():
    from datetime import datetime, timezone
    mem = {l.split(":")[0]:l.split(":")[1].strip() for l in Path("/proc/meminfo").read_text().splitlines()}
    s = os.statvfs(ROOT)
    return dict(time_UTC=datetime.now(timezone.utc).isoformat(), affinity=len(os.sched_getaffinity(0)),
        load=list(os.getloadavg()), MemAvailable=mem["MemAvailable"], disk_available_bytes=s.f_bavail*s.f_frsize,
        arbitrary_resource_caps=False, shared_settings_changed=False)


def contract():
    from .h2_non_df_pilot import contract as original_contract
    old = original_contract()
    return dict(schema="h2_missing_cases_process_contract_v1", previous_result_commit=BASE_COMMIT,
        previous_science_SOURCE=OLD_SOURCE, previous_contract_sha256=sha(OLD/"execution_contract.json"),
        science={k:old[k] for k in ["system","DF","Pauli","evaluation","compiler","numerical","environment"]},
        acquisition_cases=list(CASES), reuse_without_recomputation=list(REUSED),
        input_snapshot_sha256=sha(OLD/"run01/inputs/snapshot.npz"),
        input_receipt_sha256=sha(OLD/"run01/inputs/receipt.json"),
        execution=dict(run_namespace="artifacts/non_df_h2_process_recovery/2026-10-23/run02",
            isolated_fresh_interpreter_per_case=True, new_SCF=0, new_FCI=0, new_DF_decompositions=0,
            new_state_solves=0, fixed_scientific_cases=2, one_shot=True, resume=False, automatic_retry=False,
            original_run01_permanently_stopped=True, old_grant_reuse=False, no_new_model=True,
            no_candidate_search=True, no_H4_or_larger=True, no_independent_confirmation=True,
            no_detailed_scientific_review=True, push=False, mandatory_STOP=True))


def verify_old():
    from .h2_non_df_pilot import contract as original_contract
    inventory = read(OLD/"result_inventory.json")
    if read(OLD/"execution_contract.json") != original_contract():
        raise ValueError("ORIGINAL_CONTRACT_CHANGED")
    if read(OLD/"SOURCE.json")["commit"] != OLD_SOURCE:
        raise ValueError("ORIGINAL_SOURCE_CHANGED")
    if read(OLD/"run01/STOP.json")["status"] != "FAILED_NO_RETRY_MANDATORY_STOP":
        raise ValueError("ORIGINAL_STOP_CHANGED")
    for name, expected in inventory["raw_result_hashes"].items():
        if sha(OLD/name) != expected:
            raise ValueError("OLD_RESULT_HASH:"+name)
    # All old scientific implementation bytes must remain unchanged. This also
    # gates joining the two SOURCE cohorts without silently redefining circuits.
    source = read(OLD/"SOURCE.json")
    for name, expected in source["files"].items():
        if sha(ROOT/name) != expected:
            raise ValueError("ORIGINAL_SCIENTIFIC_SOURCE_BYTES:"+name)
    for name in REUSED:
        r = read(OLD/"run01"/name/"result.json")
        if (read(OLD/"run01"/name/"status.json")["status"] != "COMPLETE_FIXED_CASE"
                or read(OLD/"run01"/name/"validation.json")["status"] != "PASS"
                or r["provenance"]["source"]["SOURCE"] != OLD_SOURCE
                or r["provenance"]["source"]["input_snapshot_sha256"] != sha(OLD/"run01/inputs/snapshot.npz")):
            raise ValueError("REUSED_RESULT_PROVENANCE:"+name)
    return dict(status="PASS", old_raw_hashes_verified=len(inventory["raw_result_hashes"]),
        original_source_files_verified=len(source["files"]), input_sha256=sha(OLD/"run01/inputs/snapshot.npz"),
        new_SCF=0, new_DF_decompositions=0, new_FCI=0, new_state_solves=0,
        reused_result_hashes={name:sha(OLD/"run01"/name/"result.json") for name in REUSED})


def load_saved_inputs():
    import numpy as np
    from .df_hamiltonian import DFHamiltonian
    from .pauli_signal_adapter import PauliHamiltonian, array_hash
    verify_old()
    receipt, p = read(OLD/"run01/inputs/receipt.json"), read(OLD/"run01/inputs/pauli.json")
    with np.load(OLD/"run01/inputs/snapshot.npz", allow_pickle=False) as z:
        pauli = PauliHamiltonian(4, p["constant"], tuple(map(tuple,p["terms"])), json.dumps(p["metadata"], sort_keys=True))
        df = DFHamiltonian(float(z["nuclear_repulsion"]), z["df_one_body"].copy(), z["df_lambdas"].copy(),
                           tuple(g.copy() for g in z["df_g_matrices"]), receipt["DF_metadata"])
        state = z["state"].copy()
        if array_hash(state) != receipt["state_hash"] or pauli.hamiltonian_hash != receipt["Pauli_hamiltonian_hash"]:
            raise ValueError("RESTORED_INPUT_HASH")
        if len(pauli.terms) != 14 or df.n_blocks != 3:
            raise ValueError("RESTORED_INPUT_SHAPE")
    return pauli, df, state, receipt


def verify_binding(plan, source):
    if plan != contract() or digest(plan) != source["contract_digest"]:
        raise ValueError("CONTRACT_CHANGED")
    if source["commit"] != git("rev-parse", "HEAD"):
        raise ValueError("SOURCE_HEAD_CHANGED")
    for name, expected in source["files"].items():
        if sha(ROOT/name) != expected:
            raise ValueError("SOURCE_BYTES_CHANGED:"+name)
    for name, version in plan["science"]["environment"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("ENVIRONMENT_VERSION:"+name)
    verify_old()


def bind():
    if git("status", "--porcelain"):
        raise ValueError("SOURCE_REQUIRES_CLEAN_COMMIT")
    if (BASE/"SOURCE.json").exists() or (BASE/"run02").exists():
        raise FileExistsError("ALREADY_BOUND_OR_CONSUMED")
    plan = read(BASE/"execution_contract.json")
    if plan != contract():
        raise ValueError("CONTRACT_CHANGED")
    preflight = read(BASE/"technical/preflight.json")
    tests = read(BASE/"technical/regression.json")
    if preflight["status"] != "PASS" or tests["status"] != "PASS":
        raise ValueError("TECHNICAL_GATES_NOT_PASSED")
    names = [str(p.relative_to(ROOT)) for p in (ROOT/"src/trotterlib").rglob("*.py")]
    names += ["scripts/run_h2_non_df_process_recovery.py", "tests/test_h2_non_df_process_recovery.py"]
    names += [str(p.relative_to(ROOT)) for p in BASE.rglob("*") if p.is_file()]
    source = dict(commit=git("rev-parse","HEAD"), branch=git("branch","--show-current"),
        base_commit=BASE_COMMIT, previous_science_SOURCE=OLD_SOURCE, contract_digest=digest(plan),
        files={n:sha(ROOT/n) for n in sorted(names)}, python=platform.python_version(),
        resources=resources(), local_only=True)
    verify_binding(plan, source)
    save(BASE/"SOURCE.json", source)
    save(BASE/"grant.json", dict(grant_id=digest(dict(SOURCE=source["commit"], contract=digest(plan),
        authorization=sha(BASE/"authorization.md"))), SOURCE=source["commit"],
        source_manifest_sha256=sha(BASE/"SOURCE.json"), contract_digest=digest(plan),
        approved_by_user=True, authorized_cases=list(CASES), reused_cases=list(REUSED),
        previous_grant_consumed_not_reused=read(OLD/"grant.json")["grant_id"],
        automatic_retry=False, one_shot_per_case=True, mandatory_STOP=True))
    print(json.dumps(dict(status="BOUND_NO_SCIENCE_EXECUTED", SOURCE=source["commit"])), flush=True)


def checked_binding():
    plan, source, grant = read(BASE/"execution_contract.json"), read(BASE/"SOURCE.json"), read(BASE/"grant.json")
    verify_binding(plan, source)
    if (grant["SOURCE"] != source["commit"] or grant["contract_digest"] != digest(plan)
            or grant["source_manifest_sha256"] != sha(BASE/"SOURCE.json") or grant["authorized_cases"] != list(CASES)):
        raise ValueError("GRANT_BINDING")
    return plan, source, grant


def ledger(run, event):
    # Only the supervisor writes this ledger; workers have distinct directories.
    with (run/"ledger.jsonl").open("a") as stream:
        stream.write(json.dumps(dict(time=time.time(), **event), sort_keys=True)+"\n")


def worker(name, gate):
    if name not in CASES:
        raise ValueError("SCIENTIFIC_CASE_NOT_AUTHORIZED")
    plan, source, grant = checked_binding()
    run, destination = ROOT/plan["execution"]["run_namespace"], ROOT/plan["execution"]["run_namespace"]/name
    marker = read(destination/"marker.json")
    if marker["grant_id"] != grant["grant_id"] or marker["case"] != name:
        raise ValueError("CASE_GRANT_BINDING")
    # A second direct worker invocation is rejected even after failure.
    save(destination/"worker_consumed.json", dict(pid=os.getpid(), no_retry=True))
    try:
        await_release(destination, gate)
        started = time.time()
        print(f"{name}: started pid={os.getpid()} saved-input-only", flush=True)
        from .h2_non_df_pilot import evaluate_case
        from .h2_non_df_pilot_validation import validate_case
        from .pauli_signal_adapter import array_hash
        pauli, df, state, receipt = load_saved_inputs()
        representation, method = name.rsplit("_",1)
        provenance = dict(SOURCE=source["commit"], source_manifest_sha256=sha(BASE/"SOURCE.json"),
            input_snapshot_sha256=plan["input_snapshot_sha256"], input_SOURCE=OLD_SOURCE,
            grant_id=grant["grant_id"], contract_digest=digest(plan), run=str(run.relative_to(ROOT)))
        save(destination/"SOURCE.json", dict(**provenance, case=name, original_scientific_bytes_unchanged=True))
        c, a, evidence, result, rows = evaluate_case(pauli if representation=="pauli_jw" else df,
            state.copy(), representation, method, plan["science"], provenance)
        save(destination/"numerical_evidence.json", evidence)
        save(destination/"precision_rows.json", rows)
        validation = validate_case(a, state, c, result, plan["science"])
        if array_hash(state) != receipt["state_hash"]:
            raise ValueError("STATE_MUTATED")
        old_hash = (receipt["Pauli_hamiltonian_hash"] if representation=="pauli_jw" else
                    read(OLD/"run01/df_native_PR/result.json")["model"]["hamiltonian_hash"])
        if result["model"]["hamiltonian_hash"] != old_hash or result["state"]["hash"] != receipt["state_hash"]:
            raise ValueError("CASE_INPUT_IDENTITY")
        save(destination/"validation.json", validation)
        result["validation"].update(status="H2_FIXED_CASE_LOCAL_CONDITIONAL_EVIDENCE", molecular_validation=True,
            molecular_validation_scope="only H2 1.00A STO-3G T=.8 q4 delta=.2 executed technical gates")
        save(destination/"result.json", result)
        save(destination/"status.json", dict(status="COMPLETE_FIXED_CASE", SOURCE=source["commit"],
            pid=os.getpid(), started=started, ended=time.time(), one_shot=True, no_retry=True))
        print(f"{name}: COMPLETE N={rows[0]['shots']['N_total']} G={rows[0]['total_RZ']}", flush=True)
    except BaseException as exc:
        save(destination/"status.json", dict(status="FAILED_NO_RETRY", reason=str(exc), traceback=traceback.format_exc(),
                                             SOURCE=source["commit"], pid=os.getpid()))
        raise


def execute():
    plan, source, grant = checked_binding()
    run = ROOT/plan["execution"]["run_namespace"]
    run.mkdir()  # Atomic irrevocable consumption; no resume or alternative ID.
    start = time.time()
    save(run/"marker.json", dict(status="CONSUMED_NONREFUNDABLE", SOURCE=source["commit"],
        grant_id=grant["grant_id"], cases=list(CASES), resources=resources()))
    ledger(run, dict(event="GRANT_CONSUMED", cases=list(CASES), grant_id=grant["grant_id"]))
    try:
        save(run/"input_verification.json", verify_old())
        jobs, gate = [], run/"release.json"
        for name in CASES:
            destination = run/name
            destination.mkdir()
            save(destination/"marker.json", dict(case=name, SOURCE=source["commit"], grant_id=grant["grant_id"],
                status="CONSUMED_NONREFUNDABLE", input_snapshot_sha256=plan["input_snapshot_sha256"]))
            jobs.append(dict(case=name, output=destination, argv=[sys.executable,
                str(ROOT/"scripts/run_h2_non_df_process_recovery.py"), "--worker", name, "--gate", str(gate)]))
        print("One-shot consumed: launching only Pauli PR and DF PF in fresh interpreters", flush=True)
        reports = run_processes(jobs, gate, cwd=ROOT)
        for report in reports:
            ledger(run, dict(event="CHILD_REAPED", **report))
        save(run/"processes.json", reports)
        if any(r["exit_code"] != 0 for r in reports):
            raise RuntimeError("CASE_FAILED_NO_RETRY")
        for name in CASES:
            if read(run/name/"status.json")["status"] != "COMPLETE_FIXED_CASE":
                raise ValueError("CASE_NOT_ACCEPTED")
        save(run/"summary.json", dict(status="TWO_MISSING_CASES_COMPLETE", SOURCE=source["commit"],
            cases=list(CASES), new_SCF=0, new_FCI=0, new_DF_decompositions=0, new_state_solves=0,
            old_successes_recomputed=False, elapsed_seconds=time.time()-start, resources_after=resources()))
        ledger(run, dict(event="MANDATORY_STOP", status="COMPLETE"))
        save(run/"STOP.json", dict(status="COMPLETE_MANDATORY_STOP", SOURCE=source["commit"],
            grant_id=grant["grant_id"], automatic_retry=False, automatic_next_stage=False,
            elapsed_seconds=time.time()-start,
            result_hashes={str(p.relative_to(run)):sha(p) for p in sorted(run.rglob("*")) if p.is_file()}))
    except BaseException as exc:
        ledger(run, dict(event="MANDATORY_STOP", status="FAILED", reason=str(exc)))
        save(run/"STOP.json", dict(status="FAILED_NO_RETRY_MANDATORY_STOP", SOURCE=source["commit"],
            grant_id=grant["grant_id"], reason=str(exc), traceback=traceback.format_exc(),
            automatic_retry=False, automatic_next_stage=False))
        raise
