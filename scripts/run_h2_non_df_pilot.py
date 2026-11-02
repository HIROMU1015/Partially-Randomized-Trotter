#!/usr/bin/env python3
"""Bind then execute the authorized fixed H2 pilot once; no resume/retry."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback

from trotterlib.h2_non_df_pilot import build_h2_inputs, contract, evaluate_case
from trotterlib.h2_non_df_pilot_validation import validate_case, validate_df_primitives
from trotterlib.fixed_signal_evaluation import prepare_signal_adapter
from trotterlib.h2_non_df_pilot import case_config

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/"artifacts/non_df_h2_pilot/2026-10-22"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def read(path):
    return json.loads(Path(path).read_text())


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def resource_snapshot():
    mem = {line.split(':')[0]: line.split(':')[1].strip() for line in Path('/proc/meminfo').read_text().splitlines()}
    stat = os.statvfs(ROOT)
    return dict(time_JST=datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(),
                affinity_available=len(os.sched_getaffinity(0)), load_average=list(os.getloadavg()),
                MemAvailable=mem.get("MemAvailable"), filesystem_available_bytes=stat.f_bavail*stat.f_frsize,
                user_resources_modified=False, global_settings_modified=False)


def verify_binding(plan, source):
    if digest(plan) != source["contract_digest"] or plan != contract():
        raise ValueError("CONTRACT_CHANGED")
    if source["commit"] != git("rev-parse", "HEAD"):
        raise ValueError("SOURCE_HEAD_CHANGED")
    for path, expected in source["files"].items():
        if sha(ROOT/path) != expected:
            raise ValueError("SOURCE_BYTES_CHANGED:"+path)
    for name, version in plan["environment"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("ENVIRONMENT_VERSION:"+name)


def bind():
    plan = read(BASE/"execution_contract.json")
    if plan != contract():
        raise ValueError("CONTRACT_FILE_MISMATCH")
    if (BASE/"SOURCE.json").exists() or (BASE/"run01").exists():
        raise ValueError("BINDING_OR_RUN_ALREADY_EXISTS")
    if git("status", "--porcelain"):
        raise ValueError("SOURCE_BIND_REQUIRES_CLEAN_COMMIT")
    names = [str(p.relative_to(ROOT)) for p in (ROOT/"src/trotterlib").rglob("*.py")]
    names += ["scripts/run_h2_non_df_pilot.py", "tests/test_h2_non_df_pilot.py",
              str((BASE/"execution_contract.json").relative_to(ROOT)),
              str((BASE/"authorization.md").relative_to(ROOT)),
              "artifacts/resource_applicability/pr_pf_s2_recomputed/2026-10-22/fixed_provider/src/trottertracks/resource_applicability/h6_matched_accounting_v1.py",
              "artifacts/resource_applicability/pr_pf_s2_recomputed/2026-10-22/fixed_provider/src/trottertracks/resource_applicability/ax2b_numerical_accounting.py"]
    source = dict(commit=git("rev-parse", "HEAD"), branch=git("branch", "--show-current"),
                  base_commit="87e5f4a5ab0182f11c4b3af58ebf5e1c036c80d1",
                  contract_digest=digest(plan), files={p:sha(ROOT/p) for p in sorted(names)},
                  environment={p:importlib.metadata.version(p) for p in plan["environment"]},
                  python=platform.python_version(), resources=resource_snapshot(), local_only=True)
    verify_binding(plan, source)
    save(BASE/"SOURCE.json", source)
    save(BASE/"grant.json", dict(approved_by_user=True, scope="new H2 non-DF pilot including necessary science",
        authorization_sha256=sha(BASE/"authorization.md"), SOURCE=source["commit"],
        plan_digest=source["contract_digest"], source_manifest_sha256=sha(BASE/"SOURCE.json"),
        grant_id=digest(dict(SOURCE=source["commit"], plan=plan, authorization=sha(BASE/"authorization.md"))),
        one_shot=True, automatic_retry=False, mandatory_STOP=True))
    print(json.dumps(dict(status="BOUND_NO_SCIENCE_EXECUTED", SOURCE=source["commit"], plan_digest=source["contract_digest"])))


def execute():
    plan, source, grant = read(BASE/"execution_contract.json"), read(BASE/"SOURCE.json"), read(BASE/"grant.json")
    verify_binding(plan, source)
    if (grant["plan_digest"] != digest(plan) or grant["source_manifest_sha256"] != sha(BASE/"SOURCE.json")
            or grant["authorization_sha256"] != sha(BASE/"authorization.md") or grant["SOURCE"] != source["commit"]):
        raise ValueError("GRANT_BINDING_MISMATCH")
    run = ROOT/plan["execution"]["run_namespace"]
    # This mkdir is the atomic one-shot consumption gate. A failed run cannot
    # be retried, even if input generation or a later phase did not complete.
    run.mkdir()
    started = time.monotonic()
    save(run/"marker.json", dict(state="CONSUMED_NONREFUNDABLE", grant_id=grant["grant_id"],
                                  SOURCE=source["commit"], resources=resource_snapshot()))
    print("H2 pilot one-shot consumed; building new SCF/MO/JW/DF inputs", flush=True)
    try:
        pauli, df, state, receipt = build_h2_inputs(run/"inputs", plan)
        save(run/"inputs/receipt.json", receipt)
        save(run/"inputs/pauli.json", dict(constant=pauli.constant, terms=pauli.terms, metadata=pauli.metadata,
                                           Hamiltonian_hash=pauli.hamiltonian_hash))
        snapshot_hash = sha(run/"inputs/snapshot.npz")
        input_provenance = dict(SOURCE=source["commit"], source_manifest_sha256=sha(BASE/"SOURCE.json"),
                                input_snapshot_sha256=snapshot_hash, grant_id=grant["grant_id"],
                                contract_digest=digest(plan))
        full = prepare_signal_adapter(df, case_config(df, "df_native", "PF", plan))
        save(run/"inputs/DF_native_primitives.json", validate_df_primitives(full, plan))
        print(f"Input gates PASS; Pauli terms={len(pauli.terms)}, DF rank={df.n_blocks}; fixed evaluations start", flush=True)

        def task(representation, method):
            h = pauli if representation == "pauli_jw" else df
            name = representation+"_"+method
            destination = run/name
            destination.mkdir()
            try:
                config, adapter, evidence, result, rows = evaluate_case(h, state.copy(), representation, method, plan, input_provenance)
                save(destination/"numerical_evidence.json", evidence)
                save(destination/"precision_rows.json", rows)
                validation = validate_case(adapter, state, config, result, plan)
                save(destination/"validation.json", validation)
                result["validation"].update(status="H2_FIXED_CASE_LOCAL_CONDITIONAL_EVIDENCE",
                    molecular_validation=True,
                    molecular_validation_scope="only the executed H2 1.00A STO-3G fixed-case technical gates")
                save(destination/"result.json", result)
                save(destination/"status.json", dict(status="COMPLETE_FIXED_CASE", SOURCE=source["commit"],
                                                      no_retry=True, no_next_stage=True))
                print(f"{name}: PASS, N(.05)={rows[0]['shots']['N_total']}, G(.05)={rows[0]['total_RZ']}", flush=True)
                return dict(case=name, status="PASS", result=str(destination.relative_to(ROOT)))
            except BaseException as exc:
                save(destination/"status.json", dict(status="FAILED_NO_RETRY", reason=str(exc),
                                                      traceback=traceback.format_exc(), SOURCE=source["commit"]))
                raise

        reports, failures = [], []
        # Four independent predeclared evaluations, no adaptive task creation.
        with ThreadPoolExecutor() as pool:
            futures = {pool.submit(task, rep, method):(rep, method)
                       for rep in plan["evaluation"]["representations"] for method in plan["evaluation"]["methods"]}
            for future in as_completed(futures):
                try:
                    reports.append(future.result())
                except BaseException as exc:
                    failures.append(dict(case=futures[future], reason=str(exc)))
        if failures:
            save(run/"failures.json", failures)
            raise ValueError("FIXED_CASE_FAILURE_NO_RETRY")
        save(run/"summary.json", dict(status="H2_PILOT_COMPLETE_LOCAL_CONDITIONAL_EVIDENCE", SOURCE=source["commit"],
            input_receipt=receipt, cases=sorted(reports, key=lambda x:x["case"]),
            input_snapshot_sha256=snapshot_hash, elapsed_seconds=time.monotonic()-started,
            resources_after=resource_snapshot(), scientific_superiority_claimed=False,
            new_scientific_model=False, H4_or_larger=False, automatic_next_stage=False,
            independent_scientific_review=False, old_science_retried=False))
        save(run/"STOP.json", dict(status="COMPLETE_MANDATORY_STOP", SOURCE=source["commit"],
            grant_id=grant["grant_id"], automatic_retry=False, automatic_next_stage=False,
            elapsed_seconds=time.monotonic()-started,
            result_hashes={str(p.relative_to(run)):sha(p) for p in sorted(run.rglob('*')) if p.is_file()}))
    except BaseException as exc:
        save(run/"STOP.json", dict(status="FAILED_NO_RETRY_MANDATORY_STOP", reason=str(exc),
            traceback=traceback.format_exc(), SOURCE=source["commit"], grant_id=grant["grant_id"],
            automatic_retry=False, automatic_next_stage=False, elapsed_seconds=time.monotonic()-started))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--bind", action="store_true")
    group.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    bind() if args.bind else execute()
