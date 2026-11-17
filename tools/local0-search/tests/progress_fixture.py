"""Artificial SSH bootstrap for the Windows client interface acceptance."""

import argparse
import base64
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time


def write(path, body):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(body))
    temp.replace(path)


def step(path):
    time.sleep(1)
    write(path / "WORKER_STARTED.json", {"sequence": 1})
    write(path / "PREPARATION_STARTED.json", {"sequence": 2})
    time.sleep(2)
    write(path / "WORKER_RESULT.json", {"synthetic_value": sum(range(5)), "sequence": 3})
    time.sleep(2)
    result = path / "PREPARATION_RESULT.json"
    write(result, {"packet": [1, 2, 3], "sequence": 4})
    packet_hash = hashlib.sha256(result.read_bytes()).hexdigest()
    if json.loads(result.read_bytes())["packet"] != [1, 2, 3]:
        raise ValueError("Unexpected artificial preparation result")
    write(path / "PREPARATION_INTAKE.json", {"input_sha256": packet_hash, "sequence": 5})
    time.sleep(1)
    write(path / "PREPARED_EVALUATION_STARTED.json", {"input_sha256": packet_hash, "sequence": 6})
    time.sleep(1)
    write(path / "PREPARED_EVALUATION_RESULT.json", {"input_sha256": packet_hash, "sequence": 7})


def receipt(path, name):
    location = path / (name + ".json")
    return json.loads(location.read_bytes()) if location.is_file() else None


def evaluation_state(started, complete):
    if complete:
        return "EVALUATION_COMPLETED"
    return "EVALUATION_RUNNING" if started else "EVALUATION_WAITING"


def serve(transport):
    namespace = Path(__file__).resolve().parent
    request_path = Path(transport["server_config"]).resolve()
    if request_path != namespace / "ARTIFICIAL_STATE.json":
        raise ValueError("Fixture only accepts its own artificial state")
    raw = base64.b64decode(transport["request_base64"])
    request = json.loads(raw)
    action = transport["action"]
    supplied = transport.get("run_id")
    if supplied and not re.fullmatch(r"fixture-(?:failed|[0-9a-f]{20})", supplied):
        raise ValueError("Invalid artificial fixture run")
    if supplied == "fixture-failed":
        if action == "resume":
            return dict(ok=True, result=dict(run_id="fixture-continuation", status="STARTED",
                        runtime=dict(phase="RUNNING"), continuation=dict(parent_run_id=supplied)))
        return dict(ok=True, result=dict(run_id=supplied, status="REUSED_COMPLETED",
                    runtime=dict(phase="FAILED", stop_reason="RUNTIME_EXCEPTION")))
    run_id = supplied or "fixture-" + hashlib.sha256(raw).hexdigest()[:20]
    work = namespace / run_id
    with (namespace / "fixture.lock").open("ab") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        exists = (work / "START.json").exists()
        if not exists and action == "start":
            work.mkdir()
            scope = request["scope"]
            write(work / "START.json", dict(scope=scope, count=1,
                  request_sha256=hashlib.sha256(raw).hexdigest()))
            subprocess.Popen([sys.executable, "-B", __file__, "--work", str(work)],
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, start_new_session=True)
        data = json.loads((work / "START.json").read_bytes()) if (work / "START.json").exists() else {
            "scope": request.get("scope", {}), "count": 0, "request_sha256": hashlib.sha256(raw).hexdigest(),
        }
    prepared_complete = receipt(work, "PREPARED_EVALUATION_RESULT")
    prepared_started = receipt(work, "PREPARED_EVALUATION_STARTED")
    intake = receipt(work, "PREPARATION_INTAKE")
    prepared = (work / "PREPARATION_RESULT.json").is_file()
    preparation_started = (work / "PREPARATION_STARTED.json").is_file()
    ready_complete = receipt(work, "WORKER_RESULT")
    started = (work / "WORKER_STARTED.json").is_file()
    complete = bool(ready_complete and prepared_complete)
    state = evaluation_state(started, ready_complete)
    preparation_state = "PREPARATION_RUNNING" if preparation_started else "PREPARATION_WAITING"
    if intake:
        preparation_state = evaluation_state(prepared_started, prepared_complete)
    status = "STARTED" if action == "start" and not exists else "ALREADY_RUNNING" if exists and not complete else action.upper()
    budgets = data["scope"].get("preparation_budgets", {"signal": 3, "compile": 2, "audit": 1})
    return dict(ok=True, result=dict(
        run_id=run_id, status=status, runtime=dict(phase="COMPLETE" if complete else "RUNNING"),
        applied=dict(requested=data["scope"]), synthetic_launch_count=data["count"],
        synthetic_receipts=dict(ready_evaluation=ready_complete, preparation_intake=intake,
                                prepared_evaluation=prepared_complete),
        received_request_sha256=data["request_sha256"],
        progress=dict(schema="RPE_LOCAL0_PROGRESS_V1",
            effective=dict(preparation_budgets=budgets, resources=dict(max_concurrent_jobs="auto")),
            candidates=[
                dict(candidate_id="saved", method="PR_PAULI", state="REUSED_SAVED_RESULT"),
                dict(candidate_id="new-missing-input", method="PR_PAULI", state="PREPARATION_WAITING",
                     reason="SIGNAL_PACKET_MISSING"),
                dict(candidate_id="new-inputs-ready", method="PR_PAULI", state=state),
                dict(candidate_id="new-inputs-prepared", method="PR_PAULI", state=preparation_state),
            ],
            stages=[
                dict(stage="compile", queue_accepted=8, execution_started=0, execution_completed=0, execution_unit="compile"),
                dict(stage="signal", queue_accepted=data["count"], execution_started=int(preparation_started),
                     execution_completed=int(prepared), execution_unit="artificial_job"),
                dict(stage="q3", queue_accepted=data["count"] + int(bool(intake)),
                     execution_started=int(started) + int(bool(prepared_started)),
                     execution_completed=int(bool(ready_complete)) + int(bool(prepared_complete)),
                     execution_unit="artificial_job"),
            ]),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport-base64")
    parser.add_argument("--work", type=Path)
    arguments = parser.parse_args()
    if arguments.work:
        step(arguments.work)
    else:
        print(json.dumps(serve(json.loads(base64.b64decode(arguments.transport_base64)))))
