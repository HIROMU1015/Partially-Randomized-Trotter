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
    time.sleep(8)
    write(path / "WORKER_RESULT.json", {"synthetic_value": sum(range(5)), "sequence": 1})


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
    complete = (work / "WORKER_RESULT.json").is_file()
    started = (work / "WORKER_STARTED.json").is_file()
    state = "EVALUATION_COMPLETED" if complete else "EVALUATION_RUNNING" if started else "EVALUATION_WAITING"
    status = "STARTED" if action == "start" and not exists else "ALREADY_RUNNING" if exists and not complete else action.upper()
    budgets = data["scope"].get("preparation_budgets", {"signal": 3, "compile": 2, "audit": 1})
    return dict(ok=True, result=dict(
        run_id=run_id, status=status, runtime=dict(phase="COMPLETE" if complete else "RUNNING"),
        applied=dict(requested=data["scope"]), synthetic_launch_count=data["count"],
        received_request_sha256=data["request_sha256"],
        progress=dict(schema="RPE_LOCAL0_PROGRESS_V1",
            effective=dict(preparation_budgets=budgets, resources=dict(max_concurrent_jobs="auto")),
            candidates=[
                dict(candidate_id="saved", method="PR_PAULI", state="REUSED_SAVED_RESULT"),
                dict(candidate_id="new-missing-input", method="PR_PAULI", state="PREPARATION_WAITING",
                     reason="SIGNAL_PACKET_MISSING"),
                dict(candidate_id="new-inputs-ready", method="PR_PAULI", state=state),
            ],
            stages=[
                dict(stage="compile", queue_accepted=8, execution_started=0, execution_completed=0, execution_unit="compile"),
                dict(stage="q3", queue_accepted=data["count"], execution_started=int(started),
                     execution_completed=int(complete), execution_unit="artificial_job"),
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
