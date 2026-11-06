#!/usr/bin/env python3
"""Complete only the interrupted C3 q8 Reference-Y postcheck; never compile."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import run_track_d_c3_q4_q8 as c3

ROOT = c3.ROOT
PREVIOUS_SOURCE = "52b3cab50d5c5edcbd7e3249637168e9978be535"
INPUTS = "artifacts/track_d_c3_q4_q8/2026-11-06/run02_interrupted_inputs.json"
OUTPUT = "artifacts/track_d_c3_q4_q8/2026-11-06/run03_completion"
NEW = ("scripts/complete_track_d_c3_run02.py", "tests/check_track_d_c3_completion.py",
       "docs/research/track_d_c3_q4_q8_20261106/COMPLETION_PLAN.md", INPUTS)


def inventory(directory):
    """Bind scientific files only; the interrupted run is always read-only."""
    return {str(p.relative_to(directory)): dict(bytes=p.stat().st_size,
            sha256=c3.c2.prior().sha(p.read_bytes()))
            for p in sorted(Path(directory).rglob("*"))
            if p.is_file() and "matplotlib-cache" not in p.parts}


def validate_inputs(directory, packet):
    if inventory(directory) != packet["files"]:
        raise ValueError("INTERRUPTED_INPUT_CHANGED")
    old = Path(directory)
    if any((old / p).exists() for p in ("result.json", "STOP.json", "failure.json",
            "q8/reference_Y_check.json", "q8/reference_Y_observable.npz")):
        raise ValueError("EXPECTED_INCOMPLETE_RUN02")
    compiled = {q: {} for q in c3.QS}
    for q in c3.QS:
        out = old / f"q{q}"
        if json.loads((out / "precompile_checks.json").read_text())["status"] != "PASS":
            raise ValueError("RAW_NOT_PASS")
        for arm in c3.c2.ARMS:
            for axis in c3.c2.AXES:
                name = arm + "_" + axis
                marker = json.loads((out / (name + "_compile.json")).read_text())
                if marker["status"] != "PASS" or marker["compile_calls"] != 1:
                    raise ValueError("EIGHT_EXISTING_COMPILES_REQUIRED")
                if q == 8 and name == "reference_Y":
                    continue
                report = json.loads((out / (name + "_check.json")).read_text())
                if report["status"] != "PASS":
                    raise ValueError("INHERITED_POSTCHECK_NOT_PASS")
                compiled[q][name] = report
    return compiled


def preflight(origin, saved):
    packet = json.loads((ROOT / INPUTS).read_text())
    previous = ROOT / packet["previous_output"]
    compiled = validate_inputs(previous, packet)
    source = c3.preflight(origin, saved)
    current = source["execution_source"]
    prior_source = json.loads((previous / "SOURCE.json").read_text())
    comparable = dict(source, execution_source=PREVIOUS_SOURCE)
    if prior_source["execution_source"] != PREVIOUS_SOURCE or comparable != prior_source:
        raise ValueError("ORIGINAL_SOURCE_INPUT_ENVIRONMENT_CHANGED")
    for name in NEW:
        if (ROOT / name).read_bytes() != c3.c2.prior().git("show", current + ":" + name):
            raise ValueError("UNCOMMITTED_COMPLETION_SOURCE:" + name)
    source.update(completion_source={n: c3.c2.prior().sha((ROOT/n).read_bytes()) for n in NEW},
        inherited_source=PREVIOUS_SOURCE, inherited_inputs=packet,
        completion_policy="one_saved_q8_reference_Y_postcheck_no_compile_then_final_checks")
    return source, packet, compiled


def materialize(previous, output, packet):
    """Copy byte-bound q4/q8 data; never write back into the original run."""
    for name in packet["files"]:
        relative = Path(name)
        if relative.parts[0] not in ("q4", "q8"):
            continue
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as f:
            f.write((previous / relative).read_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", type=Path, required=True)
    parser.add_argument("--saved", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise PermissionError("COMPLETION_EXPLICIT_EXECUTION_ONLY")
    output = ROOT / OUTPUT
    if output.exists():
        raise ValueError("FRESH_COMPLETION_OUTPUT_NO_RETRY")
    source, packet, compiled = preflight(args.origin, args.saved)
    observation = c3.c2.prior().cpu_observation()
    threads = observation["numerical_threads"]
    observation.update(worker_processes=1, remaining_postcheck_tasks=1,
        reference_postcheck_threads=threads,
        policy="one missing task; use observed idle CPUs, no arbitrary worker/thread cap")
    output.mkdir(parents=True)
    write = c3.c2.old().write  # existing atomic JSON writer
    write(output/"SOURCE.json", source)
    write(output/"resources.json", observation)
    tick = time.monotonic()
    success = preserved = False
    os.environ["QISKIT_PARALLEL"] = "false"
    os.environ["RAYON_NUM_THREADS"] = "1"
    os.environ["MPLCONFIGDIR"] = str(output/"matplotlib-cache"/"completion")
    try:
        materialize(ROOT/packet["previous_output"], output, packet)
        write(output/"progress.json", dict(status="POSTCHECK_STARTED", q=8,
            arm="reference", axis="Y", inherited_postchecks=7, new_compile_calls=0))
        print("q8 reference_Y saved-circuit postcheck STARTED", flush=True)
        name, report = c3.c2.old().verify_one((args.origin, output/"q8", "reference", "Y", threads))
        if report["status"] != "PASS":
            raise ValueError("LAST_POSTCHECK_NOT_PASS")
        compiled[8][name] = report
        if any(len(rows) != 4 or any(r["status"] != "PASS" for r in rows.values())
               for rows in compiled.values()):
            raise ValueError("EIGHT_WRAPPER_POSTCHECKS_REQUIRED")
        print("q8 reference_Y postcheck PASS; final paired/phase checks", flush=True)
        for q in c3.QS:
            c3.paired_postcheck(output, q)
        saved = json.loads((ROOT/c3.C2_OUTPUT/"resource_comparison.json").read_text())
        write(output/"resource_comparison.json", c3.comparisons(compiled, saved))
        if source != preflight(args.origin, args.saved)[0]:
            raise ValueError("SOURCE_INPUT_CHANGED")
        for name, item in packet["files"].items():
            if Path(name).parts[0] in ("q4", "q8"):
                data = (output/name).read_bytes()
                if len(data) != item["bytes"] or c3.c2.prior().sha(data) != item["sha256"]:
                    raise ValueError("INHERITED_DATA_CHANGED:" + name)
        success = preserved = True
    except BaseException as error:
        message = str(error)
        for p in (str(ROOT), str(args.origin), str(args.saved)):
            message = message.replace(p, "<task-path>")
        write(output/"failure.json", dict(status="STOP", error_type=type(error).__name__, error=message))
    result = dict(status="TECHNICAL_PASS_MANDATORY_STOP" if success else "STOP_FAILURE",
        source_commit=source["execution_source"], inherited_source=PREVIOUS_SOURCE,
        inherited_raw_results={str(q): json.loads((ROOT/packet["previous_output"]/f"q{q}"/"precompile_checks.json").read_text()) for q in c3.QS},
        compiled={str(q): r for q,r in compiled.items()}, inherited_postchecks=7,
        new_postchecks_started=1, inherited_compile_calls=8, new_compile_calls=0,
        new_circuit_builds=0, source_input_preserved=preserved,
        wall_seconds=time.monotonic()-tick, sampling_calls=0, ground_solves=0,
        DF_refits=0, N_evaluations=0, quantum_shots=0, B2_started=False,
        q_above8=False, RPE_total_cost_evaluated=False, whole_law_qualified=False,
        scientific_GO=False, resume=1, retry=0, mandatory_stop=True)
    write(output/"result.json", result)
    write(output/"STOP.json", dict(status=result["status"], mandatory_stop=True,
        inherited_compile_calls=8, new_compile_calls=0, resume=1, retry=0,
        next_stage_authorized=False))
    write(output/"progress.json", dict(status=result["status"], completed_postchecks=sum(map(len, compiled.values()))))
    write(output/"inventory.json", inventory(output))
    print(result["status"], flush=True)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
