#!/usr/bin/env python3
"""Artificial concurrency preflight, then separately bound one-shot H2 cases."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
import threading
import traceback

from trotterlib.isolated_qiskit_execution import (
    artificial_case, await_release, compare_artificial, read, run_processes, save, sha,
)
from trotterlib import h2_non_df_process_recovery as recovery

ROOT, BASE = recovery.ROOT, recovery.BASE
ARTIFICIAL_CASES = ("pauli_jw_PF", "pauli_jw_PR", "df_native_PF", "df_native_PR")


def thread_diagnostic(destination):
    destination.mkdir()
    barrier = threading.Barrier(4)

    def task(name):
        output = destination/name
        output.mkdir()
        try:
            barrier.wait()
            artificial_case(name, output)
            return dict(case=name, status="PASS")
        except BaseException as exc:
            report = dict(case=name, status="FAILED", reason=str(exc), traceback=traceback.format_exc())
            save(output/"failure.json", report)
            return report

    with ThreadPoolExecutor(max_workers=len(ARTIFICIAL_CASES)) as pool:
        reports = list(pool.map(task, ARTIFICIAL_CASES))
    import qiskit.circuit.add_control as control_module
    # importlib avoids the public add_control function shadowing the module.
    import importlib
    control_module = importlib.import_module("qiskit.circuit.add_control")
    from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
    save(destination/"diagnostic.json", dict(pid=os.getpid(), reports=reports,
        borrow_exception_reproduced=any("borrow" in r.get("reason","").lower() for r in reports),
        add_control_uses_process_global_session=control_module.sel is SessionEquivalenceLibrary,
        scientific_cases_executed=0, Qiskit_installation_changed=False))


def preflight():
    technical = BASE/"technical"
    technical.mkdir(parents=True)
    try:
        # A bounded diagnostic workload, not a retry of any science case.
        with (technical/"thread.stdout.log").open("x") as stdout, (technical/"thread.stderr.log").open("x") as stderr:
            subprocess.run([sys.executable, __file__, "--thread-diagnostic", "--output", str(technical/"threads")],
                           check=True, stdout=stdout, stderr=stderr, cwd=ROOT)
        sequential, parallel = {}, {}
        for mode, names in [("sequential", ARTIFICIAL_CASES), ("parallel", ARTIFICIAL_CASES)]:
            folder = technical/mode
            folder.mkdir()
            batches = [(name,) for name in names] if mode=="sequential" else [names]
            for index, batch in enumerate(batches):
                gate = folder/f"release{index}.json"
                jobs = []
                for name in batch:
                    output = folder/name
                    output.mkdir()
                    jobs.append(dict(case=name, output=output, argv=[sys.executable, __file__,
                        "--artificial", name, "--output", str(output), "--gate", str(gate)]))
                reports = run_processes(jobs, gate, cwd=ROOT)
                if any(r["exit_code"] != 0 for r in reports):
                    raise RuntimeError("ARTIFICIAL_PROCESS_FAILURE")
            target = sequential if mode=="sequential" else parallel
            target.update({name:read(folder/name/"result.json") for name in names})
        result = compare_artificial(sequential, parallel)
        result["thread_diagnostic"] = read(technical/"threads/diagnostic.json")
        result["source_files"] = {name:sha(ROOT/name) for name in [
            "src/trotterlib/isolated_qiskit_execution.py", "src/trotterlib/h2_non_df_process_recovery.py",
            "scripts/run_h2_non_df_process_recovery.py"]}
        save(technical/"preflight.json", result)
        print(result["status"], "four-way process overlap", result["overlap_seconds"],
              "thread borrow reproduced", result["thread_diagnostic"]["borrow_exception_reproduced"], flush=True)
    except BaseException as exc:
        save(technical/"STOP.json", dict(status="TECHNICAL_FAILURE_NO_SCIENCE", reason=str(exc), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--thread-diagnostic", action="store_true")
    group.add_argument("--artificial", choices=ARTIFICIAL_CASES)
    group.add_argument("--worker", choices=recovery.CASES)
    group.add_argument("--bind", action="store_true")
    group.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gate", type=Path)
    args = parser.parse_args()
    if args.preflight:
        preflight()
    elif args.thread_diagnostic:
        thread_diagnostic(args.output)
    elif args.artificial:
        await_release(args.output, args.gate)
        artificial_case(args.artificial, args.output)
    elif args.worker:
        recovery.worker(args.worker, args.gate)
    elif args.bind:
        recovery.bind()
    else:
        recovery.execute()
