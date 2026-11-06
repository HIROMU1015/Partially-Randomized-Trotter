#!/usr/bin/env python3
"""One authorized B1/C1 run, independent workers, no B2/C2 or compiler."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
FAMILY = "artifacts/track_d_b1_c1_validation/2026-11-06"
OUTPUT = FAMILY + "/run01"
CARRIER = "3649b75af191916daf44abb91e14d34c43b83e8f"
STAGE_A = "artifacts/track_d_h4_paired32_reference/2026-11-06/run01"
NEW = ("src/trottertracks/track_d_b1_c1.py", "scripts/run_track_d_b1_c1.py",
       "tests/check_track_d_b1_c1.py", "docs/research/track_d_b1_c1_validation_20261106/PLAN.md")
B_INPUTS = ("artifacts/research_c_near_practical_r32/2026-11-03/run01/PR/A/identity.json",
            "artifacts/research_c_near_practical_r32/2026-11-03/run01/PR/B/references.json",
            "artifacts/research_c_near_practical_r32/2026-11-03/run01/PR/A/law.json",
            "docs/research/research_c_near_practical_r32_downstream_contract_20261103.json")


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def old_runner():
    return load("b1c1_old_paired_runner", ROOT / "scripts/run_track_d_h4_paired_reference.py")


def write(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def preflight(origin, saved):
    old = old_runner(); _, source = old.preflight(origin, saved)
    current = git("rev-parse", "HEAD").decode().strip()
    more = (*NEW, "src/trottertracks/c1_stage3_accuracy.py", "scripts/run_track_d_h4_paired_reference.py")
    for p in more:
        if (ROOT / p).read_bytes() != git("show", current + ":" + p):
            raise ValueError("UNCOMMITTED_B1_C1_SOURCE:" + p)
    bindings = {}
    for p in B_INPUTS:
        data = (ROOT / p).read_bytes()
        if data != git("show", CARRIER + ":" + p):
            raise ValueError("B1_INPUT_BINDING:" + p)
        bindings[p] = dict(commit=CARRIER, sha256=sha(data))
    root = ROOT / STAGE_A
    prior = json.loads((root / "SOURCE.json").read_text())
    stop = json.loads((root / "STOP.json").read_text())
    if (prior["execution_source"] != "2dd012dc77dcc877d5918c7d212819eb79b97557"
            or stop["status"] != "TECHNICAL_PASS_MANDATORY_STOP"
            or stop["compile_calls_started"] != 128 or not stop["mandatory_stop"]):
        raise ValueError("STAGE_A_IDENTITY_STOP")
    archived = json.loads((root / "payload_archives.json").read_text())
    archive_bindings = {}
    for i in (0, 1):
        p = f"payloads/sample{i:02d}.tar.gz"
        entry, = [x for x in archived["archives"] if x["path"] == p]
        if sha((root / p).read_bytes()) != entry["sha256"]:
            raise ValueError("SAVED_STAGE_A_ARCHIVE:" + p)
        archive_bindings[STAGE_A + "/" + p] = entry
    source.update(preparation="5f94972adf5a10e96e9400d3919c00d14e7a718a",
        new_source={p: sha((ROOT / p).read_bytes()) for p in more},
        B1_inputs=bindings, C1_stage_A_archive_bindings=archive_bindings,
        stage_A_metadata={p: sha((root / p).read_bytes()) for p in
                          ("SOURCE.json", "STOP.json", "payload_archives.json")})
    return source


def cpu_observation():
    cpus = sorted(os.sched_getaffinity(0))
    def read():
        rows = {}
        for line in Path("/proc/stat").read_text().splitlines():
            cells = line.split()
            if cells and cells[0].startswith("cpu") and cells[0][3:].isdigit():
                fields = list(map(int, cells[1:9])); rows[int(cells[0][3:])] = (sum(fields), fields[3] + fields[4])
        return rows
    before = read(); time.sleep(1); after = read()
    idle = 0.
    for cpu in cpus:
        total = after[cpu][0] - before[cpu][0]
        idle += (after[cpu][1] - before[cpu][1]) / total if total else 0.
    memory = {line.split(":")[0]: int(line.split()[1]) * 1024
              for line in Path("/proc/meminfo").read_text().splitlines()
              if line.startswith(("MemTotal:", "MemAvailable:"))}
    # Workload: one large C1 task plus a short scalar-only B1 task.
    return dict(affinity_cpus=len(cpus), observed_idle_cpu_equivalents=idle,
        numerical_threads=max(1, min(len(cpus), int(idle))), memory=memory,
        worker_processes=2, resource_caps=None, small_matrix_BLAS_threads=1)


def worker(kind, origin, saved, output, threads, stop):
    tick = time.monotonic(); out = Path(output) / kind; out.mkdir()
    records = []
    os.environ["MPLCONFIGDIR"] = str(Path(output) / "matplotlib-cache" / kind)
    try:
        if stop.is_set():
            raise RuntimeError("STOP_BEFORE_WORKER")
        base = old_runner().old_runner().bootstrap(origin)
        from trottertracks import track_d_b1_c1 as implementation
        if kind == "B1":
            result = implementation.b1(ROOT, out)
            if result["status"] != "CONDITIONAL_OPERATOR_SUFFICIENT":
                stop.set()
        else:
            checks = load("b1c1_checks", ROOT / "tests/track_d_h4_reference_checks.py")
            if stop.is_set():
                raise RuntimeError("STOP_BEFORE_C1_CORRECTNESS")
            result = implementation.c1(ROOT, saved, out, base, checks, threads)
        result.update(wall_seconds=time.monotonic() - tick,
                      peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write(out / "result.json", result)
        return kind, result
    except BaseException as error:
        stop.set()
        message = str(error)
        for path in (str(ROOT), str(origin), str(saved), str(output)):
            message = message.replace(path, "<task-path>")
        failure = dict(status="STOP_FAILURE", error_type=type(error).__name__, error=message,
                       wall_seconds=time.monotonic() - tick)
        write(out / "failure.json", failure)
        return kind, failure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", required=True, type=Path)
    parser.add_argument("--saved", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise PermissionError("B1_C1_EXPLICIT_EXECUTION_ONLY")
    output = ROOT / OUTPUT
    if output.exists():
        raise ValueError("FRESH_OUTPUT_NO_RETRY_RESUME")
    source = preflight(args.origin, args.saved)
    resources = cpu_observation()
    output.mkdir(parents=True)
    write(output / "SOURCE.json", source); write(output / "resources.json", resources)
    tick = time.monotonic(); results = {}; preserved = False
    with multiprocessing.Manager() as manager:
        stop = manager.Event()
        try:
            with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
                jobs = [pool.submit(worker, k, args.origin, args.saved, output,
                                    resources["numerical_threads"], stop) for k in ("B1", "C1")]
                for job in as_completed(jobs):
                    kind, result = job.result(); results[kind] = result
                    print(kind, result["status"], flush=True)
            if source != preflight(args.origin, args.saved):
                raise ValueError("SOURCE_INPUT_CHANGED_DURING_RUN")
            preserved = True
        except BaseException as error:
            stop.set(); results["execution_error"] = dict(error_type=type(error).__name__, error=str(error).replace(str(ROOT), "<task>"))
        passed = (results.get("B1", {}).get("status") == "CONDITIONAL_OPERATOR_SUFFICIENT"
                  and results.get("C1", {}).get("status") == "PASS" and preserved)
        result = dict(status="TECHNICAL_PASS_MANDATORY_STOP" if passed else "STOP_INCOMPLETE_OR_FAILURE",
            source_commit=source["execution_source"], results=results, source_input_preserved=preserved,
            wall_seconds=time.monotonic()-tick, compile_calls=0, price_evaluations=0,
            sampling_calls=0, ground_solves=0, DF_refits=0, quantum_shots=0, N_evaluations=0,
            B2_started=False, C2_started=False, all_round_total_cost=False,
            retry=0, resume=0, scientific_GO=False, mandatory_stop=True)
        write(output / "result.json", result)
        write(output / "STOP.json", dict(status=result["status"], mandatory_stop=True,
            B2_authorized=False, C2_authorized=False, retry=0, resume=0, compile_calls=0))
    inventory = {str(p.relative_to(output)): dict(bytes=p.stat().st_size, sha256=sha(p.read_bytes()))
                 for p in sorted(output.rglob("*")) if p.is_file() and "matplotlib-cache" not in p.parts}
    write(output / "inventory.json", inventory)
    print(result["status"], flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
