#!/usr/bin/env python3
"""Fixed C3 q4/q8: raw checks, at most eight P0 compiles/postchecks, STOP."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
from pathlib import Path
import resource
import time

import run_track_d_c2_q2 as c2

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = "artifacts/track_d_c3_q4_q8/2026-11-06/run01"
C2_COMMIT = "c8fe5091c42ec786688b6931105ad64d10c87d18"
C2_OUTPUT = "artifacts/track_d_c2_q2_compile/2026-11-06/run01"
NEW = ("src/trottertracks/track_d_c3_whole_path.py", "scripts/run_track_d_c3_q4_q8.py",
       "tests/check_track_d_c3.py", "docs/research/track_d_c3_q4_q8_20261106/PLAN.md")
QS = (4, 8)


def preflight(origin, saved):
    source, _, _ = c2.preflight(origin, saved)
    p = c2.prior(); head = p.git("rev-parse", "HEAD").decode().strip()
    for name in NEW:
        if (ROOT / name).read_bytes() != p.git("show", head + ":" + name):
            raise ValueError("UNCOMMITTED_C3_SOURCE:" + name)
    prior = {}
    for leaf in ("SOURCE.json", "STOP.json", "result.json", "resource_comparison.json"):
        name = C2_OUTPUT + "/" + leaf; data = (ROOT / name).read_bytes()
        if data != p.git("show", C2_COMMIT + ":" + name):
            raise ValueError("C2_SAVED_BINDING:" + leaf)
        prior[name] = p.sha(data)
    stop = json.loads((ROOT / C2_OUTPUT / "STOP.json").read_text())
    if stop["status"] != "TECHNICAL_PASS_MANDATORY_STOP" or not stop["mandatory_stop"]:
        raise ValueError("C2_SAVED_STOP")
    source.update(C3_source={n:p.sha((ROOT/n).read_bytes()) for n in NEW},
                  C2_result=C2_COMMIT, C2_saved=prior,
                  C3_policy="fixed_q4_q8_global_half_step_cut_legal_boundary_fusion_max8_P0")
    return source


def raw_worker(q, origin, saved, output, threads, stop):
    tick = time.monotonic(); out = Path(output) / f"q{q}"; out.mkdir()
    os.environ["MPLCONFIGDIR"] = str(out / "matplotlib-cache" / "raw")
    try:
        if stop.is_set(): raise RuntimeError("STOP_BEFORE_RAW")
        runner = c2.old(); runner.bootstrap(origin)
        from trottertracks import track_d_c3_whole_path as implementation
        contract = json.loads((Path(origin) / runner.CONTRACT).read_text())
        checks = c2.load("c3_raw_checks", ROOT / "tests/track_d_h4_reference_checks.py")
        value = implementation.raw_checks(origin, saved, contract, out, q, checks, threads)
        value.update(wall_seconds=time.monotonic()-tick,
                     peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        c2.write(out / "precompile_checks.json", value)
        return q, value
    except BaseException as error:
        stop.set()
        message = str(error)
        for path in (str(ROOT), str(origin), str(saved)): message=message.replace(path,"<task-path>")
        value = dict(status="STOP_FAILURE", error_type=type(error).__name__, error=message,
                     wall_seconds=time.monotonic()-tick)
        c2.write(out / "failure.json", value)
        return q, value


def compiled_worker(q, task, stop):
    name, report = c2.worker(task, stop)
    return q, name, report


def paired_postcheck(output, q):
    import numpy as np
    from threadpoolctl import threadpool_limits
    from trottertracks.c1_stage3_molecular import component_check
    from trottertracks import track_d_h4_reference as base
    out = Path(output) / f"q{q}"; observables = {}; rows = {}
    for arm in c2.ARMS:
        for axis in c2.AXES:
            with np.load(out / (arm+"_"+axis+"_observable.npz"), allow_pickle=False) as z:
                observables[arm,axis] = z["observable"].copy()
    chi = json.loads((out / "conditions.json").read_text())["phase_chi"]
    with threadpool_limits(limits=1):
        for axis in c2.AXES:
            row = component_check(observables["reference",axis], observables["symmetric_directional",axis], base.ATOL, label="compiled_paired_"+axis)
            if row["frobenius"] > base.SIGNAL_ATOL: raise ValueError("COMPILED_PAIRED_SIGNAL_ALLOWANCE")
            rows[axis] = row
        for arm in c2.ARMS:
            value = observables[arm,"X"][0,0] + 1j*observables[arm,"Y"][0,0]
            rows[arm+"_vacuum_chi"] = component_check(np.array([[value]]), np.array([[np.exp(1j*chi)]]), base.ATOL, label="compiled_vacuum_chi_"+arm)
    c2.write(out / "paired_observables.json", rows)


def comparisons(compiled, saved):
    result = dict(saved_q1_q2=saved, current={},
        qualification="fixed nested fixtures; path, cut and fusion differ; no isolated q effect or law inference",
        C2_cross_step_manual_fusion=False, C3_cross_step_manual_fusion=True,
        RZ_to_T_conversion=False, topology_free_depth_is_runtime=False)
    for q in QS:
        result["current"][str(q)] = c2.comparisons(compiled[q], [])["q2_paired"]
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", type=Path, required=True); parser.add_argument("--saved", type=Path, required=True)
    parser.add_argument("--execute", action="store_true"); args=parser.parse_args()
    if not args.execute: raise PermissionError("C3_EXPLICIT_EXECUTION_ONLY")
    output = ROOT / OUTPUT
    if output.exists(): raise ValueError("FRESH_OUTPUT_NO_RETRY_RESUME")
    source = preflight(args.origin, args.saved)
    observation = c2.prior().cpu_observation(); budget = observation["numerical_threads"]
    previous = json.loads((ROOT / C2_OUTPUT / "result.json").read_text())
    memory_estimate = max(v["peak_RSS_KiB"]*1024 for v in previous["compiled"].values())
    memory_slots = max(1, observation["memory"]["MemAvailable"] // memory_estimate)
    raw_workers = min(2, budget, memory_slots); compile_workers = min(8, budget, memory_slots)
    raw_threads = max(1, budget // raw_workers)
    reference_threads = max(1, (budget-min(4,compile_workers)) // min(4,compile_workers))
    observation.update(raw_workers=raw_workers, raw_threads=raw_threads, compiler_workers=compile_workers,
        reference_postcheck_threads=reference_threads, direct_postcheck_threads=1,
        saved_C2_peak_RSS_bytes_estimate=memory_estimate,
        policy="independent fixed tasks; observed idle CPU and free RAM, no imposed resource cap")
    output.mkdir(parents=True); c2.write(output/"SOURCE.json",source); c2.write(output/"resources.json",observation)
    raw = {}; compiled = {q:{} for q in QS}; success=False; preserved=False; tick=time.monotonic()
    os.environ["QISKIT_PARALLEL"]="false"; os.environ["RAYON_NUM_THREADS"]="1"
    with multiprocessing.Manager() as manager:
        stop = manager.Event()
        try:
            with ProcessPoolExecutor(max_workers=raw_workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                futures=[pool.submit(raw_worker,q,args.origin,args.saved,output,raw_threads,stop) for q in QS]
                for future in as_completed(futures):
                    q, row=future.result(); raw[q]=row; print(f"q{q} raw {row['status']}",flush=True)
            if any(raw[q]["status"]!="PASS" for q in QS): raise ValueError("RAW_CORRECTNESS_STOP_NO_COMPILE")
            tasks=[(q,(args.origin,output/f"q{q}",arm,axis,reference_threads if arm=="reference" else 1))
                   for q in QS for arm in c2.ARMS for axis in c2.AXES]
            with ProcessPoolExecutor(max_workers=compile_workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                for future in as_completed([pool.submit(compiled_worker,q,t,stop) for q,t in tasks]):
                    q,name,row=future.result(); compiled[q][name]=row; print(f"q{q} {name} {row['status']}",flush=True)
                    c2.write(output/"progress.json",{str(k):{n:r["status"] for n,r in v.items()} for k,v in compiled.items()})
            if any(len(v)!=4 or any(r["status"]!="PASS" for r in v.values()) for v in compiled.values()):
                raise ValueError("EIGHT_WRAPPER_POSTCHECKS_REQUIRED")
            os.environ["MPLCONFIGDIR"]=str(output/"matplotlib-cache"/"parent")
            c2.old().bootstrap(args.origin)
            for q in QS: paired_postcheck(output,q)
            saved=json.loads((ROOT/C2_OUTPUT/"resource_comparison.json").read_text())
            c2.write(output/"resource_comparison.json",comparisons(compiled,saved))
            if source != preflight(args.origin,args.saved): raise ValueError("SOURCE_INPUT_CHANGED")
            preserved=True; success=True
        except BaseException as error:
            stop.set(); c2.write(output/"failure.json",dict(status="STOP",error_type=type(error).__name__,error=str(error).replace(str(ROOT),"<task>")))
        count=len(list(output.glob("q*/*_compile.json")))
        result=dict(status="TECHNICAL_PASS_MANDATORY_STOP" if success else "STOP_FAILURE",
            source_commit=source["execution_source"], raw_results=raw, compiled=compiled, compile_calls_started=count,
            source_input_preserved=preserved, wall_seconds=time.monotonic()-tick,
            sampling_calls=0,ground_solves=0,DF_refits=0,N_evaluations=0,quantum_shots=0,
            B2_started=False,q_above8=False,RPE_total_cost_evaluated=False,whole_law_qualified=False,
            scientific_GO=False,retry=0,resume=0,mandatory_stop=True)
        c2.write(output/"result.json",result)
        c2.write(output/"STOP.json",dict(status=result["status"],mandatory_stop=True,compile_calls_started=count,
                                      retry=0,resume=0,next_stage_authorized=False))
    inventory={str(p.relative_to(output)):dict(bytes=p.stat().st_size,sha256=c2.prior().sha(p.read_bytes()))
               for p in sorted(output.rglob("*")) if p.is_file() and "matplotlib-cache" not in p.parts}
    c2.write(output/"inventory.json",inventory); print(result["status"],flush=True)
    return 0 if success else 1


if __name__=="__main__": raise SystemExit(main())
