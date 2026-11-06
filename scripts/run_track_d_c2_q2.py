#!/usr/bin/env python3
"""Four fixed C1 q2 wrappers: existing P0 compiler/checks, then STOP."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import importlib.util
import json
import math
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
C1_COMMIT = "d3913866e6ade4ab97ac02645fed91b2d5fcfa75"
C1 = "artifacts/track_d_b1_c1_validation/2026-11-06/run01"
Q1 = "artifacts/track_d_h4_paired32_reference/2026-11-06/run01/paired_resources.csv"
OUTPUT = "artifacts/track_d_c2_q2_compile/2026-11-06/run01"
NEW = ("scripts/run_track_d_c2_q2.py", "tests/check_track_d_c2_q2.py",
       "docs/research/track_d_c2_q2_compile_20261106/PLAN.md")
ARMS = ("symmetric_directional", "reference")
AXES = ("X", "Y")
METRICS = ("rz_count", "cx_count", "rz_depth", "cx_depth", "total_depth", "qubit_count", "circuit_size")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module); return module


def prior():
    return load("c2_prior", ROOT / "scripts/run_track_d_b1_c1.py")


def old():
    return prior().old_runner().old_runner()


def write(path, value):
    prior().write(path, value)


def preflight(origin, saved):
    p = prior(); source = p.preflight(origin, saved)
    current = p.git("rev-parse", "HEAD").decode().strip()
    for name in NEW:
        if (ROOT/name).read_bytes() != p.git("show", current+":"+name):
            raise ValueError("UNCOMMITTED_C2_SOURCE:"+name)
    records = {}
    for leaf in ("SOURCE.json", "STOP.json", "C1/result.json", "payload_archive.json", "C1_payload.tar.gz"):
        name = C1+"/"+leaf; data = (ROOT/name).read_bytes()
        if data != p.git("show", C1_COMMIT+":"+name):
            raise ValueError("C1_RESULT_BINDING:"+leaf)
        records[name] = dict(commit=C1_COMMIT, sha256=p.sha(data))
    c1 = json.loads((ROOT/C1/"C1/result.json").read_text())
    stop = json.loads((ROOT/C1/"STOP.json").read_text())
    condition = c1["conditions"]
    if (c1["status"] != "PASS" or not stop["mandatory_stop"]
            or condition["ordered_samples"] != [0,1] or condition["q"] != 2
            or condition["event_count"] != 64 or condition["h"] != "1/10"
            or condition["phase_chi"] != -.4574879096539928
            or condition["scalar_identity_occurrences"] != 2
            or condition["reference_gadgets"] != 1 or condition["short_gadget_repetition"]):
        raise ValueError("FIXED_C1_Q2_CONDITIONS")
    index = json.loads((ROOT/C1/"payload_archive.json").read_text())
    if p.sha((ROOT/C1/index["archive"]).read_bytes()) != index["sha256"]:
        raise ValueError("C1_ARCHIVE_HASH")
    if (ROOT/Q1).read_bytes() != p.git("show", "3dc9a61ee6defa3ffa8cda8601f87253cdd7f8b0:"+Q1):
        raise ValueError("Q1_SAVED_PRICES")
    source.update(C1_result=C1_COMMIT, C1_science_source="8c6eeb02e89f6cf14870789719493e75482f62f8",
        C1_inputs=records, C2_source={name:p.sha((ROOT/name).read_bytes()) for name in NEW},
        q1_saved_prices=dict(path=Q1, sha256=p.sha((ROOT/Q1).read_bytes())),
        C2_policy="exact_C1_raw_bytes_four_P0_wrappers_no_rebuild")
    return source, c1, index


def member_destination(name):
    p = Path(name)
    if p.is_absolute() or ".." in p.parts or p.parts[:1] != ("C1",):
        raise ValueError("UNEXPECTED_C1_MEMBER")
    relative = p.relative_to("C1")
    if relative != Path("references.npz") and (relative.parent != Path("circuits") or relative.suffix != ".qpy"):
        raise ValueError("UNEXPECTED_C1_MEMBER")
    return relative


def materialize(output, index):
    output = Path(output); (output/"circuits").mkdir()
    with tarfile.open(ROOT/C1/index["archive"], "r:gz") as archive:
        for item in index["members"]:
            relative = member_destination(item["path"])
            data = archive.extractfile(item["path"]).read()
            if prior().sha(data) != item["sha256"] or len(data) != item["bytes"]:
                raise ValueError("C1_MEMBER_HASH:"+str(relative))
            with (output/relative).open("xb") as f: f.write(data)


def raw_small_checks(output, condition, base):
    import numpy as np
    from trottertracks.c1_stage3_molecular import component_check
    from threadpoolctl import threadpool_limits
    records=[]
    def same(a,b,label):
        row=component_check(a,b,base.ATOL,label=label);records.append(row)
    with np.load(output/"references.npz",allow_pickle=False) as z:
        a,b,adj=[base.small_operator(base.load_qpy(output/"circuits"/(name+".qpy")))
                 for name in ("A","B","A_actual_adjoint")]
        with threadpool_limits(limits=1):
            same(a@b,z["V"],"C1_AB_order_scalar_twice")
            same(adj,a.conj().T,"C1_actual_adjoint_phase")
            same(adj@a,np.eye(256),"C1_actual_adjoint_identity")
        vacuum=np.zeros((256,1),complex);vacuum[0,0]=np.exp(1j*condition["phase_chi"])
        same(z["V"][:,:1],vacuum,"C1_vacuum_chi")
        for label,matrix in (("A",a),("B",b)):
            ray=np.zeros((256,1),complex);ray[0,0]=matrix[0,0]
            same(matrix[:,:1],ray,label+"_vacuum_ray")
        expected=np.zeros((512,512),complex);expected[:256,:256]=np.eye(256);expected[256:,256:]=z["V"]
        same(base.small_operator(base.load_qpy(output/"circuits/symmetric_directional_core.qpy")),expected,
             "strong_symmetric_both_branches")
        same(z["A"],a,"saved_A");same(z["B"],b,"saved_B");same(z["A_adjoint"],adj,"saved_actual_adjoint")
    if math.fsum(condition["per_step_chi"]) != condition["phase_chi"]:
        raise ValueError("C1_PHASE_METADATA_CHANGED")
    return dict(status="PASS",checks=records,
        raw_reference_checks="inherited22+paired_XY_PASS; exact byte-bound C1 core/wrappers/reference",
        chi=condition["phase_chi"], rebuilt_circuits=0)


def worker(task, stop):
    origin,output,arm,axis,threads=task
    os.environ["MPLCONFIGDIR"]=str(Path(output)/"matplotlib-cache"/(arm+"_"+axis))
    try:
        if stop.is_set(): raise RuntimeError("STOP_BEFORE_COMPILE")
        runner=old();runner.bootstrap(origin)
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1): runner.compile_one(task)
        if stop.is_set(): raise RuntimeError("STOP_BEFORE_POSTCHECK")
        return runner.verify_one(task)
    except BaseException as error:
        stop.set()
        message=str(error).replace(str(ROOT),"<task>").replace(str(origin),"<original SOURCE>")
        write(Path(output)/(arm+"_"+axis+"_failure.json"),dict(status="FAIL",error_type=type(error).__name__,error=message))
        return arm+"_"+axis,dict(status="FAIL",error_type=type(error).__name__,error=message)


def comparisons(compiled, q1):
    rows=[]
    for axis in AXES:
        s=compiled["symmetric_directional_"+axis]["metrics"]
        r=compiled["reference_"+axis]["metrics"]
        rows.append(dict(axis=axis,symmetric={k:s[k] for k in METRICS},
                         reference={k:r[k] for k in METRICS},delta={k:r[k]-s[k] for k in METRICS}))
    return dict(q2_paired=rows,q1_saved_individual_rows=q1,
        q1_cut="complete event16 in each single h-step",q2_cut="complete first h-step",
        qualification="one fixed q2 fixture; no population estimate or isolated q/cut effect",
        sum_of_q1_wrapper_prices_is_q2_price=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin",type=Path,required=True);parser.add_argument("--saved",type=Path,required=True)
    parser.add_argument("--execute",action="store_true");args=parser.parse_args()
    if not args.execute:raise PermissionError("C2_EXPLICIT_EXECUTION_ONLY")
    output=ROOT/OUTPUT
    if output.exists():raise ValueError("FRESH_OUTPUT_NO_RETRY_RESUME")
    source,c1,index=preflight(args.origin,args.saved)
    resources=prior().cpu_observation();budget=resources["numerical_threads"]
    workers=min(4,budget);reference_threads=budget if workers==1 else max(1,(budget-min(2,workers))//2)
    resources.update(worker_processes=workers,wrapper_tasks=4,reference_postcheck_threads=reference_threads,
        direct_postcheck_threads=1,policy="four independent compiler/check tasks; split observed CPU budget")
    output.mkdir(parents=True);write(output/"SOURCE.json",source);write(output/"resources.json",resources)
    write(output/"conditions.json",c1["conditions"]);write(output/"inherited_C1_checks.json",c1)
    tick=time.monotonic();results={};success=False;preserved=False
    with multiprocessing.Manager() as manager:
        stop=manager.Event()
        try:
            materialize(output,index)
            os.environ["MPLCONFIGDIR"]=str(output/"matplotlib-cache"/"parent")
            base=old().bootstrap(args.origin)
            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=1): check=raw_small_checks(output,c1["conditions"],base)
            write(output/"precompile_checks.json",check)
            print("raw identity/phase/adjoint checks PASS",flush=True)
            tasks=[(args.origin,output,arm,axis,reference_threads if arm=="reference" else 1) for arm in ARMS for axis in AXES]
            os.environ["QISKIT_PARALLEL"]="false";os.environ["RAYON_NUM_THREADS"]="1"
            with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn")) as pool:
                for future in as_completed([pool.submit(worker,t,stop) for t in tasks]):
                    name,row=future.result();results[name]=row;print(name,row["status"],flush=True)
                    write(output/"progress.json",dict(completed={k:v["status"] for k,v in results.items()}))
            if len(results)!=4 or any(x["status"]!="PASS" for x in results.values()):
                raise ValueError("FOUR_WRAPPER_POSTCHECKS_REQUIRED")
            import numpy as np
            from threadpoolctl import threadpool_limits
            from trottertracks.c1_stage3_molecular import component_check
            paired={};observables={}
            for arm in ARMS:
                for axis in AXES:
                    with np.load(output/(arm+"_"+axis+"_observable.npz"),allow_pickle=False) as z:
                        observables[arm,axis]=z["observable"].copy()
            with threadpool_limits(limits=1):
                for axis in AXES:
                    row=component_check(observables["reference",axis],observables["symmetric_directional",axis],base.ATOL,label="compiled_paired_"+axis)
                    if row["frobenius"]>base.SIGNAL_ATOL:raise ValueError("COMPILED_PAIRED_SIGNAL_ALLOWANCE")
                    paired[axis]=row
                for arm in ARMS:
                    value=observables[arm,"X"][0,0]+1j*observables[arm,"Y"][0,0]
                    paired[arm+"_vacuum_phase"]=component_check(np.array([[value]]),np.array([[np.exp(1j*c1['conditions']['phase_chi'])]]),base.ATOL,label="compiled_vacuum_chi_"+arm)
            write(output/"paired_observables.json",paired)
            with (ROOT/Q1).open(newline="") as f:q1=[x for x in csv.DictReader(f) if int(x["sample"]) in (0,1)]
            write(output/"resource_comparison.json",comparisons(results,q1))
            if source!=preflight(args.origin,args.saved)[0]:raise ValueError("SOURCE_INPUT_CHANGED")
            for item in index["members"]:
                if prior().sha((output/member_destination(item["path"])).read_bytes())!=item["sha256"]:
                    raise ValueError("COPIED_C1_RAW_CHANGED")
            preserved=True;success=True
        except BaseException as error:
            stop.set();write(output/"failure.json",dict(status="STOP",error_type=type(error).__name__,error=str(error).replace(str(ROOT),"<task>")))
        count=len(list(output.glob("*_compile.json")))
        result=dict(status="TECHNICAL_PASS_MANDATORY_STOP" if success else "STOP_FAILURE",
            source_commit=source["execution_source"],compiled=results,compile_calls_started=count,
            source_input_preserved=preserved,raw_input_rebuilt=False,wall_seconds=time.monotonic()-tick,
            sampling_calls=0,ground_solves=0,DF_refits=0,N_evaluations=0,quantum_shots=0,
            B2_started=False,other_path_or_q=False,RPE_total_cost_evaluated=False,
            whole_law_qualified=False,scientific_GO=False,retry=0,resume=0,mandatory_stop=True)
        write(output/"result.json",result);write(output/"STOP.json",dict(status=result["status"],mandatory_stop=True,
            compile_calls_started=count,retry=0,resume=0,next_stage_authorized=False))
    inventory={str(p.relative_to(output)):dict(bytes=p.stat().st_size,sha256=prior().sha(p.read_bytes()))
               for p in sorted(output.rglob("*")) if p.is_file() and "matplotlib-cache" not in p.parts}
    write(output/"inventory.json",inventory);print(result["status"],flush=True)
    return 0 if success else 1


if __name__=="__main__":raise SystemExit(main())
