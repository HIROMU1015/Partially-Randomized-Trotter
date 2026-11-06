#!/usr/bin/env python3
"""C5 saved00->01 only: common builder, raw checks, <=4 P0 wrappers, STOP."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import json
import multiprocessing
import os
from pathlib import Path
import tarfile

import run_track_d_c3_q4_q8 as c3
c2 = c3.c2
ROOT = c3.ROOT
OUTPUT = "artifacts/track_d_c5_q2/2026-11-06/run01"
NEW = ("src/trottertracks/track_d_c3_whole_path.py",
       "src/trottertracks/track_d_whole_path_price.py", "scripts/run_track_d_c5_q2.py",
       "tests/check_track_d_c5.py", "docs/research/track_d_c5_q2_20261106/PLAN.md")
C3_RESULT = "cc554cf6daea08df6c20ea8e1e92405259ce49b5"
C3_OUTPUT = "artifacts/track_d_c3_q4_q8/2026-11-06/run03_completion"
PRIVATE_KEYS = {"peak_RSS_KiB", "wall_seconds", "state_action_threads"}


def public_view(value):
    if isinstance(value, dict):
        return {k: public_view(v) for k,v in value.items() if k not in PRIVATE_KEYS}
    if isinstance(value, (tuple,list)):
        return [public_view(v) for v in value]
    return value


def preflight(origin, saved):
    source, _, _ = c2.preflight(origin, saved)
    p = c2.prior(); current = p.git("rev-parse", "HEAD").decode().strip()
    for name in NEW:
        if (ROOT/name).read_bytes() != p.git("show",current+":"+name):
            raise ValueError("UNCOMMITTED_C5_SOURCE:"+name)
    bindings={}
    for commit,folder,leaves in [
        (c3.C2_COMMIT,c3.C2_OUTPUT,("SOURCE.json","STOP.json","result.json","resource_comparison.json","payload_archive.json","circuit_payload.tar.gz")),
        (C3_RESULT,C3_OUTPUT,("SOURCE.json","STOP.json","result.json","resource_comparison.json"))]:
        for leaf in leaves:
            name=folder+"/"+leaf;data=(ROOT/name).read_bytes()
            if data!=p.git("show",commit+":"+name):raise ValueError("SAVED_C2_C3_BINDING:"+name)
            bindings[name]=dict(commit=commit,sha256=p.sha(data))
    if json.loads((ROOT/C3_OUTPUT/"STOP.json").read_text())["status"]!="TECHNICAL_PASS_MANDATORY_STOP":
        raise ValueError("C3_COMPLETION_REQUIRED")
    source.update(C5_source={n:p.sha((ROOT/n).read_bytes()) for n in NEW},
        C4_result="a5784c4bd3e3fb9779849a1e1fc1759559e2caea",saved_C2_C3=bindings,
        policy="q2_saved00_01_cut1_strong_cross_step_fusion_scalar_aggregation_max4_P0")
    return source


def reuse_compiles(local):
    """Reuse exact old raw/compiled bytes only; still check against new raw/CAR."""
    root=ROOT/c3.C2_OUTPUT;index=json.loads((root/"payload_archive.json").read_text())
    wanted={row["path"]:row for row in index["members"]}
    results={}
    with tarfile.open(root/index["archive"],"r:gz") as tar:
        for method in c2.ARMS:
            for axis in c2.AXES:
                name=method+"_"+axis;raw="circuits/"+name+"_raw.qpy"
                old_raw=tar.extractfile(raw).read()
                same=old_raw==(local/raw).read_bytes()
                record=dict(raw_bytes_identical=same,reused=False)
                if same:
                    compiled="circuits/"+name+"_compiled.qpy";data=tar.extractfile(compiled).read()
                    row=wanted[compiled]
                    if len(data)!=row["bytes"] or c2.prior().sha(data)!=row["sha256"]:
                        raise ValueError("SAVED_COMPILED_MEMBER")
                    with (local/compiled).open("xb") as stream:stream.write(data)
                    marker=json.loads((root/(name+"_compile.json")).read_text())
                    if marker["status"]!="PASS":raise ValueError("SAVED_COMPILE_NOT_PASS")
                    c2.old().write(local/(name+"_compile.json"),dict(status="PASS",arm=method,axis=axis,
                        compile_calls=0,compiler=marker["compiler"],reused_from=c3.C2_COMMIT))
                    record.update(reused=True,compiled_sha256=row["sha256"],source=c3.C2_COMMIT)
                results[name]=record
    return results


def worker(task, reused, stop):
    if not reused:return c2.worker(task,stop)
    origin,output,method,axis,threads=task
    os.environ["MPLCONFIGDIR"]=str(Path(output)/"matplotlib-cache"/(method+"_"+axis))
    try:
        if stop.is_set():raise RuntimeError("STOP_BEFORE_REUSED_POSTCHECK")
        return c2.old().verify_one(task)
    except BaseException:
        stop.set()
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin",type=Path,required=True);parser.add_argument("--saved",type=Path,required=True)
    parser.add_argument("--execute",action="store_true");args=parser.parse_args()
    if not args.execute:raise PermissionError("C5_EXPLICIT_EXECUTION_ONLY")
    out=ROOT/OUTPUT
    if out.exists():raise ValueError("FRESH_OUTPUT_NO_RETRY_RESUME")
    out.mkdir(parents=True);local=out/"local";local.mkdir();qdir=local/"q2";qdir.mkdir()
    write=c2.old().write;source={"execution_source":c2.prior().git("rev-parse","HEAD").decode().strip()}
    raw=None;compiled={};reuse={};success=preserved=False
    os.environ["MPLCONFIGDIR"]=str(local/"matplotlib-cache"/"parent")
    os.environ["QISKIT_PARALLEL"]="false";os.environ["RAYON_NUM_THREADS"]="1"
    try:
        source=preflight(args.origin,args.saved);write(out/"SOURCE.json",source)
        resources=c2.prior().cpu_observation();budget=resources["numerical_threads"]
        old=json.loads((ROOT/c3.C2_OUTPUT/"result.json").read_text())
        peak=max(row["peak_RSS_KiB"]*1024 for row in old["compiled"].values())
        slots=max(1,resources["memory"]["MemAvailable"]//peak)
        workers=min(4,budget,slots)
        reference_threads=budget if workers==1 else max(1,(budget-min(2,workers))//min(2,workers))
        resources.update(worker_processes=workers,reference_threads=reference_threads,raw_threads=budget)
        write(local/"resources.json",resources)
        base=c2.old().bootstrap(args.origin)
        from threadpoolctl import threadpool_limits
        from trottertracks import track_d_whole_path_price as api
        request=api.OrderedRequest(2,(0,1),1)
        checks=c2.load("c5_raw_checks",ROOT/"tests/track_d_h4_reference_checks.py")
        write(out/"progress.json",dict(status="RAW_STARTED",q=2))
        print("q2 raw STARTED",flush=True)
        with threadpool_limits(limits=1):
            context=api.c3.paired.prepare_common(args.origin,c3.load_contract())
            bundle=api.prepare(context,args.saved,request)
            for method in api.METHODS:
                for axis in api.AXES:api.wrapper_for(bundle,request,method=method,axis=axis)
            raw=api.c3.save_and_check(bundle,qdir,checks,budget)
        write(qdir/"precompile_checks.json",raw)
        if raw["status"]!="PASS":raise ValueError("RAW_CORRECTNESS_STOP_NO_COMPILE")
        print("q2 raw PASS",flush=True)
        reuse=reuse_compiles(qdir);write(out/"reuse.json",reuse)
        with multiprocessing.Manager() as manager:
            stop=manager.Event()
            tasks=[(args.origin,qdir,method,axis,reference_threads if method=="reference" else 1)
                   for method in c2.ARMS for axis in c2.AXES]
            with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn")) as pool:
                futures=[pool.submit(worker,t,reuse[t[2]+"_"+t[3]]["reused"],stop) for t in tasks]
                for future in as_completed(futures):
                    name,row=future.result();compiled[name]=row
                    write(out/"progress.json",dict(status="POSTCHECKS",completed={k:v["status"] for k,v in compiled.items()}))
                    print(name,row["status"],flush=True)
            if len(compiled)!=4 or any(row["status"]!="PASS" for row in compiled.values()):
                raise ValueError("FOUR_POSTCHECKS_REQUIRED")
        c3.paired_postcheck(local,2)
        if source!=preflight(args.origin,args.saved):raise ValueError("SOURCE_INPUT_CHANGED")
        preserved=True
        prices=[api.price_record(request,method=method,axis=axis,postcheck=compiled[method+"_"+axis],
            raw_status=raw["status"],paired_status="PASS",execution_source=source["execution_source"],
            compiler=json.loads((qdir/(method+"_"+axis+"_compile.json")).read_text())["compiler"])
            for method in c2.ARMS for axis in c2.AXES]
        write(out/"prices.json",prices)
        write(out/"resource_comparison.json",dict(current=c2.comparisons(compiled,[])["q2_paired"],
            saved_C2=json.loads((ROOT/c3.C2_OUTPUT/"resource_comparison.json").read_text()),
            saved_C3=json.loads((ROOT/C3_OUTPUT/"resource_comparison.json").read_text())["current"],
            qualification="fixed fixtures; C2 lacks C3 manual boundary fusion; no law mean, q-only effect or fit"))
        success=True
    except BaseException as error:
        message=str(error)
        for path in (str(ROOT),str(args.origin),str(args.saved)):message=message.replace(path,"<task-path>")
        write(out/"failure.json",dict(status="STOP",error_type=type(error).__name__,error=message))
    for path in sorted(qdir.glob('*.json')):
        public=out/"q2"/path.name;public.parent.mkdir(exist_ok=True)
        write(public,public_view(json.loads(path.read_text())))
    calls=sum(json.loads(p.read_text())["compile_calls"] for p in qdir.glob('*_compile.json'))
    result=dict(status="TECHNICAL_PASS_MANDATORY_STOP" if success else "STOP_FAILURE",
        source_commit=source["execution_source"],raw=public_view(raw),compiled=public_view(compiled),
        source_input_preserved=preserved,compile_calls_started=calls,
        reused_compiles=sum(r['reused'] for r in reuse.values()),
        sampling_calls=0,ground_solves=0,DF_refits=0,N_evaluations=0,quantum_shots=0,
        q_above2=False,paired_mean_calibration=False,price_model_fit=False,holdout=False,
        RPE_total_cost_evaluated=False,whole_law_qualified=False,scientific_GO=False,
        retry=0,resume=0,mandatory_stop=True)
    write(out/"result.json",result);write(out/"STOP.json",dict(status=result['status'],
        mandatory_stop=True,compile_calls_started=calls,retry=0,resume=0,next_stage_authorized=False))
    write(out/"progress.json",dict(status=result['status'],completed_postchecks=len(compiled)))
    print(result['status'],flush=True)
    return 0 if success else 1


if __name__=="__main__":raise SystemExit(main())
