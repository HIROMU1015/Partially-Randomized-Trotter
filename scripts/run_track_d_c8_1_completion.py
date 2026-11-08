#!/usr/bin/env python3
"""One authorized q64 remainder: final gate, raw7, <=28 P0/post, n8, STOP."""
import argparse
from dataclasses import asdict, replace
import json
import multiprocessing
import os
from pathlib import Path
import resource
import sys
import time

import run_track_d_c8_1_accelerated as reused
ROOT=reused.ROOT
sys.path.insert(0,str(ROOT/"src"))
from trottertracks import track_d_c8_1_handoff as evidence

OUTPUT="artifacts/track_d_c8_1_q64/2026-11-08/run04_limited_completion"
TECH="artifacts/track_d_c8_1_q64/2026-11-08/completion_technical"
FILES=("src/trottertracks/track_d_c8_1_completion.py","scripts/run_track_d_c8_1_completion.py",
       "tests/check_track_d_c8_1_completion.py","docs/research/track_d_c8_1_q64_20261108/COMPLETION_PLAN.md",
       TECH+"/fixed_inheritance.json",TECH+"/synthetic_tests.json")
REPAIR_FILES=("src/trottertracks/track_d_c8_1_nofusion.py","src/trottertracks/track_d_c8_1_diagnostics.py")
ORIGINAL_RAW=reused.frozen.raw_worker
ORIGINAL_STAGE=reused.frozen.old.stage_worker
write=reused.write


def raw_worker(row,origin,local,threads,stop):
    reused.frozen.c2.old().bootstrap(origin)
    from trottertracks import track_d_c7_3_acceleration as fast
    from trottertracks import track_d_c8_1_completion as adapter
    original=fast.install
    progress=Path(local)/row["sample_id"]/"raw_progress_private.json"
    fast.install=lambda budget:adapter.install(budget,progress,existing_install=original)
    try:
        return ORIGINAL_RAW(row,origin,local,threads,stop)
    finally:
        fast.install=original


def stage_worker(sid,task,stage,stop):
    if stage=="compile":
        return ORIGINAL_STAGE(sid,task,stage,stop)
    origin,directory,arm,axis,threads=task;directory=Path(directory);name=arm+"_"+axis
    started=time.monotonic()
    os.environ["MPLCONFIGDIR"]=str(directory/"matplotlib-cache"/("post_"+name))
    try:
        if stage!="post" or stop.is_set():raise ValueError("STOP_OR_UNKNOWN_STAGE")
        if evidence.read(directory/"precompile_checks.json")["status"]!="PASS":raise ValueError("RAW_PASS_REQUIRED")
        if evidence.read(directory/(name+"_compile.json"))["status"]!="PASS":raise ValueError("COMPILE_PASS_REQUIRED")
        if (directory/(name+"_check.json")).exists():raise ValueError("POSTCHECK_ALREADY_COMPLETED")
        write(directory/(name+"_post_stage.json"),dict(status="STARTED",postcheck_calls=1))
        boot=reused.frozen.c2.old();m=boot.bootstrap(origin)
        from trottertracks import track_d_c7_3_acceleration as fast
        from trottertracks import track_d_c8_1_completion as adapter
        from trottertracks import track_d_c8_1_diagnostics as diagnostics
        from trottertracks import track_d_c8_1_q64 as pilot
        from threadpoolctl import threadpool_limits
        import numpy as np
        engine=adapter.install(threads,directory/(name+"_progress_private.json"),existing_install=fast.install)
        checker=boot.load_file("c8_completion_postchecks",ROOT/"tests/track_d_h4_reference_checks.py")
        compiled=m.load_qpy(directory/"circuits"/(name+"_compiled.qpy"))
        with np.load(directory/"references.npz",allow_pickle=False) as archive:
            arrays={k:archive[k] for k in archive.files}
        destination=directory/(name+"_diagnostics")
        with threadpool_limits(limits=threads):
            report=diagnostics.diagnose(checker,compiled,arm,axis,arrays,m,threads,destination)
        if report["status"]!="PASS":
            stop.set();write(directory/(name+"_post_failure.json"),report)
            write(directory/(name+"_post_stage.json"),dict(status="FAIL",postcheck_calls=1))
            return sid,name,stage,report
        with np.load(destination/"observable.npz",allow_pickle=False) as archive:
            np.savez_compressed(directory/(name+"_observable.npz"),observable=archive["observable"])
        report.update(peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            wall_seconds=time.monotonic()-started,state_action_threads=threads,
            metrics=m.artificial.metrics(compiled),simulation_fusion_enabled=False)
        write(directory/(name+"_check.json"),report)
        write(directory/(name+"_features.json"),pilot.wrapper_features(compiled,arm))
        write(directory/(name+"_native_metadata_private.json"),engine.last_metadata)
        write(directory/(name+"_post_stage.json"),dict(status="PASS",postcheck_calls=1))
        return sid,name,stage,report
    except Exception as error:
        stop.set();message=str(error)
        for p in (str(ROOT),str(origin),str(directory)):message=message.replace(p,"<task-path>")
        report=dict(status="FAIL",error_type=type(error).__name__,error=message)
        write(directory/(name+"_post_failure.json"),report)
        if (directory/(name+"_post_stage.json")).exists():
            write(directory/(name+"_post_stage.json"),dict(status="FAIL",postcheck_calls=1))
        return sid,name,stage,report


def preflight(origin,saved,original,repair):
    from trottertracks import track_d_c8_1_completion as adapter
    source=reused.frozen.preflight(origin,saved);head=source["execution_source"]
    for p in (*FILES,*REPAIR_FILES):
        if (ROOT/p).read_bytes()!=reused.frozen.c2.prior().git("show",head+":"+p):
            raise ValueError("UNFIXED_COMPLETION_SOURCE:"+p)
    for p in REPAIR_FILES:
        if (ROOT/p).read_bytes()!=reused.frozen.c2.prior().git("show",adapter.REPAIR_SOURCE+":"+p):
            raise ValueError("REPAIR_NUMERICAL_CODE_CHANGED")
    packet=evidence.read(ROOT/TECH/"fixed_inheritance.json")
    if packet!=adapter.inheritance(original,repair):raise ValueError("FIXED_INHERITANCE_CHANGED")
    adapter.verify(original,repair,packet)
    if evidence.read(ROOT/TECH/"synthetic_tests.json")["status"]!="PASS":raise ValueError("ARTIFICIAL_TESTS_REQUIRED")
    source.update(completion_source={p:evidence.sha(ROOT/p) for p in (*FILES,*REPAIR_FILES)},
        inherited_original=evidence.OLD_SOURCE,inherited_repair=adapter.REPAIR_SOURCE,
        policy="new limited execution: paired sample00 first, raw7, max28 P0/post, paired8, eligible n8 statistics only",
        simulation_fusion_enabled=False,compiler_changed=False,precision_changed=False,
        original_sampling_preserved=True,sampling_new=0,raw_reused=1,compile_reused=4,postcheck_reused=4,
        raw_limit_new=7,compile_limit_new=28,postcheck_limit_new=28,paired_chi_limit_new=8)
    return source,packet


def sample00_records(local,implementation,pilot):
    """Price metadata from saved P0; never rerun a numerical postcheck."""
    directory=Path(local)/"q64_sample00";records={}
    for arm in evidence.METHODS:
        for axis in evidence.AXES:
            name=arm+"_"+axis;original=evidence.read(directory/(name+"_check.json"))
            circuit=implementation.load_qpy(directory/"circuits"/(name+"_compiled.qpy"))
            row=dict(original,metrics=implementation.artificial.metrics(circuit),
                inherited_postcheck_source="f1e89590c7daea12319352b85b959928ad166213",
                inherited_postcheck_sha256=evidence.sha(directory/(name+"_check.json")),
                postcheck_reexecuted=False)
            write(directory/(name+"_price_metadata.json"),row)
            write(directory/(name+"_features.json"),pilot.wrapper_features(circuit,arm))
            records[name]=row
    return records


def resources(original,repair):
    """Use actual availability and completed same-q peaks, not old worker caps."""
    cpu=len(os.sched_getaffinity(0));quota=Path("/sys/fs/cgroup/cpu.max")
    if quota.exists() and quota.read_text().split()[0]!="max":
        amount,period=map(int,quota.read_text().split());cpu=min(cpu,max(1,amount//period))
    raw=evidence.read(Path(original)/"local/q64_sample00/precompile_checks.json")["peak_RSS_KiB"]*1024
    posts={p.parent.name:evidence.read(p)["peak_RSS_KiB"]*1024 for p in (Path(repair)/"local").glob("*/runtime_private.json")}
    if len(posts)!=4:raise ValueError("COMPLETED_REPAIR_PEAKS_REQUIRED")
    compile_peaks=[evidence.read(p)["peak_RSS_KiB"]*1024 for p in
        (Path(original)/"local/q64_sample00").glob("*_compile_stage.json")]
    # ru_maxrss is a process-lifetime maximum: the old reused raw worker's
    # compile record equals the much larger raw peak. Other completed P0
    # workers isolate compile memory. Current MemAvailable also accounts for
    # memory retained by reused workers; do not reserve that raw peak twice.
    isolated_compile=[v for v in compile_peaks if v<raw]
    available=reused.frozen.old.frozen.memory_available()
    return dict(numerical_threads=cpu,memory=dict(MemAvailable=available),
        raw_memory_estimate=2*raw,post_memory_estimate=2*max(posts.values()),
        reference_post_memory_estimate=2*max(v for k,v in posts.items() if k.startswith("reference")),
        direct_post_memory_estimate=2*max(v for k,v in posts.items() if k.startswith("symmetric_directional")),
        compile_memory_estimate=2*max(isolated_compile or [raw]),
        policy="completed same-q peak estimates and actual available RAM admission; compile1CPU, native stages share remainder; no small worker cap")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for p in ("origin","saved","original-output","repair-output"):parser.add_argument("--"+p,type=Path,required=True)
    parser.add_argument("--execute",action="store_true");args=parser.parse_args()
    if not args.execute:raise ValueError("EXPLICIT_LIMITED_EXECUTION_REQUIRED")
    out=ROOT/OUTPUT
    if out.exists():raise ValueError("FRESH_NAMESPACE_REQUIRED_NO_RETRY")
    out.mkdir(parents=True);local=out/"local";local.mkdir();(local/".gitignore").write_text("*\n")
    for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","RAYON_NUM_THREADS"):os.environ[k]="1"
    os.environ["QISKIT_PARALLEL"]="false";os.environ["MPLCONFIGDIR"]=str(local/"matplotlib-cache"/"parent")
    source={};packet=None;raw={};compiled={};success=preserved=False
    try:
        boot=reused.frozen.c2.old();m=boot.bootstrap(args.origin)
        from trottertracks import track_d_c8_1_completion as adapter
        from trottertracks import track_d_c8_1_q64 as pilot
        source,packet=preflight(args.origin,args.saved,args.original_output,args.repair_output)
        write(out/"SOURCE.json",source);write(out/"inherited_evidence.json",packet)
        reused.copy_inherited(args.original_output,out,packet["original"])
        adapter.inherit_repair(args.repair_output,out,packet)
        events=evidence.read(out/"event_bindings.json");models=evidence.read(out/"frozen_models.json")
        for event in events:
            path=local/"events"/(event["sample_id"]+".json")
            if evidence.sha(path)!=event["record_sha256"]:raise ValueError("SEALED_EVENT_HASH")
            request=pilot.request_from_record(evidence.read(path))
            if request.execution_source!=evidence.OLD_SOURCE:raise ValueError("SAMPLING_SOURCE_BINDING")
            if event["sample_id"]=="q64_sample00" and evidence.read(local/"q64_sample00/conditions.json")["request_fingerprint"]!=request.fingerprint:
                raise ValueError("INHERITED_RAW_REQUEST_BINDING")
        raw["q64_sample00"]=evidence.read(local/"q64_sample00/precompile_checks.json")
        compiled["q64_sample00"]=sample00_records(local,m,pilot)
        allocation=resources(args.original_output,args.repair_output);write(local/"resources.json",allocation)
        write(out/"progress.json",dict(status="SAMPLE00_FINAL_PAIRED_CHI",raw_reused=1,compile_reused=4,postchecks_reused=4))
        reused.frozen.raw_worker=raw_worker;reused.frozen.old.stage_worker=stage_worker
        prior_allocation=reused.handoff.allocation
        reused.handoff.allocation=lambda allocator,cpu,memory,pending,workers,active:adapter.allocation(
            prior_allocation,allocator,cpu,memory,pending,workers,active,allocation)
        with multiprocessing.Manager() as manager:
            stop=manager.Event()
            reused.pipeline(events,args.origin,local,allocation,stop,out,raw,compiled,
                {(events[0]["sample_id"],a,x) for a in evidence.METHODS for x in evidence.AXES})
        if len(raw)!=8 or any(r["status"]!="PASS" for r in raw.values()):raise ValueError("ALL8_RAW_REQUIRED")
        if sum(len(v) for v in compiled.values())!=32 or any(r["status"]!="PASS" for v in compiled.values() for r in v.values()):
            raise ValueError("ALL32_POSTCHECKS_REQUIRED")
        paired=list(local.glob("q64_sample*/paired_observables.json"))
        if len(paired)!=8:raise ValueError("ALL8_FINAL_PAIRED_CHI_REQUIRED")
        calls=sum(evidence.read(p).get("compile_calls",0) for p in local.glob("q64_sample*/*_compile.json"))
        if calls!=32:raise ValueError("ORIGINAL4_PLUS_NEW28_COMPILE_ACCOUNTING")
        prices=[]
        for event in events:
            sid=event["sample_id"];directory=local/sid
            path=local/"events"/(sid+".json")
            if evidence.sha(path)!=event["record_sha256"]:raise ValueError("FIXED_EVENT_HASH")
            request=pilot.request_from_record(evidence.read(path))
            if evidence.read(directory/"conditions.json")["request_fingerprint"]!=request.fingerprint:raise ValueError("RAW_REQUEST_BINDING")
            for arm in evidence.METHODS:
                for axis in evidence.AXES:
                    inherited=sid=="q64_sample00";execution=adapter.REPAIR_SOURCE if inherited else source["execution_source"]
                    row=pilot.price_record(replace(request,execution_source=execution),method=arm,axis=axis,
                        postcheck=compiled[sid][arm+"_"+axis],raw_status="PASS",paired_status="PASS",execution_source=execution,
                        compiler=evidence.read(directory/(arm+"_"+axis+"_compile.json"))["compiler"])
                    row.update(sampling_source=evidence.OLD_SOURCE,raw_source=evidence.OLD_SOURCE if inherited else source["execution_source"],
                        compile_source=evidence.OLD_SOURCE if inherited else source["execution_source"],postcheck_source=execution,
                        final_paired_chi_source=source["execution_source"],price_extraction_source=source["execution_source"],
                        raw_request=asdict(request),raw_request_fingerprint=request.fingerprint)
                    prices.append(row)
        write(out/"prices.json",prices);summary=pilot.paired_statistics(prices);write(out/"paired_statistics.json",summary)
        write(out/"cx_candidate_residuals.json",pilot.cx_candidate_residuals(summary))
        write(out/"model_diagnostics.json",pilot.diagnose(summary,models));success=True
    except BaseException as error:
        message=str(error)
        for p in (str(ROOT),str(args.origin),str(args.saved),str(args.original_output),str(args.repair_output)):
            message=message.replace(p,"<task-path>")
        write(out/"failure.json",dict(status="STOP",error_type=type(error).__name__,error=message))
    try:
        preserved=bool(source) and (source,packet)==preflight(args.origin,args.saved,args.original_output,args.repair_output)
    except BaseException:
        preserved=False
    reused.frozen.collect(out,local)
    raw_files=list(local.glob("q64_sample*/precompile_checks.json"));post_files=list(local.glob("q64_sample*/*_check.json"))
    calls=sum(evidence.read(p).get("compile_calls",0) for p in local.glob("q64_sample*/*_compile.json"))
    result=dict(status="Q64_COMPLETE_EXPLORATORY_MANDATORY_STOP" if success and preserved else "STOP_FAILURE",
        source_commit=source.get("execution_source"),q=64,cut=32,sampling_new=0,
        raw_reused=1 if packet else 0,raw_completed=len(raw_files),raw_PASS=sum(evidence.read(p)["status"]=="PASS" for p in raw_files),
        compile_reused=4 if packet else 0,compile_calls_total=calls,compile_calls_new=max(0,calls-(4 if packet else 0)),
        postchecks_reused=4 if packet else 0,postchecks_PASS=sum(evidence.read(p)["status"]=="PASS" for p in post_files),
        postcheck_calls_new=sum(evidence.read(p).get("postcheck_calls",0) for p in local.glob("q64_sample*/*_post_stage.json")),
        raw_tasks_dispatched_new=max(0,len(list(local.glob("q64_sample*")))-(1 if packet else 0)),
        paired_chi_completed=len(list(local.glob("q64_sample*/paired_observables.json"))),
        source_input_preserved=preserved,old_STOP_preserved=True,model_refits=0,N_evaluations=0,
        RPE_total_cost_updated=False,q_above64=False,actual_success_guaranteed=False,whole_law_guaranteed=False,
        retry=0,resume=0,mandatory_stop=True)
    write(out/"result.json",result);write(out/"STOP.json",dict(status=result["status"],mandatory_stop=True,next_stage_authorized=False,next_actor="user/GPT"))
    write(out/"progress.json",dict(status=result["status"]));print(result["status"],flush=True)
    return 0 if success and preserved else 1


if __name__=="__main__":raise SystemExit(main())
