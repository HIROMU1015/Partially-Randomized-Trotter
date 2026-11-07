#!/usr/bin/env python3
"""Authorized saved-q64 numerical diagnosis. No build, sampling or compile."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
from importlib.metadata import version
import subprocess
import time
import resource

import run_track_d_h4_reference as boot

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = "artifacts/track_d_c8_1_q64/2026-11-08/run02_diagnostics"
FIXED = "artifacts/track_d_c8_1_q64/2026-11-08/repair_technical/fixed_inputs.json"
FILES = ("src/trottertracks/track_d_c8_1_diagnostics.py", "scripts/run_track_d_c8_1_diagnostics.py",
         "tests/check_track_d_c8_1_diagnostics.py", "docs/research/track_d_c8_1_q64_20261108/NUMERICAL_REPAIR_PLAN.md", FIXED)
BACKEND = dict(backend_changed=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixed_inputs(prior):
    prior = Path(prior)
    index = json.loads((prior / "payload_index.json").read_text())
    # Existing payload index is authoritative; retain old STOP and failure too.
    members = []
    for row in index["members"]:
        relative = "local/" + row["path"]
        path = prior / relative
        digest = sha(path)
        if digest != row["sha256"]:
            raise ValueError("PRIOR_PAYLOAD_CHANGED:" + relative)
        members.append(dict(path=relative, sha256=digest, bytes=path.stat().st_size))
    for name in ("SOURCE.json", "STOP.json", "result.json", "event_bindings.json", "payload_index.json"):
        p = prior / name
        members.append(dict(path=name, sha256=sha(p), bytes=p.stat().st_size))
    for p in sorted((prior/"local/q64_sample00").glob("*.json")):
        members.append(dict(path=str(p.relative_to(prior)), sha256=sha(p), bytes=p.stat().st_size))
    return dict(prior_publication="1e88074f1fd64e6efe5e458c0b189069801b727b",
                prior_execution="8a6bbdac6977df9a6bc430bdd48aad2659a7e191", members=members)


def verify_runtime(prior, origin):
    """Reuse the previous scientific SOURCE tables; no environment mutation."""
    source = json.loads((Path(prior)/"SOURCE.json").read_text())
    for package, expected in source["packages"].items():
        if version(package) != expected:
            raise ValueError("RUNTIME_CHANGED:"+package)
    for table in ("original_source_files","original_saved_inputs"):
        for relative, expected in source[table].items():
            if sha(Path(origin)/relative) != expected:
                raise ValueError("ORIGINAL_SOURCE_INPUT_CHANGED:"+relative)
    for table in ("new_and_reused","C8_source","accelerated_source"):
        for relative, expected in source[table].items():
            if sha(ROOT/relative) != expected:
                raise ValueError("REUSED_SOURCE_CHANGED:"+relative)
    return source["packages"]


def worker(prior, origin, arm, axis, threads, output):
    started = time.monotonic()
    os.environ["MPLCONFIGDIR"] = str(Path(output)/"local"/"matplotlib-cache"/(arm+"_"+axis))
    m = boot.bootstrap(origin)
    from trottertracks import track_d_c7_3_acceleration as fast
    from trottertracks import track_d_c8_1_diagnostics as diagnostics
    from threadpoolctl import threadpool_limits
    engine = fast.install(threads)
    checker = boot.load_file("c8_repair_historical_checks", ROOT/"tests/track_d_h4_reference_checks.py")
    import numpy as np
    directory = Path(prior)/"local/q64_sample00"
    with np.load(directory/"references.npz", allow_pickle=False) as z:
        arrays = {k:z[k] for k in z.files}
    compiled = m.load_qpy(directory/"circuits"/(arm+"_"+axis+"_compiled.qpy"))
    dest = Path(output)/"local"/(arm+"_"+axis)
    with threadpool_limits(limits=threads):
        report = diagnostics.diagnose(checker, compiled, arm, axis, arrays, m, threads, dest)
    diagnostics.save_json(dest/"runtime_private.json", dict(wall_seconds=time.monotonic()-started,
        peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, threads=threads,
        backend_metadata=engine.last_metadata, cache=fast.cache_statistics()))
    return arm+"_"+axis, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--origin", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise ValueError("EXPLICIT_EXECUTE_REQUIRED")
    source = boot.git("rev-parse", "HEAD").decode().strip()
    for path in FILES:
        if (ROOT/path).read_bytes() != boot.git("show", source+":"+path):
            raise ValueError("UNFIXED_SOURCE:"+path)
    packet = json.loads((ROOT/FIXED).read_text())
    if packet != fixed_inputs(args.prior):
        raise ValueError("FIXED_INPUT_CHANGED")
    packages = verify_runtime(args.prior,args.origin)
    output = ROOT/OUTPUT
    output.mkdir(parents=True, exist_ok=False)
    local = output/"local"; local.mkdir(); (local/".gitignore").write_text("*\n")
    for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "RAYON_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["QISKIT_PARALLEL"] = "false"
    identity = dict(execution_source=source, fixed_inputs=packet,
        files={p:sha(ROOT/p) for p in FILES}, q=64, cut=32, sample_id="q64_sample00",
        compiler_changed=False, component_atol=1e-10, signal_atol=2e-10,
        **BACKEND, global_phase_fit=False, renormalized=False,
        sampling_calls=0, raw_build_calls=0, compile_calls=0, packages=packages)
    boot.write(output/"SOURCE.json", identity)
    cpu = len(os.sched_getaffinity(0))
    tasks = [(m,a) for m in ("symmetric_directional","reference") for a in ("X","Y")]
    reports = {}
    # Same total CPU budget as actual affinity; independent four checks.
    base, remainder = divmod(cpu, len(tasks))
    try:
        with ProcessPoolExecutor(max_workers=min(cpu,len(tasks)), mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(worker,args.prior,args.origin,m,a,max(1,base+(i<remainder)),output)
                       for i,(m,a) in enumerate(tasks)]
            for future in as_completed(futures):
                name, report = future.result(); reports[name] = report
                boot.write(output/(name+"_check.json"), report)
                boot.write(output/"progress.json",dict(completed=len(reports), total=4,
                    pass_count=sum(r["status"]=="PASS" for r in reports.values())))
                print(name,report["status"],report.get("observable_residuals"),flush=True)
    except Exception as error:
        boot.write(output/"failure.json", dict(error_type=type(error).__name__,error=str(error)))
    preserved = packet == fixed_inputs(args.prior) and packages == verify_runtime(args.prior,args.origin)
    payload = [dict(path=str(p.relative_to(output)),bytes=p.stat().st_size,sha256=sha(p))
               for p in sorted(local.rglob("*.npz"))]
    boot.write(output/"payload_index.json", dict(members=payload))
    result = dict(status="DIAGNOSTICS_COMPLETE" if len(reports)==4 and preserved else "STOP_FAILURE",
        completed=len(reports), pass_count=sum(r["status"]=="PASS" for r in reports.values()),
        prior_preserved=preserved, sampling_calls=0, raw_build_calls=0, compile_calls=0,
        scientific_retry_authorized=True, acceptance_changed=False,
        remaining7_released=False, mandatory_stop=True)
    boot.write(output/"result.json",result)
    boot.write(output/"STOP.json",dict(mandatory_stop=True,reason="fixed diagnostic batch complete; no automatic follow-on dispatch"))
    print(result,flush=True)
    return 0 if result["status"]=="DIAGNOSTICS_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
