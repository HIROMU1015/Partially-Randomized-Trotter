#!/usr/bin/env python3
"""Artificial supervisor tests, saved H2 diagnosis and inactive H4 drafts only."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from trotterlib.isolated_qiskit_execution import artificial_case, compare_artificial, read, save
from trotterlib.isolated_qiskit_supervisor import WatchPolicy, run_supervised, worker_ready

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"artifacts/non_df_h4_readiness/2026-10-23"
CASES=("pauli_jw_PF","pauli_jw_PR","df_native_PF","df_native_PR")
MODES=("normal","artificial","dimension","noready","earlyexit","nonzero","hang","ignoreterm","descendants","leaf","branch","badready")


def worker(args):
    out=args.output
    if args.mode in {"leaf","branch"}:
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        if args.mode=="branch":
            leaf=subprocess.Popen([sys.executable,__file__,"--worker","leaf","--mode","leaf","--output",str(out)],start_new_session=True)
            save(out/"family.json",dict(branch=os.getpid(),leaf=leaf.pid))
        while True: time.sleep(.05)
    if args.mode=="earlyexit": raise RuntimeError("ARTIFICIAL_STARTUP_EXCEPTION")
    if args.mode=="noready":
        while True: time.sleep(.05)
    if args.mode=="badready":
        save(out/"ready.json",dict(pid=0,parent_pid=0,case=args.worker,launch_token="wrong"))
        while True: time.sleep(.05)
    worker_ready(out,args.release)
    if args.mode=="nonzero":
        print("ARTIFICIAL_NONZERO",file=sys.stderr,flush=True)
        sys.exit(7)
    if args.mode=="descendants":
        subprocess.Popen([sys.executable,__file__,"--worker","branch","--mode","branch","--output",str(out)],start_new_session=True)
    if args.mode in {"ignoreterm","descendants"}: signal.signal(signal.SIGTERM,signal.SIG_IGN)
    if args.mode in {"hang","ignoreterm","descendants"}:
        print("ARTIFICIAL_HANG_READY",flush=True)
        while True: time.sleep(.05)
    if args.mode=="artificial":
        if args.worker not in CASES: raise ValueError("SYNTHETIC_CASE_NAME")
        artificial_case(args.worker,out)
    elif args.mode=="dimension":
        if args.worker not in CASES: raise ValueError("SYNTHETIC_CASE_NAME")
        from trotterlib.non_df_synthetic_dimension_preflight import synthetic_dimension_case
        synthetic_dimension_case(args.worker,out)
    else:
        print("artificial stdout",flush=True)
        print("artificial stderr",file=sys.stderr,flush=True)
        save(out/"result.json",dict(status="PASS",artificial_only=True,pid=os.getpid()))


def supervise(spec):
    run=Path(spec["run"])
    release=run/"release.json"
    jobs=[]
    for item in spec["jobs"]:
        if item["mode"] not in MODES or item["mode"] in {"leaf","branch"}:
            raise ValueError("ARTIFICIAL_MODE_ONLY")
        out=run/item["case"]
        jobs.append(dict(case=item["case"],case_id=item.get("case_id",item["case"]),output=out,
            binding=dict(scope="ARTIFICIAL_ONLY",SOURCE=spec.get("SOURCE"),input_hash=spec.get("input_hash")),
            argv=[sys.executable,__file__,"--worker",item["case"],"--mode",item["mode"],
                  "--output",str(out),"--release",str(release)]))
    result=run_supervised(jobs,run,cwd=ROOT,policy=WatchPolicy(**spec.get("policy",{})),
                          stop_request=run/"STOP.request")
    print(json.dumps(dict(status=result["status"],run=str(run),all_children_reaped=result["all_children_reaped"])),flush=True)
    return 0 if result["status"]=="COMPLETE_MANDATORY_STOP" else 1


def preflight():
    folder=BASE/"technical/qiskit_preflight"
    folder.mkdir(parents=True)
    results={}
    for mode in ["sequential","parallel"]:
        m=folder/mode;m.mkdir()
        batches=[(name,) for name in CASES] if mode=="sequential" else [CASES]
        for i,batch in enumerate(batches):
            spec=dict(run=str(m/f"batch{i}"),jobs=[dict(case=name,mode="artificial") for name in batch],
                      policy=dict(ready_timeout=60.,run_timeout=None,terminate_grace=2.,kill_grace=2.))
            sp=m/f"spec{i}.json";save(sp,spec)
            with (m/f"supervisor{i}.stdout.log").open("x") as so,(m/f"supervisor{i}.stderr.log").open("x") as se:
                code=subprocess.run([sys.executable,__file__,"--supervise",str(sp)],cwd=ROOT,stdout=so,stderr=se).returncode
            if code!=0: raise RuntimeError("ARTIFICIAL_QISKIT_PREFLIGHT_FAILED")
        results[mode]={name:read(m/f"batch{0 if mode=='parallel' else j}"/name/"result.json") for j,name in enumerate(CASES)}
    result=compare_artificial(results["sequential"],results["parallel"])
    save(folder/"summary.json",result)
    dimension=folder/"dimension"
    dimension.mkdir()
    spec=dict(run=str(dimension/"batch"),jobs=[dict(case=name,mode="dimension") for name in CASES],
        policy=dict(ready_timeout=60.,run_timeout=None,terminate_grace=2.,kill_grace=2.))
    sp=dimension/"spec.json";save(sp,spec)
    with (dimension/"stdout.log").open("x") as so,(dimension/"stderr.log").open("x") as se:
        code=subprocess.run([sys.executable,__file__,"--supervise",str(sp)],cwd=ROOT,stdout=so,stderr=se).returncode
    if code!=0: raise RuntimeError("SYNTHETIC_EIGHT_QUBIT_PREFLIGHT_FAILED")
    rows={name:read(dimension/"batch"/name/"result.json") for name in CASES}
    overlap=min(r["ended"] for r in rows.values())-max(r["started"] for r in rows.values())
    if len({r["pid"] for r in rows.values()})!=4 or overlap<=0:
        raise ValueError("SYNTHETIC_EIGHT_QUBIT_CONCURRENCY")
    save(dimension/"summary.json",dict(status="PASS",num_qubits=8,molecular=False,cases=list(rows),
        overlap_seconds=overlap,errors={n:r["errors"] for n,r in rows.items()}))
    print(json.dumps(dict(status="PASS",overlap_seconds=result["overlap_seconds"],scope="artificial_only")),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument("--supervise",type=Path)
    g.add_argument("--worker")
    g.add_argument("--preflight",action="store_true")
    g.add_argument("--h4-execute",action="store_true")
    p.add_argument("--mode",choices=MODES,default="normal")
    p.add_argument("--output",type=Path)
    p.add_argument("--release",type=Path)
    args=p.parse_args()
    if args.h4_execute:
        p.error("H4_SCIENCE_NOT_AUTHORIZED: draft only; no input generation or scientific call")
    if args.supervise: sys.exit(supervise(read(args.supervise)))
    if args.worker: worker(args)
    else: preflight()
