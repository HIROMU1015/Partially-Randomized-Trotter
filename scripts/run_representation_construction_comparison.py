#!/usr/bin/env python3
"""Guarded limited construction/comparison batch approved by the independent review."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMBA_NUM_THREADS"):
    os.environ[key]="1"
os.environ.setdefault("MPLCONFIGDIR","/tmp/prt-representation-matplotlib")
sys.path.insert(0,str(ROOT/"src"))
FILE_CAP=32*1024**2


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,value):
    data=(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode()
    if len(data)>FILE_CAP:raise RuntimeError("32 MiB file output cap")
    path.write_bytes(data)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args();out=args.output.resolve()
    if out.exists():raise SystemExit("Refusing existing output; no overwrite/resume")
    if not out.is_relative_to(ROOT/"artifacts/representation_construction_comparison"):
        raise SystemExit("Dedicated output namespace required")
    subprocess.run(["git","diff","--quiet"],cwd=ROOT,check=True)
    subprocess.run(["git","diff","--cached","--quiet"],cwd=ROOT,check=True)
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    paths=subprocess.check_output(["git","ls-files","-z","src","scripts/run_representation_construction_comparison.py",
                                   "tests/test_representation_construction_comparison.py",
                                   "docs/research/representation_construction_comparison_scope.md",
                                   "docs/research/representation_construction_inputs"],cwd=ROOT).decode().split("\0")
    hashes={p:sha(ROOT/p) for p in paths if p and (ROOT/p).is_file()}
    out.mkdir(parents=True)
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
    resource.setrlimit(resource.RLIMIT_CPU,(600,600))
    resource.setrlimit(resource.RLIMIT_FSIZE,(FILE_CAP,FILE_CAP))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError("900 s wall cap")))
    signal.alarm(900);start=time.monotonic();status="TECHNICAL_STOP";code=1
    try:
        from trottertracks.representation_exploration.construction_comparison import run_all
        result=run_all();ir=result.pop("native_ir");result["source_commit"]=head
        save(out/"native_ir.json",ir);save(out/"result.json",result)
        status=result["status"];code=0
        print(status, "compilations",result["compiled_circuits"],flush=True)
    except Exception:
        failure={"status":status,"source_commit":head,"traceback":traceback.format_exc()}
        save(out/"failure.json",failure);traceback.print_exc()
    finally:
        signal.alarm(0);r=resource.getrusage(resource.RUSAGE_SELF)
        audit={"schema_version":1,"status":status,"source_commit":head,"source_sha256":hashes,
               "base_commit":"39345830ddfe7c3e2a488c284a0623f489764087",
               "branch":"representation-construction-comparison-20261010",
               "date":"2026-10-10","timezone":"Asia/Tokyo","python":sys.version,"platform":platform.platform(),
               "packages":{d.metadata["Name"]:d.version for d in importlib.metadata.distributions()},
               "wall_seconds":time.monotonic()-start,"cpu_seconds":r.ru_utime+r.ru_stime,"peak_rss_bytes":r.ru_maxrss*1024,
               "caps":{"wall_seconds":900,"cpu_seconds":600,"address_space_bytes":4*1024**3,
                       "file_bytes":FILE_CAP,"compilations":512,"native_gates_per_circuit":10000,
                       "total_circuit_qubits":5,"cost_trajectories_per_row":8},
               "threads":{k:os.environ[k] for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMBA_NUM_THREADS")},
               "outputs":{p.name:{"sha256":sha(p),"bytes":p.stat().st_size} for p in out.iterdir() if p.is_file()},
               "evidence_status":"local exact-data development; no immutable CI or external scientific replication",
               "next_stage_authorized":False,"central_hypothesis_adopted":None}
        save(out/"run_audit.json",audit)
    return code


if __name__=="__main__":raise SystemExit(main())
