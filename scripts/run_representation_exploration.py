#!/usr/bin/env python3
"""Bounded independent exploration runner. See docs/research/representation_exploration_scope.md."""
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

ROOT = Path(__file__).resolve().parents[1]
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[key] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/prt-representation-matplotlib")
sys.path.insert(0, str(ROOT / "src"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: object) -> None:
    encoded = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    if len(encoded) > 16 * 1024**2:
        raise RuntimeError("16 MiB per-file output cap exceeded")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(encoded)
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit("Refusing to overwrite an existing run directory")
    if not output.is_relative_to(ROOT / "artifacts" / "representation_exploration"):
        raise SystemExit("Output must be under this worktree's dedicated artifact namespace")
    subprocess.run(["git", "diff", "--quiet"], cwd=ROOT, check=True)
    subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT, check=True)
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"],cwd=ROOT,text=True).strip()
    files = subprocess.check_output(["git", "ls-files", "src", "scripts/run_representation_exploration.py",
                                    "tests/test_representation_exploration.py",
                                    "docs/research/representation_exploration_scope.md"],
                                   cwd=ROOT,text=True).splitlines()
    source_hashes = {p:digest(ROOT/p) for p in files if p.endswith((".py",".md"))}
    output.mkdir(parents=True)
    resource.setrlimit(resource.RLIMIT_AS, (4*1024**3,4*1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (240,240))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16*1024**2,16*1024**2))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("300 second wall cap")))
    signal.alarm(300)
    start = time.monotonic()
    status, code = "TECHNICAL_STOP", 1
    try:
        from trottertracks.representation_exploration.mechanisms import run_all
        result = run_all()
        events = result["candidate_b"].pop("events")
        result["source_commit"] = source_commit
        save(output/"result.json",result)
        save(output/"b_events.json",events)
        status, code = result["status"], 0
        print(f"{status}: {len(result['candidate_a']['rows'])} A rows, "
              f"{len(result['candidate_b']['rows'])} B rows, "
              f"{len(result['candidate_c']['rows'])} C rows, "
              f"{result['resource_counts']['compiled_circuits']} compilations")
    except Exception:
        save(output/"failure.json",{"status":status,"source_commit":source_commit,
                                     "traceback":traceback.format_exc()})
        traceback.print_exc()
    finally:
        signal.alarm(0)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        packages = {d.metadata["Name"]:d.version for d in importlib.metadata.distributions()}
        audit = {"schema_version":1,"source_commit":source_commit,"source_sha256":source_hashes,
                 "base_commit":"b2e1bf65e21893b6c617223b42313623d3186f12",
                 "branch":"representation-exploration-20261010","status":status,
                 "client_date":"2026-10-10","timezone":"Asia/Tokyo",
                 "python":sys.version,"platform":platform.platform(),"packages":packages,
                 "threads":{k:os.environ[k] for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMBA_NUM_THREADS")},
                 "wall_seconds":time.monotonic()-start,"cpu_seconds":usage.ru_utime+usage.ru_stime,
                 "peak_rss_bytes":usage.ru_maxrss*1024,
                 "caps":{"wall_seconds":300,"cpu_seconds":240,"address_space_bytes":4*1024**3,
                         "output_file_bytes":16*1024**2,"compilations":64,"b_events_per_row":10000,
                         "dense_system_dimension":8,"reflection_dimension":8,"total_circuit_qubits":5},
                 "outputs":{p.name:{"sha256":digest(p),"bytes":p.stat().st_size}
                            for p in output.iterdir() if p.is_file()},
                 "evidence_status":"local execution; source commit fixed; no immutable CI or external reproduction",
                 "central_hypothesis_adopted":None,"next_stage_authorized":False}
        save(output/"run_audit.json",audit)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
