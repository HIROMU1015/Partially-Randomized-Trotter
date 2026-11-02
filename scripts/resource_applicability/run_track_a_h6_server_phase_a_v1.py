#!/usr/bin/env python3
"""H6 Phase A only. Dry preparation default; execution requires a new pinned grant."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.resource_applicability.h6_server_contract_v1 import (
    flags,validate_launch,file_hash,safe_path)
from trottertracks.resource_applicability.h6_server_execution_v1 import execute_worker,orchestrate
from trottertracks.resource_applicability.ax2b_supplement_records_v1 import AtomicWriter
from trottertracks.resource_applicability.ax2a_preparation import digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true');p.add_argument('--manifest',type=Path)
    p.add_argument('--authorization',type=Path);p.add_argument('--authorization-sha256')
    p.add_argument('--output',type=Path);p.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--run-root',type=Path,help=argparse.SUPPRESS);a=p.parse_args()
    if a.worker:
        if not a.run_root or a.execute or a.manifest or a.authorization or a.authorization_sha256 or a.output:p.error('Bound internal worker only')
        return execute_worker(ROOT,a.worker,a.run_root)
    if not a.execute:
        if any((a.manifest,a.authorization,a.authorization_sha256,a.output,a.run_root)):
            p.error('Execution fields require --execute and a fresh approved grant')
        print(json.dumps(flags(),sort_keys=True));return 0
    if not all((a.manifest,a.authorization,a.authorization_sha256,a.output)) or a.run_root:
        p.error('Pinned manifest, authorization and exclusive output required')
    if file_hash(a.authorization)!=a.authorization_sha256:raise ValueError('PHASE_A_AUTHORIZATION_SHA')
    payload=a.authorization.read_bytes();m=json.loads(a.manifest.read_text());g=json.loads(payload)
    fresh=validate_launch(ROOT,m,g,a.output,requested=True)
    if file_hash(a.authorization)!=a.authorization_sha256:raise ValueError('PHASE_A_AUTHORIZATION_CHANGED')
    lock=safe_path(ROOT,m['one_shot_lock']);lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('x') as f:
        f.write(json.dumps(dict(manifest_digest=digest(m),authorization_sha256=a.authorization_sha256)));f.flush();os.fsync(f.fileno())
    out=a.output.resolve();out.mkdir(exist_ok=False)
    w=AtomicWriter(out,byte_cap=m['plan']['caps_proposed']['output_bytes'])
    w.write('launch_binding.json',dict(manifest_digest=digest(m),authorization_digest=digest(g),authorization_source_sha256=a.authorization_sha256))
    w.write('frozen_manifest.json',m);w.write('authorization.json',g)
    with (out/'authorization_source.json').open('xb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
    return orchestrate(ROOT,out,m,w,fresh)


if __name__=='__main__':raise SystemExit(main())
