#!/usr/bin/env python3
"""Preparation by default. No matched H6 work without a new pinned one-shot grant."""
import argparse
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.resource_applicability.h6_matched_contract_v1 import preparation, validate_launch, file_hash
from trottertracks.resource_applicability.h6_matched_execution_v1 import bounded_json, execute_worker, orchestrate
from trottertracks.resource_applicability.ax2b_limits import exclusive_json
from trottertracks.resource_applicability.ax2b_supplement_records_v1 import AtomicWriter
from trottertracks.resource_applicability.ax2a_preparation import digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path);p.add_argument('--execute',action='store_true')
    p.add_argument('--manifest',type=Path);p.add_argument('--authorization',type=Path)
    p.add_argument('--authorization-sha256')
    p.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--run-root',type=Path,help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.worker:
        if not a.run_root or a.execute or a.output or a.manifest or a.authorization or a.authorization_sha256:
            p.error('Internal worker requires only its bound run root.')
        return execute_worker(ROOT,a.worker,a.run_root)
    if not a.output:p.error('--output required')
    if not a.execute:
        if a.manifest or a.authorization or a.authorization_sha256 or a.run_root:
            p.error('Execution arguments require a new grant.')
        exclusive_json(a.output,preparation());print('H6_NOT_AUTHORIZED');return 0
    if not a.manifest or not a.authorization or not a.authorization_sha256 or a.run_root:
        p.error('Sealed manifest and pinned fresh authorization required.')
    if file_hash(a.authorization)!=a.authorization_sha256:raise ValueError('MATCHED_AUTHORIZATION_SHA')
    m,g=bounded_json(a.manifest),bounded_json(a.authorization)
    out=a.output.resolve();validate_launch(ROOT,m,g,out,requested=True)
    payload=a.authorization.read_bytes()
    if file_hash(a.authorization)!=a.authorization_sha256:raise ValueError('MATCHED_AUTHORIZATION_CHANGED')
    out.mkdir(exist_ok=False)
    w=AtomicWriter(out,byte_cap=m['plan']['caps_proposed']['output_bytes'])
    w.write('launch_binding.json',dict(manifest_digest=digest(m),authorization_digest=digest(g),
                                      authorization_source_sha256=a.authorization_sha256))
    w.write('frozen_manifest.json',m);w.write('authorization.json',g)
    with (out/'authorization_source.json').open('xb') as f:
        f.write(payload);f.flush();os.fsync(f.fileno())
    return orchestrate(ROOT,out,m,w)


if __name__=='__main__':raise SystemExit(main())
