#!/usr/bin/env python3
"""Dry run by default. Real s2 pilot needs its own exact, separately approved grant."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks.resource_applicability import h6_conditional_s2_contract_v1 as c
from trottertracks.resource_applicability.ax2a_preparation import digest
from trottertracks.resource_applicability.ax2b_supplement_records_v1 import AtomicWriter


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true')
    p.add_argument('--manifest',type=Path);p.add_argument('--authorization',type=Path)
    p.add_argument('--authorization-sha256');p.add_argument('--output',type=Path)
    p.add_argument('--worker',type=Path,help=argparse.SUPPRESS);p.add_argument('--run-root',type=Path,help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.worker:
        if not a.run_root or any((a.execute,a.manifest,a.authorization,a.authorization_sha256,a.output)):p.error('Bound worker only')
        from trottertracks.resource_applicability.h6_conditional_s2_execution_v1 import worker
        return worker(ROOT,a.worker,a.run_root)
    if not a.execute:
        if any((a.manifest,a.authorization,a.authorization_sha256,a.output,a.run_root)):p.error('Execution fields require new grant')
        print('CONDITIONAL_S2_PILOT_PREPARATION_ONLY; approved=false; runtime_authorization=false; allowed_cpus=[]');return 0
    if not all((a.manifest,a.authorization,a.authorization_sha256,a.output)) or a.run_root:p.error('New manifest/grant/SHA/output required')
    payload=c.private_path(a.authorization.absolute()).read_bytes()
    if c.file_hash(a.authorization)!=a.authorization_sha256:raise ValueError('PILOT_AUTHORIZATION_SHA')
    m=c.read(c.private_path(a.manifest.absolute()));g=c.read(a.authorization)
    fresh=c.launch_gate(ROOT,m,g,a.output)
    if c.file_hash(a.authorization)!=a.authorization_sha256:raise ValueError('PILOT_AUTHORIZATION_CHANGED')
    lock=c.safe_path(ROOT,m['one_shot_lock']);lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('x') as f:f.write(json.dumps(dict(manifest_digest=digest(m),authorization_sha256=a.authorization_sha256)));f.flush();os.fsync(f.fileno())
    out=a.output.resolve();out.mkdir(exist_ok=False);writer=AtomicWriter(out,byte_cap=m['base_manifest']['plan']['caps_proposed']['output_bytes'])
    writer.write('frozen_manifest.json',m);writer.write('authorization.json',g)
    with (out/'authorization_source.json').open('xb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
    writer.write('launch_binding.json',dict(manifest_digest=digest(m),authorization_digest=digest(g),authorization_source_sha256=a.authorization_sha256))
    from trottertracks.resource_applicability.h6_conditional_s2_execution_v1 import orchestrate
    return orchestrate(ROOT,out,m,writer,fresh)


if __name__=='__main__':raise SystemExit(main())
