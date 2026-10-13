#!/usr/bin/env python3
"""One explicitly approved H8 core, or its disposable owned worker."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.resource_applicability.h8_resource_map_execution_v1 import read,worker,orchestrate
from trottertracks.resource_applicability.ax2b_h4_contract_v5 import file_hash

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path);p.add_argument('--grant',type=Path)
    p.add_argument('--manifest-sha256');p.add_argument('--grant-sha256')
    p.add_argument('--execute',action='store_true');p.add_argument('--worker',type=Path)
    p.add_argument('--run-root',type=Path)
    a=p.parse_args()
    if a.worker:
        if a.run_root is None:raise ValueError('CORE_WORKER_RUN_ROOT')
        return worker(ROOT,a.worker,a.run_root)
    if not all([a.manifest,a.grant,a.manifest_sha256,a.grant_sha256,a.execute]):
        raise ValueError('CORE_EXECUTE_PINNED_BYTES_REQUIRED')
    if file_hash(a.manifest)!=a.manifest_sha256 or file_hash(a.grant)!=a.grant_sha256:
        raise ValueError('CORE_MANIFEST_GRANT_BYTES')
    return orchestrate(ROOT,read(a.manifest),read(a.grant))

if __name__=='__main__':raise SystemExit(main())
