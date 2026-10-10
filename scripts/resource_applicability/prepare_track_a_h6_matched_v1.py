#!/usr/bin/env python3
"""Seal metadata and saved input headers only. Never imports a numerical port."""
import argparse
import subprocess
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.resource_applicability.h6_matched_contract_v1 import (
    preparation, source_paths, verify_sources, verify_parent, environment, resources, file_hash, NAMESPACE)
from trottertracks.resource_applicability.ax2b_limits import exclusive_json
from trottertracks.resource_applicability.ax2a_preparation import digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-commit',required=True)
    p.add_argument('--cpus',type=int,nargs=4,required=True)
    a=p.parse_args();m=preparation()
    m.update(source_commit=a.source_commit,source_hashes={n:file_hash(ROOT/n) for n in source_paths(ROOT)},
             input_identity=verify_parent(ROOT),environment=environment(),assigned_resources=resources(a.cpus),
             exclusive_output=dict(repository_path=NAMESPACE+'launch_v1',absolute_path=str(ROOT/(NAMESPACE+'launch_v1'))),
             execution_plan_sealed=True)
    verify_sources(ROOT,a.source_commit,m['source_hashes'])
    if (ROOT/(NAMESPACE+'launch_v1')).exists():raise ValueError('OUTPUT_ALREADY_EXISTS')
    exclusive_json(a.output,m)
    print('SEALED_PREPARATION_ONLY H6_NOT_AUTHORIZED '+digest(m))


if __name__=='__main__':raise SystemExit(main())
