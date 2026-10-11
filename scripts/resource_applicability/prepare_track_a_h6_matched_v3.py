#!/usr/bin/env python3
"""Seal metadata and saved input headers only. Never imports a numerical port."""
import argparse
import subprocess
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.resource_applicability.h6_matched_contract_v3 import (
    preparation, source_paths, verify_sources, verify_parent, environment, resources, file_hash, NAMESPACE, check_inherited_limits)
from trottertracks.resource_applicability.ax2b_limits import exclusive_json
from trottertracks.resource_applicability.ax2a_preparation import digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-commit',required=True)
    p.add_argument('--cpus',type=int,nargs=4,required=True)
    a=p.parse_args();check_inherited_limits();m=preparation()
    from trottertracks.resource_applicability.h6_matched_reuse_v3 import evidence
    reused=evidence(ROOT)
    m.update(source_commit=a.source_commit,source_hashes={n:file_hash(ROOT/n) for n in source_paths(ROOT)},
             input_identity=verify_parent(ROOT),environment=environment(),assigned_resources=resources(a.cpus),
             reuse_registry=reused,validation_source_hashes={p:file_hash(ROOT/p) for p in [
                 'tests/tracks/resource_applicability/test_h6_matched_v3.py',
                 'tests/tracks/resource_applicability/test_h6_matched_v2.py',
                 'tests/tracks/resource_applicability/test_h6_matched_v1.py',
                 'tests/tracks/resource_applicability/test_ax2b_native_df_v5.py',
                 'tests/tracks/resource_applicability/test_ax2b_stream_fingerprint_v5.py']},
             exclusive_output=dict(repository_path=NAMESPACE+'launch_v1',absolute_path=str(ROOT/(NAMESPACE+'launch_v1'))),
             execution_plan_sealed=True)
    verify_sources(ROOT,a.source_commit,m['source_hashes'])
    if (ROOT/(NAMESPACE+'launch_v1')).exists():raise ValueError('OUTPUT_ALREADY_EXISTS')
    if m['input_identity']!=reused['input_identity'] or m['environment']!=reused['environment']:
        raise ValueError('SUPPLEMENT_INPUT_ENVIRONMENT_BRIDGE')
    exclusive_json(a.output,m)
    print('SEALED_PREPARATION_ONLY H6_NOT_AUTHORIZED '+digest(m))


if __name__=='__main__':raise SystemExit(main())
