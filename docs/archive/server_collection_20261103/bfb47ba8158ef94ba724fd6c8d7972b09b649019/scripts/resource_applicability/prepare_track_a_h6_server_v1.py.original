#!/usr/bin/env python3
"""Seal saved bytes and resource proposals only; never import a science port."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks.resource_applicability.h6_server_contract_v1 import (
    KIND,PREPARATION,NAMESPACE,OLD_MANIFEST,SCIENCE_SOURCE,flags,source_paths,
    verify_science,phase_scope,environment,verify_parent,file_hash,resources,
    observation_policy,grant_draft,verify_bindings)
from trottertracks.resource_applicability.h6_matched_reuse_v3 import evidence,committed_hashes
from trottertracks.resource_applicability.ax2a_preparation import digest
from trottertracks.resource_applicability.ax2b_limits import exclusive_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-commit',required=True)
    p.add_argument('--profile',type=Path,required=True);p.add_argument('--cpus',nargs=4,type=int,required=True)
    a=p.parse_args();base=ROOT/PREPARATION;base.mkdir(parents=True,exist_ok=True)
    old=json.loads((ROOT/OLD_MANIFEST).read_text());registry=evidence(ROOT);inputs=verify_parent(ROOT)
    env=environment()
    if env!=old['environment'] or env!=registry['environment']:raise ValueError('EXACT_LEGACY_ENVIRONMENT_REQUIRED')
    verify_science(ROOT);scope=phase_scope(registry)
    out=ROOT/NAMESPACE/'launch_server_v1';out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise ValueError('FRESH_OUTPUT_REQUIRED')
    profile=a.profile.resolve();profile.relative_to(ROOT)
    m=dict(schema='h6_server_phase_a_manifest_v1',kind=KIND,repository_root=str(ROOT),
        source_commit=a.source_commit,science_source=SCIENCE_SOURCE,
        source_hashes={n:file_hash(ROOT/n) for n in source_paths(ROOT)},
        plan=old['plan'],science_plan_digest=digest(old['plan']),reuse_registry=registry,
        input_identity=inputs,environment=env,phase_scope=scope,
        profile=dict(repository_path=str(profile.relative_to(ROOT)),sha256=file_hash(profile)),
        assigned_resources=resources(a.cpus),observation_policy=observation_policy(),
        exclusive_output=dict(repository_path=str(out.relative_to(ROOT)),absolute_path=str(out)),
        one_shot_lock='.server-preparation/h6-server-phase-a-v1/phase-a.lock',
        execution_plan_sealed=True,**flags())
    committed_hashes(ROOT,a.source_commit,m['source_hashes']);verify_bindings(ROOT,m)
    exclusive_json(base/'sealed_manifest_server_v1.json',m)
    exclusive_json(base/'authorization_draft_server_v1.json',grant_draft(m))
    exclusive_json(base/'receipt_server_v1.json',dict(source_commit=a.source_commit,
        science_source=SCIENCE_SOURCE,science_closure_count=224,science_source_unchanged=True,
        old_raw_paths=len(registry['raw_hashes']),frozen_input_paths=len(inputs['parent_raw_hashes']),
        reused_groups=registry['accepted_groups'],reused_wrappers=registry['accepted_wrappers'],
        signals=registry['signal_count'],b3=registry['b3_count'],missing_groups=registry['missing_groups'],
        missing_indices=[x['index'] for x in registry['missing']],phase_scope=scope,
        raw_bytes=sum((ROOT/q).stat().st_size for q in registry['raw_hashes']),
        input_files=[dict(path=q,bytes=(ROOT/q).stat().st_size,sha256=sha)
                     for q,sha in inputs['parent_raw_hashes'].items()],
        environment_exact_match=True,manifest_digest=digest(m),
        H6_science_runs=0,new_approved_grants=0,grant_consumed=False,**flags()))
    print('SEALED_NOT_AUTHORIZED',digest(m),'PHASE_A_TASK',scope['task_digest'])


if __name__=='__main__':raise SystemExit(main())
