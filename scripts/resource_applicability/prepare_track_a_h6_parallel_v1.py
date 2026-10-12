#!/usr/bin/env python3
"""Bind frozen scientific bytes and four-worker proposals. Issue no grant."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks.resource_applicability.h6_parallel_contract_v1 import (BASE,PREPARATION,NAMESPACE,
    KIND,sources,file_hash,read,proposed_resources,registrations,phase_a_receipt,carry_calls,
    observation_policy,validate_plan,grant_draft,static_demand,LOCK,SCHEMA,prior_stop_accounting,saved_result_reuse)
from trottertracks.resource_applicability.ax2b_limits import exclusive_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-commit',required=True)
    p.add_argument('--bundle-name',choices=('candidate_v1','sealed_v1'),default='candidate_v1')
    p.add_argument('--roundoff-preparations',type=Path);p.add_argument('--roundoff-receipt',type=Path);a=p.parse_args()
    if bool(a.roundoff_preparations)!=bool(a.roundoff_receipt):p.error('Archive and numerical bridge receipt are both required')
    binding=None
    if a.roundoff_preparations:
        binding=dict(archive_path=str(a.roundoff_preparations.absolute()),archive_sha256=file_hash(a.roundoff_preparations),
            archive_bytes=a.roundoff_preparations.stat().st_size,receipt_path=str(a.roundoff_receipt.absolute()),
            receipt_sha256=file_hash(a.roundoff_receipt))
    base=read(ROOT/BASE);proof=phase_a_receipt(base);out=ROOT/NAMESPACE/'launch_parallel_v1';out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise ValueError('PARALLEL_OUTPUT_USED')
    reuse=saved_result_reuse(ROOT,base);rows=registrations(base,reuse)
    m=dict(schema=SCHEMA,kind=KIND,repository_root=str(ROOT),
        source_commit=a.source_commit,source_hashes={q:file_hash(ROOT/q) for q in sources(ROOT)},
        base_manifest=base,base_sha256=file_hash(ROOT/BASE),phase_a_receipt=proof,
        registrations=rows,verified_reuse=reuse,
        carry_known_calls=carry_calls(ROOT,base,proof,roundoff=binding is not None,reuse=reuse),
        roundoff_preparation_binding=binding,
        prior_stop_accounting=prior_stop_accounting(ROOT),
        static_demand=static_demand(rows),
        proposed_resources=proposed_resources(),observation_policy=observation_policy(),
        exclusive_output=dict(repository_path=str(out.relative_to(ROOT)),absolute_path=str(out)),
        one_shot_lock=LOCK,
        execution_plan_sealed=proof is not None and binding is not None,approved=False,runtime_authorization=False,allowed_cpus=[],
        confirmation_authorized=False,H8_authorized=False,retry=False,resume=False,mandatory_stop=True)
    validate_plan(ROOT,m);prep=ROOT/PREPARATION/a.bundle_name;prep.mkdir(parents=True,exist_ok=False)
    exclusive_json(prep/'manifest_candidate_v1.json',m);exclusive_json(prep/'authorization_draft_v1.json',grant_draft(m))
    print('PARALLEL_PREPARED_NOT_AUTHORIZED; technically_sealed='+str(m['execution_plan_sealed'])+'; roundoff bridge receipt required before launch')


if __name__=='__main__':raise SystemExit(main())
