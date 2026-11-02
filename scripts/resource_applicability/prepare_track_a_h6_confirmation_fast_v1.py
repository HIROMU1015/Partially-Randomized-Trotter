#!/usr/bin/env python3
"""Metadata-only seal AFTER the original4 run has stopped; never stop/launch it."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks.resource_applicability.ax2b_limits import exclusive_json
from trottertracks.resource_applicability import h6_confirmation_fast_contract_v1 as c


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-commit',required=True);p.add_argument('--parent-run',type=Path,required=True)
    a=p.parse_args();binding=c.exploration_binding(ROOT);base=c.read(ROOT/c.BASE)
    reuse=c.parent_confirmation_receipt(a.parent_run,binding)
    out=ROOT/c.NAMESPACE/'launch_confirmation_fast_v1'
    if out.exists() or out.is_symlink():raise ValueError('FAST_OUTPUT_USED')
    m=dict(schema=c.SCHEMA,kind=c.KIND,repository_root=str(ROOT),source_commit=a.source_commit,
        source_hashes={x:c.file_hash(ROOT/x) for x in c.sources(ROOT)},
        base_manifest=base,base_sha256=c.file_hash(ROOT/c.BASE),exploration_binding=binding,
        roundoff_preparation_binding=binding['roundoff_preparation_binding'],confirmation_reuse=reuse,
        cumulative_accounting=c.continuation_accounting(binding,base,reuse),
        rare_order_plan=c.rare_order_plan(binding),hash_reuse_policy=c.hash_policy(),
        registrations=c.remaining(binding['selection'],reuse),
        proposed_resources=c.proposed_resources(),observation_policy=c.observation_policy(),
        exclusive_output=dict(repository_path=str(out.relative_to(ROOT)),absolute_path=str(out)),
        one_shot_lock=c.LOCK,candidate_selection_sealed=True,execution_plan_sealed=True,
        approved=False,runtime_authorization=False,allowed_cpus=[],confirmation_authorized=False,
        H8_authorized=False,retry=False,resume=False,mandatory_stop=True,next_stage_authorized=False,
        absolute_launch_command=None)
    c.validate_plan(ROOT,m)
    out.parent.mkdir(parents=True,exist_ok=True)
    prep=ROOT/c.PREPARATION/'sealed_v1';prep.mkdir(parents=True,exist_ok=False)
    exclusive_json(prep/'manifest_fast_v1.json',m)
    exclusive_json(prep/'authorization_draft_v1.json',c.grant_draft(m))
    print('FAST_CONTINUATION_SEALED_NOT_AUTHORIZED; complete_reuse='+str(len(reuse['accepted']))+
        '; remaining='+str(len(m['registrations']))+'; amendments='+str(m['cumulative_accounting']['amendments']))


if __name__=='__main__':raise SystemExit(main())
