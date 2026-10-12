#!/usr/bin/env python3
"""Read-only fast continuation source/profile/input/reuse audit; no launch."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks.resource_applicability import h6_confirmation_fast_contract_v1 as c


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True)
    a=p.parse_args();m=c.read(c.private_path(a.manifest.absolute()));c.validate_plan(ROOT,m,launch=True)
    out=c.private_path(m['exclusive_output']['absolute_path'])
    if out.exists() or out.is_symlink():raise ValueError('FAST_OUTPUT_USED')
    print(json.dumps(dict(status='FAST_CONTINUATION_LOCAL_AUDIT_PASS_NOT_AUTHORIZED',
        source_commit=m['source_commit'],source_files=len(c.sources(ROOT)),
        parent_complete_groups_reused=len(m['confirmation_reuse']['accepted']),
        remaining_groups=len(m['registrations']),workers=8,hash_reuse_policy=c.hash_policy(),
        source_profile_input_seed_selection_reuse_verified=True,
        approved=False,runtime_authorization=False,allowed_cpus=[],absolute_launch_command=None,
        budget_amendments=m['cumulative_accounting']['amendments'],actual_fast_calls=0),sort_keys=True))


if __name__=='__main__':raise SystemExit(main())
