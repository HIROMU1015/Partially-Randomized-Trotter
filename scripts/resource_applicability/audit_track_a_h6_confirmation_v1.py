#!/usr/bin/env python3
"""Read-only source/profile/input/selection/fixed32 audit; no grant or launch."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks.resource_applicability.h6_confirmation_contract_v1 import (
    validate_plan, read, private_path, file_hash, sources, LOCK)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    a = p.parse_args();m = read(private_path(a.manifest.absolute()))
    validate_plan(ROOT,m)
    out = private_path(m['exclusive_output']['absolute_path'])
    lock = ROOT/LOCK
    if out.exists() or out.is_symlink() or lock.exists() or lock.is_symlink():
        raise ValueError('CONFIRMATION_PREPARATION_OUTPUT_OR_LOCK_USED')
    print(json.dumps(dict(status='LOCAL_CONFIRMATION_PREPARATION_AUDIT_PASS',
        source_commit=m['source_commit'],source_files=len(sources(ROOT)),
        selected_ids=m['exploration_binding']['selection']['selected_ids'],
        confirmation_groups=len(m['registrations']),source_profile_input_saved_exploration_verified=True,
        seed_namespace_independent=True,candidate_selection_sealed=True,approved=False,
        runtime_authorization=False,allowed_cpus=[],absolute_launch_command=None,
        actual_confirmation_calls=0,new_grant_created=False,
        cap_amendment=m['cumulative_accounting']['amendments'],
        manifest_sha256=file_hash(a.manifest),output_unused=True,one_shot_unused=True),sort_keys=True))


if __name__ == '__main__':
    raise SystemExit(main())
