#!/usr/bin/env python3
"""Seal saved exploration/top2-union/fresh32 preparation; issue no grant."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks.resource_applicability.ax2b_limits import exclusive_json
from trottertracks.resource_applicability.h6_confirmation_contract_v1 import (
    BASE, PREPARATION, NAMESPACE, KIND, SCHEMA, LOCK, sources, file_hash, read,
    proposed_resources, observation_policy, exploration_binding, accounting,
    rare_order_plan, validate_plan, grant_draft)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-commit', required=True)
    a = parser.parse_args()
    binding = exploration_binding(ROOT); base = read(ROOT/BASE)
    out = ROOT/NAMESPACE/'launch_confirmation_v1'
    if out.exists() or out.is_symlink():
        raise ValueError('CONFIRMATION_OUTPUT_ALREADY_USED')
    out.parent.mkdir(parents=True, exist_ok=True)
    m = dict(schema=SCHEMA, kind=KIND, repository_root=str(ROOT),
        source_commit=a.source_commit, source_hashes={p:file_hash(ROOT/p) for p in sources(ROOT)},
        base_manifest=base, base_sha256=file_hash(ROOT/BASE), exploration_binding=binding,
        roundoff_preparation_binding=binding['roundoff_preparation_binding'],
        cumulative_accounting=accounting(binding,base), rare_order_plan=rare_order_plan(binding),
        registrations=binding['selection']['registrations'], proposed_resources=proposed_resources(),
        observation_policy=observation_policy(),
        exclusive_output=dict(repository_path=str(out.relative_to(ROOT)),absolute_path=str(out)),
        one_shot_lock=LOCK, candidate_selection_sealed=True, execution_plan_sealed=True,
        approved=False, runtime_authorization=False, allowed_cpus=[], confirmation_authorized=False,
        H8_authorized=False, retry=False, resume=False, mandatory_stop=True,
        next_stage_authorized=False, absolute_launch_command=None)
    validate_plan(ROOT,m)
    prep = ROOT/PREPARATION/'sealed_v1'; prep.mkdir(parents=True,exist_ok=False)
    exclusive_json(prep/'manifest_confirmation_v1.json',m)
    exclusive_json(prep/'authorization_draft_v1.json',grant_draft(m))
    exclusive_json(prep/'candidate_selection_v1.json',binding['selection'])
    exclusive_json(prep/'rare_order_plan_v1.json',m['rare_order_plan'])
    print('INDEPENDENT32_PREPARED_NOT_AUTHORIZED; technical_seal=true; future trajectory cap amendment requires approval')


if __name__ == '__main__':
    raise SystemExit(main())
