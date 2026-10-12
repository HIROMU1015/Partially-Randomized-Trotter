#!/usr/bin/env python3
"""Metadata-only technical seal; never decode arrays, sample, compile or approve."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks.resource_applicability import h6_conditional_s2_contract_v1 as c
from trottertracks.resource_applicability.ax2b_limits import exclusive_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-commit',required=True);a=p.parse_args()
    base=c.read(ROOT/c.BASE);prior=c.prior_binding(ROOT);out=ROOT/c.NAMESPACE/'launch_pilot_v1'
    if out.exists() or out.is_symlink() or (ROOT/c.LOCK).exists():raise ValueError('PILOT_OUTPUT_OR_LOCK_USED')
    m=dict(schema=c.SCHEMA,kind=c.KIND,repository_root=str(ROOT),source_commit=a.source_commit,
        source_hashes={p:c.file_hash(ROOT/p) for p in c.sources(ROOT)},base_manifest=base,base_sha256=c.file_hash(ROOT/c.BASE),
        prior_binding=prior,roundoff_preparation_binding=prior['roundoff_preparation_binding'],
        cumulative_accounting=c.accounting(prior,base),registrations=c.registrations(),target_cell=c.CELL,phase=c.PHASE,
        conditioning=dict(stratum='s2',scope='whole trajectory',R=16,K=2,
            count_law='Binomial(16,p2) conditional M>=1; uniform M-subset',
            component_law='original independent component probabilities',quantum_weights_changed=False),
        proposed_resources=c.proposed_resources(),observation_policy=c.observation_policy(),hash_reuse_policy=c.hash_policy(),
        exclusive_output=dict(repository_path=str(out.relative_to(ROOT)),absolute_path=str(out)),one_shot_lock=c.LOCK,
        execution_plan_sealed=True,approved=False,runtime_authorization=False,allowed_cpus=[],
        retry=False,resume=False,mandatory_stop=True,next_stage_authorized=False,H8_authorized=False,absolute_launch_command=None)
    c.validate_plan(ROOT,m,launch=True)
    out.parent.mkdir(parents=True,exist_ok=True)
    prep=ROOT/c.PREPARATION/'sealed_v1';prep.mkdir(parents=True,exist_ok=False)
    exclusive_json(prep/'manifest_pilot_v1.json',m);exclusive_json(prep/'authorization_draft_v1.json',c.grant_draft(m))
    exclusive_json(prep/'pilot_registrations_v1.json',m['registrations'])
    print('CONDITIONAL_S2_PILOT_TECHNICALLY_SEALED_NOT_AUTHORIZED; source='+a.source_commit+'; groups=8; wrappers=16; compile_cap=1736; trajectory_cap=852')


if __name__=='__main__':raise SystemExit(main())
