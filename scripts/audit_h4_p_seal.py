"""Metadata acceptance of a technically sealed P; never starts P science."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import h4_dp_contract as c
from prepare_h4_p_seal import PRIVATE, PUBLIC, ROOT, preserve


def main():
    spec_path=PRIVATE/'P_contract_TECHNICALLY_SEALED_PRIVATE.json'
    grant_path=PRIVATE/'P_grant_INVALID.json'
    spec,grant=c.read(spec_path),c.read(grant_path)
    c.require(spec['status']=='SEALED_TECHNICAL_PENDING_USER_APPROVAL'
        and not spec['approved_by_user'] and not spec['runtime_authorization']
        and not grant['approved_by_user'] and not grant['runtime_authorization'],'P_REMAINS_UNAUTHORIZED')
    c.require(c.digest(spec)==grant['spec_digest'] and spec['SOURCE']==grant['SOURCE'],'PRIVATE_SPEC_GRANT_BINDING')
    c.require(not (PUBLIC/'completion_audit.json').exists(),'NEW_TECHNICAL_SEAL_AUDIT')
    for name,h in spec['source_hashes'].items():
        c.require(c.sha(ROOT/name)==h,'P_SOURCE_BYTES')
        raw=subprocess.check_output(['git','show',spec['SOURCE']+':'+name],cwd=ROOT)
        c.require(hashlib.sha256(raw).hexdigest()==h,'P_SOURCE_GIT_BYTES')
    for item in spec['inputs']:c.require(c.sha(item['path'])==item['sha256'],'P_INPUT_BYTES')
    for name,h in spec['provider_hashes'].items():c.require(c.sha(name)==h,'P_ACTUAL_PROVIDER_CLOSURE')
    for name,h in spec['runtime_hashes'].items():c.require(c.sha(name)==h,'P_RUNTIME_CLOSURE')
    c.validate_tasks(spec)
    receipt=c.read(spec['seed_inventory_receipt'])
    coverage={x['scope']:x for x in receipt['coverage']}
    for name in ('CONDITIONAL_B','T_0.4','T_1.6','DF_POLICY_FP'):
        c.require(coverage[name]['trajectories']==32,'COMPLETE_RECORDED_H4_COHORT')
    c.require(coverage['EXPLORATION_A_N8']['trajectories']==128
        and coverage['GEOMETRY_SIX_DISTANCES']['stochastic_candidates']==1164
        and coverage['GEOMETRY_SIX_DISTANCES']['trajectories']==37248,'COMPLETE_OLD_METADATA_COVERAGE')
    c.require(len(receipt['geometry_checks'])==6 and all(not x['collisions'] for x in receipt['geometry_checks']),
              'SIX_GEOMETRY_COLLISION_ZERO')
    previous=c.read(PRIVATE/'prior_preservation_PRIVATE.json');preserve(previous)
    run=Path(spec['output']).parent
    c.require(not list(run.iterdir()),'UNUSED_P_SCIENTIFIC_RUN')
    checks=PRIVATE/'technical_checks';checks.mkdir(exist_ok=True)
    rejections=[]
    for script in ['run_h4_dp_verification.py','h4_dp_witness.py']:
        proc=subprocess.run([spec['python'],str(ROOT/'scripts'/script),'--spec',str(spec_path),
            '--grant',str(grant_path)],capture_output=True,text=True,
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1'))
        c.require(proc.returncode==1 and 'DRAFT_UNAUTHORIZED' in proc.stderr,'UNAUTHORIZED_P_ENTRYPOINT_REJECTS')
        (checks/(script+'.stderr.log')).write_text(proc.stderr)
        c.require(not Path(spec['output']).exists() and not Path(spec['lock']).exists()
                  and not list(run.iterdir()),'NO_P_OUTPUT_LOCK_OR_SCIENTIFIC_DRIVER')
        rejections.append(dict(entrypoint=script,exit_code=1,reason='DRAFT_UNAUTHORIZED',
            rejected_before_scientific_import_or_output=True))
    audit=dict(status='TECHNICAL_SEAL_COMPLETE_MANDATORY_STOP',SOURCE=spec['SOURCE'],
        private_spec_digest=c.digest(spec),P_final_contract_version=2,source_files=len(spec['source_hashes']),
        inputs=len(spec['inputs']),provider_closure_files=len(spec['provider_hashes']),runtime_files=len(spec['runtime_hashes']),
        fixed_trajectories=len(spec['tasks']),new_master_count=receipt['new_master_count'],
        new_seed_slots=receipt['new_seed_slots'],internal_seed_collisions=0,old_seed_collisions=0,
        old_explicit_seed_values=len(c.read(spec['seed_exclusions'])['values']),
        geometry_seed_slots_checked=sum(x['old_seed_slots_checked'] for x in receipt['geometry_checks']),
        prior_records_and_copies_preserved=len(previous),all_fixed_contract_checks_passed=True,
        lazy_original_A_module_import_passed=True,entrypoint_rejections=rejections,
        P_scientific_driver_started=False,P_output_exists=False,P_one_shot_exists=False,
        RTE_events_sampled=0,native_evolution=0,wrapper_compile=0,new_statistics=0,
        approved_by_user=False,runtime_authorization=False,grant_valid=False,remote_push=False,
        next_stage_authorized=False)
    c.save(PUBLIC/'completion_audit.json',audit)
    c.save(PRIVATE/'completion_audit_PRIVATE.json',audit)
    paths=[ROOT/name for name in spec['source_hashes']]+[Path(x['path']) for x in spec['inputs']]
    paths += [Path(name) for name in spec['provider_hashes']]
    paths += [PRIVATE/name for name in ['technical_preparation_authorization_PRIVATE.json',
        'P_contract_TECHNICALLY_SEALED_PRIVATE.json','P_grant_INVALID.json','A_adapter_for_P.json',
        'inventory_inputs_PRIVATE.json','old_seed_exclusions_PRIVATE.json','geometry_seed_cases_PRIVATE.json',
        'seed_inventory_receipt_PRIVATE.json','new_confirmation_seeds_PRIVATE.json','completion_audit_PRIVATE.json']]
    store=PRIVATE/'seal_preservation_store';store.mkdir(exist_ok=False);rows=[]
    for path in paths:
        h=c.sha(path);copy=store/h
        if not copy.exists():copy.write_bytes(path.read_bytes())
        c.require(c.sha(copy)==h,'P_SEAL_BACKUP_BYTES')
        rows.append(dict(path=str(path),sha256=h,copy=str(copy)))
    c.save(PRIVATE/'seal_preservation_manifest_PRIVATE.json',rows)
    c.save(PUBLIC/'preservation_projection.json',dict(prior_records=len(previous),new_records=len(rows),
        backup_copies_verified=True,private_manifest_sha256=c.sha(PRIVATE/'seal_preservation_manifest_PRIVATE.json')))
    print('P_TECHNICAL_SEAL_AUDIT_PASS; RUNNER_WITNESS_REJECTED; SCIENCE_ZERO; MANDATORY_STOP')


if __name__=='__main__':main()
