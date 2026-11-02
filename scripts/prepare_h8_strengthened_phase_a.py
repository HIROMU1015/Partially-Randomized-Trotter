"""Read-only binding and new UNAPPROVED config, never a scientific launcher.

Explicit paths identify already existing receipts. No NPZ arrays or real Gates
are restored. Existing grant, ledger, worktree, profile and results are read-only.
"""
import argparse
import json
from pathlib import Path
import subprocess

import h8_strengthened_contract as contract
import h8_strengthened_export as exported
import h8_strengthened_runtime as runtime
from h8_strengthened_signal import saved_reference

SOURCES=(
    'scripts/h8_strengthened_contract.py','scripts/h8_strengthened_export.py',
    'scripts/h8_strengthened_runtime.py','scripts/h8_strengthened_signal.py',
    'scripts/run_h8_strengthened_phase_a.py','scripts/launch_h8_strengthened_phase_a.py',
    'scripts/prepare_h8_strengthened_phase_a.py','scripts/track_a_phase_a_parallel.py',
    'scripts/h6_native_b0i_bridge_v1.py','scripts/final_pre_review_execution_v1.py',
    'scripts/pre_review_native_b0i_prototype_v1.py','scripts/d0_saved_value_reanalysis_v1.py',
    'scripts/d0_frozen_shot_rule_v1.py','src/trotterlib/pf_decomposition.py',contract.PROPOSAL)


def materialize(old_contract,old_mapping,export_path,reference_path,scalar_path,
                expected_path,font,protected,private_directory):
    old=runtime.read(old_contract);mapping=runtime.read(old_mapping)
    expected=runtime.read(expected_path)['H8']
    hashes={k:v['preparation_hash'] for k,v in expected.items()}
    audit=exported.validate_export(runtime.read(export_path),expected_hashes=hashes)
    ref=runtime.read(reference_path);saved_reference(ref,old['binding'])
    map_manifest=Path(mapping['signal_summary.json']).with_name('manifest.json')
    history=runtime.read(map_manifest)
    if history['resume_parent']['file_hashes'].get(str(reference_path))!=runtime.sha(reference_path):
        raise ValueError('REFERENCE_NOT_BOUND_TO_SAVED_COMPLETION_LINEAGE')
    original=runtime.read(Path(reference_path).with_name('worker_terminal.json'))
    accepted=history['resume_parent']['accepted']['signal']
    if (original['status']!='COMPLETE' or not any(
            row['directory']==str(Path(reference_path).parent) for row in accepted.values())):
        raise ValueError('REFERENCE_ORIGINAL_WORKER_NOT_COMPLETE')
    provider=Path(old['native_provider']['checkout'])
    for name,h in old['native_provider']['source_hashes'].items():
        raw=subprocess.check_output(['git','show',contract.PROVIDER+':'+name],cwd=provider)
        if __import__('hashlib').sha256(raw).hexdigest()!=h or runtime.sha(provider/name)!=h:
            raise ValueError('OLD_PROVIDER_CLOSURE_CHANGED')
    files=[]
    for row in old['input_receipts']:
        if row['logical_name'] in ('h8_input_snapshot.npz','snapshot_receipt.json','df_receipt.json','projection_receipt.json'):
            path=mapping[row['logical_name']]
            if Path(path).stat().st_size!=row['bytes'] or runtime.sha(path)!=row['sha256']:
                raise ValueError('OLD_INPUT_CHANGED')
            files.append(dict(row,path=path))
    if len(files)!=4:raise ValueError('FOUR_OLD_FIXED_INPUTS_REQUIRED')
    extras={'h8_preparation_export':export_path,'full_H_reference':reference_path,
            'p21_scalar_receipts':scalar_path,'preparation_identity_receipts':expected_path,'font_seed':font['path'],
            'reference_lineage_manifest':map_manifest,
            'reference_original_worker':Path(reference_path).with_name('worker_terminal.json')}
    for name,path in extras.items():
        path=Path(path)
        files.append(dict(logical_name=name,path=str(path),bytes=path.stat().st_size,sha256=runtime.sha(path)))
    allmapping={r['logical_name']:r['path'] for r in files}
    own={name:runtime.sha(contract.ROOT/name) for name in SOURCES}
    plan=contract.plan();directory=runtime.private_path(private_directory)
    if directory.exists():raise ValueError('NEW_PRIVATE_DIRECTORY_REQUIRED')
    config=dict(schema='H8_PHASE_A_PRIVATE_CONFIG_V1',SOURCE=None,execution_sealed=False,
        plan_sha256=contract.digest(plan),source_hashes=own,provider_root=str(provider),
        provider_source=contract.PROVIDER,provider_hashes=old['native_provider']['source_hashes'],
        python=old['profile']['executable'],runtime_file_hashes=old['profile']['runtime_file_hashes'],
        expected_profile_digest=old['expected_profile_digest'],binding=old['binding'],
        input_mapping=allmapping,files=files,expected_preparation_hashes=hashes,font_seed=font,
        export_sha256=runtime.sha(export_path),reference_sha256=runtime.sha(reference_path),
        epsilons=contract.proposal()['target']['epsilons'],resources=plan['resources'],
        output=str(directory/'signal_output01'),lock=str(directory/'signal_ONE_SHOT'),
        control=str(directory/'signal_control01'),protected_files=protected['files'],
        protected_worktrees=protected['worktrees'],allowed_cpus=[],approved=False,
        runtime_authorization=False,absolute_launch_command=None)
    grant=dict(schema='H8_STRENGTHENED_PHASE_A_GRANT_V1',approved_by_user=False,
        runtime_authorization=False,approval_text=None,SOURCE=None,scope=contract.SCOPE,
        config_sha256=None,plan_sha256=config['plan_sha256'],call_caps=plan['call_caps'],
        resources=plan['resources'],driver_cpu=None,monitor_cpu=None,worker_cpus=[],allowed_cpus=[],
        one_shot=True,retry=False,resume=False,next_stage_authorized=False)
    directory.mkdir(mode=0o700)
    for name,data in (('config_UNAPPROVED.json',config),('grant_INVALID_DRAFT.json',grant)):
        runtime.Writer(directory).write(name,data)
    return dict(status='BOUND_METADATA_ONLY_NOT_AUTHORIZED',source_hashes=own,
        plan_sha256=config['plan_sha256'],provider_SOURCE=contract.PROVIDER,
        provider_files=len(config['provider_hashes']),runtime_files=len(config['runtime_file_hashes']),
        runtime_full_rehash_repeated=False,input_receipts=[{k:v for k,v in r.items() if k!='path'} for r in files],
        export_metadata_audit=audit,config_sha256=runtime.sha(directory/'config_UNAPPROVED.json'),
        full_H_reference_origin_SOURCE=original['source_commit'],
        full_H_reference_completion_lineage_bound=True,
        runtime_authorization=False,execution_sealed=False,scientific_array_loads=0,gate_objects_restored=0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('old-contract','old-mapping','export','reference','scalar-receipts',
                 'preparation-receipts','font-receipt','protected-receipt','private-directory'):
        p.add_argument('--'+name,required=True)
    a=p.parse_args()
    result=materialize(a.old_contract,a.old_mapping,a.export,a.reference,a.scalar_receipts,
        a.preparation_receipts,runtime.read(a.font_receipt),runtime.read(a.protected_receipt),a.private_directory)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
