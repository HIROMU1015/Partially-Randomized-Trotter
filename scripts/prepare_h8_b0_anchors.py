"""Bind saved bytes and write a NEW invalid grant draft; never execute science."""
import argparse
import json
from pathlib import Path

import h8_b0_anchor_contract as c
import h8_b0_anchor_runtime as r
import h8_strengthened_export as exported


def prepare(previous_config,protected_path,directory,*,source=None):
    old=r.read(previous_config);before=r.read(protected_path);p=c.plan()
    directory=r.private_path(directory)
    if directory.exists():raise ValueError('NEW_PREPARATION_DIRECTORY_REQUIRED')
    expected=p['binding']['inputs'];files=[]
    for row in expected:
        path=Path(old['input_mapping'][row['logical_name']])
        if path.stat().st_size!=row['bytes'] or r.sha(path)!=row['sha256']:
            raise ValueError('FIXED_SAVED_INPUT_CHANGED')
        files.append(dict(row,path=str(path)))
    font=dict(old['font_seed']);font['name']=font.get('name',font['filename'])
    if Path(font['name']).name!=font['name'] or r.sha(font['path'])!=font['sha256']:raise ValueError('FONT_SEED')
    files.append(dict(logical_name='font_seed',path=font['path'],bytes=Path(font['path']).stat().st_size,sha256=font['sha256']))
    mapping={row['logical_name']:row['path'] for row in files}
    audit=exported.validate_export(r.read(mapping['h8_preparation_export']),expected_hashes=old['expected_preparation_hashes'])
    config=dict(schema='H8_B0_ANCHOR_PRIVATE_CONFIG_V1',SOURCE=source,execution_sealed=source is not None,
        plan_sha256=c.digest(p),source_hashes={name:r.sha(c.ROOT/name) for name in c.SOURCES},
        provider_root=old['provider_root'],provider_source=c.PROVIDER,provider_hashes=old['provider_hashes'],
        runtime_file_hashes=old['runtime_file_hashes'],expected_profile_digest=c.PROFILE,python=old['python'],
        expected_preparation_hashes=old['expected_preparation_hashes'],scientific_binding=old['binding'],
        input_mapping=mapping,files=files,font_seed=font,resources=p['resources'],
        protected_files=[dict(row,path=path) for path,row in before['protected'].items()],
        protected_worktrees=before['worktrees'],output=str(directory/'output01'),
        lock=str(directory/'B0_ANCHOR_ONE_SHOT'),control=str(directory/'control01'))
    # Own source does not become a Git object until the source publication commit.
    # The final prepared directory repeats this with its exact committed SOURCE.
    r.verify_files(config,git=source is not None);r.protected(config)
    directory.mkdir(mode=0o700)
    r.Writer(directory).write('config_UNAPPROVED.json',config)
    grant=dict(schema='H8_B0_ANCHOR_NEW_SCOPE_GRANT_V1',scope=c.SCOPE,
        approved_by_user=False,runtime_authorization=False,approval_text=None,SOURCE=source,
        config_sha256=r.sha(directory/'config_UNAPPROVED.json'),plan_sha256=c.digest(p),
        resources=p['resources'],call_caps=p['call_caps'],one_shot=True,retry=False,resume=False,
        driver_cpu=None,monitor_cpu=None,worker_cpus=[],allowed_cpus=[],command=None,
        next_stage_authorized=False,valid=False)
    r.Writer(directory).write('grant_INVALID_DRAFT.json',grant)
    receipt=dict(status='PREPARED_NOT_AUTHORIZED',SOURCE=source,execution_source_bound=source is not None,
        plan_sha256=c.digest(p),config_sha256=grant['config_sha256'],source_hashes=config['source_hashes'],
        provider_files_verified=len(config['provider_hashes']),runtime_files_verified=len(config['runtime_file_hashes']),
        runtime_mapping_sha256=c.digest(config['runtime_file_hashes']),provider_mapping_sha256=c.digest(config['provider_hashes']),
        input_receipts=[{k:v for k,v in row.items() if k!='path'} for row in files],
        saved_export_metadata=audit,protected_files_unchanged=len(config['protected_files']),
        protected_worktrees_unchanged=len(config['protected_worktrees']),new_scientific_imports=0,
        molecular_objects_restored=0,new_compile=0,new_fit=0,new_valid_grants=0,
        fresh_resource_gate_issued=False,next_stage_authorized=False,mandatory_stop=True)
    r.Writer(directory).write('preparation_receipt.json',receipt)
    return receipt,grant


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('previous-config','protected-receipt','private-directory'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source');a=p.parse_args()
    receipt,_=prepare(a.previous_config,a.protected_receipt,a.private_directory,source=a.source)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
