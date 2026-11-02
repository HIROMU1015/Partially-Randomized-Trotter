"""Materialize V1 UNAPPROVED config from existing receipts, without science."""
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_v1_materialize_runner',ROOT/'scripts/track_a_v1_runner.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)

SOURCES=('src/trotterlib/track_a_v1_contract.py','src/trotterlib/native_df_gate_model.py',
 'scripts/track_a_v1_runner.py','scripts/track_a_v1_signal.py','scripts/track_a_v1_compile.py',
 'scripts/track_a_v1_costs.py','scripts/prepare_track_a_v1.py',
 'scripts/d0_saved_value_reanalysis_v1.py','scripts/d0_frozen_shot_rule_v1.py',
 'scripts/h6_native_b0i_bridge_v1.py','scripts/final_pre_review_execution_v1.py',
 'scripts/pre_review_native_b0i_prototype_v1.py','scripts/track_a_phase_a_parallel.py',
 'scripts/track_a_v1_parallel_signal.py','scripts/launch_track_a_v1_phase_a.py')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--existing-config',required=True);p.add_argument('--private-directory',required=True)
    p.add_argument('--phase-a-parallel',action='store_true')
    args=p.parse_args();old=r.read_json(args.existing_config);directory=Path(args.private_directory)
    if directory.exists() or not directory.is_absolute() or not directory.is_relative_to('/home/AbeHiromu/projects'):
        raise ValueError('NEW_PRIVATE_DIRECTORY_REQUIRED')
    kept={'h6_input_snapshot.npz','snapshot_receipt.json','df_receipt.json','old_preparation_archive','font_seed'}
    receipts=[deepcopy(x) for x in old['files'] if x['logical_name'] in kept]
    if len(receipts)!=5:raise ValueError('FIVE_FIXED_INPUT_RECEIPTS_REQUIRED')
    for item in receipts:
        path=Path(item['path'])
        if path.stat().st_size!=item['bytes'] or r.sha(path)!=item['sha256']:raise ValueError('EXISTING_INPUT_CHANGED')
    hashes={name:r.sha(ROOT/name) for name in SOURCES}
    plan=r.contract.phase_a_plan() if args.phase_a_parallel else r.contract.plan()
    config=dict(schema='track_a_v1_private_preparation_v1',plan_sha256=r.contract.digest(plan),
        provider_root=old['provider_root'],provider_source=r.contract.PROVIDER,
        provider_hashes=old['provider_hashes'],provider_closure_digest=r.contract.digest(old['provider_hashes']),
        python=old['python'],runtime_file_hashes=old['runtime_file_hashes'],
        font_seed=old['font_seed'],
        input_mapping=old['input_mapping'],archive=old['archive'],files=receipts,source_hashes=hashes,
        compiler_options=dict(basis_gates=['rz','sx','x','cx'],optimization_level=1,seed_transpiler=17),
        binding=json.loads((ROOT/'artifacts/resource_applicability/track_a_native_df_metadata_binding/2026-10-16/binding_inventory_v1.json').read_bytes())[0]['binding'],
        profile_reference=old['profile_reference'],carry_policy='NO_RESET_NO_REFUND_UNKNOWN_RETAINED',
        resources=plan['resources'],outputs={k:str(directory/(k+'_output01')) for k in ('signal','compile_validation')},
        locks={k:str(directory/(k+'_ONE_SHOT')) for k in ('signal','compile_validation')},
        frozen_model_sha256=None,frozen_model_digest=None,approved=False,runtime_authorization=False,
        allowed_cpus=[],absolute_launch_command=None)
    if args.phase_a_parallel:
        config.update(phase_a_parallel=True,control=str(directory/'signal_control01'))
        config['outputs']={'signal':str(directory/'signal_output01')}
        config['locks']={'signal':str(directory/'signal_ONE_SHOT')}
    directory.mkdir(mode=0o700)
    with (directory/'config_draft.json').open('x') as f:json.dump(config,f,indent=2);f.write('\n')
    public=dict(schema='track_a_v1_preparation_binding_v1',plan_sha256=config['plan_sha256'],
        provider_source=config['provider_source'],provider_closure_digest=config['provider_closure_digest'],
        input_receipts=[{k:v for k,v in x.items() if k!='path'} for x in receipts],
        binding=config['binding'],profile_reference=config['profile_reference'],source_hashes=hashes,
        private_config=dict(bytes=(directory/'config_draft.json').stat().st_size,
                            sha256=r.sha(directory/'config_draft.json')),
        source_commit_binding='final committed HEAD required before grant',
        old_grants_modified=False,new_grants=0,scientific_array_loads=0,
        new_signal=0,new_compile=0,approved=False,runtime_authorization=False)
    print(json.dumps(public,indent=2))


if __name__=='__main__':main()
