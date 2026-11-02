"""Byte/metadata-only materialization; never issues a grant or imports science."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from h6_fixed_delta_plan_v1 import preparation, cells, digest, PROVIDER
from h6_fixed_delta_runner_v1 import ROOT, file_hash, private

FAMILY = Path('/home/AbeHiromu/projects/h6-handoff-evidence/20261015/h6-native-b0i-full-family-run04')
ART = ROOT/'artifacts/resource_applicability/track_a_h6_fixed_delta_readiness_v1/2026-10-15'
SOURCE_PATHS = ['scripts/h6_fixed_delta_plan_v1.py','scripts/h6_fixed_delta_science_v1.py',
    'scripts/h6_fixed_delta_runner_v1.py','scripts/prepare_h6_fixed_delta_v1.py',
    'tests/test_h6_fixed_delta_readiness_v1.py']


def load(path):return json.loads(Path(path).read_bytes())


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as handle:
        json.dump(value,handle,ensure_ascii=False,allow_nan=False,indent=2);handle.write('\n')


def receipt(name,path):
    p=private(path)
    return dict(logical_name=name,path=str(p),bytes=p.stat().st_size,sha256=file_hash(p))


def main():
    p=argparse.ArgumentParser();p.add_argument('--SOURCE',required=True);p.add_argument('--private-directory',required=True)
    a=p.parse_args();directory=private(a.private_directory)
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()!=a.SOURCE:
        raise ValueError('FIXED_HEAD_REQUIRED')
    directory.mkdir();ART.mkdir(parents=True,exist_ok=True)
    old=load(FAMILY/'final_contract_v1.json');mapping=load(FAMILY/'input_mapping_private.json')
    provider=private(old['native_provider']['checkout'])
    ref=load(provider/'artifacts/resource_applicability/track_a_h6_confirmation_preparation_v1/2026-10-12/sealed_v1/manifest_confirmation_v1.json')
    archive_ref=ref['roundoff_preparation_binding'];archive_path=private(archive_ref['archive_path'])
    # JSON metadata is inspected, but no codec decoding or array load is done.
    archive=load(archive_path)
    if file_hash(archive_path)!=archive_ref['archive_sha256']:raise ValueError('ARCHIVE_SHA')
    scalar_path=FAMILY/'run01/case_15_4_screen.json';scalar=load(scalar_path)['scalar']
    files=[]
    for name in ('h6_input_snapshot.npz','snapshot_receipt.json','df_receipt.json','signal_summary.json'):
        r=receipt(name,mapping[name]);expected=next(x for x in old['input_receipts'] if x['logical_name']==name)
        if (r['bytes'],r['sha256'])!=(expected['bytes'],expected['sha256']):raise ValueError('INPUT_BYTES_SHA')
        files.append(r)
    pilot_path=Path('/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h6-conditional-s2-pilot-preparation-20261013/artifacts/resource_applicability/track_a_h6_conditional_s2_pilot_result_v1/2026-10-13/aggregate_calls.json')
    pilot=load(pilot_path)
    files.extend([receipt('old_preparation_archive',archive_path),receipt('tail_scalar_receipt',scalar_path),
        receipt('completed_s2_pilot_cumulative_calls',pilot_path),
        receipt('carry_history',FAMILY/'carry_binding_evidence_v1.json'),receipt('completed_family_terminal',FAMILY/'run01/terminal_v1.json')])
    fonts=old['startup_font_cache'];files.append(receipt('font_seed',fonts['path']))
    profile=old['profile'];hashes={name:file_hash(ROOT/name) for name in SOURCE_PATHS}
    config=dict(kind=preparation()['kind'],plan=preparation(),SOURCE=a.SOURCE,
        source_hashes=hashes,provider_source=PROVIDER,provider_root=str(provider),
        provider_hashes=old['native_provider']['source_hashes'],python=profile['executable'],
        runtime_file_hashes=profile['runtime_file_hashes'],input_mapping={k:mapping[k] for k in (
            'h6_input_snapshot.npz','snapshot_receipt.json','df_receipt.json')},files=files,
        archive=dict(path=str(archive_path),sha256=archive_ref['archive_sha256'],binding=archive['binding']),
        tail_scalar=dict(C=scalar['C'],C_error=scalar['scalar_error_upper']),
        font_seed=dict(path=fonts['path'],name=fonts['filename'],sha256=fonts['sha256']),
        compiler_hash=old['binding']['compiler'],profile_reference=profile['full_critical_hashes_reference'],
        carry_policy='NO_RESET_NO_REFUND_UNKNOWN_RETAINED',carry=dict(
            s2_pilot_aggregate=pilot,native_b0i_history=deepcopy(old['carry']),
            completed_family_calls=load(FAMILY/'run01/full_family_results_v1.json')['calls'],
            native_compile_known_entered_or_completed=24,native_compile_unknown_held=6,
            conservative_compile_carry=1766,conservative_trajectory_carry=852,
            unknown_historical_actual=None,refund=0,reset=False),
        cumulative_caps=dict(compile=1910,trajectory=916,sampling_calls=3584),
        completed_family_calls=load(FAMILY/'run01/full_family_results_v1.json')['calls'],
        resources=dict(worker_cpus=[20,21,24,25,26,27,28,29],driver_cpu=22),
        output=str(directory/'output01'),control=str(directory/'control01'),
        approved=False,runtime_authorization=False,allowed_cpus=[],absolute_launch_command=None)
    write(directory/'config.json',config)
    # Saved T=.8 panel: retain old exploration and confirmation as distinct rows.
    saved_path=ROOT/'artifacts/resource_applicability/track_a_d0_saved_value_reanalysis_result_v1/2026-10-14/candidate_accounting_v1.json'
    saved=load(saved_path);ids={x['id'] for x in cells(4) if x['method']!='B0I'}
    selected={k:[x for x in v if x['cell_id'] in ids] for k,v in saved.items() if k.startswith('H6_')}
    scalar_record=load(scalar_path)
    selected['H6_B0I_p15_q4']=scalar_record
    write(ART/'T08_saved_panel_v1.json',selected)
    write(ART/'execution_plan_draft_v1.json',preparation())
    write(ART/'metadata_binding_v1.json',dict(SOURCE=a.SOURCE,source_hashes=hashes,
        provider_source=PROVIDER,provider_closure_digest=digest(config['provider_hashes']),
        environment_profile=profile['full_critical_hashes_reference'],
        compiler_hash=config['compiler_hash'],input_identity=old['binding'],
        received_files=[{k:v for k,v in r.items() if k!='path'} for r in files],
        private_config=dict(bytes=(directory/'config.json').stat().st_size,sha256=file_hash(directory/'config.json'),digest=digest(config)),
        saved_panel_origin=dict(commit='8dbdb68293c988a999645e21a82b53966f9ef6cd',
            path=str(saved_path.relative_to(ROOT)),sha256=file_hash(saved_path)),
        new_science_operations=0,NPZ_scientific_array_loads=0,new_grants=0,
        subagents=0,independent_review=False,approved=False,runtime_authorization=False,
        allowed_cpus=[],absolute_launch_command=None))
    print(json.dumps(dict(private_config=str(directory/'config.json'),public_entry=str(ART),SOURCE=a.SOURCE)))


if __name__=='__main__':main()
