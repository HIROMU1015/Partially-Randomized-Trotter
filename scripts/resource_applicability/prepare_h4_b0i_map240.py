#!/usr/bin/env python3
"""Byte-only full-map preparation; all runtime authorization remains false."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import signal
import subprocess
import sys

ROOT=Path(__file__).absolute().parents[2];sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS,(512*2**20,512*2**20));resource.setrlimit(resource.RLIMIT_CPU,(90,90));signal.alarm(120)
    from trottertracks.resource_applicability.h4_b0i import adapter as science
    from trottertracks.resource_applicability.h4_b0i_map import contract as c,runtime as runtime
    from trottertracks.resource_applicability.h4_geometry.identity import fingerprint
    from trottertracks.resource_applicability.h4_geometry.prelaunch_audit import private_path,streaming_sha,host_readonly
    from trottertracks.resource_applicability.h4_geometry.prelaunch_audit import environment_profile,compiler_profile
    def load(path):return json.loads(path.read_text())
    def stamp(path):
        raw=path.read_bytes();return dict(path=str(path.relative_to(ROOT)),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    def put(name,value):
        with (out/name).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    out=private_path(args.output);out.mkdir(exist_ok=False)
    pilot=ROOT/'artifacts/resource_applicability/track_a_h4_b0i_pilot6/2026-10-14'
    prior=load(pilot/'plan.json');measured=load(pilot/'compiled_wrapper_metrics.json')
    native=load(pilot/'native_resource_receipt.json')
    assert native['budget']['cumulative_actual_consumed_or_reserved']==c.CARRY['actual_invocations']
    assert native['budget']['charge_at_receipt_bytes']==c.CARRY['charged_bytes']
    assert native['budget']['recorded_wall_seconds']==c.CARRY['recorded_wall_seconds']
    reuse={c.key(t):(t,next(r for r in measured['results'] if r['task_id']==t['task_id'])) for t in prior['tasks']}
    scalar=ROOT/'artifacts/resource_applicability/track_a_h4_geometry_scalar_diagnostic/2026-10-13'
    receipts=load(scalar/'source_input_provenance_v1.json')['inputs'];phases=load(scalar/'saved_b0_phase_sensitivity_v1.json')
    phase_by_key={(v['geometry'],v['prefix'],int(v['original_template_id'].split('-q')[1].split('-')[0])):v for v in phases}
    input_root=Path(prior['tasks'][0]['input_file']['path']).parent
    freeze_path=input_root/'generation-freeze.json';freeze_stamp=streaming_sha(freeze_path)
    assert freeze_stamp==dict(bytes=27115,sha256='75d7ddc8dc71ebeec03a6c173397a9b941b492b74e4dc80814d613d83ce56c69')
    freeze=load(freeze_path);input_by_geometry={};array_metadata={}
    for row in receipts:
        g=row['geometry'];path=input_root/row['file'];actual=streaming_sha(path)
        assert actual==dict(bytes=row['bytes'],sha256=row['sha256'])
        identity=freeze['identities'][g];ids=identity['arrays'];expected=row['input_identity']
        recomputed=dict(input=fingerprint('h4-input-freeze-v1',identity),H=fingerprint('h4-H-v1',ids['H']),
            DF=fingerprint('h4-DF-v1',{k:ids[k] for k in ('lambdas','G','correction','permutation','raw_indices','raw_eigenvalues','raw_eigenvectors','signs')}),
            state=fingerprint('h4-state-v1',{k:ids[k] for k in ('sector','sector_state','state','qiskit_state')}))
        assert all(expected[k]==v for k,v in recomputed.items())
        input_by_geometry[g]=dict(input_file=dict(path=str(path),**actual),input_identity=expected)
        array_metadata[g]=dict(freeze_array_identities=ids,canonical_recomputed=recomputed,generation_source=identity['source_commit'])
    assert set(input_by_geometry)==set(c.GEOMETRIES)
    put('frozen_inputs_byte_binding.json',dict(generation_freeze=freeze_stamp,inputs=input_by_geometry,array_metadata=array_metadata,scientific_arrays_decoded=False))
    tasks=[]
    for g in c.GEOMETRIES:
        for p in c.PREFIXES:
            for q in c.QS:
                phase=phase_by_key[g,p,q]
                for axis in ('cosine','sine'):
                    reuse_key=(g,p,q,axis);old=reuse.get(reuse_key)
                    t=dict(geometry=g,prefix=p,axis=axis,template=science.template(p,q),**input_by_geometry[g],
                        source_commit=args.source,original_science_source=science.ORIGINAL_SOURCE,
                        tail_identity=phase['tail_identity'],phase_angle=phase['phase_angle'],
                        original_B0_template=phase['original_template_id'],saved_bias_B0I=phase['counterfactual_bias'],
                        normalization=1.,saved_phase_rotated_signal=phase['scalar_phase_counterfactual'],
                        saved_target=phase['target'],trajectory_seed=None,trajectory_index=None,
                        numeric_circuit_fingerprint=None,compiled_cost='UNMEASURED_IN_NEW_MAP',
                        action='VERIFY_REUSE' if old else 'COMPILE',reuse_receipt=None)
                    if old:
                        t['reuse_receipt']=dict(old_task_id=old[0]['task_id'],old_source=c.PILOT_SOURCE,
                            old_result_commit=c.PILOT_RESULT_COMMIT,expected_numeric_fingerprint=old[1]['numeric_circuit_fingerprint'],
                            old_namespace=prior['namespace'],new_namespace=c.NAMESPACE,
                            acceptance='fresh build must reproduce the exact old numeric fingerprint; otherwise STOP without compile fallback')
                        c.verify_reuse_match(t,*old)
                    t['task_id']=c.digest(t);tasks.append(t)
    env_path=prior['environment_profile'];compiler_path=prior['compiler_profile']
    env=load(ROOT/env_path);comp=load(ROOT/compiler_path)
    observed_env=environment_profile(env['dependencies'],[Path(p) for p in env['installed_sources']]);observed_comp=compiler_profile(comp['explicit_options'],comp['inherited_defaults'])
    assert observed_env==env and observed_comp==comp
    put('profile_audit.json',dict(environment_matches=True,compiler_matches=True,environment_fingerprint=env['fingerprint'],compiler_fingerprint=comp['fingerprint'],old_compiler_output_equivalence=False,shared_or_venv_changes=0))
    old_closure=prior['source_closure']
    new_paths=[*sorted((ROOT/'src/trottertracks/resource_applicability/h4_b0i_map').glob('*.py')),
        ROOT/'tests/tracks/resource_applicability/test_h4_b0i_map.py',
        *[ROOT/'scripts/resource_applicability'/n for n in ('run_h4_b0i_map240.py','prepare_h4_b0i_map240.py','run_h4_b0i_map_tests.py')]]
    closure=old_closure+[stamp(p) for p in new_paths]
    for item in closure:
        path=ROOT/item['path'];assert stamp(path)==item
        raw=subprocess.check_output(['git','show',args.source+':'+item['path']],cwd=ROOT)
        assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
    public=[ROOT/item['path'] for item in prior['bound_public_files']]
    public += [pilot/n for n in ('plan.json','authorization.json','review.json','binding.json','compiled_wrapper_metrics.json','native_resource_receipt.json','result_artifact_manifest.json')]
    public += [out/'frozen_inputs_byte_binding.json']
    root=private_path('/home/AbeHiromu/projects/h4-handoff-evidence/20261014/h4-b0i-map-preparation/future-map240');root.mkdir(exist_ok=True)
    plan=dict(schema_version='h4-b0i-map240-plan-v1',namespace=c.NAMESPACE,method=science.METHOD,semantics=science.SEMANTICS,
        source_commit=args.source,original_science_source=science.ORIGINAL_SOURCE,tasks=tasks,
        source_closure=closure,bound_public_files=[stamp(path) for path in sorted(set(public))],
        environment_profile=env_path,compiler_profile=compiler_path,compiler_options=comp['explicit_options'],
        frozen_input_binding=str((out/'frozen_inputs_byte_binding.json').relative_to(ROOT)),
        pilot6_reuse_proof=dict(plan_path=str((pilot/'plan.json').relative_to(ROOT)),metrics_path=str((pilot/'compiled_wrapper_metrics.json').relative_to(ROOT)),native_receipt_path=str((pilot/'native_resource_receipt.json').relative_to(ROOT)),
            original_source=c.PILOT_SOURCE,original_result_commit=c.PILOT_RESULT_COMMIT,unchanged_pilot_source_files=len(old_closure),numeric_FP_check='required at runtime for each of six reuse tasks'),
        caps=c.CAPS,carry=c.CARRY,conservative_prior_stage_charge_bytes=native['budget']['conservative_charge_with_full_stage_cap'],
        output=str(root/c.NAMESPACE),control=str(root/(c.NAMESPACE+'-control')),one_shot_lock=str(root/(c.NAMESPACE+'.lock')),
        approved=False,runtime_authorization=False,allowed_cpus=[],absolute_launch_command=None,
        proposed_role_cpus=dict(worker=0,driver=1,observer=2,permission=False),**c.FLAGS)
    c.validate_plan(plan);runtime.source_gate(ROOT,plan);mapped=runtime.reuse_gate(ROOT,plan)
    auth=dict(schema_version='h4-b0i-map240-authorization-v1',namespace=c.NAMESPACE,source_commit=args.source,
        plan_digest=c.digest(plan),approved=False,runtime_authorization=False,allowed_cpus=[],role_cpus={},absolute_launch_command=None,user_approval_evidence_sha256=None,**c.FLAGS)
    review=dict(schema_version='h4-b0i-map240-review-v1',approved=False,plan_digest=c.digest(plan),authorization_digest=c.digest(auth),scientific_scope='complete adopted comparison domain; final scope/claim judgment remains separate',launch_instruction_received=False)
    put('plan.json',plan);put('authorization_draft.json',auth);put('review_draft.json',review)
    put('binding.json',dict(source_commit=args.source,plan_digest=c.digest(plan),authorization_digest=c.digest(auth),review_digest=c.digest(review),technical_bound=True,runtime_sealed=False))
    put('reuse_mapping.json',dict(source_files_unchanged=len(old_closure),mappings=[dict(new_task_id=t['task_id'],**t['reuse_receipt']) for t in tasks if t['action']=='VERIFY_REUSE'],static_proof_passed=len(mapped),fresh_numeric_match_deferred=True))
    host=host_readonly(root,sample_seconds=1);put('host_readonly.json',host)
    worst_observer=(c.CAPS['stage_wall_seconds']+5)*8192+16384
    storage=dict(task_result_logical_bound=240*8192,reservation_logical_bound=234*1024,
        journal_logical_cap=c.CAPS['journal_bytes'],observer_logical_bound=worst_observer,observer_cap=c.CAPS['observer_output_bytes'],
        task_and_reservation_inodes=474,run_inode_cap=1024,output_cap=c.CAPS['stage_output_bytes'],fresh_required_bytes=c.CAPS['storage_required_bytes'],
        fresh_required_inodes=1024,temporary_library_cache_reserve_bytes=32*2**20,metadata_block_rounding_reserve_bytes=8*2**20,
        terminal_reserve_bytes=64*2**20,new_actual_upper=234,total_actual_upper=73263,total_actual_cap=74805,
        conservative_total_charge_bound=plan['conservative_prior_stage_charge_bytes']+c.CAPS['stage_output_bytes'],
        stage_wall_cap_seconds=14400,task_wall_cap_seconds=600,time_prediction='4h is a proposed STOP bound, not a runtime forecast; larger q/prefix not measured',
        physical_sibling_CPU_sample=[t for t in host['topology'] if t['cpu'] in (0,1,2)])
    assert worst_observer<c.CAPS['observer_output_bytes']
    assert worst_observer+c.CAPS['journal_bytes']+storage['temporary_library_cache_reserve_bytes']+storage['metadata_block_rounding_reserve_bytes']<c.CAPS['stage_output_bytes']
    put('storage_and_budget_projection.json',storage)
    print(json.dumps(dict(status='FULL_MAP_PREPARED_NOT_RUNTIME_AUTHORIZED',source=args.source,logical_wrappers=240,new_compile_calls=234,verified_reuse_mappings=6,source_files=len(closure),frozen_input_bytes=sum(i['input_file']['bytes'] for i in input_by_geometry.values()),stage_wall_proposal_hours=4,output_proposal_MiB=256,observer_output_proposal_MiB=160,inode_proposal=1024,actual_new_compute_calls=0)))


if __name__=='__main__':main()
